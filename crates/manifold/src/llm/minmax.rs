// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

use std::collections::HashMap;

use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};

use super::error_classification::{
    classify_http_error, map_backend_error, network_error, parse_error,
};
use super::format_contract::finalize_chat_response;
use super::retry::{RetryOutcome, retry_with_backoff};
use crate::secret::{SecretProvider, SecretString, default_secret_provider};
use converge_core::backend::{BackendError, BackendResult};
use converge_provider::{
    BoxFuture, ChatBackend, ChatRequest, ChatResponse, ChatRole, FinishReason as ChatFinishReason,
    LlmError as ChatLlmError, ResponseFormat, TokenUsage as ChatTokenUsage,
    ToolCall as ChatToolCall,
};

/// MiniMax backend — chat completions for MiniMax-Text-01 and abab series via
/// MiniMax's international API at `api.minimax.io`.
///
/// MiniMax's API is mostly OpenAI-compatible (model + messages + max_tokens),
/// but uses the path `/v1/text/chatcompletion_v2` and wraps every response in
/// a `base_resp` envelope. HTTP 200 responses can still represent
/// application-level errors (status_code 2049 = invalid api key,
/// 1008 = insufficient balance, etc.); this backend translates those into
/// the appropriate `LlmError` variants.
///
/// The China-region host (`api.minimax.chat`) is also accessible — override
/// `MINMAX_BASE_URL` if you have a domestic-only key.
pub struct MinMaxBackend {
    api_key: SecretString,
    model: String,
    base_url: String,
    client: Client,
    temperature: f32,
    max_retries: usize,
}

impl MinMaxBackend {
    /// REAL-by-default constructor. Rejects empty / whitespace keys so that
    /// missing or placeholder credentials surface immediately at construction.
    /// Production code should prefer [`Self::from_env`].
    pub fn try_new(api_key: impl Into<String>) -> BackendResult<Self> {
        let api_key: String = api_key.into();
        if api_key.trim().is_empty() {
            return Err(BackendError::Unavailable {
                message: "MINIMAX_API_KEY is empty or whitespace".to_string(),
            });
        }
        Ok(Self {
            api_key: SecretString::new(api_key),
            model: "MiniMax-Text-01".to_string(),
            base_url: "https://api.minimax.io".to_string(),
            client: Client::new(),
            temperature: 0.0,
            max_retries: 3,
        })
    }

    pub fn from_env() -> BackendResult<Self> {
        Self::from_secret_provider(default_secret_provider())
    }

    pub fn from_secret_provider(secrets: &dyn SecretProvider) -> BackendResult<Self> {
        let api_key =
            secrets
                .get_secret("MINIMAX_API_KEY")
                .map_err(|e| BackendError::Unavailable {
                    message: format!("MINIMAX_API_KEY: {e}"),
                })?;

        let model = secrets
            .get_secret("MINMAX_MODEL")
            .map(|s| s.expose().to_string())
            .unwrap_or_else(|_| "MiniMax-Text-01".to_string());
        let base_url = secrets
            .get_secret("MINMAX_BASE_URL")
            .map(|s| s.expose().to_string())
            .unwrap_or_else(|_| "https://api.minimax.io".to_string());

        Ok(Self {
            api_key,
            model,
            base_url,
            client: Client::new(),
            temperature: 0.0,
            max_retries: 3,
        })
    }

    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    #[must_use]
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    #[must_use]
    pub fn with_max_retries(mut self, retries: usize) -> Self {
        self.max_retries = retries;
        self
    }

    fn build_headers(&self) -> BackendResult<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let auth_value = format!("Bearer {}", self.api_key.expose());
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&auth_value).map_err(|e| BackendError::InvalidRequest {
                message: format!("Invalid API key: {e}"),
            })?,
        );

        Ok(headers)
    }

    fn build_request(&self, req: &ChatRequest) -> MinMaxRequest {
        let model = req.model.clone().unwrap_or_else(|| self.model.clone());
        let temperature = req.temperature.unwrap_or(self.temperature);
        let max_tokens = req.max_tokens.map(|t| t as usize).unwrap_or(4096);

        let mut messages = Vec::new();

        // Append format instruction to system prompt for all structured formats.
        // JSON also gets native response_format, but OpenAI-compatible APIs require
        // the word "json" in the messages when using json_object mode.
        let system_content = if let Some(instruction) = req.response_format.system_instruction() {
            let base = req.system.clone().unwrap_or_default();
            Some(format!("{base}\n\n{instruction}"))
        } else {
            req.system.clone()
        };

        if let Some(system) = &system_content {
            messages.push(MinMaxMessage {
                role: "system".to_string(),
                content: Some(system.clone()),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        for msg in &req.messages {
            let role = match msg.role {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
                ChatRole::Tool => "tool",
            };
            let tool_calls = if msg.tool_calls.is_empty() {
                None
            } else {
                Some(
                    msg.tool_calls
                        .iter()
                        .map(|tool_call| MinMaxResponseToolCall {
                            id: tool_call.id.clone(),
                            r#type: "function".to_string(),
                            function: MinMaxResponseFunction {
                                name: tool_call.name.clone(),
                                arguments: tool_call.arguments.clone(),
                            },
                        })
                        .collect(),
                )
            };
            let content = if msg.content.is_empty() && tool_calls.is_some() {
                None
            } else {
                Some(msg.content.clone())
            };
            messages.push(MinMaxMessage {
                role: role.to_string(),
                content,
                tool_calls,
                tool_call_id: msg.tool_call_id.clone(),
            });
        }

        let tools: Option<Vec<MinMaxTool>> = if req.tools.is_empty() {
            None
        } else {
            Some(
                req.tools
                    .iter()
                    .map(|t| MinMaxTool {
                        r#type: "function".to_string(),
                        function: MinMaxFunction {
                            name: t.name.clone(),
                            description: Some(t.description.clone()),
                            parameters: Some(t.parameters.clone()),
                        },
                    })
                    .collect(),
            )
        };

        // Only JSON has native API-level enforcement; other structured formats
        // are handled via the system prompt instruction above.
        let response_format = match req.response_format {
            ResponseFormat::Json => Some(serde_json::json!({"type": "json_object"})),
            _ => None,
        };

        let stop = if req.stop_sequences.is_empty() {
            None
        } else {
            Some(req.stop_sequences.clone())
        };

        MinMaxRequest {
            model,
            messages,
            temperature: Some(temperature),
            max_tokens: Some(max_tokens),
            tools,
            response_format,
            stop,
        }
    }

    async fn chat_async(&self, req: ChatRequest) -> Result<ChatResponse, ChatLlmError> {
        let minmax_req = self.build_request(&req);
        let model = req.model.clone().unwrap_or_else(|| self.model.clone());
        let response = self.execute_with_retries(&model, &minmax_req).await?;

        // MiniMax returns HTTP 200 even for auth / quota / model errors,
        // signalling them via `base_resp.status_code`. Translate non-zero
        // status codes into the appropriate `LlmError` variant.
        if let Some(base) = &response.base_resp
            && base.status_code != 0
        {
            return Err(translate_minmax_status(base, &model));
        }

        let choices = response.choices.as_ref();
        let choice = choices.and_then(|c| c.first());

        let content = choice
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        let tool_calls = choice
            .and_then(|c| c.message.tool_calls.as_ref())
            .map(|calls| {
                calls
                    .iter()
                    .map(|tc| ChatToolCall {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        arguments: tc.function.arguments.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        let finish_reason = choice.and_then(|c| match c.finish_reason.as_deref() {
            Some("stop") => Some(ChatFinishReason::Stop),
            Some("length") => Some(ChatFinishReason::Length),
            Some("tool_calls") => Some(ChatFinishReason::ToolCalls),
            Some("content_filter") => Some(ChatFinishReason::ContentFilter),
            _ => None,
        });

        let metadata = extract_minmax_metadata(&response);

        finalize_chat_response(
            &req,
            ChatResponse {
                content,
                tool_calls,
                usage: response.usage.map(|u| ChatTokenUsage {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                    total_tokens: u.total_tokens,
                }),
                model: Some(response.model),
                finish_reason,
                metadata,
            },
        )
    }

    async fn execute_with_retries(
        &self,
        model: &str,
        request: &MinMaxRequest,
    ) -> Result<MinMaxResponse, ChatLlmError> {
        let url = format!("{}/v1/text/chatcompletion_v2", self.base_url);
        let headers = self.build_headers().map_err(map_backend_error)?;

        retry_with_backoff(self.max_retries, || {
            let client = &self.client;
            let url = &url;
            let headers = headers.clone();
            let request = request;
            async move {
                match client.post(url).headers(headers).json(request).send().await {
                    Ok(response) => {
                        let status = response.status();
                        if status.is_success() {
                            match response.json::<MinMaxResponse>().await {
                                Ok(parsed) => RetryOutcome::Success(parsed),
                                Err(e) => RetryOutcome::Retry(parse_error(e)),
                            }
                        } else if status.as_u16() == 429 || status.as_u16() >= 500 {
                            let body = response.text().await.unwrap_or_default();
                            RetryOutcome::Retry(classify_http_error(status.as_u16(), &body, model))
                        } else {
                            let body = response.text().await.unwrap_or_default();
                            RetryOutcome::Fail(classify_http_error(status.as_u16(), &body, model))
                        }
                    }
                    Err(e) => RetryOutcome::Retry(network_error(e)),
                }
            }
        })
        .await
    }
}

fn extract_minmax_metadata(response: &MinMaxResponse) -> HashMap<String, String> {
    let mut meta = HashMap::new();
    if let Some(base) = &response.base_resp
        && base.status_code == 0
        && !base.status_msg.is_empty()
    {
        meta.insert("minmax.status_msg".to_string(), base.status_msg.clone());
    }
    meta
}

/// Translate a non-zero MiniMax `base_resp.status_code` into the right
/// `LlmError` variant. Codes that match MiniMax's documented error set:
/// 1000 = unknown error, 1001 = timeout, 1002 = rate limit,
/// 1004 = auth failed, 1008 = insufficient balance,
/// 1013 = service internal, 1027 = generated content blocked,
/// 2013 = invalid params, 2049 = invalid api key.
fn translate_minmax_status(base: &MinMaxBaseResp, model: &str) -> ChatLlmError {
    let message = if base.status_msg.is_empty() {
        format!("MiniMax error {}", base.status_code)
    } else {
        format!("MiniMax error {}: {}", base.status_code, base.status_msg)
    };
    match base.status_code {
        1004 | 2049 => ChatLlmError::AuthDenied { message },
        1002 => ChatLlmError::RateLimited {
            retry_after: std::time::Duration::from_secs(1),
            message: Some(message),
        },
        // 2013 is a generic "invalid params" — disambiguate by looking at the
        // message. MiniMax uses "unknown model 'X'" for model-not-found.
        2013 => {
            if base
                .status_msg
                .to_ascii_lowercase()
                .contains("unknown model")
                || base.status_msg.to_ascii_lowercase().contains("model not")
            {
                ChatLlmError::ModelNotFound {
                    model: model.to_string(),
                }
            } else {
                ChatLlmError::InvalidRequest { message }
            }
        }
        1008 => ChatLlmError::InvalidRequest { message }, // insufficient balance
        _ => ChatLlmError::ProviderError {
            message: format!("{message} (model={model})"),
            code: Some(base.status_code.to_string()),
        },
    }
}

impl ChatBackend for MinMaxBackend {
    type ChatFut<'a>
        = BoxFuture<'a, Result<ChatResponse, ChatLlmError>>
    where
        Self: 'a;

    fn chat(&self, req: ChatRequest) -> Self::ChatFut<'_> {
        Box::pin(async move { self.chat_async(req).await })
    }
}

// ============================================================================
// MiniMax API Types (OpenAI-compatible)
// ============================================================================

#[derive(Debug, Serialize)]
struct MinMaxRequest {
    model: String,
    messages: Vec<MinMaxMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<MinMaxTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct MinMaxMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<MinMaxResponseToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct MinMaxTool {
    r#type: String,
    function: MinMaxFunction,
}

#[derive(Debug, Serialize)]
struct MinMaxFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct MinMaxResponse {
    #[serde(default)]
    model: String,
    /// MiniMax returns `null` for `choices` when the call fails at the
    /// application layer (it still uses HTTP 200, hence the Option).
    #[serde(default)]
    choices: Option<Vec<MinMaxChoice>>,
    #[serde(default)]
    usage: Option<MinMaxUsage>,
    /// MiniMax's per-response status envelope. Present on every response.
    #[serde(default)]
    base_resp: Option<MinMaxBaseResp>,
}

#[derive(Debug, Deserialize)]
struct MinMaxBaseResp {
    #[serde(default)]
    status_code: i64,
    #[serde(default)]
    status_msg: String,
}

#[derive(Debug, Deserialize)]
struct MinMaxChoice {
    message: MinMaxResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MinMaxResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<MinMaxResponseToolCall>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MinMaxResponseToolCall {
    id: String,
    /// OpenAI-compatible APIs require `"type": "function"` on tool_calls in
    /// outgoing assistant messages. Without it, upstream routers translating
    /// to Anthropic-native format silently drop the tool_call. See
    /// `openrouter.rs` for the documented diagnosis (2026-05).
    #[serde(rename = "type", default = "default_function_type")]
    r#type: String,
    function: MinMaxResponseFunction,
}

fn default_function_type() -> String {
    "function".to_string()
}

#[derive(Debug, Serialize, Deserialize)]
struct MinMaxResponseFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct MinMaxUsage {
    /// MiniMax doesn't always emit prompt/completion breakdowns — particularly
    /// when the call fails before generation. Make these tolerant.
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use converge_core::traits::{ChatMessage, ChatRequest, ChatRole, ResponseFormat};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn test_minmax_backend_creation() {
        let backend = MinMaxBackend::try_new("test-key")
            .unwrap()
            .with_model("abab7-chat")
            .with_temperature(0.5);

        assert_eq!(backend.model, "abab7-chat");
        assert_eq!(backend.temperature, 0.5);
        assert_eq!(backend.api_key.expose(), "test-key");
        assert_eq!(backend.base_url, "https://api.minimax.io");
    }

    #[test]
    fn test_default_model_is_minimax_text_01() {
        let backend = MinMaxBackend::try_new("test-key").unwrap();
        assert_eq!(backend.model, "MiniMax-Text-01");
    }

    #[test]
    fn test_build_request_basic() {
        let backend = MinMaxBackend::try_new("test-key").unwrap();
        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "Hello".to_string(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }],
            system: None,
            tools: Vec::new(),
            response_format: ResponseFormat::default(),
            max_tokens: None,
            temperature: None,
            stop_sequences: Vec::new(),
            model: None,
        };

        let minmax_req = backend.build_request(&req);

        assert_eq!(minmax_req.model, "MiniMax-Text-01");
        assert_eq!(minmax_req.messages.len(), 1);
        assert_eq!(minmax_req.messages[0].role, "user");
        assert!(minmax_req.tools.is_none());
    }

    #[test]
    fn test_chat_with_mock_server() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = runtime.block_on(MockServer::start());

        runtime.block_on(async {
            Mock::given(method("POST"))
                .and(path("/v1/text/chatcompletion_v2"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "gen-test",
                    "model": "MiniMax-Text-01",
                    "choices": [{
                        "message": {
                            "content": "Hello from MiniMax!",
                            "tool_calls": null
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15
                    },
                    "base_resp": {
                        "status_code": 0,
                        "status_msg": ""
                    }
                })))
                .mount(&server)
                .await;
        });

        let backend = MinMaxBackend::try_new("test-key")
            .unwrap()
            .with_base_url(server.uri());

        let response = runtime
            .block_on(backend.chat(ChatRequest {
                messages: vec![ChatMessage {
                    role: ChatRole::User,
                    content: "Hello".to_string(),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                }],
                system: None,
                tools: Vec::new(),
                response_format: ResponseFormat::default(),
                max_tokens: None,
                temperature: None,
                stop_sequences: Vec::new(),
                model: None,
            }))
            .unwrap();

        assert_eq!(response.content, "Hello from MiniMax!");
        assert_eq!(response.finish_reason, Some(ChatFinishReason::Stop));
        assert_eq!(response.usage.as_ref().unwrap().total_tokens, 15);

        drop(server);
        drop(runtime);
    }
}
