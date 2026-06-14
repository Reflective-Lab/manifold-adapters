// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

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

pub struct StaikBackend {
    api_key: SecretString,
    model: String,
    base_url: String,
    client: Client,
    temperature: f32,
    max_retries: usize,
}

impl StaikBackend {
    /// REAL-by-default constructor. Rejects empty / whitespace keys so that
    /// missing or placeholder credentials surface immediately at construction.
    /// Production code should prefer [`Self::from_env`].
    pub fn try_new(api_key: impl Into<String>) -> BackendResult<Self> {
        let api_key: String = api_key.into();
        if api_key.trim().is_empty() {
            return Err(BackendError::Unavailable {
                message: "STAIK_API_KEY is empty or whitespace".to_string(),
            });
        }
        Ok(Self {
            api_key: SecretString::new(api_key),
            model: "gemma4:31b".to_string(),
            base_url: "https://api.staik.se".to_string(),
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
                .get_secret("STAIK_API_KEY")
                .map_err(|e| BackendError::Unavailable {
                    message: format!("STAIK_API_KEY: {e}"),
                })?;
        Ok(Self {
            api_key,
            model: "gemma4:31b".to_string(),
            base_url: "https://api.staik.se".to_string(),
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

    fn build_request(&self, req: &ChatRequest) -> StaikRequest {
        let model = req.model.clone().unwrap_or_else(|| self.model.clone());
        let temperature = req.temperature.unwrap_or(self.temperature);
        let max_tokens = req.max_tokens.map(|t| t as usize).unwrap_or(4096);

        let mut messages = Vec::new();

        let system_content = if let Some(instruction) = req.response_format.system_instruction() {
            let base = req.system.clone().unwrap_or_default();
            Some(format!("{base}\n\n{instruction}"))
        } else {
            req.system.clone()
        };

        if let Some(system) = &system_content {
            messages.push(StaikMessage {
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
                        .map(|tc| StaikResponseToolCall {
                            id: tc.id.clone(),
                            r#type: "function".to_string(),
                            function: StaikResponseFunction {
                                name: tc.name.clone(),
                                arguments: tc.arguments.clone(),
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
            messages.push(StaikMessage {
                role: role.to_string(),
                content,
                tool_calls,
                tool_call_id: msg.tool_call_id.clone(),
            });
        }

        let tools: Option<Vec<StaikTool>> = if req.tools.is_empty() {
            None
        } else {
            Some(
                req.tools
                    .iter()
                    .map(|t| StaikTool {
                        r#type: "function".to_string(),
                        function: StaikFunction {
                            name: t.name.clone(),
                            description: Some(t.description.clone()),
                            parameters: Some(t.parameters.clone()),
                        },
                    })
                    .collect(),
            )
        };

        let response_format = match req.response_format {
            ResponseFormat::Json => Some(serde_json::json!({"type": "json_object"})),
            _ => None,
        };

        let stop = if req.stop_sequences.is_empty() {
            None
        } else {
            Some(req.stop_sequences.clone())
        };

        StaikRequest {
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
        let staik_req = self.build_request(&req);
        let model = req.model.clone().unwrap_or_else(|| self.model.clone());
        let response = self.execute_with_retries(&model, &staik_req).await?;

        let choice = response.choices.first();

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
                metadata: Default::default(),
            },
        )
    }

    async fn execute_with_retries(
        &self,
        model: &str,
        request: &StaikRequest,
    ) -> Result<StaikResponse, ChatLlmError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
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
                            match response.json::<StaikResponse>().await {
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

impl ChatBackend for StaikBackend {
    type ChatFut<'a>
        = BoxFuture<'a, Result<ChatResponse, ChatLlmError>>
    where
        Self: 'a;

    fn chat(&self, req: ChatRequest) -> Self::ChatFut<'_> {
        Box::pin(async move { self.chat_async(req).await })
    }
}

// ============================================================================
// Staik API Types (OpenAI-compatible)
// ============================================================================

#[derive(Debug, Serialize)]
struct StaikRequest {
    model: String,
    messages: Vec<StaikMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<StaikTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct StaikMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<StaikResponseToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct StaikTool {
    r#type: String,
    function: StaikFunction,
}

#[derive(Debug, Serialize)]
struct StaikFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct StaikResponse {
    model: String,
    choices: Vec<StaikChoice>,
    usage: Option<StaikUsage>,
}

#[derive(Debug, Deserialize)]
struct StaikChoice {
    message: StaikResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StaikResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<StaikResponseToolCall>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct StaikResponseToolCall {
    id: String,
    /// OpenAI-compatible APIs require `"type": "function"` on tool_calls in
    /// outgoing assistant messages. Without it, upstream routers translating
    /// to Anthropic-native format silently drop the tool_call. See
    /// `openrouter.rs` for the documented diagnosis (2026-05).
    #[serde(rename = "type", default = "default_function_type")]
    r#type: String,
    function: StaikResponseFunction,
}

fn default_function_type() -> String {
    "function".to_string()
}

#[derive(Debug, Serialize, Deserialize)]
struct StaikResponseFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct StaikUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
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
    fn test_staik_backend_creation() {
        let backend = StaikBackend::try_new("sk-st-test")
            .unwrap()
            .with_model("qwen3.5:9b")
            .with_temperature(0.7);

        assert_eq!(backend.model, "qwen3.5:9b");
        assert_eq!(backend.temperature, 0.7);
        assert_eq!(backend.api_key.expose(), "sk-st-test");
    }

    #[test]
    fn test_build_request_default_model() {
        let backend = StaikBackend::try_new("sk-st-test").unwrap();
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

        let staik_req = backend.build_request(&req);
        assert_eq!(staik_req.model, "gemma4:31b");
        assert_eq!(staik_req.messages.len(), 1);
        assert_eq!(staik_req.messages[0].role, "user");
    }

    #[test]
    fn test_chat_happy_path() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = runtime.block_on(MockServer::start());

        runtime.block_on(async {
            Mock::given(method("POST"))
                .and(path("/v1/chat/completions"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "model": "gemma4:31b",
                    "choices": [{
                        "message": {"content": "Hej!", "tool_calls": null},
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 7
                    }
                })))
                .mount(&server)
                .await;
        });

        let backend = StaikBackend::try_new("sk-st-test")
            .unwrap()
            .with_model("gemma4:31b");
        // Override base_url for test
        let mut backend = backend;
        backend.base_url = server.uri();

        let response = runtime
            .block_on(backend.chat(ChatRequest {
                messages: vec![ChatMessage {
                    role: ChatRole::User,
                    content: "Hej!".to_string(),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                }],
                system: None,
                tools: Vec::new(),
                response_format: ResponseFormat::Text,
                max_tokens: Some(16),
                temperature: Some(0.0),
                stop_sequences: Vec::new(),
                model: None,
            }))
            .unwrap();

        assert_eq!(response.content, "Hej!");
        assert_eq!(response.finish_reason, Some(ChatFinishReason::Stop));
    }
}
