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

pub struct OpenAiBackend {
    api_key: SecretString,
    model: String,
    base_url: String,
    client: Client,
    temperature: f32,
    max_retries: usize,
}

impl OpenAiBackend {
    /// REAL-by-default constructor. Rejects empty / whitespace keys so that
    /// missing or placeholder credentials surface immediately at construction
    /// rather than at first request. Production code should prefer
    /// [`Self::from_env`].
    pub fn try_new(api_key: impl Into<String>) -> BackendResult<Self> {
        let api_key: String = api_key.into();
        if api_key.trim().is_empty() {
            return Err(BackendError::Unavailable {
                message: "OPENAI_API_KEY is empty or whitespace".to_string(),
            });
        }
        Ok(Self {
            api_key: SecretString::new(api_key),
            model: "gpt-4o".to_string(),
            base_url: "https://api.openai.com".to_string(),
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
                .get_secret("OPENAI_API_KEY")
                .map_err(|e| BackendError::Unavailable {
                    message: format!("OPENAI_API_KEY: {e}"),
                })?;
        Ok(Self {
            api_key,
            model: "gpt-4o".to_string(),
            base_url: "https://api.openai.com".to_string(),
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

    fn build_request(&self, req: &ChatRequest) -> OpenAiRequest {
        let model = req.model.clone().unwrap_or_else(|| self.model.clone());
        let temperature = req.temperature.unwrap_or(self.temperature);
        let max_tokens = req.max_tokens.map(|t| t as usize).unwrap_or(4096);

        let mut messages = Vec::new();

        // Append format instruction to system prompt for all structured formats.
        // JSON also gets native response_format, but OpenAI requires the word "json"
        // in the messages when using json_object mode.
        let system_content = if let Some(instruction) = req.response_format.system_instruction() {
            let base = req.system.clone().unwrap_or_default();
            Some(format!("{base}\n\n{instruction}"))
        } else {
            req.system.clone()
        };

        if let Some(system) = &system_content {
            messages.push(OpenAiMessage {
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
                        .map(|tool_call| OpenAiResponseToolCall {
                            id: tool_call.id.clone(),
                            r#type: "function".to_string(),
                            function: OpenAiResponseFunction {
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
            messages.push(OpenAiMessage {
                role: role.to_string(),
                content,
                tool_calls,
                tool_call_id: msg.tool_call_id.clone(),
            });
        }

        let tools: Option<Vec<OpenAiTool>> = if req.tools.is_empty() {
            None
        } else {
            Some(
                req.tools
                    .iter()
                    .map(|t| OpenAiTool {
                        r#type: "function".to_string(),
                        function: OpenAiFunction {
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

        OpenAiRequest {
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
        let openai_req = self.build_request(&req);
        let model = req.model.clone().unwrap_or_else(|| self.model.clone());
        let response = self.execute_with_retries(&model, &openai_req).await?;

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
        request: &OpenAiRequest,
    ) -> Result<OpenAiResponse, ChatLlmError> {
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
                            match response.json::<OpenAiResponse>().await {
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

impl ChatBackend for OpenAiBackend {
    type ChatFut<'a>
        = BoxFuture<'a, Result<ChatResponse, ChatLlmError>>
    where
        Self: 'a;

    fn chat(&self, req: ChatRequest) -> Self::ChatFut<'_> {
        Box::pin(async move { self.chat_async(req).await })
    }
}

// ============================================================================
// OpenAI API Types
// ============================================================================

#[derive(Debug, Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct OpenAiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiResponseToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenAiTool {
    r#type: String,
    function: OpenAiFunction,
}

#[derive(Debug, Serialize)]
struct OpenAiFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    model: String,
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiResponseToolCall>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAiResponseToolCall {
    id: String,
    /// OpenAI-compatible APIs require `"type": "function"` on tool_calls in
    /// outgoing assistant messages. Without it, upstream routers translating
    /// to Anthropic-native format silently drop the tool_call. See
    /// `openrouter.rs` for the documented diagnosis (2026-05).
    #[serde(rename = "type", default = "default_function_type")]
    r#type: String,
    function: OpenAiResponseFunction,
}

fn default_function_type() -> String {
    "function".to_string()
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAiResponseFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
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
    use converge_core::traits::{
        ChatMessage, ChatRequest, ChatRole, ResponseFormat, ToolDefinition,
    };
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn test_openai_backend_creation() {
        let backend = OpenAiBackend::try_new("test-key")
            .unwrap()
            .with_model("gpt-4o-mini")
            .with_temperature(0.5);

        assert_eq!(backend.model, "gpt-4o-mini");
        assert_eq!(backend.temperature, 0.5);
        assert_eq!(backend.api_key.expose(), "test-key");
    }

    #[test]
    fn test_build_request_basic() {
        let backend = OpenAiBackend::try_new("test-key").unwrap();
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

        let openai_req = backend.build_request(&req);

        assert_eq!(openai_req.model, "gpt-4o");
        assert_eq!(openai_req.messages.len(), 1);
        assert_eq!(openai_req.messages[0].role, "user");
        assert!(openai_req.tools.is_none());
        assert!(openai_req.response_format.is_none());
    }

    #[test]
    fn test_build_request_with_system() {
        let backend = OpenAiBackend::try_new("test-key").unwrap();
        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "Hi".to_string(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }],
            system: Some("You are helpful.".to_string()),
            tools: Vec::new(),
            response_format: ResponseFormat::default(),
            max_tokens: None,
            temperature: None,
            stop_sequences: Vec::new(),
            model: None,
        };

        let openai_req = backend.build_request(&req);

        assert_eq!(openai_req.messages.len(), 2);
        assert_eq!(openai_req.messages[0].role, "system");
        assert_eq!(
            openai_req.messages[0].content.as_deref(),
            Some("You are helpful.")
        );
        assert_eq!(openai_req.messages[1].role, "user");
    }

    #[test]
    fn test_build_request_with_tools() {
        let backend = OpenAiBackend::try_new("test-key").unwrap();
        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "What's the weather?".to_string(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }],
            system: None,
            tools: vec![ToolDefinition {
                name: "get_weather".to_string(),
                description: "Get current weather".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            }],
            response_format: ResponseFormat::default(),
            max_tokens: None,
            temperature: None,
            stop_sequences: Vec::new(),
            model: None,
        };

        let openai_req = backend.build_request(&req);
        let tools = openai_req.tools.unwrap();

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].r#type, "function");
        assert_eq!(tools[0].function.name, "get_weather");
    }

    #[test]
    fn test_build_request_json_format() {
        let backend = OpenAiBackend::try_new("test-key").unwrap();
        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "Return JSON".to_string(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }],
            system: None,
            tools: Vec::new(),
            response_format: ResponseFormat::Json,
            max_tokens: None,
            temperature: None,
            stop_sequences: Vec::new(),
            model: None,
        };

        let openai_req = backend.build_request(&req);

        assert_eq!(
            openai_req.response_format,
            Some(serde_json::json!({"type": "json_object"}))
        );
    }

    #[test]
    fn test_build_request_with_stop_sequences() {
        let backend = OpenAiBackend::try_new("test-key").unwrap();
        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "Go".to_string(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }],
            system: None,
            tools: Vec::new(),
            response_format: ResponseFormat::default(),
            max_tokens: None,
            temperature: None,
            stop_sequences: vec!["STOP".to_string()],
            model: None,
        };

        let openai_req = backend.build_request(&req);

        assert_eq!(openai_req.stop, Some(vec!["STOP".to_string()]));
    }

    #[test]
    fn test_chat_runtime_multiturn_and_tool_calls() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = runtime.block_on(MockServer::start());

        runtime.block_on(async {
            Mock::given(method("POST"))
                .and(path("/v1/chat/completions"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "chatcmpl_test",
                    "model": "gpt-4o",
                    "choices": [{
                        "message": {
                            "content": "I'll use a tool.",
                            "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup_weather",
                                    "arguments": "{\"city\":\"Paris\"}"
                                }
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 4,
                        "total_tokens": 16
                    }
                })))
                .mount(&server)
                .await;
        });

        let backend = OpenAiBackend::try_new("test-key")
            .unwrap()
            .with_base_url(server.uri());
        let response = runtime
            .block_on(backend.chat(ChatRequest {
                messages: vec![
                    ChatMessage {
                        role: ChatRole::User,
                        content: "Weather?".to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    },
                    ChatMessage {
                        role: ChatRole::Assistant,
                        content: "Let me check.".to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    },
                ],
                system: Some("You are helpful.".to_string()),
                tools: vec![ToolDefinition {
                    name: "lookup_weather".to_string(),
                    description: "Lookup weather".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {"city": {"type": "string"}}
                    }),
                }],
                response_format: ResponseFormat::Json,
                max_tokens: Some(64),
                temperature: Some(0.0),
                stop_sequences: vec!["DONE".to_string()],
                model: None,
            }))
            .unwrap();

        assert_eq!(response.content, "I'll use a tool.");
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].name, "lookup_weather");
        assert_eq!(response.finish_reason, Some(ChatFinishReason::ToolCalls));

        let requests = runtime.block_on(server.received_requests()).unwrap();
        assert_eq!(requests.len(), 1);
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][2]["role"], "assistant");
        assert_eq!(body["tools"][0]["function"]["name"], "lookup_weather");
        assert_eq!(body["response_format"]["type"], "json_object");
        assert_eq!(body["stop"][0], "DONE");

        drop(server);
        drop(runtime);
    }

    #[test]
    fn test_build_request_with_assistant_tool_call_history() {
        let backend = OpenAiBackend::try_new("test-key").unwrap();
        let req = ChatRequest {
            messages: vec![
                ChatMessage {
                    role: ChatRole::User,
                    content: "Weather in Paris?".to_string(),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                },
                ChatMessage {
                    role: ChatRole::Assistant,
                    content: String::new(),
                    tool_calls: vec![ChatToolCall {
                        id: "call_1".to_string(),
                        name: "lookup_weather".to_string(),
                        arguments: r#"{"city":"Paris"}"#.to_string(),
                    }],
                    tool_call_id: None,
                },
                ChatMessage {
                    role: ChatRole::Tool,
                    content: r#"{"temp_c":18}"#.to_string(),
                    tool_calls: Vec::new(),
                    tool_call_id: Some("call_1".to_string()),
                },
            ],
            system: None,
            tools: Vec::new(),
            response_format: ResponseFormat::default(),
            max_tokens: None,
            temperature: None,
            stop_sequences: Vec::new(),
            model: None,
        };

        let request = backend.build_request(&req);
        assert_eq!(request.messages[1].role, "assistant");
        assert!(request.messages[1].content.is_none());
        assert_eq!(
            request.messages[1].tool_calls.as_ref().map(Vec::len),
            Some(1)
        );
        assert_eq!(request.messages[2].role, "tool");
        assert_eq!(request.messages[2].tool_call_id.as_deref(), Some("call_1"));
    }
}
