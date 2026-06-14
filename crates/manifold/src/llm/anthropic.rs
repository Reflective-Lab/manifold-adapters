// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

use reqwest::Client;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::error_classification::{
    classify_http_error, map_backend_error, network_error, parse_error,
};
use super::format_contract::finalize_chat_response;
use super::retry::{RetryOutcome, retry_with_backoff};
use crate::secret::{SecretProvider, SecretString, default_secret_provider};
use converge_core::backend::{BackendError, BackendResult};
use converge_provider::{
    BoxFuture, ChatBackend, ChatRequest, ChatResponse, ChatRole, FinishReason as ChatFinishReason,
    LlmError as ChatLlmError, TokenUsage as ChatTokenUsage, ToolCall,
};

pub struct AnthropicBackend {
    api_key: SecretString,
    model: String,
    base_url: String,
    client: Client,
    temperature: f32,
    top_p: f32,
    max_retries: usize,
}

impl AnthropicBackend {
    /// REAL-by-default constructor. Rejects empty / whitespace keys so that
    /// missing or placeholder credentials surface immediately at construction.
    /// Production code should prefer [`Self::from_env`].
    pub fn try_new(api_key: impl Into<String>) -> BackendResult<Self> {
        let api_key: String = api_key.into();
        if api_key.trim().is_empty() {
            return Err(BackendError::Unavailable {
                message: "ANTHROPIC_API_KEY is empty or whitespace".to_string(),
            });
        }
        Ok(Self {
            api_key: SecretString::new(api_key),
            model: "claude-sonnet-4-6".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
            client: Client::new(),
            temperature: 0.0,
            top_p: 1.0,
            max_retries: 3,
        })
    }

    pub fn from_env() -> BackendResult<Self> {
        Self::from_secret_provider(default_secret_provider())
    }

    pub fn from_secret_provider(secrets: &dyn SecretProvider) -> BackendResult<Self> {
        let api_key =
            secrets
                .get_secret("ANTHROPIC_API_KEY")
                .map_err(|e| BackendError::Unavailable {
                    message: format!("ANTHROPIC_API_KEY: {e}"),
                })?;
        Ok(Self {
            api_key,
            model: "claude-sonnet-4-6".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
            client: Client::new(),
            temperature: 0.0,
            top_p: 1.0,
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
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
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
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(self.api_key.expose()).map_err(|e| {
                BackendError::InvalidRequest {
                    message: format!("Invalid API key: {e}"),
                }
            })?,
        );
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        Ok(headers)
    }

    fn convert_chat_request(
        &self,
        req: &ChatRequest,
    ) -> (
        Option<String>,
        Vec<AnthropicMessage>,
        Option<Vec<AnthropicTool>>,
    ) {
        let mut system = req.system.clone();
        let mut messages: Vec<AnthropicMessage> = Vec::new();

        for msg in &req.messages {
            match msg.role {
                ChatRole::System => system = Some(msg.content.clone()),
                ChatRole::User => {
                    messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicMessageContent::Text(msg.content.clone()),
                    });
                }
                ChatRole::Assistant => {
                    if !msg.tool_calls.is_empty() {
                        let mut blocks = Vec::new();
                        if !msg.content.is_empty() {
                            blocks.push(AnthropicContentBlock::Text {
                                text: msg.content.clone(),
                            });
                        }
                        for tool_call in &msg.tool_calls {
                            blocks.push(AnthropicContentBlock::ToolUse {
                                id: tool_call.id.clone(),
                                name: tool_call.name.clone(),
                                input: parse_tool_call_arguments(&tool_call.arguments),
                            });
                        }
                        messages.push(AnthropicMessage {
                            role: "assistant".to_string(),
                            content: AnthropicMessageContent::Blocks(blocks),
                        });
                    } else {
                        messages.push(AnthropicMessage {
                            role: "assistant".to_string(),
                            content: AnthropicMessageContent::Text(msg.content.clone()),
                        });
                    }
                }
                ChatRole::Tool => {
                    let tool_call_id = msg
                        .tool_call_id
                        .clone()
                        .unwrap_or_else(|| "unknown".to_string());
                    messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicMessageContent::Blocks(vec![
                            AnthropicContentBlock::ToolResult {
                                tool_use_id: tool_call_id,
                                content: msg.content.clone(),
                            },
                        ]),
                    });
                }
            }
        }

        let tools = if req.tools.is_empty() {
            None
        } else {
            Some(
                req.tools
                    .iter()
                    .map(|t| AnthropicTool {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        input_schema: t.parameters.clone(),
                    })
                    .collect(),
            )
        };

        (system, messages, tools)
    }

    fn sampling_params(&self, req: &ChatRequest) -> (Option<f32>, Option<f32>) {
        if let Some(temperature) = req.temperature {
            return (Some(temperature), None);
        }
        if self.top_p != 1.0 && self.temperature == 0.0 {
            return (None, Some(self.top_p));
        }
        (Some(self.temperature), None)
    }

    async fn chat_async(&self, req: ChatRequest) -> Result<ChatResponse, ChatLlmError> {
        let (system, messages, tools) = self.convert_chat_request(&req);
        let model = req.model.clone().unwrap_or_else(|| self.model.clone());
        let max_tokens = req.max_tokens.map(|t| t as usize).unwrap_or(4096);
        let (temperature, top_p) = self.sampling_params(&req);
        let stop_sequences = if req.stop_sequences.is_empty() {
            None
        } else {
            Some(req.stop_sequences.clone())
        };

        // Anthropic doesn't have a structured output mode — prepend a system instruction.
        let system = if let Some(instruction) = req.response_format.system_instruction() {
            let base = system.unwrap_or_default();
            Some(format!("{base}\n\n{instruction}"))
        } else {
            system
        };

        let anthropic_req = AnthropicRequest {
            model,
            max_tokens,
            temperature,
            top_p,
            system,
            messages,
            tools,
            stop_sequences,
        };

        let response = self.execute_with_retries(&anthropic_req).await?;

        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();

        for block in &response.content {
            match block {
                AnthropicContentBlock::Text { text } => text_parts.push(text.as_str()),
                AnthropicContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: serde_json::to_string(input).unwrap_or_default(),
                    });
                }
                AnthropicContentBlock::ToolResult { .. } => {}
            }
        }

        let finish_reason = match response.stop_reason.as_deref() {
            Some("end_turn" | "stop_sequence") => Some(ChatFinishReason::Stop),
            Some("max_tokens") => Some(ChatFinishReason::Length),
            Some("tool_use") => Some(ChatFinishReason::ToolCalls),
            _ => None,
        };

        finalize_chat_response(
            &req,
            ChatResponse {
                content: text_parts.join(""),
                tool_calls,
                usage: Some(ChatTokenUsage {
                    prompt_tokens: response.usage.input_tokens as u32,
                    completion_tokens: response.usage.output_tokens as u32,
                    total_tokens: (response.usage.input_tokens + response.usage.output_tokens)
                        as u32,
                }),
                model: Some(response.model),
                finish_reason,
                metadata: Default::default(),
            },
        )
    }

    #[allow(dead_code)]
    fn request_fingerprint(&self, request: &AnthropicRequest) -> String {
        let canonical = serde_json::to_string(request).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    #[allow(dead_code)]
    fn response_fingerprint(&self, response: &AnthropicResponse) -> String {
        let canonical = serde_json::to_string(response).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    async fn execute_with_retries(
        &self,
        request: &AnthropicRequest,
    ) -> Result<AnthropicResponse, ChatLlmError> {
        let url = format!("{}/v1/messages", self.base_url);
        let headers = self.build_headers().map_err(map_backend_error)?;
        let model = request.model.clone();

        retry_with_backoff(self.max_retries, || {
            let client = &self.client;
            let url = &url;
            let headers = headers.clone();
            let request = request;
            let model = &model;
            async move {
                match client.post(url).headers(headers).json(request).send().await {
                    Ok(response) => {
                        let status = response.status();
                        if status.is_success() {
                            match response.json::<AnthropicResponse>().await {
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

fn parse_tool_call_arguments(arguments: &str) -> serde_json::Value {
    serde_json::from_str(arguments)
        .unwrap_or_else(|_| serde_json::Value::String(arguments.to_string()))
}

impl ChatBackend for AnthropicBackend {
    type ChatFut<'a>
        = BoxFuture<'a, Result<ChatResponse, ChatLlmError>>
    where
        Self: 'a;

    fn chat(&self, req: ChatRequest) -> Self::ChatFut<'_> {
        Box::pin(async move { self.chat_async(req).await })
    }
}

// ============================================================================
// Anthropic API Types
// ============================================================================

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicMessageContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum AnthropicMessageContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentBlock {
    Text {
        text: String,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Debug, Deserialize, Serialize)]
struct AnthropicResponse {
    id: Option<String>,
    model: String,
    content: Vec<AnthropicContentBlock>,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize, Serialize)]
struct AnthropicUsage {
    input_tokens: usize,
    output_tokens: usize,
}

#[allow(dead_code)]
fn estimate_cost(model: &str, usage: &AnthropicUsage) -> u64 {
    let (input_per_m, output_per_m) = if model.contains("opus") {
        (15_000_000, 75_000_000)
    } else if model.contains("sonnet") {
        (3_000_000, 15_000_000)
    } else if model.contains("haiku") {
        (250_000, 1_250_000)
    } else {
        (3_000_000, 15_000_000)
    };

    let input_cost = (usage.input_tokens as u64 * input_per_m) / 1_000_000;
    let output_cost = (usage.output_tokens as u64 * output_per_m) / 1_000_000;
    input_cost + output_cost
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

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn test_backend_creation() {
        let backend = AnthropicBackend::try_new("test-key")
            .unwrap()
            .with_model("claude-haiku-4-5-20251001")
            .with_temperature(0.5);
        assert_eq!(backend.model, "claude-haiku-4-5-20251001");
        assert_eq!(backend.temperature, 0.5);
        assert_eq!(backend.api_key.expose(), "test-key");
    }

    #[test]
    fn test_convert_text_prompt() {
        let backend = AnthropicBackend::try_new("test-key").unwrap();
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
        let (system, messages, tools) = backend.convert_chat_request(&req);
        assert!(system.is_none());
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert!(tools.is_none());
    }

    #[test]
    fn test_convert_system_from_messages() {
        let backend = AnthropicBackend::try_new("test-key").unwrap();
        let req = ChatRequest {
            messages: vec![
                ChatMessage {
                    role: ChatRole::System,
                    content: "You are helpful.".to_string(),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "Hi".to_string(),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
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
        let (system, messages, _) = backend.convert_chat_request(&req);
        assert_eq!(system, Some("You are helpful.".to_string()));
        assert_eq!(messages.len(), 1);
    }

    #[test]
    fn test_tool_definitions_serialized() {
        let rt = runtime();
        let server = rt.block_on(MockServer::start());

        rt.block_on(async {
            Mock::given(method("POST"))
                .and(path("/v1/messages"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "content": [{"type": "text", "text": "I can help."}],
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 20, "output_tokens": 5}
                })))
                .mount(&server)
                .await;
        });

        let backend = AnthropicBackend::try_new("test-key")
            .unwrap()
            .with_base_url(server.uri());
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
                description: "Get weather for a city".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }),
            }],
            response_format: ResponseFormat::default(),
            max_tokens: Some(128),
            temperature: Some(0.0),
            stop_sequences: Vec::new(),
            model: None,
        };

        rt.block_on(backend.chat(req)).unwrap();

        let requests = rt.block_on(server.received_requests()).unwrap();
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["tools"][0]["name"], "get_weather");
        assert_eq!(body["tools"][0]["description"], "Get weather for a city");
        assert!(body["tools"][0]["input_schema"]["properties"]["city"].is_object());

        drop(server);
        drop(rt);
    }

    #[test]
    fn test_tool_use_response_parsed() {
        let rt = runtime();
        let server = rt.block_on(MockServer::start());

        rt.block_on(async {
            Mock::given(method("POST"))
                .and(path("/v1/messages"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "content": [
                        {"type": "text", "text": "Let me check the weather."},
                        {
                            "type": "tool_use",
                            "id": "toolu_abc123",
                            "name": "get_weather",
                            "input": {"city": "Paris"}
                        }
                    ],
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 30, "output_tokens": 15}
                })))
                .mount(&server)
                .await;
        });

        let backend = AnthropicBackend::try_new("test-key")
            .unwrap()
            .with_base_url(server.uri());
        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "Weather in Paris?".to_string(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }],
            system: None,
            tools: vec![ToolDefinition {
                name: "get_weather".to_string(),
                description: "Get weather".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            }],
            response_format: ResponseFormat::default(),
            max_tokens: Some(256),
            temperature: Some(0.0),
            stop_sequences: Vec::new(),
            model: None,
        };

        let response = rt.block_on(backend.chat(req)).unwrap();
        assert_eq!(response.content, "Let me check the weather.");
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].id, "toolu_abc123");
        assert_eq!(response.tool_calls[0].name, "get_weather");
        assert_eq!(response.tool_calls[0].arguments, r#"{"city":"Paris"}"#);
        assert_eq!(response.finish_reason, Some(ChatFinishReason::ToolCalls));

        drop(server);
        drop(rt);
    }

    #[test]
    fn test_tool_result_roundtrip() {
        let rt = runtime();
        let server = rt.block_on(MockServer::start());

        rt.block_on(async {
            Mock::given(method("POST"))
                .and(path("/v1/messages"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "content": [{"type": "text", "text": "The weather in Paris is 18°C."}],
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 50, "output_tokens": 10}
                })))
                .mount(&server)
                .await;
        });

        let backend = AnthropicBackend::try_new("test-key")
            .unwrap()
            .with_base_url(server.uri());
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
                    tool_calls: vec![ToolCall {
                        id: "toolu_abc123".to_string(),
                        name: "get_weather".to_string(),
                        arguments: r#"{"city":"Paris"}"#.to_string(),
                    }],
                    tool_call_id: None,
                },
                ChatMessage {
                    role: ChatRole::Tool,
                    content: r#"{"temp_c": 18, "condition": "sunny"}"#.to_string(),
                    tool_calls: Vec::new(),
                    tool_call_id: Some("toolu_abc123".to_string()),
                },
            ],
            system: None,
            tools: Vec::new(),
            response_format: ResponseFormat::default(),
            max_tokens: Some(256),
            temperature: Some(0.0),
            stop_sequences: Vec::new(),
            model: None,
        };

        let response = rt.block_on(backend.chat(req)).unwrap();
        assert_eq!(response.content, "The weather in Paris is 18°C.");
        assert_eq!(response.finish_reason, Some(ChatFinishReason::Stop));

        // Verify the assistant tool_use and user tool_result blocks were sent correctly
        let requests = rt.block_on(server.received_requests()).unwrap();
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        let assistant_msg = &body["messages"][1];
        assert_eq!(assistant_msg["role"], "assistant");
        assert_eq!(assistant_msg["content"][0]["type"], "tool_use");
        assert_eq!(assistant_msg["content"][0]["id"], "toolu_abc123");
        let tool_msg = &body["messages"][2];
        assert_eq!(tool_msg["role"], "user");
        assert_eq!(tool_msg["content"][0]["type"], "tool_result");
        assert_eq!(tool_msg["content"][0]["tool_use_id"], "toolu_abc123");

        drop(server);
        drop(rt);
    }

    #[test]
    fn test_json_mode_request() {
        let rt = runtime();
        let server = rt.block_on(MockServer::start());

        rt.block_on(async {
            Mock::given(method("POST"))
                .and(path("/v1/messages"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "content": [{"type": "text", "text": "{\"answer\": 42}"}],
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 15, "output_tokens": 8}
                })))
                .mount(&server)
                .await;
        });

        let backend = AnthropicBackend::try_new("test-key")
            .unwrap()
            .with_base_url(server.uri());
        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "Give me the answer as JSON".to_string(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }],
            system: Some("You are a calculator.".to_string()),
            tools: Vec::new(),
            response_format: ResponseFormat::Json,
            max_tokens: Some(128),
            temperature: Some(0.0),
            stop_sequences: Vec::new(),
            model: None,
        };

        let response = rt.block_on(backend.chat(req)).unwrap();
        assert_eq!(response.content, "{\"answer\": 42}");

        let requests = rt.block_on(server.received_requests()).unwrap();
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        let system = body["system"].as_str().unwrap();
        assert!(system.contains("You are a calculator."));
        assert!(system.contains("valid JSON only"));

        drop(server);
        drop(rt);
    }

    #[test]
    fn test_multiturn_text() {
        let rt = runtime();
        let server = rt.block_on(MockServer::start());

        rt.block_on(async {
            Mock::given(method("POST"))
                .and(path("/v1/messages"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "content": [{"type": "text", "text": "Done."}],
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 9, "output_tokens": 3}
                })))
                .mount(&server)
                .await;
        });

        let backend = AnthropicBackend::try_new("test-key")
            .unwrap()
            .with_base_url(server.uri());
        let response = rt
            .block_on(backend.chat(ChatRequest {
                messages: vec![
                    ChatMessage {
                        role: ChatRole::User,
                        content: "First".to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    },
                    ChatMessage {
                        role: ChatRole::Assistant,
                        content: "Second".to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    },
                    ChatMessage {
                        role: ChatRole::User,
                        content: "Third".to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    },
                ],
                system: Some("You are helpful.".to_string()),
                tools: Vec::new(),
                response_format: ResponseFormat::default(),
                max_tokens: Some(64),
                temperature: Some(0.0),
                stop_sequences: vec!["DONE".to_string()],
                model: None,
            }))
            .unwrap();

        assert_eq!(response.content, "Done.");
        assert_eq!(response.finish_reason, Some(ChatFinishReason::Stop));
        assert!(response.tool_calls.is_empty());

        let requests = rt.block_on(server.received_requests()).unwrap();
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["system"], "You are helpful.");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "First");
        assert_eq!(body["messages"][1]["role"], "assistant");
        assert_eq!(body["messages"][1]["content"], "Second");
        assert_eq!(body["messages"][2]["role"], "user");
        assert_eq!(body["messages"][2]["content"], "Third");
        assert_eq!(body["stop_sequences"][0], "DONE");
        assert!(body.get("tools").is_none());

        drop(server);
        drop(rt);
    }

    #[test]
    fn test_cost_estimation() {
        let usage = AnthropicUsage {
            input_tokens: 1000,
            output_tokens: 500,
        };
        let cost = estimate_cost("claude-sonnet-4-6", &usage);
        assert_eq!(cost, 10500);
    }

    #[test]
    fn test_sampling_params_temperature_wins() {
        let backend = AnthropicBackend::try_new("test-key")
            .unwrap()
            .with_top_p(0.8);
        let request = ChatRequest {
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
            temperature: Some(0.2),
            stop_sequences: Vec::new(),
            model: None,
        };
        let (temperature, top_p) = backend.sampling_params(&request);
        assert_eq!(temperature, Some(0.2));
        assert_eq!(top_p, None);
    }

    #[test]
    fn test_sampling_params_top_p_fallback() {
        let backend = AnthropicBackend::try_new("test-key")
            .unwrap()
            .with_top_p(0.8);
        let request = ChatRequest {
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
        let (temperature, top_p) = backend.sampling_params(&request);
        assert_eq!(temperature, None);
        assert_eq!(top_p, Some(0.8));
    }

    #[test]
    #[ignore = "requires ANTHROPIC_API_KEY"]
    fn test_live_anthropic_tool_call() {
        let rt = runtime();
        let backend = AnthropicBackend::from_env()
            .expect("ANTHROPIC_API_KEY required")
            .with_model("claude-haiku-4-5-20251001");

        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "What is the weather in Stockholm right now? Use the get_weather tool."
                    .to_string(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }],
            system: None,
            tools: vec![ToolDefinition {
                name: "get_weather".to_string(),
                description: "Get current weather for a city".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["city"]
                }),
            }],
            response_format: ResponseFormat::default(),
            max_tokens: Some(256),
            temperature: Some(0.0),
            stop_sequences: Vec::new(),
            model: None,
        };

        let response = rt.block_on(backend.chat(req)).unwrap();
        assert!(
            !response.tool_calls.is_empty(),
            "Expected at least one tool call"
        );
        assert_eq!(response.tool_calls[0].name, "get_weather");
        assert_eq!(response.finish_reason, Some(ChatFinishReason::ToolCalls));

        drop(rt);
    }

    #[test]
    #[ignore = "requires ANTHROPIC_API_KEY"]
    fn test_live_anthropic_tool_result_followup() {
        let rt = runtime();
        let backend = AnthropicBackend::from_env()
            .expect("ANTHROPIC_API_KEY required")
            .with_model("claude-haiku-4-5-20251001");

        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get current weather for a city".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }),
        }];

        let first = rt
            .block_on(
                backend.chat(ChatRequest {
                    messages: vec![ChatMessage {
                        role: ChatRole::User,
                        content:
                            "What is the weather in Stockholm right now? Use the get_weather tool."
                                .to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    }],
                    system: Some(
                        "After using tools, answer in one short sentence with the city name."
                            .to_string(),
                    ),
                    tools: tools.clone(),
                    response_format: ResponseFormat::default(),
                    max_tokens: Some(256),
                    temperature: Some(0.0),
                    stop_sequences: Vec::new(),
                    model: None,
                }),
            )
            .unwrap();

        assert!(
            !first.tool_calls.is_empty(),
            "Expected at least one tool call in the first Anthropic turn"
        );

        let tool_call = &first.tool_calls[0];
        let followup =
            rt.block_on(
                backend.chat(ChatRequest {
                    messages: vec![
                    ChatMessage {
                        role: ChatRole::User,
                        content:
                            "What is the weather in Stockholm right now? Use the get_weather tool."
                                .to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    },
                    ChatMessage {
                        role: ChatRole::Assistant,
                        content: first.content.clone(),
                        tool_calls: first.tool_calls.clone(),
                        tool_call_id: None,
                    },
                    ChatMessage {
                        role: ChatRole::Tool,
                        content:
                            r#"{"city":"Stockholm","temperature_c":7,"condition":"cloudy"}"#
                                .to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: Some(tool_call.id.clone()),
                    },
                ],
                    system: Some(
                        "After using tools, answer in one short sentence with the city name."
                            .to_string(),
                    ),
                    tools,
                    response_format: ResponseFormat::default(),
                    max_tokens: Some(256),
                    temperature: Some(0.0),
                    stop_sequences: Vec::new(),
                    model: None,
                }),
            );

        let followup = followup.unwrap();
        assert!(
            followup.tool_calls.is_empty(),
            "Expected Anthropic tool-result follow-up to complete without another tool call: {:?}",
            followup.tool_calls
        );
        assert!(
            !followup.content.trim().is_empty(),
            "Expected Anthropic to produce a final answer after tool result"
        );
        assert!(
            followup.content.to_ascii_lowercase().contains("stockholm"),
            "Expected final Anthropic answer to mention Stockholm, got: {:?}",
            followup.content
        );

        drop(rt);
    }
}
