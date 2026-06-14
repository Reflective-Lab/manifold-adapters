// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

use std::collections::HashMap;

use reqwest::Client;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
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

pub struct KongBackend {
    api_key: SecretString,
    model: String,
    gateway_url: String,
    route: String,
    client: Client,
    temperature: f32,
    max_retries: usize,
}

impl KongBackend {
    /// REAL-by-default constructor. Rejects empty / whitespace keys and
    /// empty gateway URLs so that missing or placeholder configuration
    /// surfaces immediately at construction. Production code should prefer
    /// [`Self::from_env`].
    pub fn try_new(
        api_key: impl Into<String>,
        gateway_url: impl Into<String>,
    ) -> BackendResult<Self> {
        let api_key: String = api_key.into();
        if api_key.trim().is_empty() {
            return Err(BackendError::Unavailable {
                message: "KONG_API_KEY is empty or whitespace".to_string(),
            });
        }
        let gateway_url: String = gateway_url.into();
        if gateway_url.trim().is_empty() {
            return Err(BackendError::Unavailable {
                message: "KONG_GATEWAY_URL is empty or whitespace".to_string(),
            });
        }
        Ok(Self {
            api_key: SecretString::new(api_key),
            model: "gpt-4o".to_string(),
            gateway_url: gateway_url.trim_end_matches('/').to_string(),
            route: "llm/v1/chat".to_string(),
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
                .get_secret("KONG_API_KEY")
                .map_err(|e| BackendError::Unavailable {
                    message: format!("KONG_API_KEY: {e}"),
                })?;

        let gateway_url =
            std::env::var("KONG_AI_GATEWAY_URL").map_err(|_| BackendError::Unavailable {
                message: "KONG_AI_GATEWAY_URL not set".to_string(),
            })?;
        let route = std::env::var("KONG_LLM_ROUTE").unwrap_or_else(|_| "llm/v1/chat".to_string());

        Ok(Self {
            api_key,
            model: "gpt-4o".to_string(),
            gateway_url: gateway_url.trim_end_matches('/').to_string(),
            route: normalize_route(route),
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
    pub fn with_route(mut self, route: impl Into<String>) -> Self {
        self.route = normalize_route(route);
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
        // Konnect uses Key Auth: apikey header with the API key value
        headers.insert(
            "apikey",
            HeaderValue::from_str(self.api_key.expose()).map_err(|e| {
                BackendError::InvalidRequest {
                    message: format!("Invalid KONG_API_KEY: {e}"),
                }
            })?,
        );
        Ok(headers)
    }

    fn build_request(&self, req: &ChatRequest) -> KongRequest {
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
            messages.push(KongMessage {
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
                        .map(|tool_call| KongResponseToolCall {
                            id: tool_call.id.clone(),
                            r#type: "function".to_string(),
                            function: KongResponseFunction {
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
            messages.push(KongMessage {
                role: role.to_string(),
                content,
                tool_calls,
                tool_call_id: msg.tool_call_id.clone(),
            });
        }

        let tools: Option<Vec<KongTool>> = if req.tools.is_empty() {
            None
        } else {
            Some(
                req.tools
                    .iter()
                    .map(|t| KongTool {
                        r#type: "function".to_string(),
                        function: KongFunction {
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

        KongRequest {
            model,
            messages,
            temperature: Some(temperature),
            max_tokens: Some(max_tokens),
            tools,
            response_format,
            stop,
        }
    }

    fn request_url(&self) -> String {
        format!("{}/{}", self.gateway_url, self.route)
    }

    async fn chat_async(&self, req: ChatRequest) -> Result<ChatResponse, ChatLlmError> {
        let kong_req = self.build_request(&req);
        let model = req.model.clone().unwrap_or_else(|| self.model.clone());
        let (resp_headers, response) = self.execute_with_retries(&model, &kong_req).await?;

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

        let metadata = extract_gateway_headers(&resp_headers);

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
        request: &KongRequest,
    ) -> Result<(HeaderMap, KongResponse), ChatLlmError> {
        let url = self.request_url();
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
                            let resp_headers = response.headers().clone();
                            match response.json::<KongResponse>().await {
                                Ok(parsed) => RetryOutcome::Success((resp_headers, parsed)),
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

fn normalize_route(route: impl Into<String>) -> String {
    let route = route
        .into()
        .trim()
        .trim_start_matches('/')
        .trim_end_matches('/')
        .to_string();
    if route.is_empty() {
        "llm/v1/chat".to_string()
    } else {
        route
    }
}

/// Extract governance-relevant headers from gateway responses.
///
/// Captures any header matching known gateway/provider prefixes. This is
/// generic — works for Kong, upstream OpenAI headers, and any future gateway
/// that follows the `x-<vendor>-*` convention.
fn extract_gateway_headers(headers: &HeaderMap) -> HashMap<String, String> {
    let prefixes = [
        "x-kong-",
        "x-ratelimit-",
        "ratelimit-",
        "x-request-id",
        "x-openai-",
        "openai-",
    ];

    headers
        .iter()
        .filter_map(|(name, value)| {
            let key = name.as_str();
            if prefixes.iter().any(|p| key.starts_with(p)) {
                value
                    .to_str()
                    .ok()
                    .map(|v| (key.to_string(), v.to_string()))
            } else {
                None
            }
        })
        .collect()
}

impl ChatBackend for KongBackend {
    type ChatFut<'a>
        = BoxFuture<'a, Result<ChatResponse, ChatLlmError>>
    where
        Self: 'a;

    fn chat(&self, req: ChatRequest) -> Self::ChatFut<'_> {
        Box::pin(async move { self.chat_async(req).await })
    }
}

// ============================================================================
// Kong AI Proxy API Types (OpenAI-compatible format)
// ============================================================================

#[derive(Debug, Serialize)]
struct KongRequest {
    model: String,
    messages: Vec<KongMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<KongTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct KongMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<KongResponseToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct KongTool {
    r#type: String,
    function: KongFunction,
}

#[derive(Debug, Serialize)]
struct KongFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct KongResponse {
    model: String,
    choices: Vec<KongChoice>,
    usage: Option<KongUsage>,
}

#[derive(Debug, Deserialize)]
struct KongChoice {
    message: KongResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct KongResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<KongResponseToolCall>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct KongResponseToolCall {
    id: String,
    /// OpenAI-compatible APIs require `"type": "function"` on tool_calls in
    /// outgoing assistant messages. Without it, upstream routers translating
    /// to Anthropic-native format silently drop the tool_call. See
    /// `openrouter.rs` for the documented diagnosis (2026-05).
    #[serde(rename = "type", default = "default_function_type")]
    r#type: String,
    function: KongResponseFunction,
}

fn default_function_type() -> String {
    "function".to_string()
}

#[derive(Debug, Serialize, Deserialize)]
struct KongResponseFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct KongUsage {
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
    fn test_kong_backend_creation() {
        let backend = KongBackend::try_new("test-key", "https://kong.example.com")
            .unwrap()
            .with_model("gpt-4o-mini")
            .with_temperature(0.5);

        assert_eq!(backend.model, "gpt-4o-mini");
        assert_eq!(backend.temperature, 0.5);
        assert_eq!(backend.api_key.expose(), "test-key");
        assert_eq!(backend.gateway_url, "https://kong.example.com");
        assert_eq!(backend.route, "llm/v1/chat");
    }

    #[test]
    fn test_kong_backend_strips_trailing_slash() {
        let backend = KongBackend::try_new("test-key", "https://kong.example.com/").unwrap();

        assert_eq!(backend.gateway_url, "https://kong.example.com");
    }

    #[test]
    fn test_kong_backend_route_normalization() {
        let backend = KongBackend::try_new("test-key", "https://kong.example.com")
            .unwrap()
            .with_route("/custom/llm/v1/chat/");

        assert_eq!(backend.route, "custom/llm/v1/chat");

        let defaulted = KongBackend::try_new("test-key", "https://kong.example.com")
            .unwrap()
            .with_route("");
        assert_eq!(defaulted.route, "llm/v1/chat");
    }

    #[test]
    fn test_build_request_basic() {
        let backend = KongBackend::try_new("test-key", "https://kong.example.com").unwrap();
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

        let kong_req = backend.build_request(&req);

        assert_eq!(kong_req.model, "gpt-4o");
        assert_eq!(kong_req.messages.len(), 1);
        assert_eq!(kong_req.messages[0].role, "user");
        assert!(kong_req.tools.is_none());
    }

    #[test]
    fn test_build_headers_uses_apikey() {
        let backend = KongBackend::try_new("my-kong-key", "https://kong.example.com").unwrap();
        let headers = backend.build_headers().unwrap();

        // Konnect Key Auth uses "apikey" header, NOT Authorization: Bearer
        assert!(headers.contains_key("apikey"));
        assert!(!headers.contains_key(reqwest::header::AUTHORIZATION));
    }

    #[test]
    fn test_build_request_with_system() {
        let backend = KongBackend::try_new("test-key", "https://kong.example.com").unwrap();
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

        let kong_req = backend.build_request(&req);

        assert_eq!(kong_req.messages.len(), 2);
        assert_eq!(kong_req.messages[0].role, "system");
        assert_eq!(
            kong_req.messages[0].content.as_deref(),
            Some("You are helpful.")
        );
    }

    #[test]
    fn test_build_request_with_tools() {
        let backend = KongBackend::try_new("test-key", "https://kong.example.com").unwrap();
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

        let kong_req = backend.build_request(&req);
        let tools = kong_req.tools.unwrap();

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].r#type, "function");
        assert_eq!(tools[0].function.name, "get_weather");
    }

    #[test]
    fn test_chat_end_to_end() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = runtime.block_on(MockServer::start());

        runtime.block_on(async {
            Mock::given(method("POST"))
                .and(path("/llm/v1/chat"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_json(serde_json::json!({
                            "id": "kongcmpl_test",
                            "model": "gpt-4o",
                            "choices": [{
                                "message": {
                                    "content": "Response from Kong",
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
                        }))
                        .insert_header("x-kong-upstream-latency", "142")
                        .insert_header("x-kong-proxy-latency", "3")
                        .insert_header("x-kong-request-id", "abc123")
                        .insert_header("x-kong-llm-model", "openai/gpt-4o")
                        .insert_header("x-ratelimit-remaining-requests", "499")
                        .insert_header("ratelimit-remaining", "498"),
                )
                .mount(&server)
                .await;
        });

        let backend = KongBackend::try_new("test-key", server.uri()).unwrap();
        let response = runtime
            .block_on(backend.chat(ChatRequest {
                messages: vec![ChatMessage {
                    role: ChatRole::User,
                    content: "Weather?".to_string(),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                }],
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
                stop_sequences: Vec::new(),
                model: None,
            }))
            .unwrap();

        assert_eq!(response.content, "Response from Kong");
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].name, "lookup_weather");
        assert_eq!(response.finish_reason, Some(ChatFinishReason::ToolCalls));

        // Verify usage was extracted
        assert!(response.usage.is_some());
        let usage = response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 12);
        assert_eq!(usage.completion_tokens, 4);
        assert_eq!(usage.total_tokens, 16);

        // Verify gateway metadata was captured from response headers
        assert_eq!(
            response.metadata.get("x-kong-upstream-latency").unwrap(),
            "142"
        );
        assert_eq!(response.metadata.get("x-kong-proxy-latency").unwrap(), "3");
        assert_eq!(
            response.metadata.get("x-kong-request-id").unwrap(),
            "abc123"
        );
        assert_eq!(
            response.metadata.get("x-kong-llm-model").unwrap(),
            "openai/gpt-4o"
        );
        assert_eq!(
            response
                .metadata
                .get("x-ratelimit-remaining-requests")
                .unwrap(),
            "499"
        );
        assert_eq!(response.metadata.get("ratelimit-remaining").unwrap(), "498");

        drop(server);
        drop(runtime);
    }

    #[test]
    fn test_chat_respects_custom_route() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = runtime.block_on(MockServer::start());

        runtime.block_on(async {
            Mock::given(method("POST"))
                .and(path("/api/llm/chat"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "kongcmpl_route",
                    "model": "gpt-4o",
                    "choices": [{
                        "message": {
                            "content": "Route test response",
                            "tool_calls": []
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 2,
                        "total_tokens": 3
                    }
                })))
                .mount(&server)
                .await;
        });

        let backend = KongBackend::try_new("test-key", server.uri())
            .unwrap()
            .with_route("api/llm/chat");
        let response = runtime
            .block_on(backend.chat(ChatRequest {
                messages: vec![ChatMessage {
                    role: ChatRole::User,
                    content: "ping".to_string(),
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

        assert_eq!(response.content, "Route test response");

        drop(server);
        drop(runtime);
    }

    #[test]
    fn test_model_override_in_request() {
        let backend = KongBackend::try_new("test-key", "https://kong.example.com").unwrap();
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
            model: Some("claude-sonnet-4-20250514".to_string()),
        };

        let kong_req = backend.build_request(&req);

        // When model is specified in ChatRequest, it should override the default
        assert_eq!(kong_req.model, "claude-sonnet-4-20250514");
    }
}
