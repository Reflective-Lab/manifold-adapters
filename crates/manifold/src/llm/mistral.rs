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

pub struct MistralBackend {
    api_key: SecretString,
    model: String,
    base_url: String,
    client: Client,
    temperature: f32,
    max_retries: usize,
}

impl MistralBackend {
    /// REAL-by-default constructor. Rejects empty / whitespace keys so that
    /// missing or placeholder credentials surface immediately at construction.
    /// Production code should prefer [`Self::from_env`].
    pub fn try_new(api_key: impl Into<String>) -> BackendResult<Self> {
        let api_key: String = api_key.into();
        if api_key.trim().is_empty() {
            return Err(BackendError::Unavailable {
                message: "MISTRAL_API_KEY is empty or whitespace".to_string(),
            });
        }
        Ok(Self {
            api_key: SecretString::new(api_key),
            model: "mistral-large-latest".to_string(),
            base_url: "https://api.mistral.ai".to_string(),
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
                .get_secret("MISTRAL_API_KEY")
                .map_err(|e| BackendError::Unavailable {
                    message: format!("MISTRAL_API_KEY: {e}"),
                })?;
        Ok(Self {
            api_key,
            model: "mistral-large-latest".to_string(),
            base_url: "https://api.mistral.ai".to_string(),
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

    fn build_request(&self, req: &ChatRequest) -> MistralRequest {
        let model = req.model.clone().unwrap_or_else(|| self.model.clone());
        let temperature = req.temperature.unwrap_or(self.temperature);
        let max_tokens = req.max_tokens.map(|t| t as usize).unwrap_or(4096);

        let mut messages = Vec::new();

        // Append format instruction to system prompt for all structured formats.
        // JSON also gets native response_format, but the API may require "json"
        // to appear in the messages when using json_object mode.
        let system_content = if let Some(instruction) = req.response_format.system_instruction() {
            let base = req.system.clone().unwrap_or_default();
            Some(format!("{base}\n\n{instruction}"))
        } else {
            req.system.clone()
        };

        if let Some(system) = &system_content {
            messages.push(MistralMessage {
                role: "system".to_string(),
                content: Some(serde_json::Value::String(system.clone())),
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
                        .map(|tool_call| MistralToolCall {
                            id: tool_call.id.clone(),
                            function: MistralToolFunction {
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
                Some(serde_json::Value::String(msg.content.clone()))
            };
            messages.push(MistralMessage {
                role: role.to_string(),
                content,
                tool_calls,
                tool_call_id: msg.tool_call_id.clone(),
            });
        }

        let tools = if req.tools.is_empty() {
            None
        } else {
            Some(
                req.tools
                    .iter()
                    .map(|tool| MistralTool {
                        r#type: "function".to_string(),
                        function: MistralFunction {
                            name: tool.name.clone(),
                            description: Some(tool.description.clone()),
                            parameters: Some(tool.parameters.clone()),
                        },
                    })
                    .collect(),
            )
        };

        let response_format = match req.response_format {
            ResponseFormat::Json => Some(serde_json::json!({ "type": "json_object" })),
            _ => None,
        };

        let stop = if req.stop_sequences.is_empty() {
            None
        } else {
            Some(req.stop_sequences.clone())
        };

        MistralRequest {
            model,
            messages,
            temperature: Some(temperature),
            max_tokens: Some(max_tokens),
            tools,
            response_format,
            stop,
        }
    }

    fn extract_text_content(content: Option<MistralMessageContent>) -> String {
        match content {
            Some(MistralMessageContent::Text(text)) => text,
            Some(MistralMessageContent::Parts(parts)) => parts
                .into_iter()
                .filter_map(|part| match part {
                    MistralContentPart::Text { text } => Some(text),
                    MistralContentPart::Unknown => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
            None => String::new(),
        }
    }

    async fn chat_async(&self, req: ChatRequest) -> Result<ChatResponse, ChatLlmError> {
        let request = self.build_request(&req);
        let model = req.model.clone().unwrap_or_else(|| self.model.clone());
        let response = self.execute_with_retries(&model, &request).await?;

        let choice = response.choices.first();
        let content = choice
            .map(|choice| Self::extract_text_content(choice.message.content.clone()))
            .unwrap_or_default();

        let tool_calls = choice
            .and_then(|choice| choice.message.tool_calls.as_ref())
            .map(|calls| {
                calls
                    .iter()
                    .map(|tool_call| ChatToolCall {
                        id: tool_call.id.clone(),
                        name: tool_call.function.name.clone(),
                        arguments: tool_call.function.arguments.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        let finish_reason = choice.and_then(|choice| match choice.finish_reason.as_deref() {
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
                usage: response.usage.map(|usage| ChatTokenUsage {
                    prompt_tokens: usage.prompt_tokens,
                    completion_tokens: usage.completion_tokens,
                    total_tokens: usage.total_tokens,
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
        request: &MistralRequest,
    ) -> Result<MistralResponse, ChatLlmError> {
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
                            match response.json::<MistralResponse>().await {
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

impl ChatBackend for MistralBackend {
    type ChatFut<'a>
        = BoxFuture<'a, Result<ChatResponse, ChatLlmError>>
    where
        Self: 'a;

    fn chat(&self, req: ChatRequest) -> Self::ChatFut<'_> {
        Box::pin(async move { self.chat_async(req).await })
    }
}

#[derive(Debug, Serialize)]
struct MistralRequest {
    model: String,
    messages: Vec<MistralMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<MistralTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct MistralMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<MistralToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct MistralTool {
    r#type: String,
    function: MistralFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct MistralToolCall {
    id: String,
    function: MistralToolFunction,
}

#[derive(Debug, Serialize)]
struct MistralFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct MistralToolFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct MistralResponse {
    model: String,
    choices: Vec<MistralChoice>,
    usage: Option<MistralUsage>,
}

#[derive(Debug, Deserialize)]
struct MistralChoice {
    message: MistralResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct MistralResponseMessage {
    content: Option<MistralMessageContent>,
    tool_calls: Option<Vec<MistralToolCall>>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum MistralMessageContent {
    Text(String),
    Parts(Vec<MistralContentPart>),
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
enum MistralContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct MistralUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use converge_core::traits::{
        ChatMessage, ChatRequest, ChatRole, ResponseFormat, ToolDefinition,
    };

    #[test]
    fn mistral_backend_creation() {
        let backend = MistralBackend::try_new("test-key")
            .unwrap()
            .with_model("mistral-medium-latest")
            .with_temperature(0.2);

        assert_eq!(backend.model, "mistral-medium-latest");
        assert_eq!(backend.temperature, 0.2);
        assert_eq!(backend.api_key.expose(), "test-key");
    }

    #[test]
    fn build_request_includes_system_tools_and_json_mode() {
        let backend = MistralBackend::try_new("test-key").unwrap();
        let request = ChatRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "Return JSON weather".to_string(),
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
            max_tokens: Some(256),
            temperature: Some(0.1),
            stop_sequences: vec!["DONE".to_string()],
            model: None,
        };

        let built = backend.build_request(&request);
        assert_eq!(built.model, "mistral-large-latest");
        assert_eq!(built.messages.len(), 2);
        assert_eq!(built.messages[0].role, "system");
        assert_eq!(built.tools.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            built.response_format,
            Some(serde_json::json!({"type": "json_object"}))
        );
        assert_eq!(built.stop, Some(vec!["DONE".to_string()]));
    }

    #[test]
    fn extract_text_content_handles_string_and_parts() {
        assert_eq!(
            MistralBackend::extract_text_content(Some(MistralMessageContent::Text(
                "Hello".to_string()
            ))),
            "Hello"
        );

        assert_eq!(
            MistralBackend::extract_text_content(Some(MistralMessageContent::Parts(vec![
                MistralContentPart::Text {
                    text: "First".to_string()
                },
                MistralContentPart::Text {
                    text: "Second".to_string()
                }
            ]))),
            "First\nSecond"
        );
    }

    #[test]
    fn parse_tool_calls_from_response() {
        let response: MistralResponse = serde_json::from_value(serde_json::json!({
            "model": "mistral-large-latest",
            "choices": [{
                "message": {
                    "content": "I'll use a tool.",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "lookup_weather",
                            "arguments": "{\"city\":\"Paris\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14
            }
        }))
        .unwrap();

        let choice = response.choices.first().unwrap();
        let tool_calls = choice.message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "lookup_weather");
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn build_request_with_assistant_tool_call_history() {
        let backend = MistralBackend::try_new("test-key").unwrap();
        let request = backend.build_request(&ChatRequest {
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
        });

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
