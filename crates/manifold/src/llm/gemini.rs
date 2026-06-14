// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

use reqwest::Client;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::error_classification::{classify_http_error, network_error, parse_error};
use super::format_contract::finalize_chat_response;
use super::retry::{RetryOutcome, retry_with_backoff};
use crate::secret::{SecretProvider, SecretString, default_secret_provider};
use converge_core::backend::{BackendError, BackendResult};
use converge_provider::{
    BoxFuture, ChatBackend, ChatRequest, ChatResponse, ChatRole, FinishReason as ChatFinishReason,
    LlmError as ChatLlmError, ResponseFormat, TokenUsage as ChatTokenUsage, ToolCall,
};

// ============================================================================
// GeminiBackend
// ============================================================================

pub struct GeminiBackend {
    api_key: SecretString,
    model: String,
    base_url: String,
    client: Client,
    temperature: f32,
    max_retries: usize,
}

impl GeminiBackend {
    /// REAL-by-default constructor. Rejects empty / whitespace keys so that
    /// missing or placeholder credentials surface immediately at construction.
    /// Production code should prefer [`Self::from_env`].
    pub fn try_new(api_key: impl Into<String>) -> BackendResult<Self> {
        let api_key: String = api_key.into();
        if api_key.trim().is_empty() {
            return Err(BackendError::Unavailable {
                message: "GEMINI_API_KEY is empty or whitespace".to_string(),
            });
        }
        Ok(Self {
            api_key: SecretString::new(api_key),
            model: "gemini-2.5-flash".to_string(),
            base_url: "https://generativelanguage.googleapis.com".to_string(),
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
                .get_secret("GEMINI_API_KEY")
                .map_err(|e| BackendError::Unavailable {
                    message: format!("GEMINI_API_KEY: {e}"),
                })?;
        Ok(Self {
            api_key,
            model: "gemini-2.5-flash".to_string(),
            base_url: "https://generativelanguage.googleapis.com".to_string(),
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

    fn build_url(&self, model: &str) -> String {
        format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.base_url,
            model,
            self.api_key.expose()
        )
    }

    fn convert_chat_request(&self, req: &ChatRequest) -> GeminiRequest {
        let mut system_text = req.system.clone();
        let mut contents = Vec::new();

        for msg in &req.messages {
            match msg.role {
                ChatRole::System => system_text = Some(msg.content.clone()),
                ChatRole::User | ChatRole::Tool => contents.push(GeminiContent {
                    role: "user".to_string(),
                    parts: vec![GeminiPart::Text {
                        text: msg.content.clone(),
                    }],
                }),
                ChatRole::Assistant => {
                    if !msg.tool_calls.is_empty() {
                        let mut parts = Vec::new();
                        if !msg.content.is_empty() {
                            parts.push(GeminiPart::Text {
                                text: msg.content.clone(),
                            });
                        }
                        for tool_call in &msg.tool_calls {
                            parts.push(GeminiPart::FunctionCall {
                                function_call: GeminiFunctionCall {
                                    name: tool_call.name.clone(),
                                    args: parse_tool_call_arguments(&tool_call.arguments),
                                },
                            });
                        }
                        contents.push(GeminiContent {
                            role: "model".to_string(),
                            parts,
                        });
                    } else {
                        contents.push(GeminiContent {
                            role: "model".to_string(),
                            parts: vec![GeminiPart::Text {
                                text: msg.content.clone(),
                            }],
                        });
                    }
                }
            }
        }

        // For non-JSON structured formats, append instruction to system text.
        // JSON uses native response_mime_type; others need prompt-based enforcement.
        let system_text = match req.response_format {
            ResponseFormat::Json | ResponseFormat::Text => system_text,
            _ => {
                let instruction = req.response_format.system_instruction().unwrap_or_default();
                let base = system_text.unwrap_or_default();
                Some(format!("{base}\n\n{instruction}"))
            }
        };

        let system_instruction = system_text.map(|text| GeminiSystemInstruction {
            parts: vec![GeminiTextPart { text }],
        });

        let max_output_tokens = req.max_tokens.map(|t| t as usize).unwrap_or(4096);
        let temperature = req.temperature.unwrap_or(self.temperature);

        // Only JSON has native Gemini enforcement via response_mime_type.
        let response_mime_type = match req.response_format {
            ResponseFormat::Json => Some("application/json".to_string()),
            _ => None,
        };

        let stop_sequences = if req.stop_sequences.is_empty() {
            None
        } else {
            Some(req.stop_sequences.clone())
        };

        let generation_config = GeminiGenerationConfig {
            max_output_tokens,
            temperature,
            stop_sequences,
            response_mime_type,
        };

        let tools = if req.tools.is_empty() {
            None
        } else {
            let declarations: Vec<GeminiFunctionDeclaration> = req
                .tools
                .iter()
                .map(|t| GeminiFunctionDeclaration {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.parameters.clone(),
                })
                .collect();
            Some(vec![GeminiTool {
                function_declarations: declarations,
            }])
        };

        GeminiRequest {
            system_instruction,
            contents,
            generation_config,
            tools,
        }
    }

    async fn chat_async(&self, req: ChatRequest) -> Result<ChatResponse, ChatLlmError> {
        let model = req.model.clone().unwrap_or_else(|| self.model.clone());
        let gemini_req = self.convert_chat_request(&req);
        let response = self.execute_with_retries(&model, &gemini_req).await?;

        let candidate = response.candidates.first();

        let mut content = String::new();
        let mut tool_calls = Vec::new();

        if let Some(candidate) = candidate {
            for part in &candidate.content.parts {
                match part {
                    GeminiResponsePart::Text { text } => {
                        if !content.is_empty() {
                            content.push('\n');
                        }
                        content.push_str(text);
                    }
                    GeminiResponsePart::FunctionCall { function_call } => {
                        tool_calls.push(ToolCall {
                            id: function_call.name.clone(),
                            name: function_call.name.clone(),
                            arguments: serde_json::to_string(&function_call.args)
                                .unwrap_or_default(),
                        });
                    }
                }
            }
        }

        let finish_reason = candidate.and_then(|c| {
            c.finish_reason.as_deref().map(|r| match r {
                "STOP" => ChatFinishReason::Stop,
                "MAX_TOKENS" => ChatFinishReason::Length,
                "SAFETY" => ChatFinishReason::ContentFilter,
                _ => ChatFinishReason::Stop,
            })
        });

        let usage = response.usage_metadata.map(|u| ChatTokenUsage {
            prompt_tokens: u.prompt_token_count,
            completion_tokens: u.candidates_token_count,
            total_tokens: u.total_token_count,
        });

        finalize_chat_response(
            &req,
            ChatResponse {
                content,
                tool_calls,
                usage,
                model: Some(model),
                finish_reason,
                metadata: Default::default(),
            },
        )
    }

    #[allow(dead_code)]
    fn request_fingerprint(&self, request: &GeminiRequest) -> String {
        let canonical = serde_json::to_string(request).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    async fn execute_with_retries(
        &self,
        model: &str,
        request: &GeminiRequest,
    ) -> Result<GeminiResponse, ChatLlmError> {
        let url = self.build_url(model);
        let mut req_headers = HeaderMap::new();
        req_headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        retry_with_backoff(self.max_retries, || {
            let client = &self.client;
            let url = &url;
            let headers = req_headers.clone();
            let request = request;
            async move {
                match client.post(url).headers(headers).json(request).send().await {
                    Ok(response) => {
                        let status = response.status();
                        if status.is_success() {
                            match response.json::<GeminiResponse>().await {
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

impl ChatBackend for GeminiBackend {
    type ChatFut<'a>
        = BoxFuture<'a, Result<ChatResponse, ChatLlmError>>
    where
        Self: 'a;

    fn chat(&self, req: ChatRequest) -> Self::ChatFut<'_> {
        Box::pin(async move { self.chat_async(req).await })
    }
}

// ============================================================================
// Gemini API Types
// ============================================================================

#[derive(Debug, Serialize)]
struct GeminiRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiSystemInstruction>,
    contents: Vec<GeminiContent>,
    generation_config: GeminiGenerationConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTool>>,
}

#[derive(Debug, Serialize)]
struct GeminiSystemInstruction {
    parts: Vec<GeminiTextPart>,
}

#[derive(Debug, Serialize)]
struct GeminiTextPart {
    text: String,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum GeminiPart {
    Text {
        text: String,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
    },
}

#[derive(Debug, Serialize)]
struct GeminiGenerationConfig {
    max_output_tokens: usize,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<String>,
}

#[derive(Debug, Serialize)]
struct GeminiTool {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiResponseContent,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponseContent {
    parts: Vec<GeminiResponsePart>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum GeminiResponsePart {
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
    },
    Text {
        text: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    args: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: u32,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: u32,
    #[serde(rename = "totalTokenCount")]
    total_token_count: u32,
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
    fn test_gemini_backend_creation() {
        let backend = GeminiBackend::try_new("test-key")
            .unwrap()
            .with_model("gemini-2.5-pro")
            .with_temperature(0.5);

        assert_eq!(backend.model, "gemini-2.5-pro");
        assert_eq!(backend.temperature, 0.5);
        assert_eq!(backend.api_key.expose(), "test-key");
    }

    #[test]
    fn test_default_model() {
        let backend = GeminiBackend::try_new("test-key").unwrap();
        assert_eq!(backend.model, "gemini-2.5-flash");
    }

    #[test]
    fn test_build_url() {
        let backend = GeminiBackend::try_new("my-key").unwrap();
        let url = backend.build_url("gemini-2.5-flash");
        assert_eq!(
            url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=my-key"
        );
    }

    #[test]
    fn test_convert_simple_request() {
        let backend = GeminiBackend::try_new("test-key").unwrap();
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

        let gemini_req = backend.convert_chat_request(&req);

        assert!(gemini_req.system_instruction.is_none());
        assert_eq!(gemini_req.contents.len(), 1);
        assert_eq!(gemini_req.contents[0].role, "user");
        assert!(gemini_req.tools.is_none());
        assert!(gemini_req.generation_config.response_mime_type.is_none());
    }

    #[test]
    fn test_convert_with_system_and_assistant() {
        let backend = GeminiBackend::try_new("test-key").unwrap();
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
                ChatMessage {
                    role: ChatRole::Assistant,
                    content: "Hello!".to_string(),
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

        let gemini_req = backend.convert_chat_request(&req);

        assert!(gemini_req.system_instruction.is_some());
        let sys = gemini_req.system_instruction.unwrap();
        assert_eq!(sys.parts[0].text, "You are helpful.");

        assert_eq!(gemini_req.contents.len(), 2);
        assert_eq!(gemini_req.contents[0].role, "user");
        assert_eq!(gemini_req.contents[1].role, "model");
    }

    #[test]
    fn test_convert_json_response_format() {
        let backend = GeminiBackend::try_new("test-key").unwrap();
        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "Give me JSON".to_string(),
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

        let gemini_req = backend.convert_chat_request(&req);
        assert_eq!(
            gemini_req.generation_config.response_mime_type,
            Some("application/json".to_string())
        );
    }

    #[test]
    fn test_convert_with_tools() {
        let backend = GeminiBackend::try_new("test-key").unwrap();
        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "Search for cats".to_string(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }],
            system: None,
            tools: vec![converge_core::traits::ToolDefinition {
                name: "search".to_string(),
                description: "Search the web".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {"query": {"type": "string"}}}),
            }],
            response_format: ResponseFormat::default(),
            max_tokens: None,
            temperature: None,
            stop_sequences: Vec::new(),
            model: None,
        };

        let gemini_req = backend.convert_chat_request(&req);
        let tools = gemini_req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function_declarations.len(), 1);
        assert_eq!(tools[0].function_declarations[0].name, "search");
    }

    #[test]
    fn test_request_fingerprint_deterministic() {
        let backend = GeminiBackend::try_new("test-key").unwrap();
        let request = GeminiRequest {
            system_instruction: None,
            contents: vec![GeminiContent {
                role: "user".to_string(),
                parts: vec![GeminiPart::Text {
                    text: "test".to_string(),
                }],
            }],
            generation_config: GeminiGenerationConfig {
                max_output_tokens: 100,
                temperature: 0.0,
                stop_sequences: None,
                response_mime_type: None,
            },
            tools: None,
        };

        let fp1 = backend.request_fingerprint(&request);
        let fp2 = backend.request_fingerprint(&request);

        assert_eq!(fp1, fp2);
        assert!(!fp1.is_empty());
    }

    #[test]
    fn test_parse_gemini_response() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello there!"}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 3,
                "totalTokenCount": 8
            }
        }"#;

        let response: GeminiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.candidates.len(), 1);
        assert_eq!(
            response.candidates[0].finish_reason.as_deref(),
            Some("STOP")
        );
        let usage = response.usage_metadata.unwrap();
        assert_eq!(usage.prompt_token_count, 5);
        assert_eq!(usage.candidates_token_count, 3);
        assert_eq!(usage.total_token_count, 8);
    }

    #[test]
    fn test_parse_function_call_response() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"functionCall": {"name": "search", "args": {"query": "cats"}}}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            }
        }"#;

        let response: GeminiResponse = serde_json::from_str(json).unwrap();
        let parts = &response.candidates[0].content.parts;
        assert_eq!(parts.len(), 1);
        match &parts[0] {
            GeminiResponsePart::FunctionCall { function_call } => {
                assert_eq!(function_call.name, "search");
                assert_eq!(function_call.args["query"], "cats");
            }
            GeminiResponsePart::Text { .. } => panic!("Expected FunctionCall"),
        }
    }

    #[test]
    fn test_chat_runtime_multiturn_tools_and_json_mode() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = runtime.block_on(MockServer::start());

        runtime.block_on(async {
            Mock::given(method("POST"))
                .and(path("/v1beta/models/gemini-2.5-flash:generateContent"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "candidates": [{
                        "content": {
                            "parts": [
                                {"text": "{\"ok\":true}"},
                                {"functionCall": {"name": "lookup", "args": {"id": 7}}}
                            ]
                        },
                        "finishReason": "STOP"
                    }],
                    "usageMetadata": {
                        "promptTokenCount": 11,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 16
                    }
                })))
                .mount(&server)
                .await;
        });

        let backend = GeminiBackend::try_new("test-key")
            .unwrap()
            .with_base_url(server.uri());
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
                    content: "Find record 7".to_string(),
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
            system: None,
            tools: vec![converge_core::traits::ToolDefinition {
                name: "lookup".to_string(),
                description: "Lookup a record".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"id": {"type": "integer"}}
                }),
            }],
            response_format: ResponseFormat::Json,
            max_tokens: Some(128),
            temperature: Some(0.0),
            stop_sequences: vec!["STOP".to_string()],
            model: None,
        };

        let response = runtime.block_on(backend.chat(req)).unwrap();
        assert_eq!(response.content, "{\"ok\":true}");
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].name, "lookup");
        assert_eq!(response.tool_calls[0].arguments, "{\"id\":7}");

        let requests = runtime.block_on(server.received_requests()).unwrap();
        assert_eq!(requests.len(), 1);
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(
            body["generation_config"]["response_mime_type"],
            "application/json"
        );
        assert_eq!(body["contents"][0]["role"], "user");
        assert_eq!(body["contents"][1]["role"], "model");
        assert_eq!(
            body["tools"][0]["function_declarations"][0]["name"],
            "lookup"
        );

        drop(server);
        drop(runtime);
    }

    #[test]
    fn test_convert_with_assistant_tool_call_history() {
        let backend = GeminiBackend::try_new("test-key").unwrap();
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
                        id: "call_1".to_string(),
                        name: "lookup_weather".to_string(),
                        arguments: r#"{"city":"Paris"}"#.to_string(),
                    }],
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

        let request = backend.convert_chat_request(&req);
        assert_eq!(request.contents[1].role, "model");
        assert_eq!(request.contents[1].parts.len(), 1);
        match &request.contents[1].parts[0] {
            GeminiPart::FunctionCall { function_call } => {
                assert_eq!(function_call.name, "lookup_weather");
                assert_eq!(function_call.args["city"], "Paris");
            }
            GeminiPart::Text { .. } => panic!("expected function call history part"),
        }
    }
}
