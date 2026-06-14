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

/// OpenRouter backend — routes to any model through `openrouter.ai/api`.
///
/// OpenRouter uses the OpenAI chat completions format. Model names follow
/// the `provider/model` convention (e.g. `anthropic/claude-sonnet-4`).
pub struct OpenRouterBackend {
    api_key: SecretString,
    model: String,
    base_url: String,
    client: Client,
    temperature: f32,
    max_retries: usize,
    site_url: String,
    site_name: String,
}

impl OpenRouterBackend {
    /// REAL-by-default constructor. Rejects empty / whitespace keys so that
    /// missing or placeholder credentials surface immediately at construction.
    /// Production code should prefer [`Self::from_env`].
    pub fn try_new(api_key: impl Into<String>) -> BackendResult<Self> {
        let api_key: String = api_key.into();
        if api_key.trim().is_empty() {
            return Err(BackendError::Unavailable {
                message: "OPENROUTER_API_KEY is empty or whitespace".to_string(),
            });
        }
        Ok(Self {
            api_key: SecretString::new(api_key),
            model: "anthropic/claude-sonnet-4".to_string(),
            base_url: "https://openrouter.ai/api".to_string(),
            client: Client::new(),
            temperature: 0.0,
            max_retries: 3,
            site_url: String::new(),
            site_name: String::new(),
        })
    }

    pub fn from_env() -> BackendResult<Self> {
        Self::from_secret_provider(default_secret_provider())
    }

    pub fn from_secret_provider(secrets: &dyn SecretProvider) -> BackendResult<Self> {
        let api_key =
            secrets
                .get_secret("OPENROUTER_API_KEY")
                .map_err(|e| BackendError::Unavailable {
                    message: format!("OPENROUTER_API_KEY: {e}"),
                })?;

        let model = secrets
            .get_secret("OPENROUTER_MODEL")
            .map(|s| s.expose().to_string())
            .unwrap_or_else(|_| "anthropic/claude-sonnet-4".to_string());
        let base_url = secrets
            .get_secret("OPENROUTER_BASE_URL")
            .map(|s| s.expose().to_string())
            .unwrap_or_else(|_| "https://openrouter.ai/api".to_string());
        let site_url = secrets
            .get_secret("OPENROUTER_SITE_URL")
            .map(|s| s.expose().to_string())
            .unwrap_or_default();
        let site_name = secrets
            .get_secret("OPENROUTER_SITE_NAME")
            .map(|s| s.expose().to_string())
            .unwrap_or_default();

        Ok(Self {
            api_key,
            model,
            base_url,
            client: Client::new(),
            temperature: 0.0,
            max_retries: 3,
            site_url,
            site_name,
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

    #[must_use]
    pub fn with_site_url(mut self, url: impl Into<String>) -> Self {
        self.site_url = url.into();
        self
    }

    #[must_use]
    pub fn with_site_name(mut self, name: impl Into<String>) -> Self {
        self.site_name = name.into();
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

        if !self.site_url.is_empty() {
            if let Ok(val) = HeaderValue::from_str(&self.site_url) {
                headers.insert("HTTP-Referer", val);
            }
        }
        if !self.site_name.is_empty() {
            if let Ok(val) = HeaderValue::from_str(&self.site_name) {
                headers.insert("X-Title", val);
            }
        }

        Ok(headers)
    }

    fn build_request(&self, req: &ChatRequest) -> OpenRouterRequest {
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
            messages.push(OpenRouterMessage {
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
                        .map(|tool_call| OpenRouterResponseToolCall {
                            id: tool_call.id.clone(),
                            r#type: "function".to_string(),
                            function: OpenRouterResponseFunction {
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
            messages.push(OpenRouterMessage {
                role: role.to_string(),
                content,
                tool_calls,
                tool_call_id: msg.tool_call_id.clone(),
            });
        }

        let tools: Option<Vec<OpenRouterTool>> = if req.tools.is_empty() {
            None
        } else {
            Some(
                req.tools
                    .iter()
                    .map(|t| OpenRouterTool {
                        r#type: "function".to_string(),
                        function: OpenRouterFunction {
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

        OpenRouterRequest {
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
        let openrouter_req = self.build_request(&req);
        let model = req.model.clone().unwrap_or_else(|| self.model.clone());
        let response = self.execute_with_retries(&model, &openrouter_req).await?;

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

        let metadata = extract_openrouter_metadata(&response);

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
        request: &OpenRouterRequest,
    ) -> Result<OpenRouterResponse, ChatLlmError> {
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
                            match response.json::<OpenRouterResponse>().await {
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

fn extract_openrouter_metadata(response: &OpenRouterResponse) -> HashMap<String, String> {
    let mut meta = HashMap::new();

    if let Some(provider) = &response.provider {
        meta.insert("provider".to_string(), provider.clone());
    }

    if let Some(usage) = &response.usage {
        if let Some(cost) = usage.cost {
            meta.insert("cost".to_string(), cost.to_string());
        }
        if let Some(byok) = usage.is_byok {
            meta.insert("is_byok".to_string(), byok.to_string());
        }
        if let Some(details) = &usage.cost_details {
            if let Some(v) = details.upstream_inference_cost {
                meta.insert("cost.upstream".to_string(), v.to_string());
            }
            if let Some(v) = details.upstream_inference_prompt_cost {
                meta.insert("cost.prompt".to_string(), v.to_string());
            }
            if let Some(v) = details.upstream_inference_completions_cost {
                meta.insert("cost.completion".to_string(), v.to_string());
            }
        }
    }

    meta
}

impl ChatBackend for OpenRouterBackend {
    type ChatFut<'a>
        = BoxFuture<'a, Result<ChatResponse, ChatLlmError>>
    where
        Self: 'a;

    fn chat(&self, req: ChatRequest) -> Self::ChatFut<'_> {
        Box::pin(async move { self.chat_async(req).await })
    }
}

// ============================================================================
// OpenRouter API Types (OpenAI-compatible)
// ============================================================================

#[derive(Debug, Serialize)]
struct OpenRouterRequest {
    model: String,
    messages: Vec<OpenRouterMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenRouterTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct OpenRouterMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenRouterResponseToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenRouterTool {
    r#type: String,
    function: OpenRouterFunction,
}

#[derive(Debug, Serialize)]
struct OpenRouterFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterResponse {
    model: String,
    choices: Vec<OpenRouterChoice>,
    usage: Option<OpenRouterUsage>,
    #[serde(default)]
    provider: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterChoice {
    message: OpenRouterResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenRouterResponseToolCall>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenRouterResponseToolCall {
    id: String,
    /// OpenAI-compatible APIs require `"type": "function"` on tool_calls
    /// in *outgoing* assistant messages (not just responses). Without it,
    /// some upstream routers (notably Anthropic via Bedrock) silently drop
    /// the tool_call from the conversation, causing the model to re-call
    /// the tool instead of consuming the tool result.
    ///
    /// `#[serde(default = "default_function_type")]` so deserialization
    /// tolerates upstreams that omit the field on response; serialization
    /// always emits it.
    #[serde(rename = "type", default = "default_function_type")]
    r#type: String,
    function: OpenRouterResponseFunction,
}

fn default_function_type() -> String {
    "function".to_string()
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenRouterResponseFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenRouterUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
    #[serde(default)]
    cost: Option<f64>,
    #[serde(default)]
    is_byok: Option<bool>,
    #[serde(default)]
    cost_details: Option<OpenRouterCostDetails>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterCostDetails {
    #[serde(default)]
    upstream_inference_cost: Option<f64>,
    #[serde(default)]
    upstream_inference_prompt_cost: Option<f64>,
    #[serde(default)]
    upstream_inference_completions_cost: Option<f64>,
}

// ============================================================================
// Streaming
// ============================================================================
//
// The streaming impl lives inline here so it can reuse `build_request`,
// `build_headers`, and the request struct without exposing them publicly.
// It is gated on both `openrouter` and `streaming` features.
//
// Wire format: OpenAI/OpenRouter SSE — each event is `data: <json>\n\n`,
// with the terminator literal `data: [DONE]\n\n`. Some upstreams send
// `:keepalive` or other comment lines we silently skip. The final
// non-DONE chunk carries `usage` when `stream_options.include_usage` is
// set (we always set it).
//
// We deliberately do not pull in `async-stream` or a direct `bytes`
// dependency: framing is hand-rolled on top of `futures::stream::unfold`
// using raw `&[u8]` slices from each `reqwest` chunk.

#[cfg(feature = "streaming")]
mod streaming_impl {
    use super::*;
    use crate::llm::{ChatEvent, ChatStream, StreamingChatBackend};
    #[cfg(test)]
    use futures::StreamExt;
    use futures::stream::{self, Stream, TryStreamExt};
    use std::collections::VecDeque;
    use std::pin::Pin;

    impl StreamingChatBackend for OpenRouterBackend {
        fn chat_stream(
            &self,
            req: ChatRequest,
        ) -> Pin<
            Box<dyn std::future::Future<Output = Result<ChatStream<'_>, ChatLlmError>> + Send + '_>,
        > {
            Box::pin(async move {
                let mut request_body =
                    serde_json::to_value(self.build_request(&req)).map_err(|e| {
                        ChatLlmError::ProviderError {
                            message: format!("serialize streaming request: {e}"),
                            code: None,
                        }
                    })?;
                // Inject `"stream": true` and request usage in the final chunk.
                if let Some(obj) = request_body.as_object_mut() {
                    obj.insert("stream".to_string(), serde_json::Value::Bool(true));
                    obj.insert(
                        "stream_options".to_string(),
                        serde_json::json!({"include_usage": true}),
                    );
                }

                let url = format!("{}/v1/chat/completions", self.base_url);
                let headers = self.build_headers().map_err(map_backend_error)?;

                let response = self
                    .client
                    .post(&url)
                    .headers(headers)
                    .json(&request_body)
                    .send()
                    .await
                    .map_err(network_error)?;

                let status = response.status();
                if !status.is_success() {
                    let body = response.text().await.unwrap_or_default();
                    return Err(classify_http_error(
                        status.as_u16(),
                        &body,
                        req.model.as_deref().unwrap_or(self.model.as_str()),
                    ));
                }

                // reqwest's `bytes_stream()` yields Result<Bytes, reqwest::Error>.
                // Convert errors into ChatLlmError and feed into the SSE parser.
                let byte_stream = response.bytes_stream().map_err(network_error);
                let event_stream = sse_event_stream(byte_stream);
                Ok(Box::pin(event_stream) as ChatStream<'_>)
            })
        }
    }

    /// Buffer state for the SSE framing/parsing state machine. One chunk
    /// can produce zero, one, or many `ChatEvent`s; `pending` queues the
    /// extras across `poll_next` calls.
    struct SseState<S> {
        byte_stream: S,
        buf: Vec<u8>,
        pending: VecDeque<ChatEvent>,
        /// Once we see the literal `[DONE]` sentinel (or upstream EOF),
        /// flag this so we drain `pending` then end the stream.
        terminated: bool,
    }

    /// Convert a byte stream into a stream of fully-parsed `ChatEvent`s.
    ///
    /// SSE framing: events are separated by `\n\n`; each event has one
    /// or more lines, but for OpenAI-compatible APIs we only care about
    /// the `data:` line. The body is JSON, except for the literal
    /// `[DONE]` terminator which ends the stream.
    fn sse_event_stream<S, B>(
        byte_stream: S,
    ) -> impl Stream<Item = Result<ChatEvent, ChatLlmError>> + Send
    where
        S: Stream<Item = Result<B, ChatLlmError>> + Send + Unpin + 'static,
        B: AsRef<[u8]> + Send,
    {
        let state = SseState {
            byte_stream,
            buf: Vec::with_capacity(4096),
            pending: VecDeque::new(),
            terminated: false,
        };

        stream::unfold(state, |mut state| async move {
            loop {
                // Drain any queued events from the previous chunk first.
                if let Some(event) = state.pending.pop_front() {
                    return Some((Ok(event), state));
                }
                if state.terminated {
                    return None;
                }

                // Pull the next chunk from upstream.
                match state.byte_stream.try_next().await {
                    Ok(Some(chunk)) => {
                        state.buf.extend_from_slice(chunk.as_ref());
                        match drain_complete_events(&mut state.buf, &mut state.pending) {
                            Ok(saw_done) => {
                                if saw_done {
                                    state.terminated = true;
                                }
                            }
                            Err(err) => {
                                state.terminated = true;
                                return Some((Err(err), state));
                            }
                        }
                        // Loop back to either yield a pending event or
                        // poll for the next chunk.
                        continue;
                    }
                    Ok(None) => {
                        // Upstream closed. Try to parse any tail bytes
                        // that don't end with `\n\n` as a best-effort
                        // final event, then end the stream.
                        state.terminated = true;
                        if !state.buf.is_empty() {
                            // Treat the leftover as a final pseudo-event.
                            let tail = std::mem::take(&mut state.buf);
                            if let Ok(s) = std::str::from_utf8(&tail) {
                                if let Some(data) = extract_data_line(s) {
                                    if data.trim() != "[DONE]" {
                                        for event in parse_openai_delta(data) {
                                            state.pending.push_back(event);
                                        }
                                    }
                                }
                            }
                        }
                        continue;
                    }
                    Err(err) => {
                        state.terminated = true;
                        return Some((Err(err), state));
                    }
                }
            }
        })
    }

    /// Parse every complete SSE event currently in `buf`, enqueueing
    /// resulting `ChatEvent`s into `pending`. Returns Ok(true) if we
    /// observed the `[DONE]` terminator (caller should stop polling
    /// upstream), Ok(false) otherwise.
    fn drain_complete_events(
        buf: &mut Vec<u8>,
        pending: &mut VecDeque<ChatEvent>,
    ) -> Result<bool, ChatLlmError> {
        let mut saw_done = false;
        while let Some(pos) = find_event_terminator(buf) {
            let event_bytes: Vec<u8> = buf.drain(..pos + 2).collect();
            // Strip the trailing `\n\n` before decoding.
            let payload_len = event_bytes.len().saturating_sub(2);
            let event_str = std::str::from_utf8(&event_bytes[..payload_len]).map_err(|e| {
                ChatLlmError::ProviderError {
                    message: format!("invalid utf-8 in SSE event: {e}"),
                    code: None,
                }
            })?;
            let Some(data) = extract_data_line(event_str) else {
                continue;
            };
            if data.trim() == "[DONE]" {
                saw_done = true;
                break;
            }
            for event in parse_openai_delta(data) {
                pending.push_back(event);
            }
        }
        Ok(saw_done)
    }

    fn find_event_terminator(buf: &[u8]) -> Option<usize> {
        buf.windows(2).position(|w| w == b"\n\n")
    }

    /// Extract the `data:` payload from a single SSE event (which may be
    /// multiple lines, e.g. a `:keepalive` comment line plus a `data:` line).
    fn extract_data_line(event: &str) -> Option<&str> {
        for line in event.split('\n') {
            // Strip a trailing `\r` for CRLF-framed streams.
            let line = line.strip_suffix('\r').unwrap_or(line);
            if let Some(rest) = line.strip_prefix("data: ") {
                return Some(rest);
            }
            if let Some(rest) = line.strip_prefix("data:") {
                return Some(rest.trim_start());
            }
        }
        None
    }

    /// Parse one OpenAI-style streaming delta into zero or more ChatEvents.
    fn parse_openai_delta(data: &str) -> Vec<ChatEvent> {
        let parsed: serde_json::Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        let mut events = Vec::new();
        if let Some(choices) = parsed.get("choices").and_then(|v| v.as_array()) {
            for choice in choices {
                if let Some(delta) = choice.get("delta") {
                    if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
                        if !text.is_empty() {
                            events.push(ChatEvent::TextDelta(text.to_string()));
                        }
                    }
                    if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                        for tc in tool_calls {
                            let index =
                                tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                            let id = tc
                                .get("id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            if let Some(function) = tc.get("function") {
                                if let Some(name) = function.get("name").and_then(|v| v.as_str()) {
                                    events.push(ChatEvent::ToolCallStart {
                                        id: id.clone(),
                                        name: name.to_string(),
                                        index,
                                    });
                                }
                                if let Some(args) =
                                    function.get("arguments").and_then(|v| v.as_str())
                                {
                                    if !args.is_empty() {
                                        events.push(ChatEvent::ToolCallArgsDelta {
                                            id: id.clone(),
                                            index,
                                            args_delta: args.to_string(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                    let mapped = match reason {
                        "stop" => Some(ChatFinishReason::Stop),
                        "length" => Some(ChatFinishReason::Length),
                        "tool_calls" => Some(ChatFinishReason::ToolCalls),
                        "content_filter" => Some(ChatFinishReason::ContentFilter),
                        _ => None,
                    };
                    if let Some(r) = mapped {
                        events.push(ChatEvent::Finish(r));
                    }
                }
            }
        }
        if let Some(usage) = parsed.get("usage") {
            let prompt = usage
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let completion = usage
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let total = usage
                .get("total_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            if total > 0 {
                events.push(ChatEvent::Usage(ChatTokenUsage {
                    prompt_tokens: prompt,
                    completion_tokens: completion,
                    total_tokens: total,
                }));
            }
        }
        events
    }

    #[cfg(test)]
    mod streaming_tests {
        use super::*;

        #[test]
        fn extract_data_line_strips_prefix_with_space() {
            let event = "data: {\"hello\":1}";
            assert_eq!(extract_data_line(event), Some("{\"hello\":1}"));
        }

        #[test]
        fn extract_data_line_strips_prefix_without_space() {
            let event = "data:{\"hello\":1}";
            assert_eq!(extract_data_line(event), Some("{\"hello\":1}"));
        }

        #[test]
        fn extract_data_line_skips_comment() {
            let event = ":keepalive\ndata: payload";
            assert_eq!(extract_data_line(event), Some("payload"));
        }

        #[test]
        fn parse_delta_emits_text() {
            let data = r#"{"choices":[{"delta":{"content":"hi"}}]}"#;
            let events = parse_openai_delta(data);
            assert!(matches!(events.as_slice(), [ChatEvent::TextDelta(s)] if s == "hi"));
        }

        #[test]
        fn parse_delta_emits_finish() {
            let data = r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
            let events = parse_openai_delta(data);
            assert!(matches!(
                events.as_slice(),
                [ChatEvent::Finish(ChatFinishReason::Stop)]
            ));
        }

        #[test]
        fn parse_delta_emits_usage() {
            let data = r#"{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}"#;
            let events = parse_openai_delta(data);
            assert!(matches!(
                events.as_slice(),
                [ChatEvent::Usage(u)]
                    if u.prompt_tokens == 3 && u.completion_tokens == 5 && u.total_tokens == 8
            ));
        }

        #[test]
        fn parse_delta_ignores_unknown_finish_reason() {
            let data = r#"{"choices":[{"delta":{},"finish_reason":"weird"}]}"#;
            assert!(parse_openai_delta(data).is_empty());
        }

        #[test]
        fn parse_delta_emits_tool_call_chunks() {
            let data = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":"{\"ci"}}]}}]}"#;
            let events = parse_openai_delta(data);
            assert_eq!(events.len(), 2);
            assert!(matches!(
                &events[0],
                ChatEvent::ToolCallStart { id, name, index } if id == "call_1" && name == "get_weather" && *index == 0
            ));
            assert!(matches!(
                &events[1],
                ChatEvent::ToolCallArgsDelta { args_delta, .. } if args_delta == "{\"ci"
            ));
        }

        #[test]
        fn find_terminator_locates_double_newline() {
            assert_eq!(find_event_terminator(b"abc\n\ndef"), Some(3));
            assert_eq!(find_event_terminator(b"no-term"), None);
        }

        #[tokio::test]
        async fn sse_event_stream_parses_simple_stream() {
            // Compose a chunk stream that splits an event across chunks
            // and includes the terminator.
            let chunks: Vec<Result<Vec<u8>, ChatLlmError>> = vec![
                Ok(b"data: {\"choices\":[{\"delta\":{\"content\":\"He".to_vec()),
                Ok(b"llo\"}}]}\n\n".to_vec()),
                Ok(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n".to_vec()),
                Ok(b"data: [DONE]\n\n".to_vec()),
            ];
            let upstream = futures::stream::iter(chunks);
            let mut events = Box::pin(sse_event_stream(upstream));
            let mut collected = Vec::new();
            while let Some(evt) = events.next().await {
                collected.push(evt.unwrap());
            }
            assert_eq!(collected.len(), 2);
            assert!(matches!(&collected[0], ChatEvent::TextDelta(s) if s == "Hello"));
            assert!(matches!(
                &collected[1],
                ChatEvent::Finish(ChatFinishReason::Stop)
            ));
        }
    }
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
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn test_openrouter_backend_creation() {
        let backend = OpenRouterBackend::try_new("test-key")
            .unwrap()
            .with_model("openai/gpt-4o")
            .with_temperature(0.5);

        assert_eq!(backend.model, "openai/gpt-4o");
        assert_eq!(backend.temperature, 0.5);
        assert_eq!(backend.api_key.expose(), "test-key");
        assert_eq!(backend.base_url, "https://openrouter.ai/api");
    }

    #[test]
    fn test_default_model_is_claude() {
        let backend = OpenRouterBackend::try_new("test-key").unwrap();
        assert_eq!(backend.model, "anthropic/claude-sonnet-4");
    }

    #[test]
    fn test_build_request_basic() {
        let backend = OpenRouterBackend::try_new("test-key").unwrap();
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

        let openrouter_req = backend.build_request(&req);

        assert_eq!(openrouter_req.model, "anthropic/claude-sonnet-4");
        assert_eq!(openrouter_req.messages.len(), 1);
        assert_eq!(openrouter_req.messages[0].role, "user");
        assert!(openrouter_req.tools.is_none());
    }

    #[test]
    fn test_build_request_with_tools() {
        let backend = OpenRouterBackend::try_new("test-key").unwrap();
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

        let openrouter_req = backend.build_request(&req);
        let tools = openrouter_req.tools.unwrap();

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
    }

    #[test]
    fn test_site_headers_set() {
        let backend = OpenRouterBackend::try_new("test-key")
            .unwrap()
            .with_site_url("https://converge.zone")
            .with_site_name("Converge");

        let headers = backend.build_headers().unwrap();

        assert_eq!(
            headers.get("HTTP-Referer").unwrap().to_str().unwrap(),
            "https://converge.zone"
        );
        assert_eq!(
            headers.get("X-Title").unwrap().to_str().unwrap(),
            "Converge"
        );
    }

    #[test]
    fn test_site_headers_omitted_when_empty() {
        let backend = OpenRouterBackend::try_new("test-key").unwrap();
        let headers = backend.build_headers().unwrap();

        assert!(headers.get("HTTP-Referer").is_none());
        assert!(headers.get("X-Title").is_none());
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
                .and(path("/v1/chat/completions"))
                .and(header("HTTP-Referer", "https://converge.zone"))
                .and(header("X-Title", "Converge"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "gen-test",
                    "model": "anthropic/claude-sonnet-4",
                    "provider": "Anthropic",
                    "choices": [{
                        "message": {
                            "content": "Hello from OpenRouter!",
                            "tool_calls": null
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "cost": 0.000_075,
                        "is_byok": false,
                        "cost_details": {
                            "upstream_inference_cost": 0.000_075,
                            "upstream_inference_prompt_cost": 0.000_03,
                            "upstream_inference_completions_cost": 0.000_045
                        }
                    }
                })))
                .mount(&server)
                .await;
        });

        let backend = OpenRouterBackend::try_new("test-key")
            .unwrap()
            .with_base_url(server.uri())
            .with_site_url("https://converge.zone")
            .with_site_name("Converge");

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

        assert_eq!(response.content, "Hello from OpenRouter!");
        assert_eq!(response.finish_reason, Some(ChatFinishReason::Stop));
        assert_eq!(response.usage.as_ref().unwrap().total_tokens, 15);

        // Verify cost/provider metadata was captured from response body
        assert_eq!(response.metadata.get("provider").unwrap(), "Anthropic");
        assert_eq!(response.metadata.get("cost").unwrap(), "0.000075");
        assert_eq!(response.metadata.get("is_byok").unwrap(), "false");
        assert_eq!(response.metadata.get("cost.upstream").unwrap(), "0.000075");
        assert_eq!(response.metadata.get("cost.prompt").unwrap(), "0.00003");
        assert_eq!(
            response.metadata.get("cost.completion").unwrap(),
            "0.000045"
        );

        drop(server);
        drop(runtime);
    }
}
