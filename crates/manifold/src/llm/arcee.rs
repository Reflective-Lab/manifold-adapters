// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

use crate::llm::OpenAiBackend;
use crate::secret::{SecretProvider, default_secret_provider};
use converge_core::backend::{BackendError, BackendResult};
use converge_provider::{BoxFuture, ChatBackend, ChatRequest, ChatResponse, LlmError};

/// Arcee AI backend — OpenAI-compatible endpoint for Arcee's small, efficient models.
///
/// Configure with `ARCEE_API_KEY`. Default model: `trinity-large-preview`.
pub struct ArceeBackend {
    inner: OpenAiBackend,
}

impl ArceeBackend {
    pub fn from_env() -> BackendResult<Self> {
        Self::from_secret_provider(default_secret_provider())
    }

    pub fn from_secret_provider(secrets: &dyn SecretProvider) -> BackendResult<Self> {
        let api_key =
            secrets
                .get_secret("ARCEE_API_KEY")
                .map_err(|e| BackendError::Unavailable {
                    message: format!("ARCEE_API_KEY: {e}"),
                })?;
        Ok(Self {
            inner: OpenAiBackend::try_new(api_key.expose().to_string())?
                .with_base_url("https://api.arcee.ai/v2")
                .with_model("trinity-large-preview"),
        })
    }

    #[must_use]
    pub fn with_model(self, model: impl Into<String>) -> Self {
        Self {
            inner: self.inner.with_model(model),
        }
    }
}

impl ChatBackend for ArceeBackend {
    type ChatFut<'a>
        = BoxFuture<'a, Result<ChatResponse, LlmError>>
    where
        Self: 'a;

    fn chat(&self, req: ChatRequest) -> Self::ChatFut<'_> {
        self.inner.chat(req)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secret::StaticSecretProvider;

    #[test]
    fn constructs_from_secret_provider() {
        let backend =
            ArceeBackend::from_secret_provider(&StaticSecretProvider::new("test-key")).unwrap();
        // with_model returns a new backend without panicking
        let _ = backend.with_model("arcee-ai/llama-3.1-70b-instruct");
    }

    #[test]
    fn missing_key_returns_unavailable() {
        use crate::secret::{SecretError, SecretProvider, SecretString};

        struct Empty;
        impl SecretProvider for Empty {
            fn get_secret(&self, key: &str) -> Result<SecretString, SecretError> {
                Err(SecretError::NotFound(key.to_string()))
            }
        }

        let err = ArceeBackend::from_secret_provider(&Empty).err().unwrap();
        assert!(matches!(
            err,
            converge_core::backend::BackendError::Unavailable { .. }
        ));
    }
}
