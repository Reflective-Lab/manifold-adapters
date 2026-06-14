// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

use std::sync::Arc;

#[cfg(feature = "anthropic")]
use crate::llm::AnthropicBackend;
#[cfg(feature = "arcee")]
use crate::llm::ArceeBackend;
#[cfg(feature = "deepseek")]
use crate::llm::DeepSeekBackend;
#[cfg(feature = "gemini")]
use crate::llm::GeminiBackend;
#[cfg(feature = "kimi")]
use crate::llm::KimiBackend;
#[cfg(feature = "kong")]
use crate::llm::KongBackend;
#[cfg(feature = "minmax")]
use crate::llm::MinMaxBackend;
#[cfg(feature = "mistral")]
use crate::llm::MistralBackend;
#[cfg(feature = "openai")]
use crate::llm::OpenAiBackend;
#[cfg(feature = "openrouter")]
use crate::llm::OpenRouterBackend;
#[cfg(feature = "perplexity")]
use crate::llm::PerplexityBackend;
#[cfg(feature = "qwen")]
use crate::llm::QwenBackend;
#[cfg(feature = "staik")]
use crate::llm::StaikBackend;
#[cfg(feature = "writer")]
use crate::llm::WriterBackend;
use crate::model_selection::{FitnessBreakdown, ModelMetadata, ProviderRegistry, SelectionResult};
use crate::secret::{SecretProvider, default_secret_provider};
use converge_provider::{
    ChatBackendCapabilities, ChatBackendDescriptor, ChatBackendRegistry,
    ChatBackendSelectionConfig, ChatMessage, ChatRequest, ChatRole, ContextWindowTokens,
    DynChatBackend, LatencyMillis, LlmError, ModelName, ProviderName, QualityScore,
    RegisteredChatBackend, RegistryValueError, ResponseFormat,
};

#[derive(Clone)]
pub struct SelectedChatBackend {
    pub backend: Arc<dyn DynChatBackend>,
    pub selection: SelectionResult,
}

impl SelectedChatBackend {
    #[must_use]
    pub fn provider(&self) -> &str {
        &self.selection.selected.provider
    }

    #[must_use]
    pub fn model(&self) -> &str {
        &self.selection.selected.model
    }
}

pub fn select_chat_backend(
    config: &ChatBackendSelectionConfig,
) -> Result<SelectedChatBackend, LlmError> {
    select_chat_backend_with_secret_provider(config, default_secret_provider())
}

pub fn select_chat_backend_with_secret_provider(
    config: &ChatBackendSelectionConfig,
    secrets: &dyn SecretProvider,
) -> Result<SelectedChatBackend, LlmError> {
    let (model_registry, registry_config) = model_registry_for_config(config, secrets)?;
    let selection =
        model_registry.select_with_details(&registry_config.criteria.to_agent_requirements())?;
    let chat_registry = chat_backend_registry_from_candidates(&selection.candidates, secrets)?;
    let resolved = chat_registry.select(&registry_config)?;
    let selection =
        selection_for_resolved_backend(selection, resolved.provider(), resolved.model());
    Ok(SelectedChatBackend {
        backend: resolved.backend(),
        selection,
    })
}

fn selection_for_resolved_backend(
    mut selection: SelectionResult,
    provider: &str,
    model: &str,
) -> SelectionResult {
    if let Some((metadata, fitness)) = selection
        .candidates
        .iter()
        .find(|(metadata, _)| metadata.provider == provider && metadata.model == model)
    {
        selection.selected = metadata.clone();
        selection.fitness = fitness.clone();
    }
    selection
}

/// Selects a chat backend with health probing — iterates ranked candidates and
/// returns the first one that responds to a minimal probe request.
///
/// Use this instead of [`select_chat_backend`] when you want automatic fallback
/// past providers whose API keys exist but are non-functional (e.g. exhausted
/// free-tier quotas, revoked keys, or temporary outages).
pub async fn select_healthy_chat_backend(
    config: &ChatBackendSelectionConfig,
) -> Result<SelectedChatBackend, LlmError> {
    select_healthy_chat_backend_with_secret_provider(config, default_secret_provider()).await
}

/// Like [`select_healthy_chat_backend`] but with an explicit secret provider.
pub async fn select_healthy_chat_backend_with_secret_provider(
    config: &ChatBackendSelectionConfig,
    secrets: &dyn SecretProvider,
) -> Result<SelectedChatBackend, LlmError> {
    let (model_registry, _registry_config) = model_registry_for_config(config, secrets)?;
    let selection = model_registry.select_with_details(&config.criteria.to_agent_requirements())?;

    let mut last_error = None;
    for (candidate, fitness) in &selection.candidates {
        let candidate_selection = SelectionResult {
            selected: candidate.clone(),
            fitness: fitness.clone(),
            candidates: selection.candidates.clone(),
            rejected: selection.rejected.clone(),
        };

        let backend = match registered_chat_backend_for_model(candidate, secrets) {
            Ok(b) => b.backend(),
            Err(e) => {
                tracing::debug!(
                    provider = %candidate.provider,
                    model = %candidate.model,
                    error = %e,
                    "skipping candidate: instantiation failed"
                );
                last_error = Some(e);
                continue;
            }
        };

        match probe_backend(&backend).await {
            Ok(()) => {
                tracing::info!(
                    provider = %candidate.provider,
                    model = %candidate.model,
                    "health probe passed"
                );
                return Ok(SelectedChatBackend {
                    backend,
                    selection: candidate_selection,
                });
            }
            Err(e) => {
                tracing::warn!(
                    provider = %candidate.provider,
                    model = %candidate.model,
                    error = %e,
                    "health probe failed, trying next candidate"
                );
                last_error = Some(e);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| LlmError::ProviderError {
        message: "No healthy provider found among candidates".into(),
        code: None,
    }))
}

fn model_registry_for_config(
    config: &ChatBackendSelectionConfig,
    secrets: &dyn SecretProvider,
) -> Result<(ProviderRegistry, ChatBackendSelectionConfig), LlmError> {
    let mut registry_config = config.clone();
    let registry = if let Some(provider) = config.provider_override.as_deref() {
        let provider = normalize_provider_name(provider).ok_or_else(|| LlmError::InvalidRequest {
            message: format!(
                "Unsupported CONVERGE_LLM_FORCE_PROVIDER={provider}. Expected one of: anthropic, openai, gemini, mistral, arcee, writer, minmax, openrouter, kong, staik, deepseek, kimi, perplexity, qwen."
            ),
        })?;

        if !is_chat_provider_available(provider, secrets) {
            return Err(LlmError::AuthDenied {
                message: format!(
                    "Requested provider {provider} is not available. Configure the matching API key first."
                ),
            });
        }

        registry_config.provider_override = Some(provider.to_string());
        ProviderRegistry::with_providers(&[provider])
    } else {
        chat_provider_registry(secrets)
    };

    Ok((registry, registry_config))
}

async fn probe_backend(backend: &Arc<dyn DynChatBackend>) -> Result<(), LlmError> {
    let request = ChatRequest {
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: "hi".to_string(),
            tool_calls: vec![],
            tool_call_id: None,
        }],
        system: None,
        tools: vec![],
        response_format: ResponseFormat::Text,
        max_tokens: Some(1),
        temperature: None,
        stop_sequences: vec![],
        model: None,
    };
    backend.chat(request).await.map(|_| ())
}

fn chat_provider_registry(secrets: &dyn SecretProvider) -> ProviderRegistry {
    let supported: Vec<&str> = [
        "anthropic",
        "openai",
        "gemini",
        "mistral",
        "arcee",
        "writer",
        "minmax",
        "openrouter",
        "kong",
        "staik",
        "deepseek",
        "kimi",
        "perplexity",
        "qwen",
    ]
    .into_iter()
    .filter(|provider| is_chat_provider_available(provider, secrets))
    .collect();
    ProviderRegistry::with_providers(&supported)
}

fn chat_backend_registry_from_candidates(
    candidates: &[(ModelMetadata, FitnessBreakdown)],
    secrets: &dyn SecretProvider,
) -> Result<ChatBackendRegistry, LlmError> {
    let mut registry = ChatBackendRegistry::new();
    for (candidate, _) in candidates {
        registry.register(registered_chat_backend_for_model(candidate, secrets)?);
    }
    Ok(registry)
}

fn registered_chat_backend_for_model(
    metadata: &ModelMetadata,
    secrets: &dyn SecretProvider,
) -> Result<RegisteredChatBackend, LlmError> {
    let descriptor = descriptor_for_model(metadata)?;
    let backend = instantiate_backend_for_model(metadata, secrets)?;
    Ok(RegisteredChatBackend::new(descriptor, backend))
}

fn descriptor_for_model(metadata: &ModelMetadata) -> Result<ChatBackendDescriptor, LlmError> {
    let capabilities = ChatBackendCapabilities::new()
        .with_reasoning(metadata.has_reasoning)
        .with_web_search(metadata.supports_web_search)
        .with_tool_use(metadata.supports_tool_use)
        .with_vision(metadata.supports_vision)
        .with_context_tokens(
            ContextWindowTokens::new(metadata.context_tokens).map_err(registry_value_error)?,
        )
        .with_structured_output(metadata.supports_structured_output)
        .with_code(metadata.supports_code)
        .with_multilingual(metadata.supports_multilingual)
        .with_content_generation(metadata.supports_content_generation)
        .with_business_acumen(metadata.supports_business_acumen);

    Ok(ChatBackendDescriptor::new(
        ProviderName::new(metadata.provider.clone()).map_err(registry_value_error)?,
        ModelName::new(metadata.model.clone()).map_err(registry_value_error)?,
        metadata.cost_class,
        LatencyMillis::new(metadata.typical_latency_ms).map_err(registry_value_error)?,
        QualityScore::new(metadata.quality).map_err(registry_value_error)?,
    )
    .with_capabilities(capabilities)
    .with_data_sovereignty(metadata.data_sovereignty)
    .with_compliance(metadata.compliance))
}

fn registry_value_error(error: RegistryValueError) -> LlmError {
    LlmError::ProviderError {
        message: error.to_string(),
        code: None,
    }
}

fn instantiate_backend_for_model(
    metadata: &ModelMetadata,
    secrets: &dyn SecretProvider,
) -> Result<Arc<dyn DynChatBackend>, LlmError> {
    let provider = metadata.provider.as_str();
    let model = metadata.model.clone();

    match provider {
        #[cfg(feature = "anthropic")]
        "anthropic" => {
            let backend = AnthropicBackend::from_secret_provider(secrets)
                .map_err(backend_error)?
                .with_model(model);
            Ok(Arc::new(backend))
        }
        #[cfg(feature = "openai")]
        "openai" => {
            let backend = OpenAiBackend::from_secret_provider(secrets)
                .map_err(backend_error)?
                .with_model(model);
            Ok(Arc::new(backend))
        }
        #[cfg(feature = "gemini")]
        "gemini" => {
            let backend = GeminiBackend::from_secret_provider(secrets)
                .map_err(backend_error)?
                .with_model(model);
            Ok(Arc::new(backend))
        }
        #[cfg(feature = "mistral")]
        "mistral" => {
            let backend = MistralBackend::from_secret_provider(secrets)
                .map_err(backend_error)?
                .with_model(model);
            Ok(Arc::new(backend))
        }
        #[cfg(feature = "openrouter")]
        "openrouter" => {
            let backend = OpenRouterBackend::from_secret_provider(secrets)
                .map_err(backend_error)?
                .with_model(model);
            Ok(Arc::new(backend))
        }
        #[cfg(feature = "kong")]
        "kong" => {
            let backend = KongBackend::from_secret_provider(secrets)
                .map_err(backend_error)?
                .with_model(model);
            Ok(Arc::new(backend))
        }
        #[cfg(feature = "staik")]
        "staik" => {
            let backend = StaikBackend::from_secret_provider(secrets)
                .map_err(backend_error)?
                .with_model(model);
            Ok(Arc::new(backend))
        }
        #[cfg(feature = "arcee")]
        "arcee" => {
            let backend = ArceeBackend::from_secret_provider(secrets)
                .map_err(backend_error)?
                .with_model(model);
            Ok(Arc::new(backend))
        }
        #[cfg(feature = "writer")]
        "writer" => {
            let backend = WriterBackend::from_secret_provider(secrets)
                .map_err(backend_error)?
                .with_model(model);
            Ok(Arc::new(backend))
        }
        #[cfg(feature = "minmax")]
        "minmax" => {
            let backend = MinMaxBackend::from_secret_provider(secrets)
                .map_err(backend_error)?
                .with_model(model);
            Ok(Arc::new(backend))
        }
        #[cfg(feature = "deepseek")]
        "deepseek" => {
            let backend = DeepSeekBackend::from_secret_provider(secrets)
                .map_err(backend_error)?
                .with_model(model);
            Ok(Arc::new(backend))
        }
        #[cfg(feature = "kimi")]
        "kimi" => {
            let backend = KimiBackend::from_secret_provider(secrets)
                .map_err(backend_error)?
                .with_model(model);
            Ok(Arc::new(backend))
        }
        #[cfg(feature = "perplexity")]
        "perplexity" => {
            let backend = PerplexityBackend::from_secret_provider(secrets)
                .map_err(backend_error)?
                .with_model(model);
            Ok(Arc::new(backend))
        }
        #[cfg(feature = "qwen")]
        "qwen" => {
            let backend = QwenBackend::from_secret_provider(secrets)
                .map_err(backend_error)?
                .with_model(model);
            Ok(Arc::new(backend))
        }
        _ => Err(LlmError::ProviderError {
            message: format!("Selected provider {provider} does not have a chat backend"),
            code: None,
        }),
    }
}

fn backend_error(error: converge_core::backend::BackendError) -> LlmError {
    LlmError::ProviderError {
        message: error.to_string(),
        code: None,
    }
}

fn is_chat_provider_available(provider: &str, secrets: &dyn SecretProvider) -> bool {
    match provider {
        #[cfg(feature = "anthropic")]
        "anthropic" => secrets.has_secret("ANTHROPIC_API_KEY"),
        #[cfg(feature = "openai")]
        "openai" => secrets.has_secret("OPENAI_API_KEY"),
        #[cfg(feature = "gemini")]
        "gemini" => secrets.has_secret("GEMINI_API_KEY"),
        #[cfg(feature = "mistral")]
        "mistral" => secrets.has_secret("MISTRAL_API_KEY"),
        #[cfg(feature = "openrouter")]
        "openrouter" => secrets.has_secret("OPENROUTER_API_KEY"),
        #[cfg(feature = "kong")]
        "kong" => {
            secrets.has_secret("KONG_API_KEY") && std::env::var("KONG_AI_GATEWAY_URL").is_ok()
        }
        #[cfg(feature = "staik")]
        "staik" => secrets.has_secret("STAIK_API_KEY"),
        #[cfg(feature = "arcee")]
        "arcee" => secrets.has_secret("ARCEE_API_KEY"),
        #[cfg(feature = "writer")]
        "writer" => secrets.has_secret("WRITER_API_KEY"),
        #[cfg(feature = "minmax")]
        "minmax" => secrets.has_secret("MINIMAX_API_KEY"),
        #[cfg(feature = "deepseek")]
        "deepseek" => secrets.has_secret("DEEPSEEK_API_KEY"),
        #[cfg(feature = "kimi")]
        "kimi" => secrets.has_secret("KIMI_API_KEY"),
        #[cfg(feature = "perplexity")]
        "perplexity" => secrets.has_secret("PERPLEXITY_API_KEY"),
        #[cfg(feature = "qwen")]
        "qwen" => secrets.has_secret("QWEN_API_KEY"),
        _ => false,
    }
}

fn normalize_provider_name(value: &str) -> Option<&'static str> {
    match value.trim().to_ascii_lowercase().as_str() {
        "anthropic" | "claude" => Some("anthropic"),
        "openai" | "gpt" => Some("openai"),
        "gemini" | "google" => Some("gemini"),
        "mistral" | "mixtral" => Some("mistral"),
        "openrouter" | "router" => Some("openrouter"),
        "kong" | "kong_gateway" | "kong_ai" => Some("kong"),
        "staik" => Some("staik"),
        "arcee" => Some("arcee"),
        "writer" | "palmyra" => Some("writer"),
        "minmax" | "minimax" => Some("minmax"),
        "deepseek" => Some("deepseek"),
        "kimi" | "moonshot" => Some("kimi"),
        "perplexity" | "sonar" => Some("perplexity"),
        "qwen" | "dashscope" => Some("qwen"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ChatBackendSelectionConfig, select_chat_backend_with_secret_provider,
        selection_for_resolved_backend,
    };
    use crate::model_selection::{FitnessBreakdown, ModelMetadata, SelectionResult};
    use crate::secret::{SecretError, SecretProvider, StaticSecretProvider};
    use converge_core::model_selection::CostClass;

    #[derive(Debug, Default)]
    struct MissingSecretProvider;

    impl SecretProvider for MissingSecretProvider {
        fn get_secret(&self, key: &str) -> Result<crate::secret::SecretString, SecretError> {
            Err(SecretError::NotFound(key.to_string()))
        }
    }

    #[test]
    #[cfg(feature = "gemini")]
    fn provider_override_selects_requested_backend_family() {
        let config = ChatBackendSelectionConfig::default().with_provider_override("gemini");
        let selected =
            select_chat_backend_with_secret_provider(&config, &StaticSecretProvider::new("test"))
                .unwrap();
        assert_eq!(selected.provider(), "gemini");
    }

    #[test]
    fn missing_secrets_fail_selection() {
        let config = ChatBackendSelectionConfig::default();
        let error = select_chat_backend_with_secret_provider(&config, &MissingSecretProvider)
            .err()
            .unwrap();
        assert!(matches!(
            error,
            converge_core::traits::LlmError::ProviderError { .. }
        ));
    }

    #[test]
    fn selection_result_tracks_backend_resolved_by_chat_registry() {
        let anthropic =
            ModelMetadata::new("anthropic", "claude-sonnet-4-6", CostClass::Low, 2500, 0.93);
        let gemini =
            ModelMetadata::new("gemini", "gemini-2.5-flash", CostClass::VeryLow, 800, 0.84);
        let anthropic_fitness = FitnessBreakdown {
            cost_score: 0.8,
            latency_score: 0.5,
            quality_score: 0.93,
            total: 0.75,
        };
        let gemini_fitness = FitnessBreakdown {
            cost_score: 1.0,
            latency_score: 0.84,
            quality_score: 0.84,
            total: 0.9,
        };

        let selection = SelectionResult {
            selected: anthropic.clone(),
            fitness: anthropic_fitness.clone(),
            candidates: vec![
                (anthropic, anthropic_fitness),
                (gemini.clone(), gemini_fitness.clone()),
            ],
            rejected: vec![],
        };

        let selection = selection_for_resolved_backend(selection, "gemini", "gemini-2.5-flash");

        assert_eq!(selection.selected.provider, "gemini");
        assert_eq!(selection.selected.model, "gemini-2.5-flash");
        assert_eq!(selection.fitness, gemini_fitness);
    }

    #[test]
    #[cfg(any(
        feature = "anthropic",
        feature = "openai",
        feature = "gemini",
        feature = "mistral",
        feature = "openrouter"
    ))]
    fn capability_driven_selection_stays_with_instantiable_backends() {
        use converge_core::model_selection::{RequiredCapabilities, SelectionCriteria};

        let config = ChatBackendSelectionConfig::default().with_criteria(
            SelectionCriteria::analysis().with_capabilities(
                RequiredCapabilities::none()
                    .with_structured_output()
                    .with_tool_use(),
            ),
        );
        let selected =
            select_chat_backend_with_secret_provider(&config, &StaticSecretProvider::new("test"))
                .unwrap();
        assert!(matches!(
            selected.provider(),
            "anthropic" | "openai" | "gemini"
        ));
    }

    #[test]
    fn normalize_provider_name_aliases() {
        use super::normalize_provider_name;

        assert_eq!(normalize_provider_name("anthropic"), Some("anthropic"));
        assert_eq!(normalize_provider_name("claude"), Some("anthropic"));
        assert_eq!(normalize_provider_name("CLAUDE"), Some("anthropic"));
        assert_eq!(normalize_provider_name("openai"), Some("openai"));
        assert_eq!(normalize_provider_name("gpt"), Some("openai"));
        assert_eq!(normalize_provider_name("gemini"), Some("gemini"));
        assert_eq!(normalize_provider_name("google"), Some("gemini"));
        assert_eq!(normalize_provider_name("mistral"), Some("mistral"));
        assert_eq!(normalize_provider_name("mixtral"), Some("mistral"));
        assert_eq!(normalize_provider_name("openrouter"), Some("openrouter"));
        assert_eq!(normalize_provider_name("router"), Some("openrouter"));
        assert_eq!(normalize_provider_name("kong"), Some("kong"));
        assert_eq!(normalize_provider_name("kong_gateway"), Some("kong"));
        assert_eq!(normalize_provider_name("kong_ai"), Some("kong"));
        assert_eq!(normalize_provider_name("unknown"), None);
        assert_eq!(normalize_provider_name(""), None);
    }

    #[test]
    fn unsupported_provider_override_fails() {
        let config = ChatBackendSelectionConfig::default().with_provider_override("cohere");
        let result =
            select_chat_backend_with_secret_provider(&config, &StaticSecretProvider::new("test"));
        let err = result.err().expect("should fail");
        assert!(err.to_string().contains("cohere"));
    }

    #[test]
    fn forced_provider_without_key_returns_auth_denied() {
        let config = ChatBackendSelectionConfig::default().with_provider_override("anthropic");
        let result = select_chat_backend_with_secret_provider(&config, &MissingSecretProvider);
        let err = result.err().expect("should fail");
        assert!(matches!(
            err,
            converge_core::traits::LlmError::AuthDenied { .. }
        ));
    }
}
