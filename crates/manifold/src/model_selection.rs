// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT
// See LICENSE file in the project root for full license information.

//! Model selection implementation with provider-specific metadata.
//!
//! This module provides concrete implementations of model selection
//! with hardcoded knowledge of all providers. The abstract interface
//! (`ModelSelectorTrait`, `AgentRequirements`) is in converge-provider.

use serde::{Deserialize, Serialize};

use crate::secret::default_secret_provider;
use converge_provider::LlmError;
use converge_provider::selection::{
    AgentRequirements, ComplianceLevel, CostClass, DataSovereignty, ModelSelectorTrait,
};

/// Breakdown of fitness score components.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitnessBreakdown {
    /// Cost efficiency score (0.0-1.0, higher = cheaper).
    /// VeryLow=1.0, Low=0.8, Medium=0.6, High=0.4, VeryHigh=0.2
    pub cost_score: f64,
    /// Latency efficiency score (0.0-1.0, higher = faster).
    /// Calculated as: 1.0 - (`typical_latency` / `max_allowed_latency`)
    pub latency_score: f64,
    /// Quality score (0.0-1.0, model's quality rating).
    pub quality_score: f64,
    /// Total weighted score: 40% cost + 30% latency + 30% quality.
    pub total: f64,
}

impl std::fmt::Display for FitnessBreakdown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.3} = 40%×cost({:.2}) + 30%×latency({:.2}) + 30%×quality({:.2})",
            self.total, self.cost_score, self.latency_score, self.quality_score
        )
    }
}

/// Result of model selection with detailed information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionResult {
    /// The selected model's metadata.
    pub selected: ModelMetadata,
    /// Fitness breakdown for the selected model.
    pub fitness: FitnessBreakdown,
    /// All candidates that were considered, with their fitness scores.
    /// Sorted by fitness (best first).
    pub candidates: Vec<(ModelMetadata, FitnessBreakdown)>,
    /// Models that were rejected and why.
    pub rejected: Vec<(ModelMetadata, RejectionReason)>,
}

/// Reason why a model was rejected during selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RejectionReason {
    /// Provider not available (no API key).
    ProviderUnavailable,
    /// Cost class exceeds budget.
    CostTooHigh {
        model_cost: CostClass,
        max_allowed: CostClass,
    },
    /// Latency exceeds limit.
    LatencyTooHigh {
        model_latency_ms: u32,
        max_allowed_ms: u32,
    },
    /// Quality below threshold.
    QualityTooLow {
        model_quality: f64,
        min_required: f64,
    },
    /// Reasoning required but not supported.
    ReasoningRequired,
    /// Web search required but not supported.
    WebSearchRequired,
    /// Tool use required but not supported.
    ToolUseRequired,
    /// Vision required but not supported.
    VisionRequired,
    /// Context window too small.
    ContextWindowTooSmall {
        model_context_tokens: usize,
        min_required_tokens: usize,
    },
    /// Structured output required but not supported.
    StructuredOutputRequired,
    /// Code capability required but not supported.
    CodeRequired,
    /// Data sovereignty mismatch.
    DataSovereigntyMismatch {
        required: DataSovereignty,
        model_has: DataSovereignty,
    },
    /// Compliance level mismatch.
    ComplianceMismatch {
        required: ComplianceLevel,
        model_has: ComplianceLevel,
    },
    /// Multilingual required but not supported.
    MultilingualRequired,
    /// Content generation capability required but not supported.
    ContentGenerationRequired,
    /// Business acumen capability required but not supported.
    BusinessAcumenRequired,
    /// Estimated cost exceeds the per-request USD budget.
    OverBudget {
        /// Estimated request cost in USD (prompt + max completion at live prices).
        estimated_cost_usd: f64,
        /// Maximum allowed cost in USD.
        max_cost_usd: f64,
    },
    /// Strict-mode budget rejected this model because it has no pricing data.
    UnpricedUnderStrictBudget,
}

impl std::fmt::Display for RejectionReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProviderUnavailable => write!(f, "provider unavailable (no API key)"),
            Self::CostTooHigh {
                model_cost,
                max_allowed,
            } => {
                write!(f, "cost {model_cost:?} exceeds max {max_allowed:?}")
            }
            Self::LatencyTooHigh {
                model_latency_ms,
                max_allowed_ms,
            } => {
                write!(
                    f,
                    "latency {model_latency_ms}ms exceeds max {max_allowed_ms}ms"
                )
            }
            Self::QualityTooLow {
                model_quality,
                min_required,
            } => {
                write!(f, "quality {model_quality:.2} below min {min_required:.2}")
            }
            Self::ReasoningRequired => write!(f, "reasoning required but not supported"),
            Self::WebSearchRequired => write!(f, "web search required but not supported"),
            Self::ToolUseRequired => write!(f, "tool use required but not supported"),
            Self::VisionRequired => write!(f, "vision required but not supported"),
            Self::ContextWindowTooSmall {
                model_context_tokens,
                min_required_tokens,
            } => {
                write!(
                    f,
                    "context window {model_context_tokens} below required {min_required_tokens}"
                )
            }
            Self::StructuredOutputRequired => {
                write!(f, "structured output required but not supported")
            }
            Self::CodeRequired => write!(f, "code capability required but not supported"),
            Self::DataSovereigntyMismatch {
                required,
                model_has,
            } => {
                write!(f, "data sovereignty {model_has:?} != required {required:?}")
            }
            Self::ComplianceMismatch {
                required,
                model_has,
            } => {
                write!(f, "compliance {model_has:?} != required {required:?}")
            }
            Self::MultilingualRequired => write!(f, "multilingual required but not supported"),
            Self::ContentGenerationRequired => {
                write!(f, "content generation required but not supported")
            }
            Self::BusinessAcumenRequired => {
                write!(f, "business acumen required but not supported")
            }
            Self::OverBudget {
                estimated_cost_usd,
                max_cost_usd,
            } => {
                write!(
                    f,
                    "estimated cost ${estimated_cost_usd:.6} exceeds budget ${max_cost_usd:.6}"
                )
            }
            Self::UnpricedUnderStrictBudget => {
                write!(f, "no pricing data and strict budget mode is active")
            }
        }
    }
}

/// Model metadata for selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct ModelMetadata {
    /// Provider name (e.g., "anthropic", "openai").
    pub provider: String,
    /// Model identifier (e.g., "claude-haiku-4-5-20251001").
    pub model: String,
    /// Cost class of this model.
    pub cost_class: CostClass,
    /// Typical latency in milliseconds.
    pub typical_latency_ms: u32,
    /// Quality score (0.0-1.0).
    pub quality: f64,
    /// Whether this model has strong reasoning capabilities.
    pub has_reasoning: bool,
    /// Whether this model supports web search.
    pub supports_web_search: bool,
    /// Data sovereignty region.
    pub data_sovereignty: DataSovereignty,
    /// Compliance level.
    pub compliance: ComplianceLevel,
    /// Whether this model supports multiple languages.
    pub supports_multilingual: bool,
    // === New fields for enhanced selection ===
    /// Context window size in tokens.
    pub context_tokens: usize,
    /// Whether this model supports tool/function calling.
    pub supports_tool_use: bool,
    /// Whether this model supports vision/images.
    pub supports_vision: bool,
    /// Whether this model supports structured output (JSON mode).
    pub supports_structured_output: bool,
    /// Whether this model is specialized for code.
    pub supports_code: bool,
    /// Whether this model excels at content generation / business writing.
    pub supports_content_generation: bool,
    /// Whether this model has business acumen (financial, strategic, market analysis).
    pub supports_business_acumen: bool,
    /// Provider's country (ISO code, e.g., "US", "FR", "CN").
    pub country: String,
    /// Provider's region (e.g., "US", "EU", "CN", "LOCAL").
    pub region: String,
    /// OpenRouter model ID for cross-referencing live catalog data.
    /// Format: `"<provider>/<model>"` (e.g., `"anthropic/claude-sonnet-4"`).
    /// When set, the model catalog can override pricing and capability fields
    /// at runtime from OpenRouter's published metadata.
    pub openrouter_id: Option<String>,
    /// Input token price in USD per million tokens (from live catalog when available).
    pub pricing_prompt_usd_per_million: Option<f64>,
    /// Output token price in USD per million tokens (from live catalog when available).
    pub pricing_completion_usd_per_million: Option<f64>,
}

impl ModelMetadata {
    /// Creates new model metadata.
    #[must_use]
    pub fn new(
        provider: impl Into<String>,
        model: impl Into<String>,
        cost_class: CostClass,
        typical_latency_ms: u32,
        quality: f64,
    ) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            cost_class,
            typical_latency_ms,
            quality: quality.clamp(0.0, 1.0),
            has_reasoning: false,
            supports_web_search: false,
            data_sovereignty: DataSovereignty::Any,
            compliance: ComplianceLevel::None,
            supports_multilingual: false,
            // New fields with defaults
            context_tokens: 8192,
            supports_tool_use: false,
            supports_vision: false,
            supports_structured_output: false,
            supports_code: false,
            supports_content_generation: false,
            supports_business_acumen: false,
            country: "US".to_string(),
            region: "US".to_string(),
            openrouter_id: None,
            pricing_prompt_usd_per_million: None,
            pricing_completion_usd_per_million: None,
        }
    }

    /// Sets reasoning capability.
    #[must_use]
    pub fn with_reasoning(mut self, has: bool) -> Self {
        self.has_reasoning = has;
        self
    }

    /// Sets web search support.
    #[must_use]
    pub fn with_web_search(mut self, supports: bool) -> Self {
        self.supports_web_search = supports;
        self
    }

    /// Sets data sovereignty.
    #[must_use]
    pub fn with_data_sovereignty(mut self, sovereignty: DataSovereignty) -> Self {
        self.data_sovereignty = sovereignty;
        self
    }

    /// Sets compliance level.
    #[must_use]
    pub fn with_compliance(mut self, compliance: ComplianceLevel) -> Self {
        self.compliance = compliance;
        self
    }

    /// Sets multilingual support.
    #[must_use]
    pub fn with_multilingual(mut self, supports: bool) -> Self {
        self.supports_multilingual = supports;
        self
    }

    /// Sets context window size.
    #[must_use]
    pub fn with_context_tokens(mut self, tokens: usize) -> Self {
        self.context_tokens = tokens;
        self
    }

    /// Sets tool/function calling support.
    #[must_use]
    pub fn with_tool_use(mut self, supports: bool) -> Self {
        self.supports_tool_use = supports;
        self
    }

    /// Sets vision support.
    #[must_use]
    pub fn with_vision(mut self, supports: bool) -> Self {
        self.supports_vision = supports;
        self
    }

    /// Sets structured output support.
    #[must_use]
    pub fn with_structured_output(mut self, supports: bool) -> Self {
        self.supports_structured_output = supports;
        self
    }

    /// Sets code capability.
    #[must_use]
    pub fn with_code(mut self, supports: bool) -> Self {
        self.supports_code = supports;
        self
    }

    /// Sets content generation / business writing capability.
    #[must_use]
    pub fn with_content_generation(mut self, supports: bool) -> Self {
        self.supports_content_generation = supports;
        self
    }

    /// Sets business acumen capability.
    #[must_use]
    pub fn with_business_acumen(mut self, supports: bool) -> Self {
        self.supports_business_acumen = supports;
        self
    }

    /// Sets provider location (country and region).
    #[must_use]
    pub fn with_location(mut self, country: impl Into<String>, region: impl Into<String>) -> Self {
        self.country = country.into();
        self.region = region.into();
        self
    }

    /// Sets the OpenRouter model ID for live catalog cross-reference.
    #[must_use]
    pub fn with_openrouter_id(mut self, id: impl Into<String>) -> Self {
        self.openrouter_id = Some(id.into());
        self
    }

    /// Sets explicit pricing in USD per million tokens.
    #[must_use]
    pub fn with_pricing(mut self, prompt_per_million: f64, completion_per_million: f64) -> Self {
        self.pricing_prompt_usd_per_million = Some(prompt_per_million);
        self.pricing_completion_usd_per_million = Some(completion_per_million);
        self
    }

    /// Checks if this model satisfies the given requirements.
    #[must_use]
    pub fn satisfies(&self, requirements: &AgentRequirements) -> bool {
        // Cost check
        if !requirements
            .max_cost_class
            .allowed_classes()
            .contains(&self.cost_class)
        {
            return false;
        }

        // Latency check
        if self.typical_latency_ms > requirements.max_latency_ms {
            return false;
        }

        // Reasoning check
        if requirements.requires_reasoning && !self.has_reasoning {
            return false;
        }

        // Web search check
        if requirements.requires_web_search && !self.supports_web_search {
            return false;
        }

        if requirements.requires_tool_use && !self.supports_tool_use {
            return false;
        }

        if requirements.requires_vision && !self.supports_vision {
            return false;
        }

        if let Some(min_context_tokens) = requirements.min_context_tokens
            && self.context_tokens < min_context_tokens
        {
            return false;
        }

        if requirements.requires_structured_output && !self.supports_structured_output {
            return false;
        }

        if requirements.requires_code && !self.supports_code {
            return false;
        }

        // Quality check
        if self.quality < requirements.min_quality {
            return false;
        }

        // Data sovereignty check
        if requirements.data_sovereignty != DataSovereignty::Any
            && self.data_sovereignty != requirements.data_sovereignty
        {
            return false;
        }

        // Compliance check
        if requirements.compliance != ComplianceLevel::None
            && self.compliance != requirements.compliance
        {
            return false;
        }

        // Multilingual check
        if requirements.requires_multilingual && !self.supports_multilingual {
            return false;
        }

        // Content generation check
        if requirements.requires_content_generation && !self.supports_content_generation {
            return false;
        }

        // Business acumen check
        if requirements.requires_business_acumen && !self.supports_business_acumen {
            return false;
        }

        true
    }

    /// Calculates a fitness score for matching requirements.
    ///
    /// Returns 0.0 if hard constraints fail; otherwise returns the fuzzy
    /// preference score in [0.0, 1.0]. See [`crate::fuzzy_fitness`] for the
    /// linguistic-variable definitions and rule set.
    #[must_use]
    pub fn fitness_score(&self, requirements: &AgentRequirements) -> f64 {
        if !self.satisfies(requirements) {
            return 0.0;
        }
        crate::fuzzy_fitness::fitness_summary(self).preference
    }

    /// Calculates a detailed fitness breakdown for matching requirements.
    ///
    /// Returns `None` if the model doesn't satisfy requirements.
    ///
    /// Field semantics under the fuzzy scoring:
    /// - `cost_score` = membership in `cost.cheap`
    /// - `latency_score` = membership in `latency.fast`
    /// - `quality_score` = membership in `quality.high`
    /// - `total` = weighted-average preference (0=weak, 0.5=moderate, 1=strong)
    #[must_use]
    pub fn fitness_breakdown(&self, requirements: &AgentRequirements) -> Option<FitnessBreakdown> {
        if !self.satisfies(requirements) {
            return None;
        }
        let summary = crate::fuzzy_fitness::fitness_summary(self);
        Some(FitnessBreakdown {
            cost_score: summary.cost_cheap,
            latency_score: summary.latency_fast,
            quality_score: summary.quality_high,
            total: summary.preference,
        })
    }

    /// Returns the IDs of the fuzzy rules that fired for this model under
    /// the standard selection rule set. Useful for explaining a selection
    /// decision to humans / governance reviewers.
    ///
    /// Returns an empty vec if the model fails hard constraints or if no
    /// rule fired.
    #[must_use]
    pub fn fitness_explanation(&self, requirements: &AgentRequirements) -> Vec<String> {
        if !self.satisfies(requirements) {
            return Vec::new();
        }
        crate::fuzzy_fitness::fitness_summary(self).activated_rule_ids
    }

    /// Determines why this model was rejected for the given requirements.
    ///
    /// Returns `None` if the model satisfies all requirements.
    #[must_use]
    pub fn rejection_reason(&self, requirements: &AgentRequirements) -> Option<RejectionReason> {
        // Cost check
        if !requirements
            .max_cost_class
            .allowed_classes()
            .contains(&self.cost_class)
        {
            return Some(RejectionReason::CostTooHigh {
                model_cost: self.cost_class,
                max_allowed: requirements.max_cost_class,
            });
        }

        // Latency check
        if self.typical_latency_ms > requirements.max_latency_ms {
            return Some(RejectionReason::LatencyTooHigh {
                model_latency_ms: self.typical_latency_ms,
                max_allowed_ms: requirements.max_latency_ms,
            });
        }

        // Reasoning check
        if requirements.requires_reasoning && !self.has_reasoning {
            return Some(RejectionReason::ReasoningRequired);
        }

        // Web search check
        if requirements.requires_web_search && !self.supports_web_search {
            return Some(RejectionReason::WebSearchRequired);
        }

        if requirements.requires_tool_use && !self.supports_tool_use {
            return Some(RejectionReason::ToolUseRequired);
        }

        if requirements.requires_vision && !self.supports_vision {
            return Some(RejectionReason::VisionRequired);
        }

        if let Some(min_context_tokens) = requirements.min_context_tokens
            && self.context_tokens < min_context_tokens
        {
            return Some(RejectionReason::ContextWindowTooSmall {
                model_context_tokens: self.context_tokens,
                min_required_tokens: min_context_tokens,
            });
        }

        if requirements.requires_structured_output && !self.supports_structured_output {
            return Some(RejectionReason::StructuredOutputRequired);
        }

        if requirements.requires_code && !self.supports_code {
            return Some(RejectionReason::CodeRequired);
        }

        // Quality check
        if self.quality < requirements.min_quality {
            return Some(RejectionReason::QualityTooLow {
                model_quality: self.quality,
                min_required: requirements.min_quality,
            });
        }

        // Data sovereignty check
        if requirements.data_sovereignty != DataSovereignty::Any
            && self.data_sovereignty != requirements.data_sovereignty
        {
            return Some(RejectionReason::DataSovereigntyMismatch {
                required: requirements.data_sovereignty,
                model_has: self.data_sovereignty,
            });
        }

        // Compliance check
        if requirements.compliance != ComplianceLevel::None
            && self.compliance != requirements.compliance
        {
            return Some(RejectionReason::ComplianceMismatch {
                required: requirements.compliance,
                model_has: self.compliance,
            });
        }

        // Multilingual check
        if requirements.requires_multilingual && !self.supports_multilingual {
            return Some(RejectionReason::MultilingualRequired);
        }

        // Content generation check
        if requirements.requires_content_generation && !self.supports_content_generation {
            return Some(RejectionReason::ContentGenerationRequired);
        }

        // Business acumen check
        if requirements.requires_business_acumen && !self.supports_business_acumen {
            return Some(RejectionReason::BusinessAcumenRequired);
        }

        None
    }
}

/// Model selector that matches requirements to models.
#[derive(Debug, Clone)]
pub struct ModelSelector {
    /// Available models with metadata.
    models: Vec<ModelMetadata>,
}

impl ModelSelector {
    /// Creates a new model selector with default models.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty selector (add models manually).
    #[must_use]
    pub fn empty() -> Self {
        Self { models: Vec::new() }
    }

    /// Adds a model to the selector.
    #[must_use]
    pub fn with_model(mut self, metadata: ModelMetadata) -> Self {
        self.models.push(metadata);
        self
    }

    /// Lists all models that satisfy the requirements.
    #[must_use]
    pub fn list_satisfying(&self, requirements: &AgentRequirements) -> Vec<&ModelMetadata> {
        self.models
            .iter()
            .filter(|m| m.satisfies(requirements))
            .collect()
    }
}

impl ModelSelectorTrait for ModelSelector {
    fn select(&self, requirements: &AgentRequirements) -> Result<(String, String), LlmError> {
        let mut candidates: Vec<(&ModelMetadata, f64)> = self
            .models
            .iter()
            .filter_map(|m| {
                // Check if provider is available before considering the model
                if !is_provider_available(&m.provider) {
                    return None;
                }

                if m.satisfies(requirements) {
                    Some((m, m.fitness_score(requirements)))
                } else {
                    None
                }
            })
            .collect();

        if candidates.is_empty() {
            return Err(LlmError::ProviderError {
                message: format!(
                    "No model found satisfying requirements: cost <= {:?}, latency <= {}ms, reasoning = {}, web_search = {}, quality >= {:.2}, data_sovereignty = {:?}, compliance = {:?}, multilingual = {}",
                    requirements.max_cost_class,
                    requirements.max_latency_ms,
                    requirements.requires_reasoning,
                    requirements.requires_web_search,
                    requirements.min_quality,
                    requirements.data_sovereignty,
                    requirements.compliance,
                    requirements.requires_multilingual
                ),
                code: None,
            });
        }

        // Sort by fitness score (descending)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return best match
        let best = candidates[0].0;
        Ok((best.provider.clone(), best.model.clone()))
    }
}

impl Default for ModelSelector {
    #[allow(clippy::too_many_lines)] // Default model catalog is comprehensive by design
    fn default() -> Self {
        // Default models with realistic metadata
        Self {
            models: vec![
                // Anthropic
                #[cfg(feature = "anthropic")]
                ModelMetadata::new(
                    "anthropic",
                    "claude-haiku-4-5-20251001",
                    CostClass::VeryLow,
                    1200,
                    0.78,
                )
                .with_tool_use(true)
                .with_vision(true)
                .with_context_tokens(200_000),
                #[cfg(feature = "anthropic")]
                ModelMetadata::new("anthropic", "claude-sonnet-4-6", CostClass::Low, 2500, 0.93)
                    .with_reasoning(true)
                    .with_tool_use(true)
                    .with_vision(true)
                    .with_structured_output(true)
                    .with_code(true)
                    .with_context_tokens(200_000),
                #[cfg(feature = "anthropic")]
                ModelMetadata::new("anthropic", "claude-opus-4-6", CostClass::High, 7000, 0.97)
                    .with_reasoning(true)
                    .with_tool_use(true)
                    .with_vision(true)
                    .with_structured_output(true)
                    .with_code(true)
                    .with_context_tokens(200_000),
                // OpenAI
                #[cfg(feature = "openai")]
                ModelMetadata::new("openai", "gpt-3.5-turbo", CostClass::VeryLow, 1200, 0.70),
                #[cfg(feature = "openai")]
                ModelMetadata::new("openai", "gpt-4", CostClass::Medium, 5000, 0.90)
                    .with_reasoning(true),
                #[cfg(feature = "openai")]
                ModelMetadata::new("openai", "gpt-4-turbo", CostClass::Medium, 4000, 0.92)
                    .with_reasoning(true),
                #[cfg(feature = "openai")]
                ModelMetadata::new("openai", "gpt-4o-mini", CostClass::VeryLow, 1200, 0.82)
                    .with_tool_use(true)
                    .with_vision(true)
                    .with_structured_output(true)
                    .with_multilingual(true)
                    .with_context_tokens(128_000)
                    .with_openrouter_id("openai/gpt-4o-mini"),
                #[cfg(feature = "openai")]
                ModelMetadata::new("openai", "gpt-4o", CostClass::Low, 2500, 0.92)
                    .with_reasoning(true)
                    .with_tool_use(true)
                    .with_vision(true)
                    .with_structured_output(true)
                    .with_code(true)
                    .with_multilingual(true)
                    .with_context_tokens(128_000)
                    .with_openrouter_id("openai/gpt-4o"),
                // o1-mini was deprecated by OpenAI in favor of o3-mini; not on OpenRouter.
                #[cfg(feature = "openai")]
                ModelMetadata::new("openai", "o1-mini", CostClass::Medium, 8000, 0.92)
                    .with_reasoning(true)
                    .with_code(true)
                    .with_context_tokens(128_000),
                #[cfg(feature = "openai")]
                ModelMetadata::new("openai", "o1", CostClass::VeryHigh, 15_000, 0.96)
                    .with_reasoning(true)
                    .with_code(true)
                    .with_structured_output(true)
                    .with_context_tokens(200_000)
                    .with_openrouter_id("openai/o1"),
                #[cfg(feature = "openai")]
                ModelMetadata::new("openai", "o3-mini", CostClass::Medium, 7000, 0.93)
                    .with_reasoning(true)
                    .with_code(true)
                    .with_structured_output(true)
                    .with_context_tokens(200_000)
                    .with_openrouter_id("openai/o3-mini"),
                // Google Gemini
                #[cfg(feature = "gemini")]
                ModelMetadata::new("gemini", "gemini-pro", CostClass::Low, 2000, 0.80)
                    .with_tool_use(true)
                    .with_structured_output(true)
                    .with_context_tokens(32000),
                #[cfg(feature = "gemini")]
                ModelMetadata::new("gemini", "gemini-1.5-flash", CostClass::VeryLow, 800, 0.78)
                    .with_tool_use(true)
                    .with_vision(true)
                    .with_structured_output(true)
                    .with_multilingual(true)
                    .with_context_tokens(1_000_000),
                #[cfg(feature = "gemini")]
                ModelMetadata::new("gemini", "gemini-2.0-flash", CostClass::VeryLow, 700, 0.82)
                    .with_tool_use(true)
                    .with_vision(true)
                    .with_structured_output(true)
                    .with_code(true)
                    .with_reasoning(true)
                    .with_multilingual(true)
                    .with_context_tokens(1_000_000),
                #[cfg(feature = "gemini")]
                ModelMetadata::new("gemini", "gemini-2.5-flash", CostClass::VeryLow, 800, 0.84)
                    .with_tool_use(true)
                    .with_vision(true)
                    .with_structured_output(true)
                    .with_code(true)
                    .with_reasoning(true)
                    .with_multilingual(true)
                    .with_context_tokens(1_000_000),
                // Perplexity (chat with built-in web search)
                #[cfg(feature = "perplexity")]
                ModelMetadata::new("perplexity", "sonar", CostClass::VeryLow, 1500, 0.78)
                    .with_web_search(true)
                    .with_context_tokens(127_000)
                    .with_openrouter_id("perplexity/sonar"),
                #[cfg(feature = "perplexity")]
                ModelMetadata::new("perplexity", "sonar-pro", CostClass::Low, 2500, 0.88)
                    .with_web_search(true)
                    .with_tool_use(true)
                    .with_context_tokens(200_000)
                    .with_openrouter_id("perplexity/sonar-pro"),
                #[cfg(feature = "perplexity")]
                ModelMetadata::new(
                    "perplexity",
                    "sonar-reasoning",
                    CostClass::Medium,
                    5000,
                    0.92,
                )
                .with_web_search(true)
                .with_reasoning(true)
                .with_context_tokens(127_000),
                #[cfg(feature = "perplexity")]
                ModelMetadata::new(
                    "perplexity",
                    "sonar-reasoning-pro",
                    CostClass::Medium,
                    6000,
                    0.94,
                )
                .with_web_search(true)
                .with_reasoning(true)
                .with_tool_use(true)
                .with_context_tokens(200_000)
                .with_openrouter_id("perplexity/sonar-reasoning-pro"),
                // Qwen (Alibaba DashScope)
                // Qwen models accessed via DashScope. Note: qwen-turbo and
                // qwen-max are DashScope-only and not published on OpenRouter,
                // so they have no openrouter_id and pricing stays static.
                #[cfg(feature = "qwen")]
                ModelMetadata::new("qwen", "qwen-turbo", CostClass::VeryLow, 1500, 0.72)
                    .with_tool_use(true)
                    .with_multilingual(true)
                    .with_context_tokens(8000),
                #[cfg(feature = "qwen")]
                ModelMetadata::new("qwen", "qwen-plus", CostClass::Low, 2500, 0.82)
                    .with_tool_use(true)
                    .with_multilingual(true)
                    .with_context_tokens(32_000)
                    .with_openrouter_id("qwen/qwen-plus"),
                #[cfg(feature = "qwen")]
                ModelMetadata::new("qwen", "qwen-max", CostClass::Medium, 3500, 0.88)
                    .with_tool_use(true)
                    .with_structured_output(true)
                    .with_multilingual(true)
                    .with_context_tokens(32_000),
                // OpenRouter — routes to any upstream model via openrouter.ai
                #[cfg(feature = "openrouter")]
                ModelMetadata::new(
                    "openrouter",
                    "anthropic/claude-haiku-4.5",
                    CostClass::VeryLow,
                    1200,
                    0.78,
                )
                .with_tool_use(true)
                .with_vision(true)
                .with_context_tokens(200_000),
                #[cfg(feature = "openrouter")]
                ModelMetadata::new(
                    "openrouter",
                    "anthropic/claude-sonnet-4",
                    CostClass::Low,
                    2500,
                    0.93,
                )
                .with_reasoning(true)
                .with_tool_use(true)
                .with_vision(true)
                .with_structured_output(true)
                .with_code(true)
                .with_context_tokens(200_000),
                #[cfg(feature = "openrouter")]
                ModelMetadata::new("openrouter", "openai/gpt-4o", CostClass::Low, 2500, 0.92)
                    .with_reasoning(true)
                    .with_tool_use(true)
                    .with_vision(true)
                    .with_structured_output(true)
                    .with_code(true)
                    .with_context_tokens(128_000),
                #[cfg(feature = "openrouter")]
                ModelMetadata::new(
                    "openrouter",
                    "openai/gpt-4o-mini",
                    CostClass::VeryLow,
                    1200,
                    0.82,
                )
                .with_tool_use(true)
                .with_vision(true)
                .with_structured_output(true)
                .with_context_tokens(128_000),
                #[cfg(feature = "openrouter")]
                ModelMetadata::new(
                    "openrouter",
                    "google/gemini-2.5-flash",
                    CostClass::VeryLow,
                    800,
                    0.84,
                )
                .with_tool_use(true)
                .with_vision(true)
                .with_structured_output(true)
                .with_code(true)
                .with_reasoning(true)
                .with_multilingual(true)
                .with_context_tokens(1_000_000),
                #[cfg(feature = "openrouter")]
                ModelMetadata::new(
                    "openrouter",
                    "meta-llama/llama-3.1-70b-instruct",
                    CostClass::VeryLow,
                    1500,
                    0.80,
                )
                .with_tool_use(true)
                .with_code(true)
                .with_context_tokens(128_000),
                #[cfg(feature = "openrouter")]
                ModelMetadata::new(
                    "openrouter",
                    "mistralai/mistral-large",
                    CostClass::Medium,
                    4000,
                    0.88,
                )
                .with_reasoning(true)
                .with_tool_use(true)
                .with_structured_output(true)
                .with_code(true)
                .with_multilingual(true)
                .with_context_tokens(128_000),
                // MinMax — only MiniMax-Text-01 is currently active; abab series
                // is deprecated as of late 2025 and rejected by the API.
                #[cfg(feature = "minmax")]
                ModelMetadata::new("minmax", "MiniMax-Text-01", CostClass::Low, 2500, 0.82)
                    .with_reasoning(true)
                    .with_tool_use(true)
                    .with_structured_output(true)
                    .with_context_tokens(1_000_000),
                // Grok
                #[cfg(feature = "grok")]
                ModelMetadata::new("grok", "grok-beta", CostClass::Medium, 3000, 0.80),
                // Mistral
                #[cfg(feature = "mistral")]
                ModelMetadata::new(
                    "mistral",
                    "mistral-large-latest",
                    CostClass::Medium,
                    4000,
                    0.88,
                )
                .with_reasoning(true)
                .with_tool_use(true)
                .with_structured_output(true)
                .with_code(true)
                .with_multilingual(true)
                .with_context_tokens(128_000),
                #[cfg(feature = "mistral")]
                ModelMetadata::new(
                    "mistral",
                    "mistral-medium-latest",
                    CostClass::Low,
                    2500,
                    0.82,
                )
                .with_reasoning(true)
                .with_tool_use(true)
                .with_structured_output(true)
                .with_code(true)
                .with_multilingual(true)
                .with_context_tokens(32_000),
                // DeepSeek
                #[cfg(feature = "deepseek")]
                ModelMetadata::new("deepseek", "deepseek-chat", CostClass::VeryLow, 1500, 0.78)
                    .with_tool_use(true)
                    .with_structured_output(true)
                    .with_code(true)
                    .with_context_tokens(64_000)
                    .with_openrouter_id("deepseek/deepseek-chat"),
                #[cfg(feature = "deepseek")]
                ModelMetadata::new("deepseek", "deepseek-reasoner", CostClass::Low, 3500, 0.88)
                    .with_reasoning(true)
                    .with_code(true)
                    .with_context_tokens(64_000)
                    .with_openrouter_id("deepseek/deepseek-r1"),
                // Baidu ERNIE (China)
                #[cfg(feature = "baidu")]
                ModelMetadata::new("baidu", "ernie-bot", CostClass::Low, 2500, 0.80)
                    .with_data_sovereignty(DataSovereignty::China)
                    .with_multilingual(true),
                #[cfg(feature = "baidu")]
                ModelMetadata::new("baidu", "ernie-bot-turbo", CostClass::VeryLow, 1500, 0.75)
                    .with_data_sovereignty(DataSovereignty::China)
                    .with_multilingual(true),
                // Zhipu GLM (China)
                #[cfg(feature = "zhipu")]
                ModelMetadata::new("zhipu", "glm-4", CostClass::Low, 2500, 0.82)
                    .with_data_sovereignty(DataSovereignty::China)
                    .with_multilingual(true),
                #[cfg(feature = "zhipu")]
                ModelMetadata::new("zhipu", "glm-4.5", CostClass::Medium, 3000, 0.88)
                    .with_data_sovereignty(DataSovereignty::China)
                    .with_reasoning(true)
                    .with_multilingual(true),
                // Kimi (Moonshot AI)
                #[cfg(feature = "kimi")]
                ModelMetadata::new("kimi", "moonshot-v1-8k", CostClass::Low, 2000, 0.80)
                    .with_multilingual(true)
                    .with_context_tokens(8_000),
                #[cfg(feature = "kimi")]
                ModelMetadata::new("kimi", "moonshot-v1-32k", CostClass::Medium, 3000, 0.85)
                    .with_reasoning(true)
                    .with_multilingual(true)
                    .with_context_tokens(32_000),
                #[cfg(feature = "kimi")]
                ModelMetadata::new("kimi", "moonshot-v1-128k", CostClass::Medium, 4000, 0.86)
                    .with_reasoning(true)
                    .with_multilingual(true)
                    .with_context_tokens(128_000),
                // Apertus (Switzerland, EU digital sovereignty)
                #[cfg(feature = "apertus")]
                ModelMetadata::new("apertus", "apertus-v1", CostClass::Medium, 4000, 0.85)
                    .with_data_sovereignty(DataSovereignty::Switzerland)
                    .with_compliance(ComplianceLevel::GDPR)
                    .with_multilingual(true),
                // Kong AI Gateway — proxies to upstream models via kong.example.com
                // Staik — Swedish OpenAI-compatible provider (EU/SE hosted)
                #[cfg(feature = "staik")]
                ModelMetadata::new("staik", "gemma4:31b", CostClass::VeryLow, 1800, 0.88)
                    .with_reasoning(true)
                    .with_vision(true)
                    .with_context_tokens(262_144)
                    .with_data_sovereignty(DataSovereignty::EU)
                    .with_compliance(ComplianceLevel::GDPR),
                #[cfg(feature = "staik")]
                ModelMetadata::new("staik", "qwen3.6:35b-a3b", CostClass::VeryLow, 2000, 0.89)
                    .with_reasoning(true)
                    .with_vision(true)
                    .with_context_tokens(262_144)
                    .with_data_sovereignty(DataSovereignty::EU)
                    .with_compliance(ComplianceLevel::GDPR),
                #[cfg(feature = "staik")]
                ModelMetadata::new("staik", "qwen3.5:9b", CostClass::VeryLow, 1200, 0.82)
                    .with_context_tokens(262_144)
                    .with_data_sovereignty(DataSovereignty::EU)
                    .with_compliance(ComplianceLevel::GDPR),
                #[cfg(feature = "kong")]
                ModelMetadata::new("kong", "gpt-4o", CostClass::Low, 2500, 0.92)
                    .with_reasoning(true)
                    .with_tool_use(true)
                    .with_vision(true)
                    .with_structured_output(true)
                    .with_code(true)
                    .with_context_tokens(128_000),
                #[cfg(feature = "kong")]
                ModelMetadata::new("kong", "gpt-4o-mini", CostClass::VeryLow, 1200, 0.85)
                    .with_tool_use(true)
                    .with_vision(true)
                    .with_structured_output(true)
                    .with_context_tokens(128_000),
                #[cfg(feature = "kong")]
                ModelMetadata::new("kong", "claude-sonnet-4", CostClass::Low, 2500, 0.93)
                    .with_reasoning(true)
                    .with_tool_use(true)
                    .with_vision(true)
                    .with_structured_output(true)
                    .with_code(true)
                    .with_context_tokens(200_000),
            ],
        }
    }
}

#[cfg(test)]
static SKIP_AVAILABILITY_CHECK: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
pub fn set_skip_availability_check(skip: bool) {
    SKIP_AVAILABILITY_CHECK.store(skip, std::sync::atomic::Ordering::SeqCst);
}

/// Checks if a provider is available (credential resolvable via Manifold secrets).
#[must_use]
pub fn is_provider_available(provider: &str) -> bool {
    // In test mode, we can bypass availability checks to allow mock providers
    #[cfg(test)]
    if SKIP_AVAILABILITY_CHECK.load(std::sync::atomic::Ordering::SeqCst) {
        return true;
    }

    #[allow(unused_variables)]
    let secrets = default_secret_provider();
    match provider {
        #[cfg(feature = "anthropic")]
        "anthropic" => secrets.has_secret("ANTHROPIC_API_KEY"),
        #[cfg(feature = "openai")]
        "openai" => secrets.has_secret("OPENAI_API_KEY"),
        #[cfg(feature = "gemini")]
        "gemini" => secrets.has_secret("GEMINI_API_KEY"),
        #[cfg(feature = "perplexity")]
        "perplexity" => secrets.has_secret("PERPLEXITY_API_KEY"),
        #[cfg(feature = "openrouter")]
        "openrouter" => secrets.has_secret("OPENROUTER_API_KEY"),
        #[cfg(feature = "qwen")]
        "qwen" => secrets.has_secret("QWEN_API_KEY"),
        #[cfg(feature = "minmax")]
        "minmax" => secrets.has_secret("MINIMAX_API_KEY"),
        #[cfg(feature = "grok")]
        "grok" => secrets.has_secret("GROK_API_KEY"),
        #[cfg(feature = "mistral")]
        "mistral" => secrets.has_secret("MISTRAL_API_KEY"),
        #[cfg(feature = "deepseek")]
        "deepseek" => secrets.has_secret("DEEPSEEK_API_KEY"),
        #[cfg(feature = "baidu")]
        "baidu" => secrets.has_secret("BAIDU_API_KEY") && secrets.has_secret("BAIDU_SECRET_KEY"),
        #[cfg(feature = "zhipu")]
        "zhipu" => secrets.has_secret("ZHIPU_API_KEY"),
        #[cfg(feature = "kimi")]
        "kimi" => secrets.has_secret("KIMI_API_KEY"),
        #[cfg(feature = "apertus")]
        "apertus" => secrets.has_secret("APERTUS_API_KEY"),
        #[cfg(feature = "staik")]
        "staik" => secrets.has_secret("STAIK_API_KEY"),
        // Search providers
        #[cfg(feature = "brave")]
        "brave" => secrets.has_secret("BRAVE_API_KEY"),
        _ => false,
    }
}

/// Checks if Brave Search is available.
#[must_use]
pub fn is_brave_available() -> bool {
    #[cfg(feature = "brave")]
    {
        is_provider_available("brave")
    }
    #[cfg(not(feature = "brave"))]
    {
        false
    }
}

/// Runtime provider registry that tracks available providers and allows
/// dynamic metadata updates.
///
/// This registry:
/// 1. Filters models by available providers (based on API keys)
/// 2. Allows dynamic updates to metadata (pricing, latency, etc.)
/// 3. Maintains requirements-based selection logic
#[derive(Debug, Clone)]
pub struct ProviderRegistry {
    /// Base selector with all models (static metadata).
    base_selector: ModelSelector,
    /// Available providers (checked at runtime).
    available_providers: std::collections::HashSet<String>,
    /// Dynamic metadata overrides (updates to pricing, latency, etc.).
    metadata_overrides: std::collections::HashMap<(String, String), ModelMetadata>,
}

impl ProviderRegistry {
    /// Creates a new registry that checks available providers from environment.
    ///
    /// Only providers with API keys set will be considered for selection.
    #[must_use]
    pub fn from_env() -> Self {
        let base_selector = ModelSelector::new();

        // Check all known providers (LLMs and search)
        let known_providers = vec![
            // LLM providers
            "anthropic",
            "openai",
            "gemini",
            "perplexity",
            "openrouter",
            "qwen",
            "minmax",
            "grok",
            "mistral",
            "deepseek",
            "baidu",
            "zhipu",
            "kimi",
            "apertus",
            "staik",
            // Search providers
            "brave",
        ];

        let available_providers: std::collections::HashSet<String> = known_providers
            .into_iter()
            .filter(|p| is_provider_available(p))
            .map(std::string::ToString::to_string)
            .collect();

        Self {
            base_selector,
            available_providers,
            metadata_overrides: std::collections::HashMap::new(),
        }
    }

    /// Creates a registry with explicit provider availability.
    ///
    /// Use this when you want to control which providers are available
    /// programmatically (e.g., from a config file or user input).
    #[must_use]
    pub fn with_providers(providers: &[&str]) -> Self {
        let base_selector = ModelSelector::new();
        let available_providers: std::collections::HashSet<String> = providers
            .iter()
            .map(std::string::ToString::to_string)
            .collect();

        Self {
            base_selector,
            available_providers,
            metadata_overrides: std::collections::HashMap::new(),
        }
    }

    /// Updates metadata for a specific model (e.g., pricing, latency).
    ///
    /// This allows dynamic updates to model characteristics without
    /// rebuilding the entire registry.
    pub fn update_metadata(
        &mut self,
        provider: impl Into<String>,
        model: impl Into<String>,
        metadata: ModelMetadata,
    ) {
        self.metadata_overrides
            .insert((provider.into(), model.into()), metadata);
    }

    /// Applies pricing and capability data from a live OpenRouter catalog as
    /// metadata overrides. Each model with an `openrouter_id` gets its pricing
    /// fields populated from the catalog (when found).
    ///
    /// Models without an `openrouter_id` or not present in the catalog are
    /// left untouched.
    #[cfg(feature = "_http")]
    pub fn apply_catalog(&mut self, catalog: &crate::model_catalog::ModelCatalog) {
        let models = self.base_selector.models.clone();
        for model in &models {
            let Some(or_id) = model.openrouter_id.as_deref() else {
                continue;
            };
            let Some(entry) = catalog.get(or_id) else {
                continue;
            };
            let Some((prompt, completion)) = catalog.pricing_per_million(or_id) else {
                continue;
            };
            let mut updated = model.clone();
            updated.pricing_prompt_usd_per_million = Some(prompt);
            updated.pricing_completion_usd_per_million = Some(completion);
            if entry.context_length > 0 {
                updated.context_tokens = entry.context_length;
            }
            // Only flip booleans ON when catalog confirms — never OFF, since our
            // hand-curated defaults may have correct truths the catalog doesn't expose.
            if entry.supports_tools() {
                updated.supports_tool_use = true;
            }
            if entry.supports_vision() {
                updated.supports_vision = true;
            }
            if entry.supports_structured_output() {
                updated.supports_structured_output = true;
            }
            if entry.supports_reasoning() {
                updated.has_reasoning = true;
            }
            self.metadata_overrides
                .insert((model.provider.clone(), model.model.clone()), updated);
        }
    }

    /// Like [`Self::from_env`] but additionally applies live pricing and
    /// capability data from an OpenRouter catalog.
    #[cfg(feature = "_http")]
    #[must_use]
    pub fn from_env_with_catalog(catalog: &crate::model_catalog::ModelCatalog) -> Self {
        let mut registry = Self::from_env();
        registry.apply_catalog(catalog);
        registry
    }

    /// Lists all available models that satisfy the requirements.
    #[must_use]
    pub fn list_available(&self, requirements: &AgentRequirements) -> Vec<&ModelMetadata> {
        self.base_selector
            .list_satisfying(requirements)
            .into_iter()
            .filter(|m| self.available_providers.contains(&m.provider))
            .collect()
    }

    /// Gets the list of available providers.
    #[must_use]
    pub fn available_providers(&self) -> Vec<&str> {
        self.available_providers
            .iter()
            .map(std::string::String::as_str)
            .collect()
    }

    /// Checks if a provider is available.
    #[must_use]
    pub fn is_available(&self, provider: &str) -> bool {
        self.available_providers.contains(provider)
    }

    /// Selects the best model with detailed information about the selection process.
    ///
    /// Returns a `SelectionResult` containing:
    /// - The selected model and its fitness breakdown
    /// - All candidates that were considered (sorted by fitness)
    /// - Models that were rejected and why
    ///
    /// # Errors
    ///
    /// Returns error if no model satisfies the requirements.
    pub fn select_with_details(
        &self,
        requirements: &AgentRequirements,
    ) -> Result<SelectionResult, LlmError> {
        let mut candidates: Vec<(ModelMetadata, FitnessBreakdown)> = Vec::new();
        let mut rejected: Vec<(ModelMetadata, RejectionReason)> = Vec::new();

        // Process all models in the base selector
        for model in &self.base_selector.models {
            // Check provider availability first
            if !self.available_providers.contains(&model.provider) {
                rejected.push((model.clone(), RejectionReason::ProviderUnavailable));
                continue;
            }

            // Use override if available
            let metadata = self
                .metadata_overrides
                .get(&(model.provider.clone(), model.model.clone()))
                .unwrap_or(model);

            // Check if model satisfies requirements
            if let Some(breakdown) = metadata.fitness_breakdown(requirements) {
                candidates.push((metadata.clone(), breakdown));
            } else if let Some(reason) = metadata.rejection_reason(requirements) {
                rejected.push((metadata.clone(), reason));
            }
        }

        if candidates.is_empty() {
            let available = self
                .available_providers
                .iter()
                .map(std::string::String::as_str)
                .collect::<Vec<_>>()
                .join(", ");
            return Err(LlmError::ProviderError {
                message: format!(
                    "No available model found satisfying requirements. Available providers: [{}]",
                    if available.is_empty() {
                        "none (set API keys)".to_string()
                    } else {
                        available
                    }
                ),
                code: None,
            });
        }

        // Sort by fitness score (descending)
        candidates.sort_by(|a, b| {
            b.1.total
                .partial_cmp(&a.1.total)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Extract the best
        let (selected, fitness) = candidates[0].clone();

        Ok(SelectionResult {
            selected,
            fitness,
            candidates,
            rejected,
        })
    }
}

impl ModelSelectorTrait for ProviderRegistry {
    fn select(&self, requirements: &AgentRequirements) -> Result<(String, String), LlmError> {
        // Get all models that satisfy requirements
        let all_candidates = self.base_selector.list_satisfying(requirements);

        // Filter by available providers and apply overrides
        let mut candidates: Vec<(&ModelMetadata, f64)> = all_candidates
            .iter()
            .filter(|m| self.available_providers.contains(&m.provider))
            .map(|m| {
                // Use override if available, otherwise use base metadata
                let metadata = self
                    .metadata_overrides
                    .get(&(m.provider.clone(), m.model.clone()))
                    .unwrap_or(m);
                (metadata, metadata.fitness_score(requirements))
            })
            .collect();

        if candidates.is_empty() {
            let available = self
                .available_providers
                .iter()
                .map(std::string::String::as_str)
                .collect::<Vec<_>>()
                .join(", ");
            return Err(LlmError::ProviderError {
                message: format!(
                    "No available model found satisfying requirements. Available providers: [{}]",
                    if available.is_empty() {
                        "none (set API keys)".to_string()
                    } else {
                        available
                    }
                ),
                code: None,
            });
        }

        // Sort by fitness score (descending)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return best match
        let best = candidates[0].0;
        Ok((best.provider.clone(), best.model.clone()))
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::from_env()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use converge_core::model_selection::CostClass;
    use parking_lot::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_gemini_rejection_when_unconfigured() {
        let _guard = TEST_LOCK.lock();
        set_skip_availability_check(false);

        let selector = ModelSelector::new();
        let reqs = AgentRequirements::balanced();

        // If GEMINI_API_KEY is not set, Gemini should NOT be selected even if it matches
        if std::env::var("GEMINI_API_KEY").is_err() {
            let result = selector.select(&reqs);
            if let Ok((provider, _)) = result {
                assert_ne!(
                    provider, "gemini",
                    "Gemini should NOT be selected when API key is missing"
                );
            }
        }
    }

    #[test]
    fn test_registry_with_explicit_providers() {
        let registry = ProviderRegistry::with_providers(&["anthropic", "openai"]);
        assert!(registry.is_available("anthropic"));
        assert!(registry.is_available("openai"));
        assert!(!registry.is_available("gemini"));
    }

    #[test]
    fn test_metadata_override() {
        let mut registry = ProviderRegistry {
            base_selector: ModelSelector::empty()
                .with_model(ModelMetadata::new(
                    "anthropic",
                    "claude-haiku-4-5-20251001",
                    CostClass::VeryLow,
                    1900,
                    0.60,
                ))
                .with_model(ModelMetadata::new(
                    "openai",
                    "gpt-3.5-turbo",
                    CostClass::VeryLow,
                    800,
                    0.70,
                )),
            available_providers: ["anthropic", "openai"]
                .into_iter()
                .map(String::from)
                .collect(),
            metadata_overrides: std::collections::HashMap::new(),
        };

        // Override latency and quality enough to change candidate ranking.
        let updated = ModelMetadata::new(
            "anthropic",
            "claude-haiku-4-5-20251001",
            CostClass::VeryLow,
            500,
            0.95,
        );
        registry.update_metadata("anthropic", "claude-haiku-4-5-20251001", updated);

        let reqs = AgentRequirements::fast_cheap();
        let (provider, model) = registry.select(&reqs).unwrap();
        assert_eq!(provider, "anthropic");
        assert_eq!(model, "claude-haiku-4-5-20251001");
    }

    #[test]
    fn test_model_selection() {
        let _guard = TEST_LOCK.lock();
        set_skip_availability_check(true);
        let selector = ModelSelector::empty()
            .with_model(ModelMetadata::new(
                "anthropic",
                "claude-haiku-test",
                CostClass::VeryLow,
                1200,
                0.78,
            ))
            .with_model(ModelMetadata::new(
                "openai",
                "gpt-4-test",
                CostClass::Medium,
                5000,
                0.90,
            ));
        let reqs = AgentRequirements::fast_cheap();

        let (provider, model) = selector.select(&reqs).unwrap();
        assert_eq!(provider, "anthropic");
        assert_eq!(model, "claude-haiku-test");
    }

    #[test]
    fn test_selection_requires_reasoning_and_web_search() {
        let _guard = TEST_LOCK.lock();
        set_skip_availability_check(true);
        let selector = ModelSelector::empty()
            .with_model(ModelMetadata::new(
                "alpha",
                "basic",
                CostClass::Low,
                1200,
                0.85,
            ))
            .with_model(
                ModelMetadata::new("beta", "reasoning-only", CostClass::Low, 1400, 0.88)
                    .with_reasoning(true),
            )
            .with_model(
                ModelMetadata::new("gamma", "reasoning-search", CostClass::Low, 1500, 0.87)
                    .with_reasoning(true)
                    .with_web_search(true),
            );

        let reqs = AgentRequirements::new(CostClass::Low, 5000, true).with_web_search(true);
        let (provider, model) = selector.select(&reqs).unwrap();
        assert_eq!(provider, "gamma");
        assert_eq!(model, "reasoning-search");
    }

    #[test]
    fn test_selection_respects_data_sovereignty_and_compliance() {
        let _guard = TEST_LOCK.lock();
        set_skip_availability_check(true);
        let selector = ModelSelector::empty()
            .with_model(
                ModelMetadata::new("us", "us-model", CostClass::Low, 1500, 0.85)
                    .with_data_sovereignty(DataSovereignty::US),
            )
            .with_model(
                ModelMetadata::new("eu", "eu-gdpr", CostClass::Low, 1800, 0.86)
                    .with_data_sovereignty(DataSovereignty::EU)
                    .with_compliance(ComplianceLevel::GDPR),
            );

        let reqs = AgentRequirements::balanced()
            .with_data_sovereignty(DataSovereignty::EU)
            .with_compliance(ComplianceLevel::GDPR);
        let (provider, model) = selector.select(&reqs).unwrap();
        assert_eq!(provider, "eu");
        assert_eq!(model, "eu-gdpr");
    }

    #[test]
    fn test_selection_requires_multilingual() {
        let _guard = TEST_LOCK.lock();
        set_skip_availability_check(true);
        let selector = ModelSelector::empty()
            .with_model(
                ModelMetadata::new("mono", "fast", CostClass::VeryLow, 800, 0.80)
                    .with_multilingual(false),
            )
            .with_model(
                ModelMetadata::new("multi", "polyglot", CostClass::Low, 1200, 0.82)
                    .with_multilingual(true),
            );

        let reqs = AgentRequirements::new(CostClass::Low, 2000, false).with_multilingual(true);
        let (provider, model) = selector.select(&reqs).unwrap();
        assert_eq!(provider, "multi");
        assert_eq!(model, "polyglot");
    }

    #[test]
    fn test_selection_respects_context_window() {
        let _guard = TEST_LOCK.lock();
        set_skip_availability_check(true);
        let selector = ModelSelector::empty()
            .with_model(
                ModelMetadata::new("gemini", "flash", CostClass::VeryLow, 700, 0.82)
                    .with_context_tokens(1_000_000),
            )
            .with_model(
                ModelMetadata::new("gemini", "pro", CostClass::Medium, 3000, 0.88)
                    .with_context_tokens(2_000_000),
            );

        let reqs = AgentRequirements::balanced().with_min_context(2_000_000);
        let (provider, model) = selector.select(&reqs).unwrap();
        assert_eq!(provider, "gemini");
        assert_eq!(model, "pro");
    }

    #[test]
    fn test_selection_respects_tool_use_and_structured_output() {
        let _guard = TEST_LOCK.lock();
        set_skip_availability_check(true);
        let selector = ModelSelector::empty()
            .with_model(
                ModelMetadata::new("plain", "text-only", CostClass::Low, 1000, 0.90)
                    .with_tool_use(false)
                    .with_structured_output(false),
            )
            .with_model(
                ModelMetadata::new("agentic", "tool-json", CostClass::Low, 1200, 0.88)
                    .with_tool_use(true)
                    .with_structured_output(true),
            );

        let reqs = AgentRequirements::balanced()
            .with_tool_use(true)
            .with_structured_output(true);
        let (provider, model) = selector.select(&reqs).unwrap();
        assert_eq!(provider, "agentic");
        assert_eq!(model, "tool-json");
    }
}
