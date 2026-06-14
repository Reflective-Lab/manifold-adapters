// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Generic adapter implementations for Converge contracts.
//!
//! Manifold owns interchangeable operational adapters. It imports Converge
//! contracts and external SDKs; Converge does not import Manifold.

#[cfg(feature = "brave")]
pub mod brave;
#[cfg(any(
    feature = "brave",
    feature = "tavily",
    feature = "perplexity",
    feature = "fetch",
    feature = "feed"
))]
mod capability_registry;
pub mod contract;
pub mod cost_routing;
#[cfg(feature = "qwen")]
pub mod embedding;
pub mod experience;
#[cfg(feature = "extract")]
pub mod extract;
#[cfg(feature = "feed")]
pub mod feed;
#[cfg(feature = "fetch")]
pub mod fetch;
pub mod fuzzy_fitness;
pub mod llm;
#[cfg(feature = "_http")]
pub mod model_catalog;
pub mod object_storage;
#[cfg(feature = "pagination")]
pub mod pagination;
#[cfg(feature = "perplexity")]
pub mod perplexity;
#[cfg(feature = "registry")]
pub mod registry_loader;
#[cfg(feature = "qwen")]
pub mod reranker;
#[cfg(any(
    feature = "brave",
    feature = "tavily",
    feature = "perplexity",
    feature = "fetch",
    feature = "feed"
))]
pub mod search;
pub mod secret;
#[cfg(feature = "tavily")]
pub mod tavily;
#[cfg(feature = "tools")]
pub mod tools;
pub mod vector;
#[cfg(feature = "xml")]
pub mod xml;

pub mod model_selection;

#[cfg(feature = "brave")]
pub use brave::{
    BraveCapability, BraveSearchError, BraveSearchProvider, BraveSearchRequest,
    BraveSearchResponse, BraveSearchResult,
};
#[cfg(any(
    feature = "brave",
    feature = "tavily",
    feature = "perplexity",
    feature = "fetch",
    feature = "feed"
))]
pub use capability_registry::{
    CapabilityRegistry, CapabilityRequirements, SearchProviderFeature, SearchProviderFeatures,
    SearchProviderMeta, WebSearchRequirements,
};
pub use contract::{
    CallTimer, Capability, ProviderCallContext, ProviderMeta, ProviderObservation, Region,
    TokenUsage, canonical_hash,
};
pub use converge_storage::{
    GetResult, ObjectPath, ObjectStore, PutResult, StorageConfig, StorageError, StorageUri,
};
#[cfg(feature = "extract")]
pub use extract::{ExtractError, ExtractedNode, HtmlExtractBackend, ScraperHtmlBackend};
#[cfg(feature = "feed")]
pub use feed::{
    FeedByteLimit, FeedCandidateLimit, FeedDiscoverySource, FeedEndpointCandidate, FeedError,
    FeedFetchBackend, FeedFetchRequest, FeedFetchResponse, FeedFormat, FeedItem, FeedProbeRequest,
    FeedProbeResponse, FeedTimeoutMs, FeedUrl, HttpFeedProvider, HttpStatusCode,
};
#[cfg(feature = "fetch")]
pub use fetch::HttpFetchProvider;
pub use llm::*;
#[cfg(feature = "perplexity")]
pub use perplexity::PerplexitySearchProvider;
#[cfg(any(
    feature = "brave",
    feature = "tavily",
    feature = "perplexity",
    feature = "fetch",
    feature = "feed"
))]
pub use search::{
    SearchDepth, SearchResponsePart, SearchResponseParts, SearchTopic, WebFetchBackend,
    WebFetchByteLimit, WebFetchError, WebFetchMethod, WebFetchRequest, WebFetchResponse,
    WebFetchTimeoutMs, WebFetchUrl, WebSearchBackend, WebSearchError, WebSearchImage,
    WebSearchRequest, WebSearchResponse, WebSearchResult,
};
pub use secret::{
    DefaultSecretProvider, EnvSecretProvider, SecretError, SecretProvider, SecretString,
    default_secret_provider,
};
#[cfg(target_os = "macos")]
pub use secret::KeychainSecretProvider;
#[cfg(feature = "gcp-secrets")]
pub use secret::GcpSecretProvider;
#[cfg(feature = "tavily")]
pub use tavily::TavilySearchProvider;
#[cfg(feature = "tools")]
pub use tools::{
    GraphQlConfig, GraphQlConverter, GraphQlOperationType, InlineToolConfig, InputSchema,
    OpenApiConfig, OpenApiConverter, SourceFilter, ToolCall, ToolDefinition, ToolError,
    ToolErrorKind, ToolFormat, ToolHandler, ToolRegistry, ToolResult, ToolResultContent,
    ToolSource, ToolsConfig, ToolsConfigError,
};
