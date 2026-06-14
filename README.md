# manifold

[![CI](https://github.com/Reflective-Lab/manifold-adapters/actions/workflows/ci.yml/badge.svg)](https://github.com/Reflective-Lab/manifold-adapters/actions/workflows/ci.yml)
[![Coverage](https://github.com/Reflective-Lab/manifold-adapters/actions/workflows/coverage.yml/badge.svg)](https://github.com/Reflective-Lab/manifold-adapters/actions/workflows/coverage.yml)
[![Security](https://github.com/Reflective-Lab/manifold-adapters/actions/workflows/security.yml/badge.svg)](https://github.com/Reflective-Lab/manifold-adapters/actions/workflows/security.yml)
[![Stability](https://github.com/Reflective-Lab/manifold-adapters/actions/workflows/stability.yml/badge.svg)](https://github.com/Reflective-Lab/manifold-adapters/actions/workflows/stability.yml)
[![Crates.io](https://img.shields.io/crates/v/converge-manifold-adapters.svg)](https://crates.io/crates/converge-manifold-adapters)
[![docs.rs](https://docs.rs/converge-manifold-adapters/badge.svg)](https://docs.rs/converge-manifold-adapters)
[![dependency status](https://deps.rs/repo/github/Reflective-Lab/manifold-adapters/status.svg)](https://deps.rs/repo/github/Reflective-Lab/manifold-adapters)
![MSRV](https://img.shields.io/badge/MSRV-1.96.0-blue)
<img alt="gitleaks badge" src="https://img.shields.io/badge/protected%20by-gitleaks-blue">
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Generic adapter implementations for Converge contracts.

`manifold` is a Converge extension for provider, storage, vector, experience,
fetch, feed, search, LLM, embedding, and tool adapters where the concrete
vendor should be hidden behind an interchangeable capability.

Cargo package: `converge-manifold-adapters`. Rust library name remains
`manifold`.

## Why It Exists

Converge owns contracts and authority. Manifold owns concrete adapter code.
This keeps the foundation small while still giving deployments reusable
implementations for common operational backends.

## What Manifold Owns

- Object-store adapter builders.
- Experience-store adapters.
- Vector recall adapters.
- Generic LLM provider adapters.
- Search, fetch, feed, embedding, reranking, vector, and tool adapters.

## Boundary

| Layer | Responsibility |
|---|---|
| Converge | Storage, vector, provider, and experience contracts. |
| Manifold | Concrete generic adapters for those contracts. |
| Embassy | Source-specific ports where foreign-system identity is part of the API. |
| Products | Runtime assembly, credentials, tenancy, and provider selection. |

Use Manifold when two providers can plausibly be swapped behind the same
contract. Use `../embassy` when the API must name the external source.

## Repository Layout

```text
crates/manifold/
  src/llm/             LLM chat adapters and provider selection helpers
  config/models.yaml   LLM provider/model registry used by selection
  src/brave.rs         Brave search adapter
  src/tavily.rs        Tavily search adapter
  src/fetch.rs         HTTP web fetch adapter
  src/feed.rs          RSS/Atom/JSON Feed adapter
  src/embedding/       Qwen-VL embedding adapter
  src/reranker/        Qwen-VL reranking adapter
  src/tools/           OpenAPI/GraphQL tool conversion and registry
  src/object_storage/  Local, S3, and GCS object-store builders
  src/experience/      SurrealDB and LanceDB experience stores
  src/vector/          LanceDB vector recall adapter
  src/secret.rs        Secret-provider abstraction for adapter credentials
  src/model_selection.rs Provider/model metadata used by LLM selection
  src/registry_loader.rs YAML model registry loader
  src/lib.rs           Public adapter surface and Converge re-exports
```

## Current Adapter Families

| Feature | Adapter |
|---|---|
| `object-local` | Local filesystem object store |
| `object-s3` | S3-compatible object store |
| `object-gcs` | Google Cloud Storage object store |
| `experience-surrealdb` | SurrealDB experience store |
| `experience-lancedb` | LanceDB vector-indexed experience store |
| `vector-lancedb` | LanceDB vector recall |
| `anthropic` | Anthropic Claude chat adapter |
| `openai` | OpenAI chat adapter |
| `gemini` | Google Gemini chat adapter |
| `mistral` | Mistral chat adapter |
| `openrouter` | OpenRouter chat adapter |
| `kong` | Kong AI Gateway chat adapter |
| `staik` | Staik chat adapter |
| `arcee`, `writer`, `minmax` | OpenAI-compatible chat adapters |
| `brave` | Brave web search adapter |
| `tavily` | Tavily web search adapter |
| `fetch` | HTTP web fetch adapter |
| `feed` | HTTP RSS/Atom/JSON Feed adapter |
| `qwen` | Qwen-VL embedding and reranking adapters |
| `tools` | OpenAPI/GraphQL tool conversion and registry |

## Feature Flags

- Default: `object-local`.
- `object-all`: local, S3, and GCS object storage.
- `all-storage`: all current object, experience, and vector adapters.
- `llm-all`: all current LLM chat adapter modules and selection metadata.
- `registry`: YAML model registry loader and compiled-in model catalog.
- `search-all`: Brave, Tavily, HTTP fetch, and feed adapters.
- `all-vector`: LanceDB vector adapter plus in-memory/vector helper surface.
- `tools`: OpenAPI/GraphQL tool conversion and registry.

## Usage

```rust
use manifold::object_storage::build_store;
use manifold::{StorageConfig, StorageUri};

let config = StorageConfig {
    uri: StorageUri::Local("./objects".into()),
    prefix: None,
    public: false,
    endpoint: None,
    region: None,
};

let store = build_store(&config)?;
```

```rust
use manifold::{AnthropicBackend, default_secret_provider};
use converge_provider::{ChatBackend, ChatRequest};

// Keys resolve from macOS Keychain (dev) or GCP Secret Manager (deployed).
let backend = AnthropicBackend::from_secret_provider(default_secret_provider())?;
// Product or Runtime Runway assembly registers the backend handle through
// converge_provider::ChatBackendRegistry.
```

### Credentials

Manifold does not read committed `.env` files for API keys. `DefaultSecretProvider`
resolves secrets in order:

1. **macOS Keychain** — `OPENAI_API_KEY` → service `openai-api-key`
2. **GCP Secret Manager** (`gcp-secrets` feature) — Runway names like
   `dev-platform-openai-api-key`
3. **Environment** — only when `MANIFOLD_ALLOW_ENV_SECRETS=1` (CI)

Non-secret configuration (gateway URLs, regions) remains in env vars.

## Development

```sh
just check
just check-all
just test
just lint
just doc
```

Converge platform dependencies resolve from crates.io at `3.9.1` or newer.

## Project Files

- [AGENTS.md](AGENTS.md) - agent entrypoint and boundary rules.
- [CHANGELOG.md](CHANGELOG.md) - release notes.
- [CONTRIBUTING.md](CONTRIBUTING.md) - contribution guide.
- [SECURITY.md](SECURITY.md) - vulnerability reporting and operator notes.
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - community expectations.

## Status

Scaffolded on 2026-05-05 as the generic adapter home for extracted Converge
extension functionality.

## License

MIT - see [LICENSE](LICENSE).
