# Changelog

All notable changes to manifold will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic
Versioning before 1.0 with the usual pre-1.0 compatibility caveats.

## [Unreleased]

### Added

- `DefaultSecretProvider` resolves adapter credentials from macOS Keychain
  (local dev) and GCP Secret Manager (`gcp-secrets` feature, deployed hosts).
  Environment variables are opt-in via `MANIFOLD_ALLOW_ENV_SECRETS=1` for CI —
  not the default dev path.
- `KeychainSecretProvider` (macOS) and `GcpSecretProvider` (Runway naming:
  `{env}-platform-{name}` / `{env}-{app}-{name}`).
- Logical-key aliases (`KONG_API_KEY` ↔ `KONG_CONSUMER_API_KEY`,
  `MINIMAX_API_KEY` ↔ `MINMAX_API_KEY`, `GROK_API_KEY` ↔ `XAI_API_KEY`).

### Changed

- `select_chat_backend`, LLM `from_env()` constructors, and
  `is_provider_available` now use `default_secret_provider()` instead of
  reading env vars directly.

### Fixed

- `select_chat_backend` now reports `SelectedChatBackend::provider()` and
  `::model()` from the backend actually resolved by `ChatBackendRegistry`, so
  downstream provenance cannot pair one provider with another provider's model.

## [1.1.1] - 2026-05-17

### Changed

- Bump `converge-core`, `converge-experience`, `converge-pack`,
  `converge-provider`, `converge-storage` to `3.9.1`. No public API changes.
- First clean `just release-check` run including all five gates.

## [1.1.0] - 2026-05-07

### Added

- HuggingFace `ObjectStore` (read-only) under `object_storage::huggingface`.
- Generic `HtmlExtractBackend` trait with a `scraper`-backed implementation
  (`extract` module).
- Codex-style LLM adapter wiring under `llm`.

### Changed

- Cargo package renamed from `manifold` to `converge-manifold-adapters`; Rust
  library name remains `manifold`.
- Switched the dotenv dev-dep from the unmaintained `dotenv` crate to
  `dotenvy 0.15`.
- `deny.toml` updated to keep the security gate green under the foundation
  baseline.

### Fixed

- Code reformatted with `cargo fmt`; the `Format` and `Lint` CI jobs are
  now green.

## [0.1.0] - 2026-05-05

### Added

- Workspace scaffold for generic Converge adapters.
- Object-store adapter builders for local, S3, and GCS backends.
- SurrealDB and LanceDB experience-store adapters.
- LanceDB vector recall adapter.
- Standard GitHub community health files.
- `AGENTS.md` and `Justfile` workflow entrypoints.
