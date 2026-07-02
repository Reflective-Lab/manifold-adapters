// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Secret management for Manifold adapter credentials.
//!
//! Manifold resolves API keys from secret backends — not from committed
//! `.env` files. Resolution order for [`DefaultSecretProvider`]:
//!
//! 1. **macOS Keychain** (local dev) — service names like `openai-api-key`
//! 2. **GCP Secret Manager** (deployed; feature `gcp-secrets`) — Runway naming
//! 3. **Environment variables** — only when `MANIFOLD_ALLOW_ENV_SECRETS=1` (CI)
//!
//! Non-secret configuration (gateway URLs, model overrides) remains in env.

mod names;

#[cfg(target_os = "macos")]
mod keychain;

#[cfg(feature = "gcp-secrets")]
mod gcp;

use std::sync::OnceLock;

use thiserror::Error;

pub use names::{gcp_short_name, keychain_services, logical_key_aliases};

#[cfg(target_os = "macos")]
pub use keychain::KeychainSecretProvider;

#[cfg(feature = "gcp-secrets")]
pub use gcp::GcpSecretProvider;

/// Error loading a secret.
#[derive(Debug, Error)]
pub enum SecretError {
    /// The requested secret was not found.
    #[error("secret not found: {0}")]
    NotFound(String),

    /// Access to the secret was denied.
    #[error("access denied: {0}")]
    AccessDenied(String),

    /// The secret backend is unavailable.
    #[error("backend unavailable: {0}")]
    Unavailable(String),
}

/// A string that holds a secret value.
///
/// - Never appears in `Debug` output
/// - With `secure` feature: zeroed from memory on drop via `zeroize`
#[derive(Clone)]
pub struct SecretString {
    inner: String,
}

impl SecretString {
    /// Wraps a string as a secret.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            inner: value.into(),
        }
    }

    /// Returns the secret value. Use sparingly — only at the point
    /// where the key is placed into an HTTP header or request body.
    #[must_use]
    pub fn expose(&self) -> &str {
        &self.inner
    }
}

impl std::fmt::Debug for SecretString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[REDACTED]")
    }
}

impl std::fmt::Display for SecretString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[REDACTED]")
    }
}

#[cfg(feature = "secure")]
impl Drop for SecretString {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.inner.zeroize();
    }
}

/// Trait for loading secrets from a backend.
///
/// Implementations must be thread-safe.
pub trait SecretProvider: Send + Sync {
    /// Loads a secret by key name.
    ///
    /// # Errors
    ///
    /// Returns `SecretError` if the secret cannot be loaded.
    fn get_secret(&self, key: &str) -> Result<SecretString, SecretError>;

    /// Checks whether a secret exists without loading it.
    fn has_secret(&self, key: &str) -> bool {
        self.get_secret(key).is_ok()
    }
}

/// Loads secrets from environment variables.
///
/// Intended for CI and explicit local overrides when
/// `MANIFOLD_ALLOW_ENV_SECRETS=1` is set — not the default dev path.
#[derive(Debug, Default, Clone)]
pub struct EnvSecretProvider;

impl SecretProvider for EnvSecretProvider {
    fn get_secret(&self, key: &str) -> Result<SecretString, SecretError> {
        std::env::var(key)
            .map(SecretString::new)
            .map_err(|_| SecretError::NotFound(key.to_string()))
    }
}

/// Default Manifold secret resolver.
///
/// Keychain on macOS, GCP Secret Manager when configured, env only with
/// `MANIFOLD_ALLOW_ENV_SECRETS=1`.
#[derive(Debug, Default)]
pub struct DefaultSecretProvider {
    env: EnvSecretProvider,
    #[cfg(target_os = "macos")]
    keychain: KeychainSecretProvider,
    #[cfg(feature = "gcp-secrets")]
    gcp: Option<GcpSecretProvider>,
}

impl DefaultSecretProvider {
    /// Builds the default resolver for the current host.
    #[must_use]
    pub fn new() -> Self {
        Self {
            env: EnvSecretProvider,
            #[cfg(target_os = "macos")]
            keychain: KeychainSecretProvider,
            #[cfg(feature = "gcp-secrets")]
            gcp: GcpSecretProvider::from_env(),
        }
    }
}

impl SecretProvider for DefaultSecretProvider {
    fn get_secret(&self, key: &str) -> Result<SecretString, SecretError> {
        #[cfg(target_os = "macos")]
        if let Ok(secret) = self.keychain.get_secret(key) {
            return Ok(secret);
        }

        #[cfg(feature = "gcp-secrets")]
        if let Some(gcp) = &self.gcp {
            if let Ok(secret) = gcp.get_secret(key) {
                return Ok(secret);
            }
        }

        if env_secrets_allowed() {
            if let Ok(secret) = self.env.get_secret(key) {
                return Ok(secret);
            }
        }

        Err(SecretError::NotFound(key.to_string()))
    }
}

/// Shared default secret provider for adapter construction and selection.
#[must_use]
pub fn default_secret_provider() -> &'static DefaultSecretProvider {
    static PROVIDER: OnceLock<DefaultSecretProvider> = OnceLock::new();
    PROVIDER.get_or_init(DefaultSecretProvider::new)
}

fn env_secrets_allowed() -> bool {
    matches!(
        std::env::var("MANIFOLD_ALLOW_ENV_SECRETS").ok().as_deref(),
        Some("1" | "true" | "yes")
    )
}

/// A static secret provider for testing.
///
/// Returns the same secret for any key. Never use in production.
#[derive(Clone)]
pub struct StaticSecretProvider {
    value: SecretString,
}

impl StaticSecretProvider {
    /// Creates a provider that always returns the given value.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            value: SecretString::new(value),
        }
    }
}

impl std::fmt::Debug for StaticSecretProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticSecretProvider")
            .field("value", &"[REDACTED]")
            .finish()
    }
}

impl SecretProvider for StaticSecretProvider {
    fn get_secret(&self, _key: &str) -> Result<SecretString, SecretError> {
        Ok(self.value.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_string_debug_is_redacted() {
        let s = SecretString::new("super-secret-key");
        assert_eq!(format!("{s:?}"), "[REDACTED]");
    }

    #[test]
    fn secret_string_display_is_redacted() {
        let s = SecretString::new("super-secret-key");
        assert_eq!(format!("{s}"), "[REDACTED]");
    }

    #[test]
    fn secret_string_expose_returns_value() {
        let s = SecretString::new("my-key-123");
        assert_eq!(s.expose(), "my-key-123");
    }

    #[test]
    fn env_provider_returns_not_found_for_missing_var() {
        let provider = EnvSecretProvider;
        let result = provider.get_secret("CONVERGE_TEST_NONEXISTENT_KEY_12345");
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), SecretError::NotFound(k) if k == "CONVERGE_TEST_NONEXISTENT_KEY_12345")
        );
    }

    #[test]
    fn static_provider_returns_value_for_any_key() {
        let provider = StaticSecretProvider::new("test-secret");
        let s1 = provider.get_secret("ANY_KEY").unwrap();
        let s2 = provider.get_secret("OTHER_KEY").unwrap();
        assert_eq!(s1.expose(), "test-secret");
        assert_eq!(s2.expose(), "test-secret");
    }

    #[test]
    fn static_provider_debug_is_redacted() {
        let provider = StaticSecretProvider::new("secret");
        let debug = format!("{provider:?}");
        assert!(!debug.contains("secret"));
        assert!(debug.contains("REDACTED"));
    }

    #[test]
    fn has_secret_delegates_to_get_secret() {
        let provider = StaticSecretProvider::new("val");
        assert!(provider.has_secret("anything"));

        let env_provider = EnvSecretProvider;
        assert!(!env_provider.has_secret("CONVERGE_TEST_NONEXISTENT_KEY_12345"));
    }
}
