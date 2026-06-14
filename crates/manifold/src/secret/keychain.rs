// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! macOS Keychain secret provider.

use std::process::Command;

use super::names::keychain_services;
use super::{SecretError, SecretProvider, SecretString};

/// Loads secrets from the macOS Keychain via the `security` CLI.
///
/// Service names follow `manifold::secret::names` (e.g. `OPENAI_API_KEY` →
/// `openai-api-key`). Store entries with:
///
/// ```text
/// security add-generic-password -a "$USER" -s openai-api-key -w
/// ```
#[derive(Debug, Default, Clone)]
pub struct KeychainSecretProvider;

impl SecretProvider for KeychainSecretProvider {
    fn get_secret(&self, key: &str) -> Result<SecretString, SecretError> {
        for service in keychain_services(key) {
            if let Some(value) = read_keychain_service(&service) {
                return Ok(SecretString::new(value));
            }
        }
        Err(SecretError::NotFound(key.to_string()))
    }
}

fn read_keychain_service(service: &str) -> Option<String> {
    let output = Command::new("security")
        .args(["find-generic-password", "-s", service, "-w"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}
