// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! GCP Secret Manager provider (Runtime Runway naming convention).

use std::collections::HashMap;
use std::sync::Mutex;

use reqwest::StatusCode;
use reqwest::blocking::Client;

use super::names::{gcp_short_name, logical_key_aliases};
use super::{SecretError, SecretProvider, SecretString};

const METADATA_TOKEN_URL: &str =
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";

/// Loads secrets from GCP Secret Manager.
///
/// Secret names follow Runtime Runway:
/// - App-scoped: `{env}-{app}-{short_name}`
/// - Platform-scoped: `{env}-platform-{short_name}`
/// - Bare short name (dev / shared vaults): `{short_name}`
///
/// Configure with `GCP_PROJECT` or `GOOGLE_CLOUD_PROJECT`, plus optional
/// `ENV` (default `dev`) and `APP_NAME`.
#[derive(Debug)]
pub struct GcpSecretProvider {
    project_id: String,
    env: String,
    app: Option<String>,
    client: Client,
    cache: Mutex<HashMap<String, SecretString>>,
}

impl GcpSecretProvider {
    /// Builds a provider when `GCP_PROJECT` or `GOOGLE_CLOUD_PROJECT` is set.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        let project_id = std::env::var("GCP_PROJECT")
            .or_else(|_| std::env::var("GOOGLE_CLOUD_PROJECT"))
            .ok()?;
        Some(Self::new(
            project_id,
            std::env::var("ENV").unwrap_or_else(|_| "dev".into()),
            std::env::var("APP_NAME").ok(),
        ))
    }

    #[must_use]
    pub fn new(project_id: impl Into<String>, env: impl Into<String>, app: Option<String>) -> Self {
        Self {
            project_id: project_id.into(),
            env: env.into(),
            app,
            #[allow(clippy::disallowed_methods)]
            client: Client::new(),
            cache: Mutex::new(HashMap::new()),
        }
    }

    fn candidate_secret_names(&self, logical_key: &str) -> Vec<String> {
        let short = gcp_short_name(logical_key);
        let mut names = Vec::new();
        names.push(format!("{}-platform-{}", self.env, short));
        if let Some(app) = &self.app {
            names.push(format!("{}-{}-{}", self.env, app, short));
        }
        names.push(short);
        names
    }

    fn fetch_raw(&self, full_name: &str) -> Result<SecretString, SecretError> {
        let url = format!(
            "https://secretmanager.googleapis.com/v1/projects/{}/secrets/{}/versions/latest:access",
            self.project_id, full_name
        );
        let token = metadata_access_token(&self.client)?;
        let resp = self
            .client
            .get(&url)
            .bearer_auth(&token)
            .send()
            .map_err(|e| SecretError::Unavailable(e.to_string()))?;
        if resp.status() == StatusCode::NOT_FOUND {
            return Err(SecretError::NotFound(full_name.to_string()));
        }
        if !resp.status().is_success() {
            return Err(SecretError::Unavailable(format!(
                "GCP Secret Manager HTTP {} for {full_name}",
                resp.status()
            )));
        }
        let body: serde_json::Value = resp
            .json()
            .map_err(|e| SecretError::Unavailable(e.to_string()))?;
        let encoded = body["payload"]["data"]
            .as_str()
            .ok_or_else(|| SecretError::NotFound(full_name.to_string()))?;
        let decoded = base64_decode(encoded)
            .map_err(|e| SecretError::Unavailable(format!("decode {full_name}: {e}")))?;
        Ok(SecretString::new(decoded))
    }
}

impl SecretProvider for GcpSecretProvider {
    fn get_secret(&self, key: &str) -> Result<SecretString, SecretError> {
        if let Ok(cache) = self.cache.lock() {
            if let Some(hit) = cache.get(key) {
                return Ok(hit.clone());
            }
        }

        let mut last_err = SecretError::NotFound(key.to_string());
        for alias in logical_key_aliases(key) {
            for full_name in self.candidate_secret_names(alias.as_ref()) {
                match self.fetch_raw(&full_name) {
                    Ok(secret) => {
                        if let Ok(mut cache) = self.cache.lock() {
                            cache.insert(key.to_string(), secret.clone());
                        }
                        return Ok(secret);
                    }
                    Err(err) => last_err = err,
                }
            }
        }
        Err(last_err)
    }
}

fn metadata_access_token(client: &Client) -> Result<String, SecretError> {
    let resp = client
        .get(METADATA_TOKEN_URL)
        .header("Metadata-Flavor", "Google")
        .send()
        .map_err(|e| SecretError::Unavailable(format!("GCE metadata token: {e}")))?;
    if !resp.status().is_success() {
        return Err(SecretError::Unavailable(format!(
            "GCE metadata token HTTP {}",
            resp.status()
        )));
    }
    let body: serde_json::Value = resp
        .json()
        .map_err(|e| SecretError::Unavailable(e.to_string()))?;
    body["access_token"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| SecretError::Unavailable("missing access_token in metadata response".into()))
}

fn base64_decode(s: &str) -> Result<String, String> {
    let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, s)
        .map_err(|e| e.to_string())?;
    String::from_utf8(bytes).map_err(|e| e.to_string())
}
