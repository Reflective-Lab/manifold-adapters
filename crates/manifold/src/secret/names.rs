// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Maps Manifold logical secret keys (e.g. `OPENAI_API_KEY`) to Keychain
//! service names and GCP Secret Manager short names.

use std::borrow::Cow;

/// Returns alternate logical keys that may hold the same credential.
#[must_use]
pub fn logical_key_aliases(key: &str) -> Vec<Cow<'_, str>> {
    let mut keys = vec![Cow::Borrowed(key)];
    match key {
        "KONG_API_KEY" => keys.push(Cow::Borrowed("KONG_CONSUMER_API_KEY")),
        "KONG_CONSUMER_API_KEY" => keys.push(Cow::Borrowed("KONG_API_KEY")),
        "MINIMAX_API_KEY" => keys.push(Cow::Borrowed("MINMAX_API_KEY")),
        "MINMAX_API_KEY" => keys.push(Cow::Borrowed("MINIMAX_API_KEY")),
        "GROK_API_KEY" => keys.push(Cow::Borrowed("XAI_API_KEY")),
        "XAI_API_KEY" => keys.push(Cow::Borrowed("GROK_API_KEY")),
        _ => {}
    }
    keys
}

/// Keychain generic-password service names to try for a logical secret key.
#[must_use]
pub fn keychain_services(logical_key: &str) -> Vec<String> {
    let mut services = Vec::new();
    for alias in logical_key_aliases(logical_key) {
        services.extend(explicit_keychain_services(alias.as_ref()));
        let derived = derive_keychain_service(alias.as_ref());
        if !services.iter().any(|s| s == &derived) {
            services.push(derived);
        }
    }
    services
}

/// GCP Secret Manager short name (before `{env}-{app}-` / `{env}-platform-` prefix).
#[must_use]
pub fn gcp_short_name(logical_key: &str) -> String {
    derive_keychain_service(logical_key)
}

fn explicit_keychain_services(logical_key: &str) -> Vec<String> {
    match logical_key {
        "KONG_API_KEY" | "KONG_CONSUMER_API_KEY" => {
            vec![
                "kong-consumer-api-key".to_string(),
                "kong-api-key".to_string(),
            ]
        }
        "MINIMAX_API_KEY" | "MINMAX_API_KEY" => {
            vec!["minimax-api-key".to_string(), "minmax-api-key".to_string()]
        }
        "GROK_API_KEY" | "XAI_API_KEY" => {
            vec!["grok-api-key".to_string(), "xai-api-key".to_string()]
        }
        _ => Vec::new(),
    }
}

fn derive_keychain_service(logical_key: &str) -> String {
    let stem = logical_key
        .trim_end_matches("_API_KEY")
        .trim_end_matches("_SECRET_KEY")
        .trim_end_matches("_KEY");
    let slug = stem.to_ascii_lowercase().replace('_', "-");
    if logical_key.ends_with("_SECRET_KEY") {
        format!("{slug}-secret-key")
    } else {
        format!("{slug}-api-key")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_openai_keychain_service() {
        assert_eq!(
            keychain_services("OPENAI_API_KEY"),
            vec!["openai-api-key".to_string()]
        );
    }

    #[test]
    fn kong_aliases_include_consumer_service() {
        let services = keychain_services("KONG_API_KEY");
        assert!(services.contains(&"kong-consumer-api-key".to_string()));
        assert!(services.contains(&"kong-api-key".to_string()));
    }

    #[test]
    fn baidu_secret_key_uses_secret_suffix() {
        assert_eq!(gcp_short_name("BAIDU_SECRET_KEY"), "baidu-secret-key");
    }
}
