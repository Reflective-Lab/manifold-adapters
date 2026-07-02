// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Feed provider — probes, fetches, and parses RSS, Atom, and JSON Feed.
//!
//! Providers produce observations, not editorial decisions. This module keeps
//! feed transport and parsing generic: downstream packs decide source trust,
//! rights, gates, and domain meaning.

use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use converge_pack::{BackendId, BasisPoints, ContentHash};
use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};
use reqwest::Url;
use serde::de;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::fetch::HttpFetchProvider;
use crate::search::{
    MAX_WEB_FETCH_BYTES, MAX_WEB_FETCH_TIMEOUT_MS, WebFetchBackend, WebFetchError, WebFetchRequest,
    WebFetchResponse, validate_public_http_url,
};

const MAX_FEED_CANDIDATES: usize = 256;

/// Absolute URL used by feed discovery and fetch observations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct FeedUrl(String);

impl FeedUrl {
    pub fn new(value: impl Into<String>) -> Result<Self, FeedError> {
        let value = value.into();
        let url = Url::parse(&value).map_err(|error| FeedError::InvalidUrl(error.to_string()))?;
        validate_public_http_url(&url).map_err(|error| FeedError::InvalidUrl(error.to_string()))?;
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn parse(&self) -> Result<Url, FeedError> {
        Url::parse(self.as_str()).map_err(|error| FeedError::InvalidUrl(error.to_string()))
    }
}

impl std::fmt::Display for FeedUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for FeedUrl {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

/// Positive byte limit for feed fetches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct FeedByteLimit(NonZeroUsize);

impl FeedByteLimit {
    pub fn new(value: usize) -> Result<Self, FeedError> {
        let value = NonZeroUsize::new(value)
            .ok_or_else(|| FeedError::InvalidLimit("max_bytes must be greater than zero".into()))?;
        if value.get() > MAX_WEB_FETCH_BYTES {
            return Err(FeedError::InvalidLimit(format!(
                "max_bytes must be <= {MAX_WEB_FETCH_BYTES}"
            )));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn get(self) -> usize {
        self.0.get()
    }
}

impl<'de> Deserialize<'de> for FeedByteLimit {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = usize::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

/// Positive candidate limit for feed discovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct FeedCandidateLimit(NonZeroUsize);

impl FeedCandidateLimit {
    pub fn new(value: usize) -> Result<Self, FeedError> {
        let value = NonZeroUsize::new(value).ok_or_else(|| {
            FeedError::InvalidLimit("max_candidates must be greater than zero".into())
        })?;
        if value.get() > MAX_FEED_CANDIDATES {
            return Err(FeedError::InvalidLimit(format!(
                "max_candidates must be <= {MAX_FEED_CANDIDATES}"
            )));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn get(self) -> usize {
        self.0.get()
    }
}

impl<'de> Deserialize<'de> for FeedCandidateLimit {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = usize::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

/// Positive timeout in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct FeedTimeoutMs(NonZeroU64);

impl FeedTimeoutMs {
    pub fn new(value: u64) -> Result<Self, FeedError> {
        let value = NonZeroU64::new(value).ok_or_else(|| {
            FeedError::InvalidLimit("timeout_ms must be greater than zero".into())
        })?;
        if value.get() > MAX_WEB_FETCH_TIMEOUT_MS {
            return Err(FeedError::InvalidLimit(format!(
                "timeout_ms must be <= {MAX_WEB_FETCH_TIMEOUT_MS}"
            )));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

impl<'de> Deserialize<'de> for FeedTimeoutMs {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u64::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

/// Valid HTTP status code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct HttpStatusCode(u16);

impl HttpStatusCode {
    pub fn new(value: u16) -> Result<Self, FeedError> {
        if (100..=599).contains(&value) {
            Ok(Self(value))
        } else {
            Err(FeedError::InvalidStatus(value))
        }
    }

    #[must_use]
    pub fn get(self) -> u16 {
        self.0
    }
}

impl<'de> Deserialize<'de> for HttpStatusCode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u16::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

/// Feed format identified by probe hints or parser detection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedFormat {
    Rss,
    Atom,
    JsonFeed,
    #[default]
    Unknown,
}

/// How a candidate feed endpoint was discovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedDiscoverySource {
    AlternateLink,
    CommonPath,
    DirectUrl,
}

/// Request to discover likely feed endpoints for a site or direct feed URL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedProbeRequest {
    pub url: FeedUrl,
    #[serde(default = "default_probe_common_paths")]
    pub probe_common_paths: bool,
    #[serde(default = "default_max_candidates")]
    pub max_candidates: FeedCandidateLimit,
    #[serde(default = "default_max_bytes")]
    pub max_bytes: FeedByteLimit,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: FeedTimeoutMs,
}

fn default_probe_common_paths() -> bool {
    true
}

fn default_max_candidates() -> FeedCandidateLimit {
    FeedCandidateLimit::new(16).expect("default candidate limit is non-zero")
}

fn default_max_bytes() -> FeedByteLimit {
    FeedByteLimit::new(1_048_576).expect("default byte limit is non-zero")
}

fn default_timeout_ms() -> FeedTimeoutMs {
    FeedTimeoutMs::new(30_000).expect("default timeout is non-zero")
}

impl FeedProbeRequest {
    pub fn new(url: impl Into<String>) -> Result<Self, FeedError> {
        Ok(Self {
            url: FeedUrl::new(url)?,
            probe_common_paths: default_probe_common_paths(),
            max_candidates: default_max_candidates(),
            max_bytes: default_max_bytes(),
            timeout_ms: default_timeout_ms(),
        })
    }
}

/// Candidate feed endpoint discovered during probing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedEndpointCandidate {
    pub url: FeedUrl,
    pub format_hint: FeedFormat,
    pub discovery_source: FeedDiscoverySource,
    pub confidence_bps: BasisPoints,
}

/// Feed probe response. Candidates are observations and require downstream
/// promotion before use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedProbeResponse {
    pub provider: BackendId,
    pub input_url: FeedUrl,
    pub candidates: Vec<FeedEndpointCandidate>,
}

/// Request to fetch and parse one feed endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedFetchRequest {
    pub url: FeedUrl,
    #[serde(default)]
    pub headers: Vec<(String, String)>,
    #[serde(default = "default_max_bytes")]
    pub max_bytes: FeedByteLimit,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: FeedTimeoutMs,
}

impl FeedFetchRequest {
    pub fn new(url: impl Into<String>) -> Result<Self, FeedError> {
        Ok(Self {
            url: FeedUrl::new(url)?,
            headers: Vec::new(),
            max_bytes: default_max_bytes(),
            timeout_ms: default_timeout_ms(),
        })
    }

    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }
}

/// Normalized feed item observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedItem {
    pub id: Option<String>,
    pub title: Option<String>,
    pub link: Option<FeedUrl>,
    pub summary: Option<String>,
    pub published_at: Option<String>,
    pub updated_at: Option<String>,
    pub authors: Vec<String>,
    pub categories: Vec<String>,
    pub item_hash: ContentHash,
}

/// Feed fetch response. Raw body is retained so callers can store the exact
/// representation used to derive normalized items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedFetchResponse {
    pub provider: BackendId,
    pub url: FeedUrl,
    pub status: HttpStatusCode,
    pub content_type: Option<String>,
    pub format: FeedFormat,
    pub raw_hash: ContentHash,
    pub raw_body: String,
    pub truncated: bool,
    pub feed_title: Option<String>,
    pub feed_link: Option<FeedUrl>,
    pub feed_updated_at: Option<String>,
    pub items: Vec<FeedItem>,
}

/// Feed provider errors.
#[derive(Debug, thiserror::Error)]
pub enum FeedError {
    #[error("fetch error: {0}")]
    Fetch(String),
    #[error("invalid url: {0}")]
    InvalidUrl(String),
    #[error("invalid limit: {0}")]
    InvalidLimit(String),
    #[error("invalid HTTP status: {0}")]
    InvalidStatus(u16),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("unsupported feed format")]
    UnsupportedFormat,
}

impl From<WebFetchError> for FeedError {
    fn from(error: WebFetchError) -> Self {
        Self::Fetch(error.to_string())
    }
}

/// Executable contract for provider-local feed adapters.
pub trait FeedFetchBackend: Send + Sync {
    fn provider_name(&self) -> &'static str;

    fn probe(&self, request: &FeedProbeRequest) -> Result<FeedProbeResponse, FeedError>;

    fn fetch_feed(&self, request: &FeedFetchRequest) -> Result<FeedFetchResponse, FeedError>;
}

/// HTTP-backed feed provider.
#[derive(Clone)]
pub struct HttpFeedProvider {
    fetch_backend: Arc<dyn WebFetchBackend>,
}

impl std::fmt::Debug for HttpFeedProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpFeedProvider").finish_non_exhaustive()
    }
}

impl HttpFeedProvider {
    /// Creates a new `HttpFeedProvider`.
    ///
    /// # Errors
    ///
    /// Propagates [`WebFetchError::Network`] if the underlying HTTP client
    /// cannot be initialised (e.g. missing system TLS stack).
    pub fn new() -> Result<Self, crate::search::WebFetchError> {
        Ok(Self {
            fetch_backend: Arc::new(HttpFetchProvider::new()?),
        })
    }

    #[must_use]
    pub fn with_fetch_backend(fetch_backend: Arc<dyn WebFetchBackend>) -> Self {
        Self { fetch_backend }
    }
}

impl FeedFetchBackend for HttpFeedProvider {
    fn provider_name(&self) -> &'static str {
        "http-feed"
    }

    fn probe(&self, request: &FeedProbeRequest) -> Result<FeedProbeResponse, FeedError> {
        let input_url = request.url.parse()?;
        let mut candidates = Vec::new();

        if looks_like_feed_url(input_url.path()) {
            candidates.push(FeedEndpointCandidate {
                url: FeedUrl::new(input_url.to_string())?,
                format_hint: format_hint_from_url(input_url.path()),
                discovery_source: FeedDiscoverySource::DirectUrl,
                confidence_bps: BasisPoints::new(9_000).expect("static basis points are valid"),
            });
        }

        let fetch = WebFetchRequest::new(input_url.as_str())?
            .with_max_bytes(request.max_bytes.get())?
            .with_timeout_ms(request.timeout_ms.get())?;
        if let Ok(response) = self.fetch_backend.fetch(&fetch) {
            candidates.extend(discover_alternate_links(&response.body, &response.url)?);
        }

        if request.probe_common_paths {
            candidates.extend(common_feed_candidates(&input_url));
        }

        dedup_candidates(&mut candidates);
        candidates.truncate(request.max_candidates.get());

        Ok(FeedProbeResponse {
            provider: BackendId::new(self.provider_name()),
            input_url: request.url.clone(),
            candidates,
        })
    }

    fn fetch_feed(&self, request: &FeedFetchRequest) -> Result<FeedFetchResponse, FeedError> {
        let fetch_request = web_fetch_request_from_feed(request)?;
        let response = self.fetch_backend.fetch(&fetch_request)?;
        parse_feed_response(self.provider_name(), response)
    }
}

fn web_fetch_request_from_feed(request: &FeedFetchRequest) -> Result<WebFetchRequest, FeedError> {
    let mut fetch = WebFetchRequest::new(request.url.as_str())?
        .with_max_bytes(request.max_bytes.get())?
        .with_timeout_ms(request.timeout_ms.get())?;
    for (name, value) in &request.headers {
        fetch = fetch.with_header(name, value);
    }
    Ok(fetch)
}

fn parse_feed_response(
    provider_name: &str,
    response: WebFetchResponse,
) -> Result<FeedFetchResponse, FeedError> {
    let raw_hash = sha256(&response.body);
    let parsed = parse_feed(&response.body)?;

    Ok(FeedFetchResponse {
        provider: BackendId::new(provider_name),
        url: FeedUrl::new(response.url)?,
        status: HttpStatusCode::new(response.status)?,
        content_type: response.content_type,
        format: parsed.format,
        raw_hash,
        raw_body: response.body,
        truncated: response.truncated,
        feed_title: parsed.feed_title,
        feed_link: parsed.feed_link,
        feed_updated_at: parsed.feed_updated_at,
        items: parsed.items,
    })
}

#[derive(Debug, Default)]
struct ParsedFeed {
    format: FeedFormat,
    feed_title: Option<String>,
    feed_link: Option<FeedUrl>,
    feed_updated_at: Option<String>,
    items: Vec<FeedItem>,
}

fn parse_feed(body: &str) -> Result<ParsedFeed, FeedError> {
    if let Ok(feed) = parse_json_feed(body) {
        return Ok(feed);
    }

    parse_xml_feed(body)
}

#[derive(Debug, Deserialize)]
struct JsonFeed {
    title: Option<String>,
    home_page_url: Option<String>,
    feed_url: Option<String>,
    items: Vec<JsonFeedItem>,
}

#[derive(Debug, Deserialize)]
struct JsonFeedItem {
    id: Option<String>,
    title: Option<String>,
    url: Option<String>,
    external_url: Option<String>,
    summary: Option<String>,
    content_text: Option<String>,
    date_published: Option<String>,
    date_modified: Option<String>,
    author: Option<JsonFeedAuthor>,
    authors: Option<Vec<JsonFeedAuthor>>,
    tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct JsonFeedAuthor {
    name: Option<String>,
}

fn parse_json_feed(body: &str) -> Result<ParsedFeed, FeedError> {
    let json: JsonFeed =
        serde_json::from_str(body).map_err(|error| FeedError::Parse(error.to_string()))?;
    let feed_link = json
        .home_page_url
        .or(json.feed_url)
        .and_then(parse_optional_url);
    let items = json
        .items
        .into_iter()
        .map(|item| {
            let mut authors = item
                .authors
                .unwrap_or_default()
                .into_iter()
                .filter_map(|author| author.name)
                .collect::<Vec<_>>();
            if let Some(author) = item.author.and_then(|author| author.name) {
                authors.push(author);
            }
            authors.sort();
            authors.dedup();
            let summary = item.summary.or(item.content_text);
            let link = item.url.or(item.external_url).and_then(parse_optional_url);
            let item_hash = item_hash(
                item.id.as_ref(),
                item.title.as_ref(),
                link.as_ref(),
                summary.as_ref(),
            );

            FeedItem {
                id: item.id,
                title: item.title,
                link,
                summary,
                published_at: item.date_published,
                updated_at: item.date_modified,
                authors,
                categories: item.tags.unwrap_or_default(),
                item_hash,
            }
        })
        .collect();

    Ok(ParsedFeed {
        format: FeedFormat::JsonFeed,
        feed_title: json.title,
        feed_link,
        feed_updated_at: None,
        items,
    })
}

#[derive(Debug, Default)]
struct FeedItemBuilder {
    id: Option<String>,
    title: Option<String>,
    link: Option<FeedUrl>,
    summary: Option<String>,
    published_at: Option<String>,
    updated_at: Option<String>,
    authors: Vec<String>,
    categories: Vec<String>,
}

impl FeedItemBuilder {
    fn build(self) -> FeedItem {
        let item_hash = item_hash(
            self.id.as_ref(),
            self.title.as_ref(),
            self.link.as_ref(),
            self.summary.as_ref(),
        );
        FeedItem {
            id: self.id,
            title: self.title,
            link: self.link,
            summary: self.summary,
            published_at: self.published_at,
            updated_at: self.updated_at,
            authors: self.authors,
            categories: self.categories,
            item_hash,
        }
    }
}

fn parse_xml_feed(body: &str) -> Result<ParsedFeed, FeedError> {
    let mut reader = Reader::from_str(body);
    reader.config_mut().trim_text(true);
    let mut feed = ParsedFeed::default();
    let mut current_item: Option<FeedItemBuilder> = None;

    loop {
        match reader.read_event() {
            Ok(Event::Start(start)) => {
                let name = local_name(start.name().as_ref());
                match name.as_str() {
                    "rss" | "rdf" => feed.format = FeedFormat::Rss,
                    "feed" => feed.format = FeedFormat::Atom,
                    "item" | "entry" => current_item = Some(FeedItemBuilder::default()),
                    _ if current_item.is_some() => {
                        read_item_start(&mut reader, &start, &name, current_item.as_mut());
                    }
                    "title" => feed.feed_title = read_text(&mut reader, &start)?,
                    "link" => {
                        if feed.format == FeedFormat::Atom {
                            feed.feed_link =
                                attr_value(&reader, &start, b"href").and_then(parse_optional_url);
                        } else {
                            feed.feed_link =
                                read_text(&mut reader, &start)?.and_then(parse_optional_url);
                        }
                    }
                    "updated" | "lastbuilddate" => {
                        feed.feed_updated_at = read_text(&mut reader, &start)?;
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(empty)) => {
                let name = local_name(empty.name().as_ref());
                if name == "link" {
                    if let Some(item) = current_item.as_mut() {
                        item.link = item.link.take().or_else(|| {
                            attr_value(&reader, &empty, b"href").and_then(parse_optional_url)
                        });
                    } else if feed.format == FeedFormat::Atom {
                        feed.feed_link = feed.feed_link.take().or_else(|| {
                            attr_value(&reader, &empty, b"href").and_then(parse_optional_url)
                        });
                    }
                }
            }
            Ok(Event::End(end)) => {
                let name = local_name(end.name().as_ref());
                if matches!(name.as_str(), "item" | "entry") {
                    if let Some(item) = current_item.take() {
                        feed.items.push(item.build());
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(error) => return Err(FeedError::Parse(error.to_string())),
            _ => {}
        }
    }

    if feed.format == FeedFormat::Unknown {
        return Err(FeedError::UnsupportedFormat);
    }

    Ok(feed)
}

fn read_item_start(
    reader: &mut Reader<&[u8]>,
    start: &BytesStart<'_>,
    name: &str,
    item: Option<&mut FeedItemBuilder>,
) {
    let Some(item) = item else {
        return;
    };

    match name {
        "title" => item.title = read_text(reader, start).ok().flatten(),
        "link" => {
            item.link = attr_value(reader, start, b"href")
                .or_else(|| read_text(reader, start).ok().flatten())
                .and_then(parse_optional_url);
        }
        "guid" | "id" => item.id = read_text(reader, start).ok().flatten(),
        "description" | "summary" | "content" | "encoded" => {
            if item.summary.is_none() {
                item.summary = read_text(reader, start).ok().flatten();
            }
        }
        "pubdate" | "published" => item.published_at = read_text(reader, start).ok().flatten(),
        "updated" => item.updated_at = read_text(reader, start).ok().flatten(),
        "creator" | "author" | "name" => {
            if let Some(author) = read_text(reader, start).ok().flatten() {
                item.authors.push(author);
            }
        }
        "category" => {
            if let Some(category) = read_text(reader, start).ok().flatten() {
                item.categories.push(category);
            }
        }
        _ => {}
    }
}

fn read_text(
    reader: &mut Reader<&[u8]>,
    start: &BytesStart<'_>,
) -> Result<Option<String>, FeedError> {
    let text = reader
        .read_text(start.name())
        .map_err(|error| FeedError::Parse(error.to_string()))?;
    let text = text
        .decode()
        .map_err(|error| FeedError::Parse(error.to_string()))?;
    let trimmed = text.trim();
    Ok((!trimmed.is_empty()).then_some(trimmed.to_string()))
}

fn local_name(name: &[u8]) -> String {
    let raw = String::from_utf8_lossy(name).to_ascii_lowercase();
    raw.rsplit(':').next().unwrap_or(&raw).to_string()
}

fn attr_value(reader: &Reader<&[u8]>, start: &BytesStart<'_>, key: &[u8]) -> Option<String> {
    start
        .attributes()
        .flatten()
        .find(|attribute| local_name(attribute.key.as_ref()).as_bytes() == key)
        .and_then(|attribute| {
            attribute
                .decoded_and_normalized_value(quick_xml::XmlVersion::Implicit1_0, reader.decoder())
                .ok()
                .map(std::borrow::Cow::into_owned)
        })
}

fn discover_alternate_links(
    body: &str,
    base_url: &str,
) -> Result<Vec<FeedEndpointCandidate>, FeedError> {
    let base = Url::parse(base_url).map_err(|error| FeedError::InvalidUrl(error.to_string()))?;
    let link_re = regex_lite::Regex::new(r"(?is)<link\s+[^>]*>")
        .map_err(|error| FeedError::Parse(error.to_string()))?;
    let attr_re = regex_lite::Regex::new(r#"(?is)([a-zA-Z_:.-]+)\s*=\s*["']([^"']+)["']"#)
        .map_err(|error| FeedError::Parse(error.to_string()))?;
    let mut candidates = Vec::new();

    for link_match in link_re.find_iter(body) {
        let tag = link_match.as_str();
        let mut rel = None;
        let mut content_type = None;
        let mut href = None;

        for capture in attr_re.captures_iter(tag) {
            let Some(name) = capture.get(1) else {
                continue;
            };
            let Some(value) = capture.get(2) else {
                continue;
            };
            match name.as_str().to_ascii_lowercase().as_str() {
                "rel" => rel = Some(value.as_str().to_ascii_lowercase()),
                "type" => content_type = Some(value.as_str().to_ascii_lowercase()),
                "href" => href = Some(value.as_str().to_string()),
                _ => {}
            }
        }

        if !rel.as_deref().unwrap_or_default().contains("alternate") {
            continue;
        }
        let format_hint = content_type
            .as_deref()
            .map(format_hint_from_content_type)
            .unwrap_or(FeedFormat::Unknown);
        if format_hint == FeedFormat::Unknown {
            continue;
        }
        let Some(href) = href else {
            continue;
        };
        let url = base
            .join(&href)
            .map_err(|error| FeedError::InvalidUrl(error.to_string()))?;
        candidates.push(FeedEndpointCandidate {
            url: FeedUrl::new(url.to_string())?,
            format_hint,
            discovery_source: FeedDiscoverySource::AlternateLink,
            confidence_bps: BasisPoints::new(8_500).expect("static basis points are valid"),
        });
    }

    Ok(candidates)
}

fn common_feed_candidates(base_url: &Url) -> Vec<FeedEndpointCandidate> {
    let common_paths = [
        ("/feed", FeedFormat::Rss),
        ("/feed/", FeedFormat::Rss),
        ("/rss", FeedFormat::Rss),
        ("/rss.xml", FeedFormat::Rss),
        ("/feed.xml", FeedFormat::Rss),
        ("/atom.xml", FeedFormat::Atom),
        ("/index.xml", FeedFormat::Rss),
        ("/feed.json", FeedFormat::JsonFeed),
    ];
    common_paths
        .into_iter()
        .filter_map(|(path, format_hint)| {
            base_url.join(path).ok().map(|url| FeedEndpointCandidate {
                url: FeedUrl::new(url.to_string()).expect("joined feed URL is valid"),
                format_hint,
                discovery_source: FeedDiscoverySource::CommonPath,
                confidence_bps: BasisPoints::new(4_000).expect("static basis points are valid"),
            })
        })
        .collect()
}

fn dedup_candidates(candidates: &mut Vec<FeedEndpointCandidate>) {
    candidates.sort_by(|a, b| {
        b.confidence_bps
            .cmp(&a.confidence_bps)
            .then_with(|| a.url.as_str().cmp(b.url.as_str()))
    });
    candidates.dedup_by(|a, b| a.url == b.url);
}

#[allow(clippy::case_sensitive_file_extension_comparisons)]
fn looks_like_feed_url(path: &str) -> bool {
    let path = path.to_ascii_lowercase();
    path.ends_with(".rss")
        || path.ends_with(".xml")
        || path.ends_with(".atom")
        || path.ends_with(".json")
        || path.ends_with("/feed")
        || path.ends_with("/feed/")
        || path.ends_with("/rss")
}

#[allow(clippy::case_sensitive_file_extension_comparisons)]
fn format_hint_from_url(path: &str) -> FeedFormat {
    let path = path.to_ascii_lowercase();
    if path.ends_with(".json") {
        FeedFormat::JsonFeed
    } else if path.contains("atom") {
        FeedFormat::Atom
    } else if looks_like_feed_url(&path) {
        FeedFormat::Rss
    } else {
        FeedFormat::Unknown
    }
}

fn format_hint_from_content_type(content_type: &str) -> FeedFormat {
    if content_type.contains("json") {
        FeedFormat::JsonFeed
    } else if content_type.contains("atom") {
        FeedFormat::Atom
    } else if content_type.contains("rss") || content_type.contains("xml") {
        FeedFormat::Rss
    } else {
        FeedFormat::Unknown
    }
}

fn item_hash(
    id: Option<&String>,
    title: Option<&String>,
    link: Option<&FeedUrl>,
    summary: Option<&String>,
) -> ContentHash {
    sha256(&format!("{id:?}\n{title:?}\n{link:?}\n{summary:?}"))
}

fn parse_optional_url(value: String) -> Option<FeedUrl> {
    FeedUrl::new(value).ok()
}

fn sha256(value: &str) -> ContentHash {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&digest);
    ContentHash::new(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct StaticFetchBackend {
        response: WebFetchResponse,
    }

    impl WebFetchBackend for StaticFetchBackend {
        fn provider_name(&self) -> &'static str {
            "static"
        }

        fn fetch(&self, _request: &WebFetchRequest) -> Result<WebFetchResponse, WebFetchError> {
            Ok(self.response.clone())
        }
    }

    #[test]
    fn parses_rss_items_and_hashes_raw_body() {
        let response = WebFetchResponse {
            url: "https://example.test/feed.xml".into(),
            status: 200,
            content_type: Some("application/rss+xml".into()),
            body: r#"
                <rss version="2.0">
                  <channel>
                    <title>Local News</title>
                    <link>https://example.test</link>
                    <item>
                      <guid>abc</guid>
                      <title>Council update</title>
                      <link>https://example.test/a</link>
                      <description>Short summary.</description>
                      <pubDate>Thu, 30 Apr 2026 08:00:00 GMT</pubDate>
                      <category>Civic</category>
                    </item>
                  </channel>
                </rss>
            "#
            .into(),
            truncated: false,
        };
        let provider =
            HttpFeedProvider::with_fetch_backend(Arc::new(StaticFetchBackend { response }));
        let parsed = provider
            .fetch_feed(&FeedFetchRequest::new("https://example.test/feed.xml").unwrap())
            .unwrap();

        assert_eq!(parsed.format, FeedFormat::Rss);
        assert_eq!(parsed.feed_title.as_deref(), Some("Local News"));
        assert_eq!(parsed.items.len(), 1);
        assert_eq!(parsed.items[0].title.as_deref(), Some("Council update"));
        assert_eq!(parsed.items[0].id.as_deref(), Some("abc"));
        assert_eq!(parsed.raw_hash.to_hex().len(), 64);
        assert_eq!(parsed.items[0].item_hash.to_hex().len(), 64);
    }

    #[test]
    fn parses_atom_entries() {
        let response = WebFetchResponse {
            url: "https://example.test/atom.xml".into(),
            status: 200,
            content_type: Some("application/atom+xml".into()),
            body: r#"
                <feed xmlns="http://www.w3.org/2005/Atom">
                  <title>Atom News</title>
                  <link href="https://example.test"/>
                  <entry>
                    <id>tag:example.test,2026:1</id>
                    <title>Match report</title>
                    <link href="https://example.test/match"/>
                    <summary>Färjestad update.</summary>
                    <updated>2026-04-30T10:00:00Z</updated>
                  </entry>
                </feed>
            "#
            .into(),
            truncated: false,
        };
        let provider =
            HttpFeedProvider::with_fetch_backend(Arc::new(StaticFetchBackend { response }));
        let parsed = provider
            .fetch_feed(&FeedFetchRequest::new("https://example.test/atom.xml").unwrap())
            .unwrap();

        assert_eq!(parsed.format, FeedFormat::Atom);
        assert_eq!(
            parsed.items[0].link.as_ref().map(FeedUrl::as_str),
            Some("https://example.test/match")
        );
        assert_eq!(
            parsed.items[0].updated_at.as_deref(),
            Some("2026-04-30T10:00:00Z")
        );
    }

    #[test]
    fn parses_json_feed_items() {
        let response = WebFetchResponse {
            url: "https://example.test/feed.json".into(),
            status: 200,
            content_type: Some("application/feed+json".into()),
            body: r#"
                {
                  "version": "https://jsonfeed.org/version/1.1",
                  "title": "JSON News",
                  "home_page_url": "https://example.test",
                  "items": [
                    {
                      "id": "1",
                      "url": "https://example.test/1",
                      "title": "Coffee company expands",
                      "summary": "Local business update.",
                      "date_published": "2026-04-30T09:00:00Z",
                      "tags": ["business"]
                    }
                  ]
                }
            "#
            .into(),
            truncated: false,
        };
        let provider =
            HttpFeedProvider::with_fetch_backend(Arc::new(StaticFetchBackend { response }));
        let parsed = provider
            .fetch_feed(&FeedFetchRequest::new("https://example.test/feed.json").unwrap())
            .unwrap();

        assert_eq!(parsed.format, FeedFormat::JsonFeed);
        assert_eq!(parsed.feed_title.as_deref(), Some("JSON News"));
        assert_eq!(parsed.items[0].categories, vec!["business"]);
    }

    #[test]
    fn probe_discovers_alternate_feed_links_and_common_paths() {
        let response = WebFetchResponse {
            url: "https://example.test/".into(),
            status: 200,
            content_type: Some("text/html".into()),
            body: r#"
                <html>
                  <head>
                    <link rel="alternate" type="application/rss+xml" href="/rss.xml">
                    <link rel="alternate" type="application/atom+xml" href="https://example.test/atom.xml">
                  </head>
                </html>
            "#
            .into(),
            truncated: false,
        };
        let provider =
            HttpFeedProvider::with_fetch_backend(Arc::new(StaticFetchBackend { response }));
        let probe = provider
            .probe(&FeedProbeRequest::new("https://example.test/").unwrap())
            .unwrap();

        assert!(probe.candidates.iter().any(|candidate| {
            candidate.discovery_source == FeedDiscoverySource::AlternateLink
                && candidate.url.as_str() == "https://example.test/rss.xml"
        }));
        assert!(probe.candidates.iter().any(|candidate| {
            candidate.discovery_source == FeedDiscoverySource::CommonPath
                && candidate.url.as_str() == "https://example.test/feed"
        }));
    }

    #[test]
    fn request_deserialization_rejects_zero_limits_and_invalid_urls() {
        let zero_limit = r#"{"url":"https://example.test/feed.xml","max_bytes":0}"#;
        assert!(serde_json::from_str::<FeedFetchRequest>(zero_limit).is_err());

        let invalid_url = r#"{"url":"not a url"}"#;
        assert!(serde_json::from_str::<FeedFetchRequest>(invalid_url).is_err());

        let localhost_url = r#"{"url":"http://127.0.0.1/feed.xml"}"#;
        assert!(serde_json::from_str::<FeedFetchRequest>(localhost_url).is_err());

        let huge_limit = format!(
            r#"{{"url":"https://example.test/feed.xml","max_bytes":{}}}"#,
            MAX_WEB_FETCH_BYTES + 1
        );
        assert!(serde_json::from_str::<FeedFetchRequest>(&huge_limit).is_err());
    }

    #[test]
    fn candidate_deserialization_rejects_invalid_basis_points() {
        let json = r#"{
            "url":"https://example.test/feed.xml",
            "format_hint":"rss",
            "discovery_source":"direct_url",
            "confidence_bps":12000
        }"#;
        assert!(serde_json::from_str::<FeedEndpointCandidate>(json).is_err());
    }
}
