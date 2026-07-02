// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Tag-shaped XML extraction helpers.
//!
//! Designed for the common Embassy use cases:
//! - SOAP envelope responses where the consumer wants the text content
//!   of named elements (e.g., `<valid>`, `<name>`, `<address>`).
//! - Government data feeds that publish XML and use simple element
//!   structures (TED procurement notices, EU FSF if a consumer points
//!   the live provider at the canonical XML feed).
//!
//! What this module does NOT do:
//! - Schema-bound deserialization. Use `quick-xml`'s serde integration
//!   or `serde-xml-rs` directly for that.
//! - Namespace-aware matching. We match on local element name only;
//!   `<ns2:checkVatResponse>` and `<tns:checkVatResponse>` both match a
//!   `"checkVatResponse"` query. That's the right call for SOAP where
//!   the namespace prefix varies by sender.

use quick_xml::Reader;
use quick_xml::events::Event;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum XmlExtractError {
    #[error("xml parse failed: {0}")]
    Parse(String),
}

/// Return the text content of the first element whose local name
/// matches `local_name`. Returns `None` if the element is not present.
///
/// Namespaces are stripped: `<ns2:valid>true</ns2:valid>` matches a
/// `local_name` of `"valid"`.
pub fn extract_first_text(xml: &str, local_name: &str) -> Result<Option<String>, XmlExtractError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let target = local_name.as_bytes();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(start)) => {
                if strip_namespace(start.name().as_ref()) == target {
                    let name = start.name();
                    let text = reader
                        .read_text(name)
                        .map_err(|e| XmlExtractError::Parse(e.to_string()))?;
                    let text = text
                        .decode()
                        .map_err(|e| XmlExtractError::Parse(e.to_string()))?;
                    return Ok(Some(text.into_owned()));
                }
            }
            Ok(Event::Empty(empty)) => {
                if strip_namespace(empty.name().as_ref()) == target {
                    return Ok(Some(String::new()));
                }
            }
            Ok(Event::Eof) => return Ok(None),
            Err(e) => {
                return Err(XmlExtractError::Parse(format!(
                    "at position {}: {e}",
                    reader.error_position()
                )));
            }
            _ => {}
        }
        buf.clear();
    }
}

/// Return the text content of every element whose local name matches
/// `local_name`, in document order.
pub fn extract_all_texts(xml: &str, local_name: &str) -> Result<Vec<String>, XmlExtractError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let target = local_name.as_bytes();
    let mut out = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(start)) => {
                if strip_namespace(start.name().as_ref()) == target {
                    let name = start.name();
                    let text = reader
                        .read_text(name)
                        .map_err(|e| XmlExtractError::Parse(e.to_string()))?;
                    let text = text
                        .decode()
                        .map_err(|e| XmlExtractError::Parse(e.to_string()))?;
                    out.push(text.into_owned());
                }
            }
            Ok(Event::Empty(empty)) => {
                if strip_namespace(empty.name().as_ref()) == target {
                    out.push(String::new());
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(XmlExtractError::Parse(format!(
                    "at position {}: {e}",
                    reader.error_position()
                )));
            }
            _ => {}
        }
        buf.clear();
    }
    Ok(out)
}

fn strip_namespace(name: &[u8]) -> &[u8] {
    if let Some(idx) = name.iter().position(|&b| b == b':') {
        &name[idx + 1..]
    } else {
        name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal SOAP-shaped fixture covering: namespaces on element
    /// names, multiple matches, and missing elements — the failure
    /// modes the helper has to handle for real responses.
    const SOAP_RESPONSE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"
               xmlns:ns2="urn:ec.europa.eu:taxud:vies:services:checkVat:types">
  <soap:Body>
    <ns2:checkVatResponse>
      <ns2:countryCode>SE</ns2:countryCode>
      <ns2:vatNumber>556036080501</ns2:vatNumber>
      <ns2:requestDate>2026-05-23+01:00</ns2:requestDate>
      <ns2:valid>true</ns2:valid>
      <ns2:name>VOLVO AKTIEBOLAG</ns2:name>
      <ns2:address>VOLVO LUNDBYHOLM 405 08, GOTEBORG</ns2:address>
    </ns2:checkVatResponse>
  </soap:Body>
</soap:Envelope>"#;

    #[test]
    fn extract_first_text_strips_namespace_prefix() {
        // Intent: SOAP responses use namespace prefixes that vary by
        // sender. The extractor must match on local element name so
        // a consumer writing `extract_first_text(xml, "valid")` keeps
        // working when the prefix changes from `ns2:` to `tns:`.
        let valid = extract_first_text(SOAP_RESPONSE, "valid").unwrap();
        assert_eq!(valid.as_deref(), Some("true"));
        let name = extract_first_text(SOAP_RESPONSE, "name").unwrap();
        assert_eq!(name.as_deref(), Some("VOLVO AKTIEBOLAG"));
    }

    #[test]
    fn missing_element_returns_none() {
        let absent = extract_first_text(SOAP_RESPONSE, "doesNotExist").unwrap();
        assert_eq!(absent, None);
    }

    #[test]
    fn extract_all_texts_preserves_document_order() {
        // Intent: when a feed has multiple `<entry>` or `<item>`
        // children, the extractor must return them in document order so
        // pagination and dedupe-by-index stay deterministic.
        let multi = r#"<root>
            <item>first</item>
            <item>second</item>
            <other>skip</other>
            <item>third</item>
        </root>"#;
        let items = extract_all_texts(multi, "item").unwrap();
        assert_eq!(items, vec!["first", "second", "third"]);
    }

    #[test]
    fn unclosed_target_element_surfaces_parse_error() {
        // Intent: when the target element is unterminated, the helper
        // must surface a parse error rather than silently return None.
        // A silent miss could mask broken upstream responses.
        let malformed = r#"<root><target>oops"#;
        let result = extract_first_text(malformed, "target");
        assert!(result.is_err(), "expected parse error, got {result:?}");
    }

    #[test]
    fn empty_element_yields_empty_string() {
        // Intent: a self-closing or empty element (`<flag/>`) is a real
        // signal — match found, no content. Distinguish from "element
        // absent" (which is `None`).
        let xml = r#"<root><flag/></root>"#;
        let result = extract_first_text(xml, "flag").unwrap();
        assert_eq!(result.as_deref(), Some(""));
    }
}
