use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read};

use fastrag_store::schema::{FieldDef, TypedKind, TypedValue};
use thiserror::Error;

/// Configuration for JSONL ingest.
#[derive(Debug, Clone)]
pub struct JsonlIngestConfig {
    /// Fields whose text content is concatenated (with "\n\n") to form the record body.
    pub text_fields: Vec<String>,
    /// Field to use as the external record identifier.
    pub id_field: String,
    /// Fields to extract as typed metadata.
    pub metadata_fields: Vec<String>,
    /// Explicit type overrides for metadata fields. Fields not listed are inferred.
    pub metadata_types: BTreeMap<String, TypedKind>,
    /// Fields that should always be treated as `Array` even if the JSON value is scalar.
    pub array_fields: Vec<String>,
    /// Name of the record field holding the CWE numeric id. Written to the
    /// corpus manifest so query-time CWE hierarchy expansion can find it.
    pub cwe_field: Option<String>,
}

/// A single parsed record ready for downstream ingestion.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedRecord {
    /// The value of `id_field` from the source JSON.
    pub external_id: String,
    /// blake3 hex digest of the raw JSON line.
    pub content_hash: String,
    /// Concatenated text from `text_fields`.
    pub text: String,
    /// The original raw JSON line, preserved for re-parsing.
    pub source_json: String,
    /// Extracted metadata as (name, value) pairs.
    pub metadata: Vec<(String, TypedValue)>,
}

/// Errors that can occur during JSONL parsing.
#[derive(Debug, Error)]
pub enum JsonlError {
    #[error("parse error at line {line}: {message}")]
    ParseError { line: usize, message: String },

    #[error("missing required field '{field}' at line {line}")]
    MissingField { line: usize, field: String },

    #[error("type mismatch at line {line} for field '{field}': expected {expected:?}, got {got}")]
    TypeMismatch {
        line: usize,
        field: String,
        expected: TypedKind,
        got: String,
    },
}

/// Infer a `TypedKind` from a JSON value.
///
/// - `bool` → `Bool`
/// - number → `Numeric`
/// - string matching `YYYY-MM-DD` → `Date`, otherwise `String`
/// - array → `Array` (inner type inferred from first element, defaults to `String`)
/// - `null` / object → `None`
pub fn infer_type(value: &serde_json::Value) -> Option<TypedKind> {
    match value {
        serde_json::Value::Bool(_) => Some(TypedKind::Bool),
        serde_json::Value::Number(_) => Some(TypedKind::Numeric),
        serde_json::Value::String(s) => {
            if is_date_string(s) {
                Some(TypedKind::Date)
            } else {
                Some(TypedKind::String)
            }
        }
        serde_json::Value::Array(arr) => {
            // Inner kind inferred from first element; Array wrapper is the outer kind.
            let _ = arr.first().and_then(infer_type); // used for docs, Array is always Array
            Some(TypedKind::Array)
        }
        serde_json::Value::Null | serde_json::Value::Object(_) => None,
    }
}

/// Convert a JSON value to a `TypedValue` given an expected `TypedKind`.
///
/// Returns `None` if the conversion is not possible.
pub fn to_typed_value(value: &serde_json::Value, kind: &TypedKind) -> Option<TypedValue> {
    match kind {
        TypedKind::Bool => value.as_bool().map(TypedValue::Bool),
        TypedKind::Numeric => value.as_f64().map(TypedValue::Numeric),
        TypedKind::String => value.as_str().map(|s| TypedValue::String(s.to_string())),
        TypedKind::Date => {
            let s = value.as_str()?;
            chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
                .ok()
                .map(TypedValue::Date)
        }
        TypedKind::Array => {
            let arr = value.as_array()?;
            // Infer element kind from first element, default to String.
            let elem_kind = arr
                .first()
                .and_then(infer_type)
                .unwrap_or(TypedKind::String);
            let converted: Vec<TypedValue> = arr
                .iter()
                .filter_map(|v| to_typed_value(v, &elem_kind))
                .collect();
            Some(TypedValue::Array(converted))
        }
    }
}

/// Parse all lines from `reader` according to `config`.
///
/// Returns `(records, field_defs)` where `field_defs` contains one entry per
/// unique metadata field seen across all records.
pub fn parse_jsonl<R: Read>(
    reader: R,
    config: &JsonlIngestConfig,
) -> Result<(Vec<ParsedRecord>, Vec<FieldDef>), JsonlError> {
    let buf = BufReader::new(reader);
    let mut records: Vec<ParsedRecord> = Vec::new();
    // Track inferred kinds per field; explicit overrides are seeded first.
    let mut inferred_kinds: BTreeMap<String, TypedKind> = config.metadata_types.clone();

    for (zero_idx, line_result) in buf.lines().enumerate() {
        let line_no = zero_idx + 1;
        let raw = line_result.map_err(|e| JsonlError::ParseError {
            line: line_no,
            message: e.to_string(),
        })?;

        if raw.trim().is_empty() {
            continue;
        }

        let obj: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| JsonlError::ParseError {
                line: line_no,
                message: e.to_string(),
            })?;

        // --- external_id ---
        let id_val = obj
            .get(&config.id_field)
            .ok_or_else(|| JsonlError::MissingField {
                line: line_no,
                field: config.id_field.clone(),
            })?;
        let external_id = match id_val {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Number(n) => n.to_string(),
            other => {
                return Err(JsonlError::TypeMismatch {
                    line: line_no,
                    field: config.id_field.clone(),
                    expected: TypedKind::String,
                    got: json_type_name(other).to_string(),
                });
            }
        };

        // --- content_hash ---
        let content_hash = blake3::hash(raw.as_bytes()).to_hex().to_string();

        // --- text ---
        let text = config
            .text_fields
            .iter()
            .filter_map(|f| obj.get(f).and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join("\n\n");

        // --- metadata ---
        let mut metadata: Vec<(String, TypedValue)> = Vec::new();
        for field in &config.metadata_fields {
            let Some(val) = obj.get(field) else {
                continue; // absent fields are silently skipped
            };
            if val.is_null() {
                continue;
            }

            let force_array = config.array_fields.contains(field);

            let kind = if force_array {
                inferred_kinds
                    .entry(field.clone())
                    .or_insert(TypedKind::Array);
                TypedKind::Array
            } else if let Some(explicit) = config.metadata_types.get(field) {
                inferred_kinds.entry(field.clone()).or_insert(*explicit);
                *explicit
            } else {
                match infer_type(val) {
                    Some(k) => {
                        inferred_kinds.entry(field.clone()).or_insert(k);
                        k
                    }
                    None => continue,
                }
            };

            // For Array, if the value isn't already an array, wrap it.
            let effective_val = if kind == TypedKind::Array && !val.is_array() {
                serde_json::Value::Array(vec![val.clone()])
            } else {
                val.clone()
            };

            let typed =
                to_typed_value(&effective_val, &kind).ok_or_else(|| JsonlError::TypeMismatch {
                    line: line_no,
                    field: field.clone(),
                    expected: kind,
                    got: json_type_name(val).to_string(),
                })?;

            metadata.push((field.clone(), typed));
        }

        records.push(ParsedRecord {
            external_id,
            content_hash,
            text,
            source_json: raw,
            metadata,
        });
    }

    // Build FieldDefs from everything we've seen.
    let field_defs: Vec<FieldDef> = inferred_kinds
        .iter()
        .map(|(name, kind)| FieldDef {
            name: name.clone(),
            typed: *kind,
            indexed: true,
            stored: true,
            positions: false,
        })
        .collect();

    Ok((records, field_defs))
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn is_date_string(s: &str) -> bool {
    chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").is_ok()
}

fn json_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> JsonlIngestConfig {
        JsonlIngestConfig {
            text_fields: vec!["title".to_string(), "description".to_string()],
            id_field: "id".to_string(),
            metadata_fields: vec![
                "severity".to_string(),
                "cvss_score".to_string(),
                "tags".to_string(),
            ],
            metadata_types: BTreeMap::new(),
            array_fields: vec!["tags".to_string()],
            cwe_field: None,
        }
    }

    #[test]
    fn parse_single_record() {
        let line = r#"{"id":"CVE-2024-0001","title":"Buffer overflow","description":"A heap overflow in foo","severity":"high","cvss_score":9.8,"tags":["memory","rce"]}"#;
        let config = default_config();
        let (records, field_defs) = parse_jsonl(line.as_bytes(), &config).unwrap();

        assert_eq!(records.len(), 1);
        let rec = &records[0];

        assert_eq!(rec.external_id, "CVE-2024-0001");
        assert_eq!(rec.text, "Buffer overflow\n\nA heap overflow in foo");
        assert!(
            !rec.content_hash.is_empty(),
            "content_hash must not be empty"
        );
        assert_eq!(rec.metadata.len(), 3);

        // Check field_defs types
        let severity_def = field_defs.iter().find(|f| f.name == "severity").unwrap();
        assert_eq!(severity_def.typed, TypedKind::String);

        let cvss_def = field_defs.iter().find(|f| f.name == "cvss_score").unwrap();
        assert_eq!(cvss_def.typed, TypedKind::Numeric);

        let tags_def = field_defs.iter().find(|f| f.name == "tags").unwrap();
        assert_eq!(tags_def.typed, TypedKind::Array);
    }

    #[test]
    fn parse_multiple_records() {
        let input = concat!(
            r#"{"id":"CVE-2024-0001","title":"Title A","description":"Desc A","severity":"high","cvss_score":9.8,"tags":["rce"]}"#,
            "\n",
            r#"{"id":"CVE-2024-0002","title":"Title B","description":"Desc B","severity":"low","cvss_score":3.1,"tags":["info"]}"#,
        );
        let config = default_config();
        let (records, _) = parse_jsonl(input.as_bytes(), &config).unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].external_id, "CVE-2024-0001");
        assert_eq!(records[0].text, "Title A\n\nDesc A");
        assert_eq!(records[1].external_id, "CVE-2024-0002");
        assert_eq!(records[1].text, "Title B\n\nDesc B");
    }

    #[test]
    fn missing_id_field_errors() {
        let line = r#"{"title":"No ID here","description":"desc"}"#;
        let config = default_config();
        let err = parse_jsonl(line.as_bytes(), &config).unwrap_err();
        match err {
            JsonlError::MissingField { line, field } => {
                assert_eq!(line, 1);
                assert_eq!(field, "id");
            }
            other => panic!("expected MissingField, got {other:?}"),
        }
    }

    #[test]
    fn malformed_json_errors() {
        let line = r#"{"id": "x", bad json"#;
        let config = default_config();
        let err = parse_jsonl(line.as_bytes(), &config).unwrap_err();
        match err {
            JsonlError::ParseError { line, .. } => assert_eq!(line, 1),
            other => panic!("expected ParseError, got {other:?}"),
        }
    }

    #[test]
    fn blank_lines_skipped() {
        let input = concat!(
            r#"{"id":"1","title":"A","description":"a","severity":"high","cvss_score":1.0,"tags":["x"]}"#,
            "\n\n",
            r#"{"id":"2","title":"B","description":"b","severity":"low","cvss_score":2.0,"tags":["y"]}"#,
        );
        let config = default_config();
        let (records, _) = parse_jsonl(input.as_bytes(), &config).unwrap();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn source_json_round_trip() {
        let line = r#"{"id":"CVE-2024-0001","title":"Title","description":"Desc","severity":"critical","cvss_score":10.0,"tags":["rce"],"nested":{"key":"value"}}"#;
        let config = default_config();
        let (records, _) = parse_jsonl(line.as_bytes(), &config).unwrap();
        assert_eq!(records.len(), 1);

        // Round-trip: re-parse source_json and verify nested fields survive.
        let reparsed: serde_json::Value =
            serde_json::from_str(&records[0].source_json).expect("source_json must be valid JSON");
        assert_eq!(
            reparsed["nested"]["key"].as_str().unwrap(),
            "value",
            "nested field must survive round-trip"
        );
        assert_eq!(reparsed["id"].as_str().unwrap(), "CVE-2024-0001");
    }
}
