use crate::error::{StoreError, StoreResult};
use serde::{Deserialize, Serialize};

/// The kind of a typed field — used for schema compatibility checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TypedKind {
    String,
    Numeric,
    Bool,
    Date,
    Array,
}

/// A concrete value that can be stored in a dynamic field.
///
/// Variant order matters for untagged serde deserialization — more-specific
/// types must appear before more-general ones so the deserializer tries them
/// first (e.g. `Date` before `String`, `Bool` before `Numeric`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TypedValue {
    Bool(bool),
    Date(chrono::NaiveDate),
    Numeric(f64),
    Array(Vec<TypedValue>),
    String(String),
}

/// Definition of a single user-defined field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDef {
    pub name: String,
    pub typed: TypedKind,
    pub indexed: bool,
    pub stored: bool,
    pub positions: bool,
}

/// Schema that grows as new records are ingested.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DynamicSchema {
    pub user_fields: Vec<FieldDef>,
}

impl DynamicSchema {
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge a new field definition into the schema.
    ///
    /// - If the field is new, it is appended.
    /// - If the field already exists with the same type, this is a no-op.
    /// - If the field already exists with a different type, returns
    ///   [`StoreError::SchemaConflict`].
    pub fn merge(&mut self, field: FieldDef) -> StoreResult<()> {
        if let Some(existing) = self.user_fields.iter().find(|f| f.name == field.name) {
            if existing.typed != field.typed {
                return Err(StoreError::SchemaConflict {
                    field: field.name,
                    existing: existing.typed,
                    incoming: field.typed,
                });
            }
            // same type — idempotent, nothing to do
            return Ok(());
        }
        self.user_fields.push(field);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn string_field(name: &str) -> FieldDef {
        FieldDef {
            name: name.to_string(),
            typed: TypedKind::String,
            indexed: true,
            stored: true,
            positions: false,
        }
    }

    fn numeric_field(name: &str) -> FieldDef {
        FieldDef {
            name: name.to_string(),
            typed: TypedKind::Numeric,
            indexed: true,
            stored: true,
            positions: false,
        }
    }

    #[test]
    fn typed_kind_serde_round_trip() {
        let variants = [
            TypedKind::String,
            TypedKind::Numeric,
            TypedKind::Bool,
            TypedKind::Date,
            TypedKind::Array,
        ];
        for kind in variants {
            let json = serde_json::to_string(&kind).unwrap();
            let back: TypedKind = serde_json::from_str(&json).unwrap();
            assert_eq!(kind, back, "round-trip failed for {kind:?}");
        }
        // also verify the lowercase wire format
        assert_eq!(serde_json::to_string(&TypedKind::String).unwrap(), "\"string\"");
        assert_eq!(serde_json::to_string(&TypedKind::Numeric).unwrap(), "\"numeric\"");
    }

    #[test]
    fn typed_value_serde_round_trip() {
        let values: Vec<TypedValue> = vec![
            TypedValue::Bool(true),
            TypedValue::Numeric(3.14),
            TypedValue::String("hello".to_string()),
            TypedValue::Date(chrono::NaiveDate::from_ymd_opt(2024, 6, 1).unwrap()),
            TypedValue::Array(vec![TypedValue::String("a".to_string()), TypedValue::Numeric(1.0)]),
        ];
        for value in &values {
            let json = serde_json::to_string(value).unwrap();
            let back: TypedValue = serde_json::from_str(&json).unwrap();
            assert_eq!(*value, back, "round-trip failed for {value:?}");
        }
    }

    #[test]
    fn schema_merge_adds_new_fields() {
        let mut schema = DynamicSchema::new();
        schema.merge(string_field("title")).unwrap();
        schema.merge(numeric_field("score")).unwrap();
        assert_eq!(schema.user_fields.len(), 2);
        assert!(schema.user_fields.iter().any(|f| f.name == "title"));
        assert!(schema.user_fields.iter().any(|f| f.name == "score"));
    }

    #[test]
    fn schema_merge_allows_same_type() {
        let mut schema = DynamicSchema::new();
        schema.merge(string_field("title")).unwrap();
        schema.merge(string_field("title")).unwrap(); // same type — no error
        assert_eq!(schema.user_fields.len(), 1, "duplicate field must not be added");
    }

    #[test]
    fn schema_merge_rejects_type_conflict() {
        let mut schema = DynamicSchema::new();
        schema.merge(string_field("value")).unwrap();
        let err = schema.merge(numeric_field("value")).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("schema conflict"),
            "error message must mention 'schema conflict', got: {msg}"
        );
    }

    #[test]
    fn schema_serde_round_trip() {
        let mut schema = DynamicSchema::new();
        schema
            .merge(FieldDef {
                name: "tags".to_string(),
                typed: TypedKind::Array,
                indexed: false,
                stored: true,
                positions: false,
            })
            .unwrap();
        let json = serde_json::to_string(&schema).unwrap();
        let back: DynamicSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(back.user_fields.len(), 1);
        assert_eq!(back.user_fields[0].name, "tags");
        assert_eq!(back.user_fields[0].typed, TypedKind::Array);
    }
}
