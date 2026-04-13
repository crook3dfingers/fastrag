use fastrag_store::schema::TypedValue;
use serde::{Deserialize, Serialize};

/// A filter expression that can be evaluated against a record's metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FilterExpr {
    Eq {
        field: String,
        value: TypedValue,
    },
    Neq {
        field: String,
        value: TypedValue,
    },
    Gt {
        field: String,
        value: TypedValue,
    },
    Gte {
        field: String,
        value: TypedValue,
    },
    Lt {
        field: String,
        value: TypedValue,
    },
    Lte {
        field: String,
        value: TypedValue,
    },
    In {
        field: String,
        values: Vec<TypedValue>,
    },
    #[serde(rename = "not_in")]
    NotIn {
        field: String,
        values: Vec<TypedValue>,
    },
    Contains {
        field: String,
        value: TypedValue,
    },
    All {
        field: String,
        values: Vec<TypedValue>,
    },
    And(Vec<FilterExpr>),
    Or(Vec<FilterExpr>),
    Not(Box<FilterExpr>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_round_trip_simple_eq() {
        let expr = FilterExpr::Eq {
            field: "severity".to_string(),
            value: TypedValue::String("HIGH".to_string()),
        };
        let json = serde_json::to_string(&expr).unwrap();
        let back: FilterExpr = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, back);
    }

    #[test]
    fn serde_round_trip_complex_nested() {
        let expr = FilterExpr::And(vec![
            FilterExpr::Or(vec![
                FilterExpr::Eq {
                    field: "severity".to_string(),
                    value: TypedValue::String("HIGH".to_string()),
                },
                FilterExpr::Gte {
                    field: "cvss_score".to_string(),
                    value: TypedValue::Numeric(7.0),
                },
            ]),
            FilterExpr::Not(Box::new(FilterExpr::Eq {
                field: "status".to_string(),
                value: TypedValue::String("resolved".to_string()),
            })),
            FilterExpr::In {
                field: "tags".to_string(),
                values: vec![
                    TypedValue::String("rce".to_string()),
                    TypedValue::String("sqli".to_string()),
                ],
            },
            FilterExpr::Contains {
                field: "labels".to_string(),
                value: TypedValue::String("critical".to_string()),
            },
        ]);
        let json = serde_json::to_string(&expr).unwrap();
        let back: FilterExpr = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, back);
    }

    #[test]
    fn serde_not_in_uses_custom_tag() {
        let expr = FilterExpr::NotIn {
            field: "status".to_string(),
            values: vec![TypedValue::String("closed".to_string())],
        };
        let json = serde_json::to_string(&expr).unwrap();
        assert!(
            json.contains("not_in"),
            "expected 'not_in' tag in JSON, got: {json}"
        );
        let back: FilterExpr = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, back);
    }
}
