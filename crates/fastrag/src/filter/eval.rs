use crate::filter::ast::FilterExpr;
use fastrag_store::schema::TypedValue;

/// Evaluate a filter expression against a set of field-value pairs.
///
/// Returns `true` if the record matches the filter.
pub fn matches(expr: &FilterExpr, fields: &[(String, TypedValue)]) -> bool {
    match expr {
        FilterExpr::Eq { field, value } => lookup(fields, field).is_some_and(|v| v == value),
        FilterExpr::Neq { field, value } => lookup(fields, field) != Some(value),
        FilterExpr::Gt { field, value } => {
            cmp_ord(fields, field, value, |o| o == std::cmp::Ordering::Greater)
        }
        FilterExpr::Gte { field, value } => {
            cmp_ord(fields, field, value, |o| o != std::cmp::Ordering::Less)
        }
        FilterExpr::Lt { field, value } => {
            cmp_ord(fields, field, value, |o| o == std::cmp::Ordering::Less)
        }
        FilterExpr::Lte { field, value } => {
            cmp_ord(fields, field, value, |o| o != std::cmp::Ordering::Greater)
        }
        FilterExpr::In { field, values } => {
            lookup(fields, field).is_some_and(|v| values.contains(v))
        }
        FilterExpr::NotIn { field, values } => {
            lookup(fields, field).is_none_or(|v| !values.contains(v))
        }
        FilterExpr::Contains { field, value } => lookup(fields, field).is_some_and(|v| match v {
            TypedValue::Array(arr) => arr.contains(value),
            other => other == value,
        }),
        FilterExpr::All { field, values } => lookup(fields, field).is_some_and(|v| match v {
            TypedValue::Array(arr) => values.iter().all(|needed| arr.contains(needed)),
            _ => false,
        }),
        FilterExpr::And(exprs) => exprs.iter().all(|e| matches(e, fields)),
        FilterExpr::Or(exprs) => exprs.iter().any(|e| matches(e, fields)),
        FilterExpr::Not(inner) => !matches(inner, fields),
    }
}

fn lookup<'a>(fields: &'a [(String, TypedValue)], name: &str) -> Option<&'a TypedValue> {
    fields.iter().find(|(k, _)| k == name).map(|(_, v)| v)
}

fn cmp_ord(
    fields: &[(String, TypedValue)],
    field: &str,
    value: &TypedValue,
    pred: impl Fn(std::cmp::Ordering) -> bool,
) -> bool {
    let Some(field_val) = lookup(fields, field) else {
        return false;
    };
    match (field_val, value) {
        (TypedValue::Numeric(a), TypedValue::Numeric(b)) => a.partial_cmp(b).is_some_and(&pred),
        (TypedValue::Date(a), TypedValue::Date(b)) => pred(a.cmp(b)),
        (TypedValue::String(a), TypedValue::String(b)) => pred(a.cmp(b)),
        // Bool and Array are not orderable
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn fields(pairs: &[(&str, TypedValue)]) -> Vec<(String, TypedValue)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn eq_string_match() {
        let f = fields(&[("severity", TypedValue::String("HIGH".to_string()))]);
        let expr = FilterExpr::Eq {
            field: "severity".to_string(),
            value: TypedValue::String("HIGH".to_string()),
        };
        assert!(matches(&expr, &f));
    }

    #[test]
    fn eq_string_no_match() {
        let f = fields(&[("severity", TypedValue::String("LOW".to_string()))]);
        let expr = FilterExpr::Eq {
            field: "severity".to_string(),
            value: TypedValue::String("HIGH".to_string()),
        };
        assert!(!matches(&expr, &f));
    }

    #[test]
    fn neq_match() {
        let f = fields(&[("status", TypedValue::String("open".to_string()))]);
        let expr = FilterExpr::Neq {
            field: "status".to_string(),
            value: TypedValue::String("closed".to_string()),
        };
        assert!(matches(&expr, &f));
    }

    #[test]
    fn gte_numeric_positive() {
        let f = fields(&[("cvss", TypedValue::Numeric(8.5))]);
        let expr = FilterExpr::Gte {
            field: "cvss".to_string(),
            value: TypedValue::Numeric(7.0),
        };
        assert!(matches(&expr, &f));
    }

    #[test]
    fn gte_numeric_negative() {
        let f = fields(&[("cvss", TypedValue::Numeric(3.0))]);
        let expr = FilterExpr::Gte {
            field: "cvss".to_string(),
            value: TypedValue::Numeric(7.0),
        };
        assert!(!matches(&expr, &f));
    }

    #[test]
    fn lt_date() {
        let f = fields(&[(
            "due",
            TypedValue::Date(NaiveDate::from_ymd_opt(2024, 3, 1).unwrap()),
        )]);
        let expr = FilterExpr::Lt {
            field: "due".to_string(),
            value: TypedValue::Date(NaiveDate::from_ymd_opt(2024, 6, 1).unwrap()),
        };
        assert!(matches(&expr, &f));
    }

    #[test]
    fn in_set() {
        let f = fields(&[("severity", TypedValue::String("HIGH".to_string()))]);
        let expr = FilterExpr::In {
            field: "severity".to_string(),
            values: vec![
                TypedValue::String("HIGH".to_string()),
                TypedValue::String("CRITICAL".to_string()),
            ],
        };
        assert!(matches(&expr, &f));
    }

    #[test]
    fn not_in_set() {
        let f = fields(&[("status", TypedValue::String("open".to_string()))]);
        let expr = FilterExpr::NotIn {
            field: "status".to_string(),
            values: vec![
                TypedValue::String("closed".to_string()),
                TypedValue::String("resolved".to_string()),
            ],
        };
        assert!(matches(&expr, &f));
    }

    #[test]
    fn contains_array_positive() {
        let f = fields(&[(
            "tags",
            TypedValue::Array(vec![
                TypedValue::String("rce".to_string()),
                TypedValue::String("sqli".to_string()),
            ]),
        )]);
        let expr = FilterExpr::Contains {
            field: "tags".to_string(),
            value: TypedValue::String("rce".to_string()),
        };
        assert!(matches(&expr, &f));
    }

    #[test]
    fn contains_array_negative() {
        let f = fields(&[(
            "tags",
            TypedValue::Array(vec![TypedValue::String("xss".to_string())]),
        )]);
        let expr = FilterExpr::Contains {
            field: "tags".to_string(),
            value: TypedValue::String("rce".to_string()),
        };
        assert!(!matches(&expr, &f));
    }

    #[test]
    fn all_array_positive() {
        let f = fields(&[(
            "tags",
            TypedValue::Array(vec![
                TypedValue::String("rce".to_string()),
                TypedValue::String("remote".to_string()),
                TypedValue::String("critical".to_string()),
            ]),
        )]);
        let expr = FilterExpr::All {
            field: "tags".to_string(),
            values: vec![
                TypedValue::String("rce".to_string()),
                TypedValue::String("remote".to_string()),
            ],
        };
        assert!(matches(&expr, &f));
    }

    #[test]
    fn all_array_negative() {
        let f = fields(&[(
            "tags",
            TypedValue::Array(vec![TypedValue::String("rce".to_string())]),
        )]);
        let expr = FilterExpr::All {
            field: "tags".to_string(),
            values: vec![
                TypedValue::String("rce".to_string()),
                TypedValue::String("remote".to_string()),
            ],
        };
        assert!(!matches(&expr, &f));
    }

    #[test]
    fn and_short_circuit() {
        let f = fields(&[("a", TypedValue::Bool(false))]);
        // First clause fails → second should never matter
        let expr = FilterExpr::And(vec![
            FilterExpr::Eq {
                field: "a".to_string(),
                value: TypedValue::Bool(true),
            },
            FilterExpr::Eq {
                field: "nonexistent".to_string(),
                value: TypedValue::Bool(true),
            },
        ]);
        assert!(!matches(&expr, &f));
    }

    #[test]
    fn or_short_circuit() {
        let f = fields(&[("a", TypedValue::Bool(true))]);
        // First clause succeeds → second doesn't matter
        let expr = FilterExpr::Or(vec![
            FilterExpr::Eq {
                field: "a".to_string(),
                value: TypedValue::Bool(true),
            },
            FilterExpr::Eq {
                field: "nonexistent".to_string(),
                value: TypedValue::Bool(true),
            },
        ]);
        assert!(matches(&expr, &f));
    }

    #[test]
    fn not_inverts() {
        let f = fields(&[("active", TypedValue::Bool(true))]);
        let expr = FilterExpr::Not(Box::new(FilterExpr::Eq {
            field: "active".to_string(),
            value: TypedValue::Bool(true),
        }));
        assert!(!matches(&expr, &f));
    }

    #[test]
    fn missing_field_no_match() {
        let f: Vec<(String, TypedValue)> = vec![];
        let expr = FilterExpr::Eq {
            field: "severity".to_string(),
            value: TypedValue::String("HIGH".to_string()),
        };
        assert!(!matches(&expr, &f));
    }

    #[test]
    fn missing_field_neq_returns_true() {
        let f: Vec<(String, TypedValue)> = vec![];
        let expr = FilterExpr::Neq {
            field: "status".to_string(),
            value: TypedValue::String("closed".to_string()),
        };
        assert!(matches(&expr, &f));
    }

    #[test]
    fn missing_field_not_in_returns_true() {
        let f: Vec<(String, TypedValue)> = vec![];
        let expr = FilterExpr::NotIn {
            field: "status".to_string(),
            values: vec![TypedValue::String("closed".to_string())],
        };
        assert!(matches(&expr, &f));
    }

    #[test]
    fn bool_ordering_returns_false() {
        let f = fields(&[("flag", TypedValue::Bool(true))]);
        let expr = FilterExpr::Gt {
            field: "flag".to_string(),
            value: TypedValue::Bool(false),
        };
        assert!(!matches(&expr, &f));
    }
}
