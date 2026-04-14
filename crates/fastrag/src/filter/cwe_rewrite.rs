//! AST walker that expands filter predicates on the configured CWE field
//! into their descendant closures, using an embedded MITRE taxonomy.

use std::collections::BTreeSet;

use fastrag_cwe::Taxonomy;
use fastrag_store::schema::TypedValue;

use super::ast::FilterExpr;

/// Rewriter that expands equality/membership predicates on the CWE field
/// to include all descendant CWEs. Non-CWE predicates and range operators
/// pass through unchanged.
pub struct CweRewriter<'a> {
    taxonomy: &'a Taxonomy,
    cwe_field: &'a str,
}

impl<'a> CweRewriter<'a> {
    pub fn new(taxonomy: &'a Taxonomy, cwe_field: &'a str) -> Self {
        Self {
            taxonomy,
            cwe_field,
        }
    }

    /// Recursively rewrite `expr`. Returns a fresh tree; nodes not affected
    /// are returned unchanged.
    pub fn rewrite(&self, expr: FilterExpr) -> FilterExpr {
        match expr {
            FilterExpr::Eq { field, value } if field == self.cwe_field => {
                if let Some(n) = as_cwe_u32(&value) {
                    let values = expand_to_typed(self.taxonomy, &[n]);
                    FilterExpr::In { field, values }
                } else {
                    FilterExpr::Eq { field, value }
                }
            }
            FilterExpr::Neq { field, value } if field == self.cwe_field => {
                if let Some(n) = as_cwe_u32(&value) {
                    let values = expand_to_typed(self.taxonomy, &[n]);
                    FilterExpr::NotIn { field, values }
                } else {
                    FilterExpr::Neq { field, value }
                }
            }
            FilterExpr::In { field, values } if field == self.cwe_field => {
                let ids = collect_cwe_u32(&values);
                if ids.is_empty() {
                    FilterExpr::In { field, values }
                } else {
                    let expanded = expand_to_typed(self.taxonomy, &ids);
                    FilterExpr::In {
                        field,
                        values: expanded,
                    }
                }
            }
            FilterExpr::NotIn { field, values } if field == self.cwe_field => {
                let ids = collect_cwe_u32(&values);
                if ids.is_empty() {
                    FilterExpr::NotIn { field, values }
                } else {
                    let expanded = expand_to_typed(self.taxonomy, &ids);
                    FilterExpr::NotIn {
                        field,
                        values: expanded,
                    }
                }
            }
            FilterExpr::And(children) => {
                FilterExpr::And(children.into_iter().map(|c| self.rewrite(c)).collect())
            }
            FilterExpr::Or(children) => {
                FilterExpr::Or(children.into_iter().map(|c| self.rewrite(c)).collect())
            }
            FilterExpr::Not(inner) => FilterExpr::Not(Box::new(self.rewrite(*inner))),
            other => other,
        }
    }
}

fn as_cwe_u32(v: &TypedValue) -> Option<u32> {
    match v {
        TypedValue::Numeric(n) if *n >= 0.0 && n.fract() == 0.0 && *n <= u32::MAX as f64 => {
            Some(*n as u32)
        }
        TypedValue::String(s) => s.parse::<u32>().ok(),
        _ => None,
    }
}

fn collect_cwe_u32(values: &[TypedValue]) -> Vec<u32> {
    values.iter().filter_map(as_cwe_u32).collect()
}

fn expand_to_typed(tx: &Taxonomy, ids: &[u32]) -> Vec<TypedValue> {
    let mut merged: BTreeSet<u32> = BTreeSet::new();
    for id in ids {
        for d in tx.expand(*id) {
            merged.insert(d);
        }
    }
    merged
        .into_iter()
        .map(|n| TypedValue::Numeric(n as f64))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_taxonomy() -> Taxonomy {
        let json = r#"{
            "version": "test",
            "view": "1000",
            "closure": {
                "89":  [89, 564, 943],
                "79":  [79, 80, 81]
            }
        }"#;
        Taxonomy::from_json(json.as_bytes()).unwrap()
    }

    fn numeric_values(vs: &[TypedValue]) -> Vec<u32> {
        vs.iter()
            .map(|v| match v {
                TypedValue::Numeric(n) => *n as u32,
                other => panic!("expected Numeric, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn eq_on_cwe_field_expands_to_in() {
        let tx = tiny_taxonomy();
        let r = CweRewriter::new(&tx, "cwe_id");
        let input = FilterExpr::Eq {
            field: "cwe_id".into(),
            value: TypedValue::Numeric(89.0),
        };
        let out = r.rewrite(input);
        match out {
            FilterExpr::In { field, values } => {
                assert_eq!(field, "cwe_id");
                let mut ids = numeric_values(&values);
                ids.sort();
                assert_eq!(ids, vec![89, 564, 943]);
            }
            other => panic!("expected In, got {other:?}"),
        }
    }

    #[test]
    fn in_on_cwe_field_expands_and_dedups() {
        let tx = tiny_taxonomy();
        let r = CweRewriter::new(&tx, "cwe_id");
        let input = FilterExpr::In {
            field: "cwe_id".into(),
            values: vec![TypedValue::Numeric(89.0), TypedValue::Numeric(79.0)],
        };
        let out = r.rewrite(input);
        match out {
            FilterExpr::In { field, values } => {
                assert_eq!(field, "cwe_id");
                let mut ids = numeric_values(&values);
                ids.sort();
                assert_eq!(ids, vec![79, 80, 81, 89, 564, 943]);
            }
            other => panic!("expected In, got {other:?}"),
        }
    }

    #[test]
    fn neq_on_cwe_field_expands_to_not_in() {
        let tx = tiny_taxonomy();
        let r = CweRewriter::new(&tx, "cwe_id");
        let input = FilterExpr::Neq {
            field: "cwe_id".into(),
            value: TypedValue::Numeric(89.0),
        };
        let out = r.rewrite(input);
        match out {
            FilterExpr::NotIn { field, values } => {
                assert_eq!(field, "cwe_id");
                let mut ids = numeric_values(&values);
                ids.sort();
                assert_eq!(ids, vec![89, 564, 943]);
            }
            other => panic!("expected NotIn, got {other:?}"),
        }
    }

    #[test]
    fn non_cwe_field_passes_through_unchanged() {
        let tx = tiny_taxonomy();
        let r = CweRewriter::new(&tx, "cwe_id");
        let input = FilterExpr::Eq {
            field: "severity".into(),
            value: TypedValue::String("HIGH".into()),
        };
        let out = r.rewrite(input.clone());
        assert_eq!(out, input);
    }

    #[test]
    fn range_operators_on_cwe_pass_through() {
        let tx = tiny_taxonomy();
        let r = CweRewriter::new(&tx, "cwe_id");
        let input = FilterExpr::Gt {
            field: "cwe_id".into(),
            value: TypedValue::Numeric(89.0),
        };
        let out = r.rewrite(input.clone());
        assert_eq!(out, input);
    }

    #[test]
    fn nested_and_or_not_recurses() {
        let tx = tiny_taxonomy();
        let r = CweRewriter::new(&tx, "cwe_id");
        let input = FilterExpr::And(vec![
            FilterExpr::Eq {
                field: "severity".into(),
                value: TypedValue::String("HIGH".into()),
            },
            FilterExpr::Or(vec![
                FilterExpr::Eq {
                    field: "cwe_id".into(),
                    value: TypedValue::Numeric(89.0),
                },
                FilterExpr::Not(Box::new(FilterExpr::Neq {
                    field: "cwe_id".into(),
                    value: TypedValue::Numeric(79.0),
                })),
            ]),
        ]);
        let out = r.rewrite(input);
        let mut found_in = false;
        let mut found_not_in = false;
        fn visit(e: &FilterExpr, fi: &mut bool, fni: &mut bool) {
            match e {
                FilterExpr::In { field, .. } if field == "cwe_id" => *fi = true,
                FilterExpr::NotIn { field, .. } if field == "cwe_id" => *fni = true,
                FilterExpr::And(cs) | FilterExpr::Or(cs) => {
                    cs.iter().for_each(|c| visit(c, fi, fni))
                }
                FilterExpr::Not(inner) => visit(inner, fi, fni),
                _ => {}
            }
        }
        visit(&out, &mut found_in, &mut found_not_in);
        assert!(found_in, "expected an In on cwe_id after rewrite");
        assert!(found_not_in, "expected a NotIn on cwe_id after rewrite");
    }

    #[test]
    fn unknown_cwe_preserves_value_as_singleton() {
        let tx = tiny_taxonomy();
        let r = CweRewriter::new(&tx, "cwe_id");
        let input = FilterExpr::Eq {
            field: "cwe_id".into(),
            value: TypedValue::Numeric(9999.0),
        };
        let out = r.rewrite(input);
        match out {
            FilterExpr::In { field, values } => {
                assert_eq!(field, "cwe_id");
                let ids = numeric_values(&values);
                assert_eq!(ids, vec![9999]);
            }
            other => panic!("expected In singleton, got {other:?}"),
        }
    }
}
