use std::collections::BTreeMap;

use fastrag_store::schema::TypedKind;

use super::jsonl::JsonlIngestConfig;

/// Return a `JsonlIngestConfig` pre-filled for tarmo-vuln-core Finding records.
pub fn tarmo_finding_preset() -> JsonlIngestConfig {
    JsonlIngestConfig {
        text_fields: vec![
            "title".into(),
            "description".into(),
            "impact".into(),
            "remediation".into(),
            "mitigation".into(),
            "replication_steps".into(),
            "finding_guidance".into(),
        ],
        id_field: "id".into(),
        metadata_fields: vec![
            "severity".into(),
            "status".into(),
            "source_tool".into(),
            "cvss_score".into(),
            "cvss_v4_score".into(),
            "cwe_id".into(),
            "owasp_id".into(),
            "published".into(),
            "false_positive".into(),
            "raw_ref".into(),
            "remediation_due".into(),
            "tags".into(),
            "attack_ids".into(),
            "attack_tactics".into(),
            "affected_hosts".into(),
            "compliance_refs".into(),
            "source_tools".into(),
            "methodology_items".into(),
        ],
        metadata_types: BTreeMap::from([
            ("cvss_score".into(), TypedKind::Numeric),
            ("cvss_v4_score".into(), TypedKind::Numeric),
            ("cwe_id".into(), TypedKind::Numeric),
            ("published".into(), TypedKind::Bool),
            ("false_positive".into(), TypedKind::Bool),
            ("remediation_due".into(), TypedKind::Date),
        ]),
        array_fields: vec![
            "tags".into(),
            "attack_ids".into(),
            "attack_tactics".into(),
            "affected_hosts".into(),
            "compliance_refs".into(),
            "source_tools".into(),
            "methodology_items".into(),
        ],
        cwe_field: Some("cwe_id".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_fields_cover_prose_columns() {
        let cfg = tarmo_finding_preset();
        let expected = [
            "title",
            "description",
            "impact",
            "remediation",
            "mitigation",
            "replication_steps",
            "finding_guidance",
        ];
        assert_eq!(cfg.text_fields, expected);
    }

    #[test]
    fn id_field_is_id() {
        let cfg = tarmo_finding_preset();
        assert_eq!(cfg.id_field, "id");
    }

    #[test]
    fn metadata_fields_count() {
        let cfg = tarmo_finding_preset();
        assert_eq!(cfg.metadata_fields.len(), 18, "expected 18 metadata fields");
    }

    #[test]
    fn array_fields_match_list_types() {
        let cfg = tarmo_finding_preset();
        let expected = [
            "tags",
            "attack_ids",
            "attack_tactics",
            "affected_hosts",
            "compliance_refs",
            "source_tools",
            "methodology_items",
        ];
        assert_eq!(cfg.array_fields, expected);
    }

    #[test]
    fn metadata_types_override_numeric_bool_date() {
        let cfg = tarmo_finding_preset();
        assert_eq!(
            cfg.metadata_types.get("cvss_score"),
            Some(&TypedKind::Numeric)
        );
        assert_eq!(
            cfg.metadata_types.get("cvss_v4_score"),
            Some(&TypedKind::Numeric)
        );
        assert_eq!(cfg.metadata_types.get("cwe_id"), Some(&TypedKind::Numeric));
        assert_eq!(cfg.metadata_types.get("published"), Some(&TypedKind::Bool));
        assert_eq!(
            cfg.metadata_types.get("false_positive"),
            Some(&TypedKind::Bool)
        );
        assert_eq!(
            cfg.metadata_types.get("remediation_due"),
            Some(&TypedKind::Date)
        );
        assert_eq!(cfg.metadata_types.len(), 6, "exactly 6 type overrides");
    }

    #[test]
    fn tarmo_preset_sets_cwe_field() {
        let cfg = tarmo_finding_preset();
        assert_eq!(cfg.cwe_field.as_deref(), Some("cwe_id"));
    }

    #[test]
    fn all_array_fields_appear_in_metadata_fields() {
        let cfg = tarmo_finding_preset();
        for af in &cfg.array_fields {
            assert!(
                cfg.metadata_fields.contains(af),
                "array field `{af}` missing from metadata_fields"
            );
        }
    }
}
