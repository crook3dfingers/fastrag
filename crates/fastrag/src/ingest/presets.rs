use std::collections::BTreeMap;

use fastrag_store::schema::TypedKind;

use super::jsonl::JsonlIngestConfig;

/// Return a `JsonlIngestConfig` pre-filled for the VIPER Assist corpus
/// (`artifacts/viper-assist-corpus.jsonl` from `crook3dfingers/VIPER_Dashboard`).
///
/// Field set follows the live producer in `build/build_viper_assist_corpus.py`:
/// section-level chunks of static dashboard pages (playbooks, tools, SOPs,
/// engagements, references) with category, ports, tools/tags, CVE/MITRE
/// cross-refs, and risk metadata.
pub fn viper_assist_preset() -> JsonlIngestConfig {
    JsonlIngestConfig {
        text_fields: vec![
            "text".into(),
            "title".into(),
            "section".into(),
            "summary".into(),
        ],
        id_field: "id".into(),
        metadata_fields: vec![
            "url".into(),
            "category".into(),
            "ports".into(),
            "tools".into(),
            "tags".into(),
            "cves".into(),
            "mitre_ids".into(),
            "risk_signals".into(),
            "requires_credentials".into(),
            "risk_level".into(),
            "source_file".into(),
            "content_hash".into(),
            "schema_version".into(),
        ],
        metadata_types: BTreeMap::from([
            ("requires_credentials".into(), TypedKind::Bool),
            ("schema_version".into(), TypedKind::Numeric),
        ]),
        array_fields: vec![
            "ports".into(),
            "tools".into(),
            "tags".into(),
            "cves".into(),
            "mitre_ids".into(),
            "risk_signals".into(),
        ],
        cwe_field: None,
    }
}

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

    // ── viper_assist_preset ──────────────────────────────────────────────────

    #[test]
    fn viper_text_fields_cover_chunk_prose() {
        let cfg = viper_assist_preset();
        assert_eq!(cfg.text_fields, ["text", "title", "section", "summary"]);
    }

    #[test]
    fn viper_id_field_is_id() {
        let cfg = viper_assist_preset();
        assert_eq!(cfg.id_field, "id");
    }

    #[test]
    fn viper_metadata_fields_count() {
        let cfg = viper_assist_preset();
        assert_eq!(
            cfg.metadata_fields.len(),
            13,
            "expected 13 metadata fields"
        );
    }

    #[test]
    fn viper_array_fields_match_list_types() {
        let cfg = viper_assist_preset();
        let expected = [
            "ports",
            "tools",
            "tags",
            "cves",
            "mitre_ids",
            "risk_signals",
        ];
        assert_eq!(cfg.array_fields, expected);
    }

    #[test]
    fn viper_metadata_types_override_bool_and_numeric() {
        let cfg = viper_assist_preset();
        assert_eq!(
            cfg.metadata_types.get("requires_credentials"),
            Some(&TypedKind::Bool)
        );
        assert_eq!(
            cfg.metadata_types.get("schema_version"),
            Some(&TypedKind::Numeric)
        );
        assert_eq!(cfg.metadata_types.len(), 2, "exactly 2 type overrides");
    }

    #[test]
    fn viper_preset_has_no_cwe_field() {
        let cfg = viper_assist_preset();
        assert!(cfg.cwe_field.is_none());
    }

    #[test]
    fn viper_all_array_fields_appear_in_metadata_fields() {
        let cfg = viper_assist_preset();
        for af in &cfg.array_fields {
            assert!(
                cfg.metadata_fields.contains(af),
                "array field `{af}` missing from metadata_fields"
            );
        }
    }
}
