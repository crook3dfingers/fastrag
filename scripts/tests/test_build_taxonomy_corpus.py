from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the emitter importable without turning scripts/ into a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_taxonomy_corpus import emit_cwe_jsonl, emit_kev_jsonl  # noqa: E402

FIXTURE_CWE = Path(__file__).parent / "fixtures" / "cwe-tree-mini.xml"


def test_emit_cwe_jsonl_contains_sql_injection(tmp_path: Path) -> None:
    out = tmp_path / "cwe.jsonl"
    n = emit_cwe_jsonl(source=FIXTURE_CWE, dest=out)
    assert n >= 3
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    sqli = next(r for r in lines if r["cwe_id"] == 89)
    assert sqli["name"].startswith("Improper Neutralization of Special Elements")
    assert 943 in sqli["parents"]
    assert sqli["description"]  # non-empty


def test_emit_cwe_jsonl_children_inverted(tmp_path: Path) -> None:
    out = tmp_path / "cwe.jsonl"
    emit_cwe_jsonl(source=FIXTURE_CWE, dest=out)
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    by_id = {r["cwe_id"]: r for r in lines}
    # CWE-943 is parent of CWE-89, so 89 should appear in 943's children
    assert 89 in by_id[943]["children"]
    # CWE-74 is parent of CWE-943
    assert 943 in by_id[74]["children"]
    # CWE-74 has no parents
    assert by_id[74]["parents"] == []


def test_emit_cwe_jsonl_applicable_platforms(tmp_path: Path) -> None:
    out = tmp_path / "cwe.jsonl"
    emit_cwe_jsonl(source=FIXTURE_CWE, dest=out)
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    by_id = {r["cwe_id"]: r for r in lines}
    assert "Not Language-Specific" in by_id[89]["applicable_platforms"]
    # CWE-943 has both Language and Technology entries
    assert "Not Language-Specific" in by_id[943]["applicable_platforms"]
    assert "Database Server" in by_id[943]["applicable_platforms"]


def test_emit_kev_jsonl_flags_ransomware(tmp_path: Path) -> None:
    src = tmp_path / "kev.json"
    src.write_text(json.dumps({
        "vulnerabilities": [
            {
                "cveID": "CVE-2023-1234",
                "vendorProject": "Acme",
                "product": "WidgetX",
                "vulnerabilityName": "Acme WidgetX RCE",
                "shortDescription": "Unauth RCE via X-Y header",
                "requiredAction": "Apply patch per vendor advisory",
                "dateAdded": "2023-05-01",
                "dueDate": "2023-05-22",
                "knownRansomwareCampaignUse": "Known",
            }
        ]
    }))
    out = tmp_path / "kev.jsonl"
    n = emit_kev_jsonl(src, out)
    assert n == 1
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["cve_id"] == "CVE-2023-1234"
    assert rec["known_ransomware_campaign_use"] is True


def test_emit_kev_jsonl_unknown_ransomware_is_false(tmp_path: Path) -> None:
    src = tmp_path / "kev.json"
    src.write_text(json.dumps({
        "vulnerabilities": [{
            "cveID": "CVE-2024-0001", "vendorProject": "V", "product": "P",
            "vulnerabilityName": "N", "shortDescription": "D", "requiredAction": "A",
            "dateAdded": "2024-01-01", "dueDate": "2024-01-22",
            "knownRansomwareCampaignUse": "Unknown",
        }]
    }))
    out = tmp_path / "kev.jsonl"
    emit_kev_jsonl(src, out)
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["known_ransomware_campaign_use"] is False
