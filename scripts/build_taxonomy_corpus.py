"""CWE + KEV JSONL emitters for fastrag ingest.

Pure functions — no CLI, no __main__. Task B3 will add argparse wiring.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

_NS = "http://cwe.mitre.org/cwe-7"
_W = f"{{{_NS}}}"  # namespace prefix for ElementTree queries


def _collapse(text: str | None) -> str:
    """Strip and collapse interior whitespace runs to a single space."""
    if not text:
        return ""
    return " ".join(text.split())


def emit_cwe_jsonl(source: Path, dest: Path) -> int:
    """Parse MITRE CWE XML and emit one JSONL record per weakness.

    Returns the number of records written.
    """
    tree = ET.parse(source)
    root = tree.getroot()

    # Collect all weaknesses in a first pass so we can invert the parent map.
    records: list[dict] = []

    for weakness in root.iter(f"{_W}Weakness"):
        cwe_id = int(weakness.attrib["ID"])
        name = weakness.attrib.get("Name", "")

        desc_el = weakness.find(f"{_W}Description")
        description = _collapse(desc_el.text if desc_el is not None else None)

        ext_el = weakness.find(f"{_W}Extended_Description")
        extended_description = _collapse(ext_el.text if ext_el is not None else None)

        # Parents: ChildOf relationships in View 1000 only.
        parents: list[int] = []
        seen_parents: set[int] = set()
        rw_container = weakness.find(f"{_W}Related_Weaknesses")
        if rw_container is not None:
            for rw in rw_container.findall(f"{_W}Related_Weakness"):
                if rw.attrib.get("Nature") == "ChildOf" and rw.attrib.get("View_ID") == "1000":
                    pid = int(rw.attrib["CWE_ID"])
                    if pid not in seen_parents:
                        parents.append(pid)
                        seen_parents.add(pid)

        # Applicable platforms: Language/Technology/Operating_System elements.
        platforms: list[str] = []
        seen_platforms: set[str] = set()
        ap_container = weakness.find(f"{_W}Applicable_Platforms")
        if ap_container is not None:
            for tag in (f"{_W}Language", f"{_W}Technology", f"{_W}Operating_System"):
                for el in ap_container.findall(tag):
                    label = el.attrib.get("Name") or el.attrib.get("Class", "")
                    if label and label not in seen_platforms:
                        platforms.append(label)
                        seen_platforms.add(label)

        records.append({
            "cwe_id": cwe_id,
            "name": name,
            "description": description,
            "extended_description": extended_description,
            "parents": parents,
            "children": [],  # filled in second pass
            "applicable_platforms": platforms,
        })

    # Build id→index map and invert the parent relationship.
    id_to_idx = {r["cwe_id"]: i for i, r in enumerate(records)}
    for rec in records:
        for pid in rec["parents"]:
            if pid in id_to_idx:
                records[id_to_idx[pid]]["children"].append(rec["cwe_id"])

    # Write JSONL.
    with dest.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    return len(records)


def emit_kev_jsonl(source: Path, dest: Path) -> int:
    """Parse CISA KEV catalog JSON and emit one JSONL record per vulnerability.

    Returns the number of records written.
    """
    catalog = json.loads(source.read_text(encoding="utf-8"))
    count = 0
    with dest.open("w", encoding="utf-8") as fh:
        for v in catalog["vulnerabilities"]:
            record = {
                "cve_id": v["cveID"],
                "vendor_project": v.get("vendorProject", ""),
                "product": v.get("product", ""),
                "vulnerability_name": v.get("vulnerabilityName", ""),
                "short_description": v.get("shortDescription", ""),
                "required_action": v.get("requiredAction", ""),
                "date_added": v.get("dateAdded", ""),
                "due_date": v.get("dueDate", ""),
                "known_ransomware_campaign_use": (
                    v.get("knownRansomwareCampaignUse", "Unknown") == "Known"
                ),
            }
            fh.write(json.dumps(record) + "\n")
            count += 1
    return count
