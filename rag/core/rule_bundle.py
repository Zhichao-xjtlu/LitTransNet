"""Deterministic writing and hashing of versioned five-table rule bundles."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


RULE_SCHEMA_VERSION = "4.0"
COMPILER_VERSION = "evidence-graph-reaction-compiler/4.0"
REGISTRY_ARTIFACTS = frozenset(
    {
        "entity_registry.jsonl",
        "entity_forms.jsonl",
        "entity_classes.jsonl",
        "entity_class_memberships.jsonl",
        "evidence_inventory.jsonl",
        "fragment_evidence.jsonl",
    }
)


@dataclass(frozen=True)
class RuleTable:
    columns: tuple[str, ...]
    rows: tuple[Mapping[str, object], ...]


def serialize_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value).strip()


def _csv_bytes(table: RuleTable) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(table.columns), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in table.rows:
        writer.writerow({column: serialize_cell(row.get(column, "")) for column in table.columns})
    return buffer.getvalue().encode("utf-8-sig")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_rule_bundle(
    output_dir: Path,
    tables: Mapping[str, RuleTable],
    *,
    registry_artifacts: Mapping[str, Path],
    compiler_version: str = COMPILER_VERSION,
) -> dict[str, object]:
    expected_names = {
        "compound_rules.csv",
        "transformation_rules.csv",
        "diagnostic_fragment_rules.csv",
        "neutral_loss_rules.csv",
        "biosynthetic_component_rules.csv",
    }
    if set(tables) != expected_names:
        missing = sorted(expected_names - set(tables))
        extra = sorted(set(tables) - expected_names)
        raise ValueError(f"rule bundle requires exactly five tables; missing={missing}, extra={extra}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, dict[str, object]] = {}
    for filename in sorted(tables):
        table = tables[filename]
        payload = _csv_bytes(table)
        _atomic_write(output_dir / filename, payload)
        metadata[filename] = {
            "required_columns": list(table.columns),
            "row_count": len(table.rows),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    registry_metadata: dict[str, dict[str, object]] = {}
    if set(registry_artifacts) != REGISTRY_ARTIFACTS:
        missing = sorted(REGISTRY_ARTIFACTS - set(registry_artifacts))
        extra = sorted(set(registry_artifacts) - REGISTRY_ARTIFACTS)
        raise ValueError(
            "schema 4.0 rule bundle requires exactly six registry artifacts; "
            f"missing={missing}, extra={extra}"
        )
    for filename in sorted(REGISTRY_ARTIFACTS):
        payload = Path(registry_artifacts[filename]).read_bytes()
        _atomic_write(output_dir / filename, payload)
        registry_metadata[filename] = {
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = {
        "schema_version": RULE_SCHEMA_VERSION,
        "compiler_version": compiler_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "encoding": "utf-8-sig",
        "tables": metadata,
        "registry_artifacts": registry_metadata,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(output_dir / "rules_manifest.json", manifest_bytes)
    return manifest

