"""Schema and integrity validation for versioned five-table rule bundles."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


RULE_SCHEMA_VERSION = "4.0"
REQUIRED_REGISTRY_ARTIFACTS = frozenset(
    {
        "entity_registry.jsonl",
        "entity_forms.jsonl",
        "entity_classes.jsonl",
        "entity_class_memberships.jsonl",
        "evidence_inventory.jsonl",
        "fragment_evidence.jsonl",
    }
)
REQUIRED_TABLES = frozenset(
    {
        "compound_rules.csv",
        "transformation_rules.csv",
        "diagnostic_fragment_rules.csv",
        "neutral_loss_rules.csv",
        "biosynthetic_component_rules.csv",
    }
)

SCHEMA_REQUIRED_COLUMNS = {
    "compound_rules.csv": frozenset(
        {
            "entity_id",
            "target_origin",
            "compound_name",
            "formula",
            "exact_mass",
            "ion_mode",
            "adduct",
            "reported_fragments",
            "derivation_id",
            "evidence_ids",
        }
    ),
    "transformation_rules.csv": frozenset(
        {
            "rule_id",
            "source_entity_id",
            "target_entity_id",
            "source_entity",
            "target_entity",
            "evidence_type",
            "reactant_entities",
            "product_entities",
            "anchor_reactant_index",
            "network_anchor_role",
            "reaction_type",
            "reaction_operator",
            "formula_equation",
            "reactant_form_ids",
            "product_form_ids",
            "chemical_validation_status",
            "product_resolution_status",
            "derivation_id",
            "delta_mass",
            "direction",
            "evidence_ids",
        }
    ),
    "diagnostic_fragment_rules.csv": frozenset(
        {"compound_class", "subclass", "fragment_mz", "ion_mode", "required", "evidence_ids"}
    ),
    "neutral_loss_rules.csv": frozenset(
        {"compound_class", "subclass", "loss_mass", "ion_mode", "evidence_ids"}
    ),
    "biosynthetic_component_rules.csv": frozenset(
        {
            "entity_id",
            "form_id",
            "entity_class_id",
            "component_name",
            "role",
            "exact_mass",
            "delta_mass_to_product",
            "evidence_ids",
        }
    ),
}


class RuleManifestError(ValueError):
    """Raised before rule parsing when a bundle is absent or inconsistent."""


@dataclass(frozen=True)
class ValidatedRuleManifest:
    schema_version: str
    compiler_version: str
    table_hashes: dict[str, str]
    registry_hashes: dict[str, str]
    required_columns: dict[str, tuple[str, ...]]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuleManifestError(
                f"invalid JSON in {path.name} line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise RuleManifestError(
                f"{path.name} line {line_number} must contain a JSON object"
            )
        rows.append(value)
    return rows


def _split_ids(value: object) -> tuple[str, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            return tuple(str(item).strip() for item in parsed if str(item).strip())
    return tuple(item.strip() for item in text.replace("|", ";").split(";") if item.strip())


def _validate_registry_references(root: Path) -> None:
    entity_rows = _read_jsonl(root / "entity_registry.jsonl")
    form_rows = _read_jsonl(root / "entity_forms.jsonl")
    class_rows = _read_jsonl(root / "entity_classes.jsonl")
    membership_rows = _read_jsonl(root / "entity_class_memberships.jsonl")
    evidence_rows = _read_jsonl(root / "evidence_inventory.jsonl")
    fragment_rows = _read_jsonl(root / "fragment_evidence.jsonl")
    entity_ids = {str(row.get("entity_id", "")).strip() for row in entity_rows}
    form_ids = {str(row.get("form_id", "")).strip() for row in form_rows}
    class_ids = {str(row.get("entity_class_id", "")).strip() for row in class_rows}
    evidence_ids = {
        str(row.get("evidence_id", "")).strip() for row in evidence_rows
    }
    if "" in entity_ids or "" in form_ids or "" in class_ids:
        raise RuleManifestError("registry artifacts contain a blank stable ID")
    for row in form_rows:
        if str(row.get("entity_id", "")).strip() not in entity_ids:
            raise RuleManifestError("entity_forms.jsonl references an unknown entity_id")
    for row in membership_rows:
        if str(row.get("entity_id", "")).strip() not in entity_ids:
            raise RuleManifestError(
                "entity_class_memberships.jsonl references an unknown entity_id"
            )
        if str(row.get("entity_class_id", "")).strip() not in class_ids:
            raise RuleManifestError(
                "entity_class_memberships.jsonl references an unknown entity_class_id"
            )
    valid_roles = {
        "explicit_target_diagnostic",
        "target_product_ion",
        "class_diagnostic",
        "reaction_supporting_fragment",
        "neutral_loss",
        "theoretical_catalog",
        "unassigned_peak",
    }
    for row in fragment_rows:
        fragment_id = str(row.get("fragment_id", "")).strip()
        if not fragment_id:
            raise RuleManifestError(
                "fragment_evidence.jsonl contains a blank fragment_id"
            )
        entity_id = str(row.get("entity_id", "")).strip()
        if entity_id and entity_id not in entity_ids:
            raise RuleManifestError(
                "fragment_evidence.jsonl references an unknown entity_id"
            )
        entity_class_id = str(row.get("entity_class_id", "")).strip()
        if entity_class_id and entity_class_id not in class_ids:
            raise RuleManifestError(
                "fragment_evidence.jsonl references an unknown entity_class_id"
            )
        role = str(row.get("evidence_role", "")).strip()
        if role not in valid_roles:
            raise RuleManifestError(
                f"fragment_evidence.jsonl has invalid evidence_role {role!r}"
            )
        ion_mode = str(row.get("ion_mode", "")).strip()
        if ion_mode not in {"positive", "negative", "not_reported"}:
            raise RuleManifestError(
                f"fragment_evidence.jsonl has invalid ion_mode {ion_mode!r}"
            )
        for evidence_id in _split_ids(row.get("evidence_ids", "")):
            if evidence_id not in evidence_ids:
                raise RuleManifestError(
                    "fragment_evidence.jsonl references an unknown evidence_id"
                )
    table_checks = {
        "compound_rules.csv": {
            "entity_id": entity_ids,
        },
        "transformation_rules.csv": {
            "source_entity_id": entity_ids,
            "target_entity_id": entity_ids,
            "reactant_form_ids": form_ids,
            "product_form_ids": form_ids,
        },
        "biosynthetic_component_rules.csv": {
            "entity_id": entity_ids,
            "form_id": form_ids,
            "entity_class_id": class_ids,
        },
    }
    for filename, checks in table_checks.items():
        with (root / filename).open("r", encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            for row_number, row in enumerate(rows, start=2):
                for field, allowed in checks.items():
                    for reference in _split_ids(row.get(field, "")):
                        if reference not in allowed:
                            raise RuleManifestError(
                                f"{filename} row {row_number} field {field} "
                                f"references unknown ID {reference!r}"
                            )


def validate_rules_manifest(
    rules_dir: Path | str,
) -> ValidatedRuleManifest:
    root = Path(rules_dir)
    manifest_path = root / "rules_manifest.json"
    if not manifest_path.exists():
        raise RuleManifestError(f"missing rules_manifest.json in {root}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuleManifestError(f"invalid rules_manifest.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuleManifestError("rules_manifest.json must contain a JSON object")
    version = str(payload.get("schema_version", "")).strip()
    if version != RULE_SCHEMA_VERSION:
        raise RuleManifestError(
            f"unsupported schema_version {version!r}; expected {RULE_SCHEMA_VERSION!r}"
        )
    tables = payload.get("tables")
    if not isinstance(tables, dict):
        raise RuleManifestError("rules_manifest.json tables must be an object")
    if set(tables) != REQUIRED_TABLES:
        missing = sorted(REQUIRED_TABLES - set(tables))
        extra = sorted(set(tables) - REQUIRED_TABLES)
        raise RuleManifestError(f"manifest must describe exactly five tables; missing={missing}, extra={extra}")
    hashes: dict[str, str] = {}
    columns_by_table: dict[str, tuple[str, ...]] = {}
    for filename in sorted(REQUIRED_TABLES):
        metadata = tables.get(filename)
        if not isinstance(metadata, dict):
            raise RuleManifestError(f"manifest metadata for {filename} must be an object")
        required_columns = metadata.get("required_columns")
        if (
            not isinstance(required_columns, list)
            or not required_columns
            or any(not isinstance(item, str) or not item for item in required_columns)
        ):
            raise RuleManifestError(f"manifest required_columns for {filename} is invalid")
        path = root / filename
        if not path.exists():
            raise RuleManifestError(f"required rule table is missing: {filename}")
        raw = path.read_bytes()
        actual_hash = hashlib.sha256(raw).hexdigest()
        expected_hash = str(metadata.get("sha256", "")).strip().lower()
        if not expected_hash or actual_hash != expected_hash:
            raise RuleManifestError(
                f"hash mismatch for {filename}: expected={expected_hash!r}, actual={actual_hash!r}"
            )
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        missing_columns = sorted(set(required_columns) - set(header))
        if missing_columns:
            raise RuleManifestError(
                f"{filename} missing required columns declared by manifest: {missing_columns}"
            )
        schema_missing = sorted(SCHEMA_REQUIRED_COLUMNS[filename] - set(header))
        if schema_missing:
            raise RuleManifestError(
                f"{filename} missing schema {RULE_SCHEMA_VERSION} columns: {schema_missing}"
            )
        hashes[filename] = actual_hash
        columns_by_table[filename] = tuple(required_columns)
    registry_hashes: dict[str, str] = {}
    registry = payload.get("registry_artifacts")
    if not isinstance(registry, dict) or set(registry) != REQUIRED_REGISTRY_ARTIFACTS:
        missing = sorted(
            REQUIRED_REGISTRY_ARTIFACTS
            - (set(registry) if isinstance(registry, dict) else set())
        )
        extra = sorted(
            (set(registry) if isinstance(registry, dict) else set())
            - REQUIRED_REGISTRY_ARTIFACTS
        )
        raise RuleManifestError(
            "manifest must describe exactly six registry artifacts; "
            f"missing={missing}, extra={extra}"
        )
    for filename in sorted(REQUIRED_REGISTRY_ARTIFACTS):
        metadata = registry[filename]
        if not isinstance(metadata, dict):
            raise RuleManifestError(
                f"manifest metadata for registry artifact {filename} must be an object"
            )
        path = root / filename
        if not path.exists():
            raise RuleManifestError(f"required registry artifact is missing: {filename}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_hash = str(metadata.get("sha256", "")).strip().lower()
        if not expected_hash or actual_hash != expected_hash:
            raise RuleManifestError(
                f"hash mismatch for {filename}: expected={expected_hash!r}, "
                f"actual={actual_hash!r}"
            )
        registry_hashes[filename] = actual_hash
    _validate_registry_references(root)
    return ValidatedRuleManifest(
        schema_version=version,
        compiler_version=str(payload.get("compiler_version", "")).strip(),
        table_hashes=hashes,
        registry_hashes=registry_hashes,
        required_columns=columns_by_table,
    )
