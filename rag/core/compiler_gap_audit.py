"""Deterministic terminal audit for evidence-graph compiler gaps.

This module never performs retrieval and never calls an LLM.  It inspects the
frozen claims, evidence inventory, registry/compiler diagnostics and emitted
rules, then assigns every detected gap a terminal status.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


AUDIT_SCHEMA_VERSION = "compiler-gap-audit/1.0"
TERMINAL_GAP_STATUSES = frozenset(
    {
        "resolved_existing_evidence",
        "resolved_deterministic_derivation",
        "unresolved_missing_literature_evidence",
        "unresolved_broken_provenance",
        "unresolved_ambiguous_entity",
        "nonmaterializable_explanatory_only",
        "invalid_chemical_or_schema_state",
    }
)


def _clean(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _values(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = [_clean(item) for item in value]
    else:
        text = _clean(value)
        if not text:
            return ()
        items = [_clean(item) for item in re.split(r"[;,|]", text)]
    return tuple(dict.fromkeys(item for item in items if item and item != "[]"))


def _stable_id(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return "audit_" + hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CompilerGapRecord:
    gap_id: str
    schema_version: str
    source_stage: str
    gap_type: str
    terminal_status: str
    claim_id: str = ""
    rule_id: str = ""
    entity_id: str = ""
    reaction_template_id: str = ""
    derivation_id: str = ""
    missing_fields: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()
    deterministic_repair_action: str = "none"
    affects_network_materialization: bool = False
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _record(
    *,
    source_stage: str,
    gap_type: str,
    terminal_status: str,
    claim_id: str = "",
    rule_id: str = "",
    entity_id: str = "",
    reaction_template_id: str = "",
    derivation_id: str = "",
    missing_fields: Iterable[str] = (),
    evidence_ids: Iterable[str] = (),
    chunk_ids: Iterable[str] = (),
    deterministic_repair_action: str = "none",
    affects_network_materialization: bool = False,
    detail: str = "",
) -> CompilerGapRecord:
    if terminal_status not in TERMINAL_GAP_STATUSES:
        raise ValueError(f"Unsupported terminal compiler-gap status: {terminal_status}")
    identity = {
        "source_stage": source_stage,
        "gap_type": gap_type,
        "claim_id": claim_id,
        "rule_id": rule_id,
        "entity_id": entity_id,
        "reaction_template_id": reaction_template_id,
        "derivation_id": derivation_id,
        "terminal_status": terminal_status,
    }
    return CompilerGapRecord(
        gap_id=_stable_id(identity),
        schema_version=AUDIT_SCHEMA_VERSION,
        source_stage=source_stage,
        gap_type=gap_type,
        terminal_status=terminal_status,
        claim_id=_clean(claim_id),
        rule_id=_clean(rule_id),
        entity_id=_clean(entity_id),
        reaction_template_id=_clean(reaction_template_id),
        derivation_id=_clean(derivation_id),
        missing_fields=tuple(dict.fromkeys(_clean(item) for item in missing_fields if _clean(item))),
        evidence_ids=tuple(dict.fromkeys(_clean(item) for item in evidence_ids if _clean(item))),
        chunk_ids=tuple(dict.fromkeys(_clean(item) for item in chunk_ids if _clean(item))),
        deterministic_repair_action=_clean(deterministic_repair_action) or "none",
        affects_network_materialization=bool(affects_network_materialization),
        detail=_clean(detail),
    )


def _claim_name(claim: Mapping[str, Any]) -> str:
    for field in (
        "compound_name",
        "precursor_name",
        "component_name",
        "entity_name",
    ):
        value = _clean(claim.get(field))
        if value:
            return value
    return ""


def audit_compiler_gaps(
    *,
    claims: Sequence[Mapping[str, Any]],
    derivation_rows: Sequence[Mapping[str, Any]],
    registry_audits: Sequence[Mapping[str, Any]],
    validation_warnings: Sequence[str],
    rule_rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]],
    evidence_inventory_rows: Sequence[Mapping[str, Any]],
) -> list[CompilerGapRecord]:
    """Return deterministic, terminal records for all detectable compiler gaps."""

    records: list[CompilerGapRecord] = []
    claim_ids = {_clean(claim.get("claim_id")) for claim in claims if _clean(claim.get("claim_id"))}
    claim_chunk_ids = {
        chunk_id
        for claim in claims
        for chunk_id in (
            _clean(claim.get("chunk_id")),
            *_values(claim.get("source_chunk_ids")),
        )
        if chunk_id
    }
    inventory_evidence_ids = {
        _clean(row.get("evidence_id"))
        for row in evidence_inventory_rows
        if _clean(row.get("evidence_id"))
    }
    inventory_chunk_ids = {
        _clean(row.get("chunk_id"))
        for row in evidence_inventory_rows
        if _clean(row.get("chunk_id"))
    }
    valid_provenance_ids = (
        claim_ids | claim_chunk_ids | inventory_evidence_ids | inventory_chunk_ids
    )
    formula_names = {
        _claim_name(claim).casefold()
        for claim in claims
        if _claim_name(claim) and _clean(claim.get("formula"))
    }

    for claim in claims:
        claim_id = _clean(claim.get("claim_id"))
        claim_type = _clean(claim.get("claim_type"))
        name = _claim_name(claim)
        evidence_ids = _values(claim.get("evidence_ids"))
        chunk_ids = tuple(
            value
            for value in (
                _clean(claim.get("chunk_id")),
                *_values(claim.get("source_chunk_ids")),
            )
            if value
        )
        if not set(evidence_ids + chunk_ids) & valid_provenance_ids:
            records.append(
                _record(
                    source_stage="claim_freeze",
                    gap_type="broken_claim_provenance",
                    terminal_status="unresolved_broken_provenance",
                    claim_id=claim_id,
                    entity_id=_clean(claim.get("entity_id")),
                    missing_fields=("traceable evidence_id or chunk_id",),
                    evidence_ids=evidence_ids,
                    chunk_ids=chunk_ids,
                    affects_network_materialization=True,
                    detail="Claim has no identifier resolving to the frozen evidence inventory or corpus chunks.",
                )
            )
        if (
            claim_type in {"compound", "precursor", "structural_component"}
            and name
            and not _clean(claim.get("formula"))
            and name.casefold() not in formula_names
        ):
            records.append(
                _record(
                    source_stage="entity_registry",
                    gap_type="missing_entity_formula",
                    terminal_status="unresolved_missing_literature_evidence",
                    claim_id=claim_id,
                    entity_id=_clean(claim.get("entity_id")),
                    missing_fields=("formula",),
                    evidence_ids=evidence_ids,
                    chunk_ids=chunk_ids,
                    affects_network_materialization=True,
                    detail=f"No formula-bearing evidence claim resolves entity {name!r}.",
                )
            )
        if claim_type in {"diagnostic_fragment", "neutral_loss"} and not _clean(
            claim.get("ion_mode")
        ):
            records.append(
                _record(
                    source_stage="rule_compiler",
                    gap_type="missing_ion_mode",
                    terminal_status="unresolved_missing_literature_evidence",
                    claim_id=claim_id,
                    missing_fields=("ion_mode",),
                    evidence_ids=evidence_ids,
                    chunk_ids=chunk_ids,
                    affects_network_materialization=True,
                    detail="Fragment or neutral-loss evidence cannot cross polarity without an ion-mode assertion.",
                )
            )
        if claim_type == "transformation" and (
            not _clean(claim.get("source_entity"))
            or not _clean(claim.get("target_entity"))
        ):
            records.append(
                _record(
                    source_stage="rule_compiler",
                    gap_type="missing_source_target_relation",
                    terminal_status="unresolved_missing_literature_evidence",
                    claim_id=claim_id,
                    missing_fields=("source_entity", "target_entity"),
                    evidence_ids=evidence_ids,
                    chunk_ids=chunk_ids,
                    affects_network_materialization=True,
                    detail="Transformation lacks both executable endpoints.",
                )
            )

    for table_name, rows in rule_rows_by_table.items():
        for row in rows:
            rule_id = _clean(row.get("rule_id"))
            evidence_ids = _values(row.get("evidence_ids"))
            resolved = tuple(value for value in evidence_ids if value in valid_provenance_ids)
            if not resolved:
                records.append(
                    _record(
                        source_stage="rule_compiler",
                        gap_type="broken_rule_provenance",
                        terminal_status="unresolved_broken_provenance",
                        rule_id=rule_id,
                        entity_id=_clean(
                            row.get("entity_id")
                            or row.get("target_entity_id")
                            or row.get("source_entity_id")
                        ),
                        missing_fields=("resolvable evidence_ids",),
                        evidence_ids=evidence_ids,
                        affects_network_materialization=True,
                        detail=f"{table_name} row does not resolve to the frozen claim/evidence inventory.",
                    )
                )

    for row in derivation_rows:
        status = _clean(row.get("status"))
        chemical_status = _clean(row.get("chemical_validation_status"))
        product_status = _clean(row.get("product_resolution_status"))
        common = {
            "source_stage": "reaction_compiler",
            "reaction_template_id": _clean(row.get("template_claim_id")),
            "derivation_id": _clean(row.get("derivation_id")),
            "entity_id": _clean(row.get("target_entity_id") or row.get("anchor_entity_id")),
            "affects_network_materialization": True,
            "detail": _clean(row.get("detail")),
        }
        if chemical_status and chemical_status not in {"valid", "derived_formula_valid"}:
            records.append(
                _record(
                    **common,
                    gap_type="invalid_reaction_chemistry",
                    terminal_status="invalid_chemical_or_schema_state",
                    missing_fields=("balanced formula equation",),
                )
            )
            continue
        if product_status in {"ambiguous_reported_entities", "unresolved_product_class"}:
            records.append(
                _record(
                    **common,
                    gap_type="ambiguous_product_isomers",
                    terminal_status="unresolved_ambiguous_entity",
                    missing_fields=("unique product entity",),
                )
            )
            continue
        if status == "combination_limit_exceeded":
            records.append(
                _record(
                    **common,
                    gap_type="combination_limit_exceeded",
                    terminal_status="nonmaterializable_explanatory_only",
                    missing_fields=("bounded reactant combination set",),
                )
            )
            continue
        if status and status != "materialized":
            records.append(
                _record(
                    **common,
                    gap_type=status,
                    terminal_status="nonmaterializable_explanatory_only",
                    missing_fields=("materializable reaction state",),
                )
            )
            continue
        if status == "materialized" and product_status == "mechanism_derived_entity":
            records.append(
                _record(
                    **common,
                    gap_type="deterministic_reaction_derivation",
                    terminal_status="resolved_deterministic_derivation",
                    deterministic_repair_action="formula-balanced mechanism-derived entity construction",
                )
            )
        elif status == "materialized" and product_status == "unique_reported_entity":
            records.append(
                _record(
                    **common,
                    gap_type="reported_reaction_materialization",
                    terminal_status="resolved_existing_evidence",
                    deterministic_repair_action="stable-ID join to unique reported product",
                )
            )

    for row in registry_audits:
        status = _clean(row.get("status"))
        if status not in {"conflict", "rejected"}:
            continue
        detail = _clean(row.get("detail"))
        ambiguous = "ambig" in detail.casefold() or "conflict" in detail.casefold()
        records.append(
            _record(
                source_stage="entity_registry",
                gap_type="registry_entity_resolution",
                terminal_status=(
                    "unresolved_ambiguous_entity"
                    if ambiguous
                    else "invalid_chemical_or_schema_state"
                ),
                claim_id=_clean(row.get("record_id")),
                entity_id=_clean(row.get("entity_id")),
                missing_fields=("unique chemically consistent entity",),
                affects_network_materialization=True,
                detail=detail,
            )
        )

    for warning in validation_warnings:
        detail = _clean(warning)
        if not detail:
            continue
        ambiguous = "ambig" in detail.casefold() or "unresolved" in detail.casefold()
        records.append(
            _record(
                source_stage="rule_compiler",
                gap_type="compiler_validation_warning",
                terminal_status=(
                    "unresolved_ambiguous_entity"
                    if ambiguous
                    else "invalid_chemical_or_schema_state"
                ),
                affects_network_materialization=True,
                detail=detail,
            )
        )

    deduplicated = {record.gap_id: record for record in records}
    return sorted(
        deduplicated.values(),
        key=lambda item: (item.terminal_status, item.gap_type, item.gap_id),
    )


def _csv_value(value: object) -> object:
    if isinstance(value, (tuple, list)):
        return ";".join(str(item) for item in value)
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return value


def write_compiler_gap_audit(
    records: Sequence[CompilerGapRecord], output_dir: Path | str
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    jsonl_path = root / "compiler_gap_audit.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")

    fieldnames = list(CompilerGapRecord.__dataclass_fields__)
    with (root / "compiler_gap_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {key: _csv_value(value) for key, value in record.as_dict().items()}
            )

    status_counts = {status: 0 for status in sorted(TERMINAL_GAP_STATUSES)}
    type_counts: dict[str, int] = {}
    for record in records:
        status_counts[record.terminal_status] += 1
        type_counts[record.gap_type] = type_counts.get(record.gap_type, 0) + 1
    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "gap_count": len(records),
        "network_materialization_gap_count": sum(
            record.affects_network_materialization for record in records
        ),
        "terminal_status_counts": status_counts,
        "gap_type_counts": dict(sorted(type_counts.items())),
        "retrieval_triggered": False,
        "llm_calls_triggered": 0,
    }
    (root / "compiler_gap_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
