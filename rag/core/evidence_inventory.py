"""Immutable, chunk-traceable literature evidence inventory."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from .evidence_models import EvidenceRecord, FragmentEvidence, clean_text


EVIDENCE_SCOPES = frozenset(
    {
        "target_specific",
        "target_associated",
        "class_diagnostic",
        "component_specific",
        "reaction_associated",
        "unassigned_peak",
    }
)

FRAGMENT_EVIDENCE_ROLES = frozenset(
    {
        "explicit_target_diagnostic",
        "target_product_ion",
        "class_diagnostic",
        "reaction_supporting_fragment",
        "neutral_loss",
        "theoretical_catalog",
        "unassigned_peak",
    }
)

SOURCE_STRUCTURES = frozenset(
    {
        "narrative",
        "structured_table",
        "structured_identification_table",
        "supplementary_catalog",
        "unknown",
    }
)

_DIAGNOSTIC_LANGUAGE = re.compile(
    r"\b(?:diagnostic|characteristic|marker|signature)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class RejectedEvidence:
    claim_id: str
    reason: str
    chunk_id: str = ""
    detail: str = ""


@dataclass(frozen=True)
class EvidenceInventory:
    evidence: tuple[EvidenceRecord, ...]
    fragments: tuple[FragmentEvidence, ...]
    rejected: tuple[RejectedEvidence, ...]


def _stable_id(prefix: str, payload: Mapping[str, object]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha1(canonical.encode('utf-8')).hexdigest()[:16]}"


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _traceability_text(value: object) -> str:
    text = str(value or "").casefold()
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    return " ".join(text.split())


def _quote_is_traceable(quote: str, chunk_text: object) -> bool:
    normalized_quote = _traceability_text(quote)
    normalized_chunk = _traceability_text(chunk_text)
    return bool(normalized_quote and normalized_quote in normalized_chunk)


def _fragment_values(value: object) -> tuple[tuple[float, int | None], ...]:
    if isinstance(value, (list, tuple)):
        raw_values = [str(item) for item in value]
    else:
        raw_values = re.split(r"[;,|]", str(value or ""))
    parsed: list[tuple[float, int | None]] = []
    seen: set[float] = set()
    for raw in raw_values:
        match = re.search(r"(?<![A-Za-z0-9])(\d{1,5}(?:\.\d+)?)(?![A-Za-z0-9])", raw.strip())
        if not match:
            continue
        number = float(match.group(1))
        if not math.isfinite(number) or number <= 0:
            continue
        rounded = round(number, 9)
        if rounded in seen:
            continue
        seen.add(rounded)
        decimals = len(match.group(1).partition(".")[2]) if "." in match.group(1) else 0
        parsed.append((number, decimals))
    return tuple(parsed)


def _ion_mode(value: object) -> str:
    text = clean_text(value).casefold()
    if text in {"positive", "pos", "+", "p"}:
        return "positive"
    if text in {"negative", "neg", "-", "n"}:
        return "negative"
    return "not_reported"


def _ion_mode_from_chunks(
    chunk_ids: Sequence[str],
    chunks_by_id: Mapping[str, Mapping[str, object]],
) -> str:
    def detect(text: str) -> str:
        normalized = (
            text.replace("\u2212", "-")
            .replace("\u2013", "-")
            .replace("\u2014", "-")
        )
        positive = bool(
            re.search(
                r"\bpositive\b.{0,30}\bion\s+mode\b",
                normalized,
                re.IGNORECASE,
            )
            or re.search(r"\bESI\s*\+", normalized, re.IGNORECASE)
            or re.search(
                r"\[\s*M\s*\+\s*(?:H|Na|NH4)\s*\]\s*\+",
                normalized,
                re.IGNORECASE,
            )
        )
        negative = bool(
            re.search(
                r"\bnegative\b.{0,30}\bion\s+mode\b",
                normalized,
                re.IGNORECASE,
            )
            or re.search(r"\bESI\s*-", normalized, re.IGNORECASE)
            or re.search(
                r"\[\s*M\s*(?:-\s*H|\+\s*HCOO)\s*\]\s*-",
                normalized,
                re.IGNORECASE,
            )
        )
        if positive == negative:
            return "not_reported"
        return "positive" if positive else "negative"

    local_text = " ".join(
        clean_text(chunks_by_id[chunk_id].get("text"))
        for chunk_id in chunk_ids
        if chunk_id in chunks_by_id
    )
    local_mode = detect(local_text)
    if local_mode != "not_reported":
        return local_mode

    source_files = {
        clean_text(chunks_by_id[chunk_id].get("source_file"))
        for chunk_id in chunk_ids
        if chunk_id in chunks_by_id
    }
    source_files.discard("")
    if not source_files:
        return "not_reported"
    document_text = " ".join(
        clean_text(chunk.get("text"))
        for chunk in chunks_by_id.values()
        if clean_text(chunk.get("source_file")) in source_files
    )
    return detect(document_text)


def _scope_for_claim(claim: Mapping[str, object]) -> str:
    explicit = clean_text(
        claim.get("specificity_scope")
        or claim.get("evidence_scope")
        or claim.get("fragment_evidence_scope")
    ).casefold()
    if explicit in EVIDENCE_SCOPES:
        return explicit
    claim_type = clean_text(claim.get("claim_type")).casefold()
    if claim_type == "diagnostic_fragment":
        return "class_diagnostic"
    return "unassigned_peak"


def _source_structure(claim: Mapping[str, object]) -> str:
    explicit = clean_text(claim.get("source_structure")).casefold()
    if explicit in SOURCE_STRUCTURES:
        return explicit
    claim_source = clean_text(claim.get("claim_source")).casefold()
    if "structured" in claim_source or "table" in claim_source:
        return "structured_table"
    return "unknown"


def normalize_fragment_evidence_role(
    claim: Mapping[str, object],
    *,
    entity_id: str,
    source_structure: str,
) -> tuple[str, str]:
    """Return a conservative semantic role and specificity scope.

    The normalization uses only claim provenance and wording. Structured peak
    catalogs are never promoted to diagnostic evidence without an explicit
    diagnostic statement tied to a resolved entity.
    """

    requested = clean_text(claim.get("evidence_role")).casefold()
    if requested not in FRAGMENT_EVIDENCE_ROLES:
        requested = ""
    claim_type = clean_text(claim.get("claim_type")).casefold()
    quote = clean_text(
        claim.get("evidence_quote") or claim.get("evidence_sentence")
    )
    diagnostic_language = bool(_DIAGNOSTIC_LANGUAGE.search(quote))
    supplied_scope = _scope_for_claim(claim)
    supplied_scope_is_explicit = bool(
        clean_text(
            claim.get("specificity_scope")
            or claim.get("evidence_scope")
            or claim.get("fragment_evidence_scope")
        )
    )

    if source_structure == "structured_identification_table":
        if requested == "explicit_target_diagnostic" and entity_id and diagnostic_language:
            return "explicit_target_diagnostic", "target_specific"
        if requested == "target_product_ion" and entity_id:
            return "target_product_ion", "target_specific"
        if supplied_scope == "class_diagnostic" and diagnostic_language:
            return "class_diagnostic", "class_diagnostic"
        return (
            "theoretical_catalog",
            "target_associated" if entity_id else "unassigned_peak",
        )

    if source_structure in {"structured_table", "supplementary_catalog"}:
        if (
            requested == "explicit_target_diagnostic"
            and entity_id
            and diagnostic_language
        ):
            return "explicit_target_diagnostic", "target_specific"
        if supplied_scope == "class_diagnostic" and diagnostic_language:
            return "class_diagnostic", "class_diagnostic"
        return (
            "theoretical_catalog",
            "target_associated" if entity_id else "unassigned_peak",
        )

    if requested == "explicit_target_diagnostic":
        if entity_id and diagnostic_language:
            return "explicit_target_diagnostic", "target_specific"
        return "class_diagnostic", "class_diagnostic"
    if requested == "target_product_ion":
        if entity_id:
            return "target_product_ion", "target_specific"
        return "unassigned_peak", "unassigned_peak"
    if requested == "class_diagnostic":
        return "class_diagnostic", "class_diagnostic"
    if requested == "reaction_supporting_fragment":
        return "reaction_supporting_fragment", "reaction_associated"
    if requested == "neutral_loss":
        return "neutral_loss", "reaction_associated"
    if requested == "theoretical_catalog":
        return (
            "theoretical_catalog",
            "target_associated" if entity_id else "unassigned_peak",
        )
    if requested == "unassigned_peak":
        return "unassigned_peak", "unassigned_peak"

    if supplied_scope == "class_diagnostic" or claim_type == "diagnostic_fragment":
        if entity_id and diagnostic_language:
            return "explicit_target_diagnostic", "target_specific"
        return "class_diagnostic", "class_diagnostic"
    if supplied_scope in {"component_specific", "reaction_associated"}:
        return "reaction_supporting_fragment", supplied_scope
    if claim_type == "neutral_loss":
        return "neutral_loss", "reaction_associated"
    if supplied_scope == "unassigned_peak" and supplied_scope_is_explicit:
        return "unassigned_peak", "unassigned_peak"
    if entity_id:
        return "target_product_ion", "target_specific"
    return "unassigned_peak", "unassigned_peak"


def _evidence_type(claim: Mapping[str, object]) -> str:
    claim_type = clean_text(claim.get("claim_type")).casefold()
    if claim_type in {"fragment", "diagnostic_fragment"}:
        return "fragment"
    if claim_type == "reaction_template":
        return "reaction"
    if claim_type == "entity_class_membership":
        return "class_membership"
    if claim_type in {"compound", "entity"}:
        return "entity"
    return claim_type or "other"


def _entity_id(claim: Mapping[str, object]) -> str:
    direct = clean_text(claim.get("entity_id"))
    if direct:
        return direct
    name = clean_text(
        claim.get("compound_name")
        or claim.get("component_name")
        or claim.get("precursor_name")
    )
    compound_class = clean_text(claim.get("compound_class"))
    if not name:
        return ""
    return _stable_id(
        "entity",
        {"compound_class": compound_class.casefold(), "reported_name": name.casefold()},
    )


def _claim_chunk_ids(claim: Mapping[str, object]) -> tuple[str, ...]:
    values: list[str] = []
    direct = clean_text(claim.get("chunk_id"))
    if direct:
        values.append(direct)
    source_ids = claim.get("source_chunk_ids")
    if isinstance(source_ids, (list, tuple)):
        values.extend(clean_text(item) for item in source_ids)
    elif source_ids:
        values.extend(clean_text(item) for item in re.split(r"[;,|]", str(source_ids)))
    return tuple(dict.fromkeys(item for item in values if item))


def build_evidence_inventory(
    claims: Sequence[Mapping[str, object]],
    chunks_by_id: Mapping[str, Mapping[str, object]],
) -> EvidenceInventory:
    """Validate claim provenance and split fragment cells into atomic records."""

    evidence_by_id: dict[str, EvidenceRecord] = {}
    fragments_by_id: dict[str, FragmentEvidence] = {}
    rejected: list[RejectedEvidence] = []
    for claim in claims:
        claim_id = clean_text(claim.get("claim_id"))
        quote = clean_text(claim.get("evidence_quote") or claim.get("evidence_sentence"))
        if not quote:
            rejected.append(RejectedEvidence(claim_id, "missing_evidence_quote"))
            continue
        chunk_ids = _claim_chunk_ids(claim)
        if not chunk_ids:
            rejected.append(RejectedEvidence(claim_id, "untraceable_evidence"))
            continue
        accepted_ids: list[str] = []
        for chunk_id in chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                rejected.append(RejectedEvidence(claim_id, "untraceable_evidence", chunk_id))
                continue
            status = clean_text(claim.get("traceability_status")).casefold()
            if status not in {"exact", "normalized", "fuzzy", "structured"} and not _quote_is_traceable(
                quote, chunk.get("text")
            ):
                rejected.append(RejectedEvidence(claim_id, "quote_not_in_chunk", chunk_id))
                continue
            record_payload = {
                "claim_id": claim_id,
                "chunk_id": chunk_id,
                "evidence_quote": quote,
                "evidence_type": _evidence_type(claim),
            }
            evidence_id = _stable_id("evidence", record_payload)
            evidence_by_id[evidence_id] = EvidenceRecord(
                evidence_id=evidence_id,
                evidence_type=record_payload["evidence_type"],
                source_file=clean_text(chunk.get("source_file")),
                chunk_id=chunk_id,
                page=_optional_int(chunk.get("page")),
                sheet_name=clean_text(chunk.get("sheet_name")),
                table_id=clean_text(chunk.get("table_id") or chunk.get("section")),
                row_index=_optional_int(chunk.get("row_index")),
                evidence_quote=quote,
                extraction_method=clean_text(claim.get("claim_source")) or "llm",
                review_status="accepted",
            )
            accepted_ids.append(evidence_id)
        if not accepted_ids:
            continue
        fragment_source = claim.get("fragment_mz")
        if not clean_text(fragment_source):
            fragment_source = claim.get("reported_fragments")
        values = _fragment_values(fragment_source)
        if not values:
            continue
        entity_id = _entity_id(claim)
        source_structure = _source_structure(claim)
        role, scope = normalize_fragment_evidence_role(
            claim,
            entity_id=entity_id,
            source_structure=source_structure,
        )
        requested_role = clean_text(claim.get("evidence_role")).casefold()
        if requested_role and requested_role != role:
            rejected.append(
                RejectedEvidence(
                    claim_id,
                    "fragment_role_downgraded",
                    detail=f"{requested_role}->{role}",
                )
            )
        ion_mode = _ion_mode(claim.get("ion_mode"))
        if ion_mode == "not_reported":
            ion_mode = _ion_mode_from_chunks(chunk_ids, chunks_by_id)
        for fragment_mz, precision in values:
            fragment_payload = {
                "entity_id": entity_id,
                "entity_class_id": clean_text(claim.get("entity_class_id")),
                "fragment_mz": round(fragment_mz, 9),
                "ion_mode": ion_mode,
                "evidence_scope": scope,
                "evidence_role": role,
                "source_structure": source_structure,
                "adduct": clean_text(claim.get("adduct")),
                "evidence_ids": sorted(accepted_ids),
            }
            fragment_id = _stable_id("fragment", fragment_payload)
            fragments_by_id[fragment_id] = FragmentEvidence(
                fragment_id=fragment_id,
                entity_id=entity_id,
                fragment_mz=fragment_mz,
                ion_mode=fragment_payload["ion_mode"],
                assignment=clean_text(claim.get("assignment")),
                evidence_scope=scope,
                reported_precision=precision,
                evidence_ids=tuple(sorted(accepted_ids)),
                evidence_role=role,
                specificity_scope=scope,
                source_structure=source_structure,
                adduct=fragment_payload["adduct"],
                entity_class_id=fragment_payload["entity_class_id"],
            )
    return EvidenceInventory(
        evidence=tuple(evidence_by_id[key] for key in sorted(evidence_by_id)),
        fragments=tuple(fragments_by_id[key] for key in sorted(fragments_by_id)),
        rejected=tuple(rejected),
    )


def derive_fragment_specificity(
    fragments: Sequence[FragmentEvidence],
    *,
    tolerance_da: float = 0.1,
    low_sharing_fraction: float = 0.1,
) -> tuple[FragmentEvidence, ...]:
    """Derive polarity-local entity sharing without metabolite knowledge."""

    if tolerance_da <= 0:
        raise ValueError("tolerance_da must be positive")
    if not 0 < low_sharing_fraction <= 1:
        raise ValueError("low_sharing_fraction must be in (0, 1]")
    rows = tuple(fragments)
    by_mode: dict[str, list[FragmentEvidence]] = {}
    for row in rows:
        by_mode.setdefault(row.ion_mode, []).append(row)
    derived: dict[str, FragmentEvidence] = {}
    for mode, mode_rows in by_mode.items():
        sorted_rows = sorted(mode_rows, key=lambda item: item.fragment_mz)
        values = [row.fragment_mz for row in sorted_rows]
        entity_universe = {
            row.entity_id for row in sorted_rows if row.entity_id
        }
        denominator = max(len(entity_universe), 1)
        for row in sorted_rows:
            if mode not in {"positive", "negative"}:
                derived[row.fragment_id] = replace(
                    row,
                    competitor_entity_count=0,
                    competitor_entity_fraction=0.0,
                    discriminative_status="polarity_unresolved",
                )
                continue
            if not row.entity_id:
                derived[row.fragment_id] = replace(
                    row,
                    competitor_entity_count=0,
                    competitor_entity_fraction=0.0,
                    discriminative_status="unassigned",
                )
                continue
            left = bisect_left(values, row.fragment_mz - tolerance_da)
            right = bisect_right(values, row.fragment_mz + tolerance_da)
            competing_entities = {
                candidate.entity_id
                for candidate in sorted_rows[left:right]
                if candidate.entity_id
            }
            count = len(competing_entities)
            fraction = count / denominator
            if count <= 1:
                status = "unique"
            elif fraction <= low_sharing_fraction:
                status = "low_sharing"
            else:
                status = "shared"
            derived[row.fragment_id] = replace(
                row,
                competitor_entity_count=count,
                competitor_entity_fraction=fraction,
                discriminative_status=status,
            )
    return tuple(derived.get(row.fragment_id, row) for row in rows)


def load_fragment_evidence(path: Path) -> tuple[FragmentEvidence, ...]:
    rows: list[FragmentEvidence] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"{path.name} line {line_number} must contain a JSON object"
            )
        rows.append(FragmentEvidence(**value))
    return tuple(rows)


def write_fragment_evidence_registry(
    fragments: Sequence[FragmentEvidence],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(fragments, key=lambda item: item.fragment_id):
            handle.write(
                json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


def write_evidence_inventory(inventory: EvidenceInventory, output_dir: Path) -> None:
    """Write inventory artifacts in stable UTF-8 order."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fragments = derive_fragment_specificity(inventory.fragments)
    for filename, rows in (
        ("evidence_inventory.jsonl", inventory.evidence),
        ("fragment_evidence.jsonl", fragments),
    ):
        with (output_dir / filename).open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
    with (output_dir / "evidence_inventory_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["claim_id", "reason", "chunk_id", "detail"])
        writer.writeheader()
        for row in inventory.rejected:
            writer.writerow(asdict(row))


def load_evidence_inventory(input_dir: Path) -> EvidenceInventory:
    """Load a previously frozen inventory without rebuilding the corpus.

    This supports deterministic compiler-only rebuilds after Entity Registry
    improvements.  It never reconstructs evidence from rule tables and keeps
    the immutable evidence and fragment identifiers unchanged.
    """

    root = Path(input_dir)

    def load_jsonl(path: Path) -> list[dict[str, object]]:
        if not path.exists():
            raise FileNotFoundError(f"evidence inventory artifact not found: {path}")
        rows: list[dict[str, object]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path.name} line {line_number} must contain a JSON object"
                )
            rows.append(value)
        return rows

    evidence = tuple(
        EvidenceRecord(**row)
        for row in load_jsonl(root / "evidence_inventory.jsonl")
    )
    fragments = tuple(
        FragmentEvidence(
            **{
                **row,
                "evidence_ids": tuple(row.get("evidence_ids", ())),
            }
        )
        for row in load_jsonl(root / "fragment_evidence.jsonl")
    )
    rejected: list[RejectedEvidence] = []
    audit_path = root / "evidence_inventory_audit.csv"
    if audit_path.exists():
        with audit_path.open(encoding="utf-8-sig", newline="") as handle:
            rejected.extend(RejectedEvidence(**row) for row in csv.DictReader(handle))
    return EvidenceInventory(
        evidence=evidence,
        fragments=fragments,
        rejected=tuple(rejected),
    )
