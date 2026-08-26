"""Evidence provenance models shared by the Agentic RAG core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class EvidenceValidationError(ValueError):
    """Raised when a claim cannot be traced to a literature chunk."""


def clean_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


@dataclass(frozen=True)
class EvidenceRef:
    chunk_id: str
    evidence_quote: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceRef":
        chunk_id = clean_text(value.get("chunk_id"))
        quote = clean_text(value.get("evidence_quote") or value.get("evidence_sentence"))
        if not chunk_id:
            raise EvidenceValidationError("chunk_id is required")
        if not quote:
            raise EvidenceValidationError("evidence_quote is required")
        return cls(chunk_id=chunk_id, evidence_quote=quote)


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    evidence_type: str
    source_file: str
    chunk_id: str
    page: int | None
    sheet_name: str
    table_id: str
    row_index: int | None
    evidence_quote: str
    extraction_method: str
    review_status: str


@dataclass(frozen=True)
class FragmentEvidence:
    fragment_id: str
    entity_id: str
    fragment_mz: float
    ion_mode: str
    assignment: str
    evidence_scope: str
    reported_precision: int | None
    evidence_ids: tuple[str, ...]
    evidence_role: str = "unassigned_peak"
    specificity_scope: str = "unassigned_peak"
    source_structure: str = "unknown"
    adduct: str = ""
    entity_class_id: str = ""
    competitor_entity_count: int = 0
    competitor_entity_fraction: float = 0.0
    discriminative_status: str = "not_evaluated"

