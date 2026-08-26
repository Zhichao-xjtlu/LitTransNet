"""Structured, variable-arity literature reaction claim models.

This module validates data shape and provenance only. It contains no compound-
class knowledge and performs no chemical inference.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .evidence_models import EvidenceRef, EvidenceValidationError, clean_text


ALLOWED_SCOPES = frozenset({"specific_entity", "entity_class", "unresolved"})


class ClaimValidationError(ValueError):
    """Raised when a structured evidence claim violates its data contract."""


def stable_structured_id(prefix: str, payload: Mapping[str, object]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ClaimValidationError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True)
class EntityClaim:
    claim_id: str
    entity_id: str
    entity_name: str
    compound_class: str
    entity_kind: str
    reported_names: tuple[str, ...]
    formula: str
    exact_mass: str
    evidence: EvidenceRef


@dataclass(frozen=True)
class EntityClassMembershipClaim:
    claim_id: str
    entity_id: str
    entity_class_id: str
    membership_role: str
    evidence: EvidenceRef


@dataclass(frozen=True)
class ReactionParticipant:
    entity_id: str
    entity_name: str
    entity_class_id: str
    role: str
    coefficient: int
    scope: str
    form_id: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], field: str) -> "ReactionParticipant":
        scope = clean_text(value.get("scope"))
        if scope not in ALLOWED_SCOPES:
            raise ClaimValidationError(f"{field}.scope must be one of {sorted(ALLOWED_SCOPES)}")
        role = clean_text(value.get("role"))
        if not role:
            raise ClaimValidationError(f"{field}.role is required")
        entity_id = clean_text(value.get("entity_id"))
        entity_name = clean_text(value.get("entity_name"))
        entity_class_id = clean_text(value.get("entity_class_id"))
        form_id = clean_text(value.get("form_id"))
        if scope == "specific_entity" and not (entity_id or entity_name):
            raise ClaimValidationError(f"{field} specific_entity requires entity_id or entity_name")
        if scope == "entity_class" and not entity_class_id:
            raise ClaimValidationError(f"{field} entity_class_id is required for entity_class scope")
        return cls(
            entity_id=entity_id,
            entity_name=entity_name,
            entity_class_id=entity_class_id,
            form_id=form_id,
            role=role,
            coefficient=_positive_integer(value.get("coefficient"), f"{field}.coefficient"),
            scope=scope,
        )


@dataclass(frozen=True)
class ReactionTemplateClaim:
    claim_id: str
    compound_class: str
    reaction_name: str
    reaction_type: str
    network_anchor_role: str
    anchor_reactant_index: int
    reactants: tuple[ReactionParticipant, ...]
    products: tuple[ReactionParticipant, ...]
    reaction_operator: str
    formula_delta: str
    stoichiometry_status: str
    evidence: EvidenceRef

    @property
    def anchor_participant(self) -> ReactionParticipant:
        return self.reactants[self.anchor_reactant_index]


StructuredClaim = EntityClaim | EntityClassMembershipClaim | ReactionTemplateClaim


def _claim_id(prefix: str, supplied: object, payload: Mapping[str, object]) -> str:
    return clean_text(supplied) or stable_structured_id(prefix, payload)


def _participants(value: object, field: str) -> tuple[ReactionParticipant, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ClaimValidationError(f"{field} must be a non-empty array")
    parsed: list[ReactionParticipant] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise ClaimValidationError(f"{field}[{index}] must be an object")
        parsed.append(ReactionParticipant.from_mapping(row, f"{field}[{index}]"))
    return tuple(parsed)


def _parse_reaction(value: Mapping[str, Any]) -> ReactionTemplateClaim:
    try:
        evidence = EvidenceRef.from_mapping(value)
    except EvidenceValidationError as exc:
        raise ClaimValidationError(str(exc)) from exc
    reactants = _participants(value.get("reactants"), "reactants")
    products = _participants(value.get("products"), "products")
    anchor_value = value.get("anchor_reactant_index")
    if isinstance(anchor_value, bool) or not isinstance(anchor_value, int):
        raise ClaimValidationError("anchor_reactant_index must be an integer")
    if anchor_value < 0 or anchor_value >= len(reactants):
        raise ClaimValidationError("anchor_reactant_index is out of range")
    anchor_role = clean_text(value.get("network_anchor_role"))
    if not anchor_role:
        raise ClaimValidationError("network_anchor_role is required")
    if reactants[anchor_value].role != anchor_role:
        raise ClaimValidationError("network_anchor_role does not match the indexed reactant role")
    normalized = {
        "claim_type": "reaction_template",
        "compound_class": clean_text(value.get("compound_class")),
        "reaction_name": clean_text(value.get("reaction_name")),
        "reaction_type": clean_text(value.get("reaction_type")),
        "network_anchor_role": anchor_role,
        "anchor_reactant_index": anchor_value,
        "reactants": [asdict(item) for item in reactants],
        "products": [asdict(item) for item in products],
        "reaction_operator": clean_text(value.get("reaction_operator")),
        "formula_delta": clean_text(value.get("formula_delta")),
        "stoichiometry_status": clean_text(value.get("stoichiometry_status")),
        "chunk_id": evidence.chunk_id,
        "evidence_quote": evidence.evidence_quote,
    }
    return ReactionTemplateClaim(
        claim_id=_claim_id("reaction", value.get("claim_id"), normalized),
        compound_class=normalized["compound_class"],
        reaction_name=normalized["reaction_name"],
        reaction_type=normalized["reaction_type"],
        network_anchor_role=anchor_role,
        anchor_reactant_index=anchor_value,
        reactants=reactants,
        products=products,
        reaction_operator=normalized["reaction_operator"],
        formula_delta=normalized["formula_delta"],
        stoichiometry_status=normalized["stoichiometry_status"],
        evidence=evidence,
    )


def _parse_membership(value: Mapping[str, Any]) -> EntityClassMembershipClaim:
    try:
        evidence = EvidenceRef.from_mapping(value)
    except EvidenceValidationError as exc:
        raise ClaimValidationError(str(exc)) from exc
    entity_id = clean_text(value.get("entity_id"))
    entity_class_id = clean_text(value.get("entity_class_id"))
    membership_role = clean_text(value.get("membership_role"))
    if not entity_id:
        raise ClaimValidationError("entity_id is required for entity_class_membership")
    if not entity_class_id:
        raise ClaimValidationError("entity_class_id is required for entity_class_membership")
    if not membership_role:
        raise ClaimValidationError("membership_role is required for entity_class_membership")
    normalized = {
        "claim_type": "entity_class_membership",
        "entity_id": entity_id,
        "entity_class_id": entity_class_id,
        "membership_role": membership_role,
        "chunk_id": evidence.chunk_id,
        "evidence_quote": evidence.evidence_quote,
    }
    return EntityClassMembershipClaim(
        claim_id=_claim_id("membership", value.get("claim_id"), normalized),
        entity_id=entity_id,
        entity_class_id=entity_class_id,
        membership_role=membership_role,
        evidence=evidence,
    )


def _parse_entity(value: Mapping[str, Any]) -> EntityClaim:
    try:
        evidence = EvidenceRef.from_mapping(value)
    except EvidenceValidationError as exc:
        raise ClaimValidationError(str(exc)) from exc
    entity_id = clean_text(value.get("entity_id"))
    entity_name = clean_text(value.get("entity_name") or value.get("compound_name"))
    if not entity_id or not entity_name:
        raise ClaimValidationError("entity claims require entity_id and entity_name")
    normalized = {
        "claim_type": "entity",
        "entity_id": entity_id,
        "entity_name": entity_name,
        "compound_class": clean_text(value.get("compound_class")),
        "entity_kind": clean_text(value.get("entity_kind")) or "molecule",
        "reported_names": [
            clean_text(item)
            for item in value.get("reported_names", [])
            if clean_text(item)
        ]
        if isinstance(value.get("reported_names"), Sequence)
        and not isinstance(value.get("reported_names"), (str, bytes))
        else [],
        "formula": clean_text(value.get("formula")),
        "exact_mass": clean_text(value.get("exact_mass")),
        "chunk_id": evidence.chunk_id,
        "evidence_quote": evidence.evidence_quote,
    }
    return EntityClaim(
        claim_id=_claim_id("entity", value.get("claim_id"), normalized),
        entity_id=entity_id,
        entity_name=entity_name,
        compound_class=normalized["compound_class"],
        entity_kind=normalized["entity_kind"],
        reported_names=tuple(normalized["reported_names"]),
        formula=normalized["formula"],
        exact_mass=normalized["exact_mass"],
        evidence=evidence,
    )


def parse_structured_claim(value: Mapping[str, Any]) -> StructuredClaim:
    claim_type = clean_text(value.get("claim_type"))
    if claim_type == "reaction_template":
        return _parse_reaction(value)
    if claim_type == "entity_class_membership":
        return _parse_membership(value)
    if claim_type in {"entity", "compound"}:
        return _parse_entity(value)
    raise ClaimValidationError(f"unsupported structured claim_type {claim_type!r}")


def adapt_legacy_unary_claim(value: Mapping[str, Any]) -> ReactionTemplateClaim:
    source = clean_text(value.get("source_entity"))
    target = clean_text(value.get("target_entity"))
    if not source or not target:
        raise ClaimValidationError("legacy unary transformation requires source_entity and target_entity")
    adapted = {
        "claim_id": clean_text(value.get("claim_id")),
        "claim_type": "reaction_template",
        "compound_class": clean_text(value.get("compound_class")),
        "reaction_name": clean_text(value.get("transformation_name")),
        "reaction_type": clean_text(value.get("transformation_name")) or "single_reactant_transformation",
        "network_anchor_role": "source",
        "anchor_reactant_index": 0,
        "reactants": [
            {
                "entity_name": source,
                "role": "source",
                "coefficient": 1,
                "scope": "specific_entity",
            }
        ],
        "products": [
            {
                "entity_name": target,
                "role": "product",
                "coefficient": 1,
                "scope": "specific_entity",
            }
        ],
        "reaction_operator": clean_text(value.get("reaction_operator")),
        "formula_delta": clean_text(value.get("formula_delta")),
        "stoichiometry_status": clean_text(value.get("stoichiometry_status")) or "legacy_unary",
        "chunk_id": clean_text(value.get("chunk_id")),
        "evidence_quote": clean_text(value.get("evidence_quote") or value.get("evidence_sentence")),
    }
    return _parse_reaction(adapted)

