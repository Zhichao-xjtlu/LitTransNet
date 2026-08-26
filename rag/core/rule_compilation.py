#!/usr/bin/env python3
"""Agent 3: compile literature evidence claims into generic rule tables.

This stage does not read PDFs, perform retrieval, or call an LLM. It only
converts Agent 2 evidence claims into the five CSV tables consumed by the
generic network rule loader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from rag.core.chemical_consistency import formula_exact_mass
from rag.core.compiler_gap_audit import audit_compiler_gaps, write_compiler_gap_audit
from rag.core.entity_registry import (
    EntityForm,
    EntityRecord as RegistryEntityRecord,
    EntityRegistry,
    load_entity_registry,
    normalize_registry_name,
    registry_name_keys,
    stable_registry_id,
    write_entity_registry,
)
from rag.core.evidence_inventory import (
    derive_fragment_specificity,
    load_fragment_evidence,
    write_fragment_evidence_registry,
)
from rag.core.evidence_models import FragmentEvidence
from rag.core.fragment_contracts import compile_fragment_evidence_contract
from rag.core.io_utils import atomic_write_json as write_json, clean_text, safe_float
from rag.core.reaction_compiler import EntityRecord, materialize_reaction_templates
from rag.core.reaction_models import (
    ClaimValidationError,
    EntityClassMembershipClaim,
    ReactionTemplateClaim,
    parse_structured_claim,
)
from rag.core.rule_bundle import RuleTable, write_rule_bundle
from rag.core.knowledge_enrichment import (
    interpret_neutral_loss,
    interpret_transformation_delta,
)
from rag.core.diagnostic_evidence_miner import run_diagnostic_evidence_mining
from rag.core.literature_supported_inference import (
    run_literature_supported_inference,
)


RULE_TABLES = {
    "compound": "compound_rules.csv",
    "transformation": "transformation_rules.csv",
    "diagnostic_fragment": "diagnostic_fragment_rules.csv",
    "neutral_loss": "neutral_loss_rules.csv",
    "biosynthetic_component": "biosynthetic_component_rules.csv",
}

COMPOUND_COLUMNS = [
    "rule_id",
    "entity_id",
    "entity_origin",
    "target_origin",
    "compound_name",
    "synonyms",
    "compound_class",
    "subclass",
    "formula",
    "exact_mass",
    "ion_mode",
    "adduct",
    "reported_precursor_mz",
    "reported_fragments",
    "reported_neutral_losses",
    "literature_status",
    "derivation_id",
    "evidence_ids",
    "review_status",
    "review_note",
]

TRANSFORMATION_COLUMNS = [
    "rule_id",
    "compound_class",
    "subclass",
    "source_entity",
    "target_entity",
    "source_entity_id",
    "target_entity_id",
    "transformation_name",
    "evidence_type",
    "claim_source",
    "component_name",
    "component_delta_formula",
    "relation_evidence_status",
    "template_support_level",
    "candidate_policy",
    "propagation_policy",
    "reaction_arity",
    "reactant_entities",
    "product_entities",
    "anchor_reactant_index",
    "network_anchor_role",
    "resolved_network_anchor_count",
    "anchor_resolution_stage",
    "reaction_type",
    "reaction_operator",
    "formula_equation",
    "reactant_form_ids",
    "product_form_ids",
    "fragment_evidence_contract",
    "calculated_delta_mass",
    "chemical_validation_status",
    "product_resolution_status",
    "derivation_id",
    "delta_mass",
    "direction",
    "repeatable",
    "max_repeat",
    "required_context",
    "evidence_ids",
    "review_status",
    "review_note",
]

TRANSFORMATION_EVIDENCE_TYPES = {
    "explicit_report",
    "literature_inferred",
    "mechanism_derived",
    "delta_only",
}

DIAGNOSTIC_FRAGMENT_COLUMNS = [
    "rule_id",
    "compound_class",
    "subclass",
    "fragment_mz",
    "ion_mode",
    "fragment_assignment",
    "diagnostic_origin",
    "support_compound_count",
    "support_compound_fraction",
    "support_rank",
    "required",
    "evidence_ids",
    "review_status",
    "review_note",
]

NEUTRAL_LOSS_COLUMNS = [
    "rule_id",
    "compound_class",
    "subclass",
    "loss_name",
    "loss_mass",
    "ion_mode",
    "interpretation",
    "required_context",
    "evidence_ids",
    "review_status",
    "review_note",
]

BIOSYNTHETIC_COMPONENT_COLUMNS = [
    "rule_id",
    "entity_id",
    "form_id",
    "entity_class_id",
    "entity_scope",
    "compound_class",
    "component_type",
    "component_name",
    "formula",
    "exact_mass",
    "role",
    "reaction_logic",
    "delta_mass_to_product",
    "decarboxy_delta_mass",
    "evidence_ids",
    "review_status",
    "review_note",
]

class RuleCompilationError(RuntimeError):
    """Raised when Agent 3 cannot compile rule tables."""


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def format_float(value: Any, digits: int = 6) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    text = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def stable_id(prefix: str, *parts: Any) -> str:
    basis = "|".join(clean_text(part).lower() for part in parts)
    return f"{prefix}_" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:14]


def resolve_registry_entity_id(
    registry: EntityRegistry,
    name: Any,
    *,
    preferred_id: Any = "",
    formula: Any = "",
    exact_mass: Any = None,
) -> str:
    """Resolve a rule row to one chemically coherent Registry entity.

    A valid preferred ID wins. Otherwise exact normalized names are narrowed
    by reported formula and mass. Duplicate records with one chemical
    signature are deterministic aliases; conflicting signatures remain
    unresolved.
    """

    preferred = clean_text(preferred_id)
    if preferred in registry.entities:
        return preferred
    candidate_ids = tuple(
        entity_id
        for entity_id in registry.name_index.get(
            normalize_registry_name(name), ()
        )
        if entity_id in registry.entities
    )
    if not candidate_ids:
        return ""
    requested_formula = clean_text(formula)
    if requested_formula:
        formula_matches = tuple(
            entity_id
            for entity_id in candidate_ids
            if registry.entities[entity_id].formula == requested_formula
        )
        if formula_matches:
            candidate_ids = formula_matches
    requested_mass = safe_float(exact_mass)
    if requested_mass is not None:
        mass_matches = tuple(
            entity_id
            for entity_id in candidate_ids
            if registry.entities[entity_id].exact_mass is not None
            and abs(
                float(registry.entities[entity_id].exact_mass)
                - requested_mass
            )
            <= 0.01
        )
        if mass_matches:
            candidate_ids = mass_matches
    formula_bearing = tuple(
        entity_id
        for entity_id in candidate_ids
        if registry.entities[entity_id].formula
    )
    if formula_bearing:
        candidate_ids = formula_bearing
    signatures = {
        (
            registry.entities[entity_id].formula,
            (
                round(float(registry.entities[entity_id].exact_mass), 6)
                if registry.entities[entity_id].exact_mass is not None
                else None
            ),
        )
        for entity_id in candidate_ids
    }
    if len(candidate_ids) == 1 or len(signatures) == 1:
        return sorted(candidate_ids)[0]
    return ""


def _fragment_claim_entity_name(claim: dict[str, Any]) -> str:
    return first_nonempty(
        claim.get("compound_name"),
        claim.get("component_name"),
        claim.get("precursor_name"),
    )


def _fragment_claim_provisional_entity_id(claim: dict[str, Any]) -> str:
    direct = clean_text(claim.get("entity_id"))
    if direct:
        return direct
    name = _fragment_claim_entity_name(claim)
    if not name:
        return ""
    return stable_registry_id(
        "entity",
        {
            "compound_class": clean_text(claim.get("compound_class")).casefold(),
            "reported_name": name.casefold(),
        },
    )


def _fragment_claim_resolution_names(value: Any) -> tuple[str, ...]:
    """Return exact evidence name plus a variant without table footnote marks."""

    name = clean_text(value)
    if not name:
        return ()
    values = [name]
    without_footnote = re.sub(r"\s*[\*†‡]+\s*$", "", name).strip()
    if without_footnote and without_footnote != name:
        values.append(without_footnote)
    return tuple(values)


def reconcile_fragment_entity_references(
    fragments: tuple[FragmentEvidence, ...],
    claims: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    registry: EntityRegistry,
    *,
    provisional_id_overrides: dict[str, str] | None = None,
) -> tuple[tuple[FragmentEvidence, ...], dict[str, int]]:
    """Resolve provisional fragment entities or conservatively unassign them.

    Fragment extraction and Registry construction are independent.  A fragment
    claim may therefore carry a deterministic provisional ID for a table name
    that resolves to a different canonical Registry ID.  Unknown IDs must not
    enter a schema-valid rule bundle.  We relink only through a unique
    evidence-explicit name; otherwise the peak is preserved as unassigned.
    """

    overrides = provisional_id_overrides or {}
    candidates: dict[str, set[str]] = defaultdict(set)
    for claim in claims:
        claim_id = clean_text(claim.get("claim_id"))
        provisional_id = overrides.get(
            claim_id,
            _fragment_claim_provisional_entity_id(claim),
        )
        if not provisional_id or provisional_id in registry.entities:
            continue
        name = _fragment_claim_entity_name(claim)
        for candidate_name in _fragment_claim_resolution_names(name):
            resolved = resolve_registry_entity_id(registry, candidate_name)
            if resolved:
                candidates[provisional_id].add(resolved)

    unique_mapping = {
        provisional_id: next(iter(entity_ids))
        for provisional_id, entity_ids in candidates.items()
        if len(entity_ids) == 1
    }
    reconciled: list[FragmentEvidence] = []
    resolved_count = 0
    downgraded_count = 0
    downgraded_class_count = 0
    for row in fragments:
        replacements: dict[str, Any] = {}
        if row.entity_id and row.entity_id not in registry.entities:
            resolved_id = unique_mapping.get(row.entity_id, "")
            if resolved_id:
                replacements["entity_id"] = resolved_id
                resolved_count += 1
            else:
                replacements["entity_id"] = ""
                downgraded_count += 1
                if row.evidence_role in {
                    "target_product_ion",
                    "explicit_target_diagnostic",
                    "theoretical_catalog",
                }:
                    replacements.update(
                        evidence_role="unassigned_peak",
                        evidence_scope="unassigned_peak",
                        specificity_scope="unassigned_peak",
                    )
        if row.entity_class_id and row.entity_class_id not in registry.classes:
            replacements["entity_class_id"] = ""
            downgraded_class_count += 1
            if row.evidence_role == "class_diagnostic":
                replacements.update(
                    evidence_role="unassigned_peak",
                    evidence_scope="unassigned_peak",
                    specificity_scope="unassigned_peak",
                )
        if replacements:
            reconciled.append(replace(row, **replacements))
        else:
            reconciled.append(row)
    return tuple(reconciled), {
        "resolved_unknown_entity_count": resolved_count,
        "downgraded_unknown_entity_count": downgraded_count,
        "downgraded_unknown_class_count": downgraded_class_count,
        "ambiguous_provisional_entity_count": sum(
            1 for entity_ids in candidates.values() if len(entity_ids) > 1
        ),
    }


def resolve_fragment_supported_alias_entity_id(
    registry: EntityRegistry | None,
    target_entity_id: str,
    fragments: tuple[FragmentEvidence, ...],
) -> str:
    """Link an explicit reported alias to one fragment-bearing entity.

    Alias linkage requires a name key explicitly present in the literature
    name plus formula or bounded mass agreement. It never selects by formula
    equality alone.
    """

    if registry is None:
        return target_entity_id
    target = registry.entities.get(target_entity_id)
    if target is None:
        return target_entity_id
    fragment_counts: dict[str, int] = defaultdict(int)
    for row in fragments:
        if row.evidence_role in {
            "target_product_ion",
            "explicit_target_diagnostic",
        }:
            fragment_counts[row.entity_id] += 1
    if fragment_counts.get(target_entity_id, 0):
        return target_entity_id
    alias_keys = {
        key
        for name in (target.canonical_name, *target.reported_names)
        for key in registry_name_keys(name)
    }
    candidate_ids = {
        entity_id
        for key in alias_keys
        for entity_id in registry.name_index.get(key, ())
        if entity_id != target_entity_id
        and fragment_counts.get(entity_id, 0)
    }
    compatible: list[str] = []
    for entity_id in sorted(candidate_ids):
        candidate = registry.entities.get(entity_id)
        if candidate is None:
            continue
        formula_agrees = bool(
            target.formula
            and candidate.formula
            and target.formula == candidate.formula
        )
        mass_agrees = bool(
            target.exact_mass is not None
            and candidate.exact_mass is not None
            and abs(float(target.exact_mass) - float(candidate.exact_mass))
            <= 0.01
        )
        if formula_agrees or mass_agrees:
            compatible.append(entity_id)
    return compatible[0] if len(compatible) == 1 else target_entity_id


def split_values(value: Any) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    parts = [clean_text(item) for item in re.split(r"[;|]+", text)]
    return [item for item in parts if item]


def join_unique(values: list[Any]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        for item in split_values(value):
            key = item.lower()
            if key not in seen:
                seen.add(key)
                out.append(item)
    return ";".join(out)


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = clean_text(value)
        if text:
            return text
    return ""


def evidence_id(claim: dict[str, Any]) -> str:
    return first_nonempty(claim.get("evidence_ids"), claim.get("claim_id"), claim.get("chunk_id"))


def load_claims(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuleCompilationError(f"Evidence claims file does not exist: {path}")
    claims: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuleCompilationError(f"Invalid claim at {path}:{line_number}")
            claims.append(row)
    return claims


def load_concepts(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "compound_class": "",
            "concepts": [],
            "single_subclass": "",
            "compound_to_subclass": {},
            "component_claims": [],
            "neutral_loss_claims": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuleCompilationError(f"Concepts JSON must contain an object: {path}")
    concepts: list[dict[str, Any]] = []
    for key in ("accepted_concepts", "review_concepts", "concepts"):
        value = payload.get(key, [])
        if isinstance(value, list):
            concepts.extend(item for item in value if isinstance(item, dict))

    compound_class = clean_text(payload.get("compound_class"))
    subclasses = [
        clean_text(concept.get("name"))
        for concept in concepts
        if clean_text(concept.get("type")) == "subclass" and clean_text(concept.get("name"))
    ]
    unique_subclasses = sorted({item.lower(): item for item in subclasses}.values(), key=str.lower)
    single_subclass = unique_subclasses[0] if len(unique_subclasses) == 1 else ""

    compound_to_subclass: dict[str, str] = {}
    for concept in concepts:
        if clean_text(concept.get("type")) != "compound":
            continue
        name = clean_text(concept.get("name"))
        subclass = first_nonempty(
            concept.get("subclass"),
            concept.get("parent_subclass"),
            concept.get("subclass_name"),
        )
        if name and subclass:
            compound_to_subclass[name.lower()] = subclass

    component_claims: list[dict[str, Any]] = []
    neutral_loss_claims: list[dict[str, Any]] = []
    for concept in concepts:
        concept_type = clean_text(concept.get("type"))
        source_ids = split_values(concept.get("evidence_ids")) or [
            clean_text(item) for item in concept.get("source_chunk_ids", []) if clean_text(item)
        ]
        if concept_type == "neutral_loss":
            explicit_mass = first_nonempty(
                concept.get("neutral_loss_mass"),
                concept.get("loss_mass"),
            )
            if not explicit_mass:
                for text in (
                    concept.get("name"),
                    concept.get("evidence_quote"),
                    concept.get("evidence_summary"),
                ):
                    match = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*Da\b", clean_text(text), re.I)
                    if match:
                        explicit_mass = match.group(1)
                        break
            if explicit_mass and source_ids:
                neutral_loss_claims.append(
                    {
                        "claim_id": join_unique(source_ids),
                        "claim_type": "neutral_loss",
                        "compound_class": compound_class,
                        "neutral_loss_mass": explicit_mass,
                        "loss_name": clean_text(concept.get("name")),
                        "interpretation": clean_text(concept.get("evidence_summary")),
                        "required_context": "concept_evidence",
                        "evidence_summary": clean_text(concept.get("evidence_summary")),
                    }
                )
            continue
        if concept_type not in {"precursor", "structural_component", "biosynthetic_component"}:
            continue
        component_name = clean_text(concept.get("name"))
        if not component_name or not source_ids:
            continue
        component_claims.append(
            {
                "claim_id": join_unique(source_ids),
                "claim_type": concept_type,
                "compound_class": compound_class,
                "component_name": component_name if concept_type != "precursor" else "",
                "precursor_name": component_name if concept_type == "precursor" else "",
                "formula": clean_text(concept.get("formula")),
                "exact_mass": clean_text(concept.get("exact_mass")),
                "role": clean_text(concept.get("role")) or concept_type,
                "reaction_logic": clean_text(concept.get("evidence_summary")),
                "evidence_summary": clean_text(concept.get("evidence_summary")),
            }
        )

    return {
        "compound_class": compound_class,
        "concepts": concepts,
        "single_subclass": single_subclass,
        "compound_to_subclass": compound_to_subclass,
        "component_claims": component_claims,
        "neutral_loss_claims": neutral_loss_claims,
    }


def enrich_subclass(claim: dict[str, Any], concept_context: dict[str, Any]) -> str:
    explicit = clean_text(claim.get("subclass"))
    if explicit:
        return explicit
    compound_name = clean_text(claim.get("compound_name"))
    compound_to_subclass = concept_context.get("compound_to_subclass", {})
    if compound_name and isinstance(compound_to_subclass, dict):
        mapped = clean_text(compound_to_subclass.get(compound_name.lower()))
        if mapped:
            return mapped
    return clean_text(concept_context.get("single_subclass"))


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: clean_text(row.get(column, "")) for column in columns})


def collect_formula_warnings(claims: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for claim in claims:
        formula = clean_text(claim.get("formula"))
        exact_mass = safe_float(claim.get("exact_mass"))
        if not formula or exact_mass is None:
            continue
        try:
            calculated = formula_exact_mass(formula)
        except ValueError:
            warnings.append(f"{evidence_id(claim)}: formula could not be parsed: {formula}")
            continue
        if abs(calculated - exact_mass) > 0.05:
            warnings.append(
                f"{evidence_id(claim)}: formula mass {calculated:.4f} differs from exact_mass {exact_mass:.4f}"
            )
    return warnings


def merge_rule(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if key == "evidence_ids":
            merged[key] = join_unique([existing.get(key, ""), value])
        elif key in {"reported_fragments", "reported_neutral_losses"}:
            merged[key] = join_unique([existing.get(key, ""), value])
        elif not clean_text(merged.get(key, "")) and clean_text(value):
            merged[key] = value
    return merged


def add_merged(target: dict[tuple[Any, ...], dict[str, Any]], key: tuple[Any, ...], row: dict[str, Any]) -> None:
    if key in target:
        target[key] = merge_rule(target[key], row)
    else:
        target[key] = row


def normalized_fragment_mz(value: Any) -> str:
    return format_float(value, digits=2)


def reported_fragment_values(claim: dict[str, Any]) -> list[str]:
    return [value for value in (normalized_fragment_mz(item) for item in split_values(claim.get("reported_fragments"))) if value]


def count_reported_fragment_compound_support(claims: list[dict[str, Any]]) -> dict[str, int]:
    fragment_to_compounds: dict[str, set[str]] = defaultdict(set)
    for claim in claims:
        if clean_text(claim.get("claim_type")) != "compound":
            continue
        compound_key = first_nonempty(claim.get("compound_name"), evidence_id(claim))
        if not compound_key:
            continue
        for fragment_mz in reported_fragment_values(claim):
            fragment_to_compounds[fragment_mz].add(compound_key.lower())
    return {fragment_mz: len(compounds) for fragment_mz, compounds in fragment_to_compounds.items()}


def count_reported_fragment_instances(claims: list[dict[str, Any]]) -> int:
    return sum(
        len(reported_fragment_values(claim))
        for claim in claims
        if clean_text(claim.get("claim_type")) == "compound"
    )


def select_class_consensus_fragments(
    claims: list[dict[str, Any]],
    *,
    min_compound_count: int,
    min_support_fraction: float,
    max_fragments_per_class: int,
) -> list[dict[str, Any]]:
    """Select a bounded diagnostic set by distinct-compound prevalence."""

    compounds_by_group: dict[tuple[str, str], set[str]] = defaultdict(set)
    fragment_compounds: dict[
        tuple[str, str, str], set[str]
    ] = defaultdict(set)
    fragment_evidence: dict[
        tuple[str, str, str], list[str]
    ] = defaultdict(list)
    class_labels: dict[str, str] = {}
    for claim in claims:
        if clean_text(claim.get("claim_type")) != "compound":
            continue
        compound_class = clean_text(claim.get("compound_class"))
        compound_name = clean_text(claim.get("compound_name"))
        fragments = sorted(set(reported_fragment_values(claim)), key=float)
        if not compound_class or not compound_name or not fragments:
            continue
        class_key = compound_class.casefold()
        class_labels.setdefault(class_key, compound_class)
        ion_mode = clean_text(claim.get("ion_mode")).casefold()
        if ion_mode not in {"positive", "negative"}:
            ion_mode = ""
        group_key = (class_key, ion_mode)
        compound_key = compound_name.casefold()
        compounds_by_group[group_key].add(compound_key)
        ev_id = evidence_id(claim)
        for fragment_mz in fragments:
            key = (class_key, ion_mode, fragment_mz)
            fragment_compounds[key].add(compound_key)
            if ev_id:
                fragment_evidence[key].append(ev_id)

    selected: list[dict[str, Any]] = []
    for (class_key, ion_mode), compound_ids in sorted(
        compounds_by_group.items()
    ):
        denominator = len(compound_ids)
        if denominator == 0:
            continue
        eligible: list[tuple[str, int, float, str]] = []
        for key, supported_compounds in fragment_compounds.items():
            fragment_class, fragment_mode, fragment_mz = key
            if (fragment_class, fragment_mode) != (class_key, ion_mode):
                continue
            support_count = len(supported_compounds)
            support_fraction = support_count / denominator
            if (
                support_count < min_compound_count
                or support_fraction < min_support_fraction
            ):
                continue
            eligible.append(
                (
                    fragment_mz,
                    support_count,
                    support_fraction,
                    join_unique(fragment_evidence[key]),
                )
            )
        eligible.sort(key=lambda row: (-row[1], float(row[0])))
        for rank, (
            fragment_mz,
            support_count,
            support_fraction,
            evidence_ids,
        ) in enumerate(eligible[:max_fragments_per_class], start=1):
            selected.append(
                {
                    "compound_class": class_labels[class_key],
                    "ion_mode": ion_mode,
                    "fragment_mz": fragment_mz,
                    "support_compound_count": support_count,
                    "support_compound_fraction": support_fraction,
                    "support_rank": rank,
                    "evidence_ids": evidence_ids,
                }
            )
    return selected


def compile_compound_rules(
    claims: list[dict[str, Any]],
    concept_context: dict[str, Any],
    enrichment_statistics: dict[str, int],
) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for claim in claims:
        if clean_text(claim.get("claim_type")) != "compound":
            continue
        compound_name = clean_text(claim.get("compound_name"))
        if not compound_name:
            continue
        ev_id = evidence_id(claim)
        if not ev_id:
            continue
        compound_class = clean_text(claim.get("compound_class"))
        original_subclass = clean_text(claim.get("subclass"))
        subclass = enrich_subclass(claim, concept_context)
        if subclass and not original_subclass:
            enrichment_statistics["subclass_values_added_from_concepts"] += 1
        key = (compound_name.lower(), compound_class.lower())
        entity_id = clean_text(claim.get("entity_id")) or stable_id("entity", compound_class, compound_name)
        row = {
            "rule_id": stable_id("cmp", compound_class, compound_name),
            "entity_id": entity_id,
            "entity_origin": "reported",
            "compound_name": compound_name,
            "synonyms": clean_text(claim.get("synonyms")),
            "compound_class": compound_class,
            "subclass": subclass,
            "formula": clean_text(claim.get("formula")),
            "exact_mass": format_float(claim.get("exact_mass")),
            "ion_mode": clean_text(claim.get("ion_mode")),
            "adduct": clean_text(claim.get("adduct")),
            "reported_precursor_mz": format_float(claim.get("reported_precursor_mz") or claim.get("precursor_mz")),
            "reported_fragments": first_nonempty(clean_text(claim.get("reported_fragments")), format_float(claim.get("fragment_mz"))),
            "reported_neutral_losses": first_nonempty(
                clean_text(claim.get("reported_neutral_losses")),
                format_float(claim.get("neutral_loss_mass")),
            ),
            "literature_status": "reported",
            "derivation_id": "",
            "evidence_ids": ev_id,
            "review_status": "candidate",
            "review_note": clean_text(claim.get("evidence_summary")),
        }
        add_merged(merged, key, row)
    return list(merged.values())


def compile_diagnostic_fragment_rules(
    claims: list[dict[str, Any]],
    concept_context: dict[str, Any],
    enrichment_statistics: dict[str, int],
    min_reported_fragment_compound_count: int = 20,
    min_reported_fragment_support_fraction: float = 0.20,
    max_class_consensus_fragments: int = 20,
) -> tuple[list[dict[str, Any]], int]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    removed = 0
    reported_fragment_support = count_reported_fragment_compound_support(claims)
    enrichment_statistics["reported_fragments_seen"] = len(reported_fragment_support)
    enrichment_statistics["reported_fragment_instances_seen"] = count_reported_fragment_instances(claims)
    enrichment_statistics["reported_fragment_min_compound_count"] = min_reported_fragment_compound_count
    enrichment_statistics["reported_fragment_min_support_fraction"] = (
        min_reported_fragment_support_fraction
    )
    enrichment_statistics["reported_fragment_max_per_class"] = (
        max_class_consensus_fragments
    )
    consensus_rows = select_class_consensus_fragments(
        claims,
        min_compound_count=min_reported_fragment_compound_count,
        min_support_fraction=min_reported_fragment_support_fraction,
        max_fragments_per_class=max_class_consensus_fragments,
    )
    enrichment_statistics["reported_fragments_promoted_to_diagnostic"] = len(
        consensus_rows
    )
    for consensus in consensus_rows:
        compound_class = clean_text(consensus.get("compound_class"))
        ion_mode = clean_text(consensus.get("ion_mode"))
        fragment_mz = clean_text(consensus.get("fragment_mz"))
        support_count = int(consensus.get("support_compound_count") or 0)
        support_fraction = float(
            consensus.get("support_compound_fraction") or 0.0
        )
        key = (
            compound_class.casefold(),
            "",
            fragment_mz,
            ion_mode,
        )
        row = {
            "rule_id": stable_id(
                "frag_consensus",
                compound_class,
                ion_mode,
                fragment_mz,
            ),
            "compound_class": compound_class,
            "subclass": "",
            "fragment_mz": fragment_mz,
            "ion_mode": ion_mode,
            "fragment_assignment": (
                f"class consensus fragment ({support_count} compounds)"
            ),
            "diagnostic_origin": "class_consensus_frequency",
            "support_compound_count": str(support_count),
            "support_compound_fraction": format_float(
                support_fraction, digits=4
            ),
            "support_rank": str(consensus.get("support_rank") or ""),
            "required": "FALSE",
            "evidence_ids": clean_text(consensus.get("evidence_ids")),
            "review_status": "candidate",
            "review_note": (
                "Promoted by distinct-compound prevalence within the "
                f"reported {compound_class} fragment catalog."
            ),
        }
        add_merged(merged, key, row)
    for claim in claims:
        claim_type = clean_text(claim.get("claim_type"))
        if claim_type != "diagnostic_fragment":
            continue
        ev_id = evidence_id(claim)
        if not ev_id:
            removed += 1
            continue
        fragment_values = [format_float(claim.get("fragment_mz"))]
        fragment_values = [value for value in fragment_values if value]
        if not fragment_values:
            removed += 1
            continue
        compound_class = clean_text(claim.get("compound_class"))
        original_subclass = clean_text(claim.get("subclass"))
        subclass = enrich_subclass(claim, concept_context)
        if subclass and not original_subclass:
            enrichment_statistics["subclass_values_added_from_concepts"] += 1
        for fragment_mz in fragment_values:
            ion_mode = clean_text(claim.get("ion_mode"))
            key = (
                compound_class.lower(),
                subclass.lower(),
                fragment_mz,
                ion_mode.lower(),
            )
            assignment = first_nonempty(claim.get("fragment_assignment"), claim.get("evidence_summary"))
            row = {
                "rule_id": stable_id("frag", compound_class, subclass, fragment_mz),
                "compound_class": compound_class,
                "subclass": subclass,
                "fragment_mz": fragment_mz,
                "ion_mode": ion_mode,
                "fragment_assignment": assignment,
                "diagnostic_origin": "explicit_evidence",
                "support_compound_count": "",
                "support_compound_fraction": "",
                "support_rank": "",
                "required": "FALSE",
                "evidence_ids": ev_id,
                "review_status": "candidate",
                "review_note": clean_text(claim.get("evidence_summary")),
            }
            add_merged(merged, key, row)
    return list(merged.values()), removed


def compile_neutral_loss_rules(
    claims: list[dict[str, Any]],
    concept_context: dict[str, Any],
    enrichment_statistics: dict[str, int],
) -> tuple[list[dict[str, Any]], int]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    removed = 0
    concept_claims = concept_context.get("neutral_loss_claims", [])
    all_claims = (
        list(claims)
        + _expand_component_claims(claims)
        + [claim for claim in concept_claims if isinstance(claim, dict)]
    )
    for claim in all_claims:
        if clean_text(claim.get("claim_type")) != "neutral_loss":
            continue
        loss_mass = format_float(claim.get("neutral_loss_mass") or claim.get("loss_mass"))
        ev_id = evidence_id(claim)
        if not loss_mass or not ev_id:
            removed += 1
            continue
        compound_class = clean_text(claim.get("compound_class"))
        original_subclass = clean_text(claim.get("subclass"))
        subclass = enrich_subclass(claim, concept_context)
        if subclass and not original_subclass:
            enrichment_statistics["subclass_values_added_from_concepts"] += 1
        interpretation = clean_text(claim.get("interpretation"))
        loss_name = first_nonempty(claim.get("loss_name"), claim.get("assignment"))
        if not interpretation:
            interpretation = interpret_neutral_loss(loss_mass)
            if interpretation:
                enrichment_statistics["neutral_loss_interpretations_added"] += 1
        if not loss_name and interpretation:
            loss_name = interpretation
        key = (
            compound_class.lower(),
            loss_mass,
            clean_text(claim.get("ion_mode")).lower(),
            clean_text(claim.get("required_context")).lower(),
        )
        required_context = clean_text(claim.get("required_context"))
        if clean_text(claim.get("source_ion_type")) == "product_ion":
            ion_context = ";".join(
                value
                for value in (
                    "product_ion_cascade",
                    f"source_ion_mz={clean_text(claim.get('source_ion_mz'))}"
                    if clean_text(claim.get("source_ion_mz"))
                    else "",
                    f"product_ion_mz={clean_text(claim.get('product_ion_mz'))}"
                    if clean_text(claim.get("product_ion_mz"))
                    else "",
                )
                if value
            )
            required_context = ion_context
        row = {
            "rule_id": stable_id("loss", compound_class, loss_mass),
            "compound_class": compound_class,
            "subclass": subclass,
            "loss_name": loss_name,
            "loss_mass": loss_mass,
            "ion_mode": clean_text(claim.get("ion_mode")),
            "interpretation": interpretation,
            "required_context": required_context,
            "evidence_ids": ev_id,
            "review_status": "candidate",
            "review_note": clean_text(claim.get("evidence_summary")),
        }
        add_merged(merged, key, row)
    return list(merged.values()), removed


def transformation_delta(claim: dict[str, Any], warnings: list[str]) -> str:
    explicit = format_float(claim.get("delta_mass"))
    source_mass = safe_float(claim.get("source_exact_mass"))
    target_mass = safe_float(claim.get("target_exact_mass"))
    if explicit:
        if source_mass is not None and target_mass is not None:
            calculated = target_mass - source_mass
            if abs(calculated - float(explicit)) > 0.05:
                warnings.append(
                    f"{evidence_id(claim)}: transformation delta_mass differs from target-source by >0.05 Da"
                )
        return explicit
    if source_mass is not None and target_mass is not None:
        return format_float(target_mass - source_mass)
    return ""


def transformation_evidence_type(claim: dict[str, Any]) -> str:
    explicit = clean_text(claim.get("evidence_type")).lower()
    if explicit in TRANSFORMATION_EVIDENCE_TYPES:
        return explicit
    inference_type = clean_text(claim.get("inference_type"))
    traceability_status = clean_text(claim.get("traceability_status")).lower()
    if inference_type or traceability_status == "inferred_from_literature_compounds":
        return "literature_inferred"
    source_entity = clean_text(claim.get("source_entity"))
    target_entity = clean_text(claim.get("target_entity"))
    if source_entity and target_entity:
        return "explicit_report"
    return "delta_only"


def compile_transformation_rules(
    claims: list[dict[str, Any]],
    warnings: list[str],
    concept_context: dict[str, Any],
    enrichment_statistics: dict[str, int],
) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for claim in claims:
        if clean_text(claim.get("claim_type")) != "transformation":
            continue
        ev_id = evidence_id(claim)
        if not ev_id:
            continue
        compound_class = clean_text(claim.get("compound_class"))
        original_subclass = clean_text(claim.get("subclass"))
        subclass = enrich_subclass(claim, concept_context)
        if subclass and not original_subclass:
            enrichment_statistics["subclass_values_added_from_concepts"] += 1
        source_entity = clean_text(claim.get("source_entity"))
        target_entity = clean_text(claim.get("target_entity"))
        evidence_type = transformation_evidence_type(claim)
        delta_mass = transformation_delta(claim, warnings)
        transformation_name = clean_text(claim.get("transformation_name"))
        if not transformation_name:
            transformation_name = interpret_transformation_delta(delta_mass)
            if transformation_name:
                enrichment_statistics["transformation_names_added"] += 1
        key = (source_entity.lower(), target_entity.lower(), transformation_name.lower(), compound_class.lower())
        row = {
            "rule_id": stable_id("tr", compound_class, source_entity, target_entity, transformation_name),
            "compound_class": compound_class,
            "subclass": subclass,
            "source_entity": source_entity,
            "target_entity": target_entity,
            "transformation_name": transformation_name,
            "evidence_type": evidence_type,
            "claim_source": clean_text(claim.get("claim_source")),
            "component_name": clean_text(claim.get("component_name")),
            "component_delta_formula": clean_text(
                claim.get("component_delta_formula")
            ),
            "relation_evidence_status": clean_text(
                claim.get("relation_evidence_status")
            ),
            "delta_mass": delta_mass,
            "direction": clean_text(claim.get("direction")) or "source_to_target",
            "repeatable": clean_text(claim.get("repeatable")) or "FALSE",
            "max_repeat": clean_text(claim.get("max_repeat")) or "1",
            "required_context": clean_text(claim.get("required_context")),
            "evidence_ids": ev_id,
            "review_status": "candidate",
            "review_note": clean_text(claim.get("evidence_summary")),
        }
        add_merged(merged, key, row)
    return list(merged.values())


def materialize_direct_unary_transformation(
    row: dict[str, Any],
    registry: EntityRegistry,
) -> None:
    """Project a direct source-to-target literature claim into schema 4.0.

    Reaction-template claims are structured by ``reaction_compiler``. Direct
    transformation claims are a separate unary pathway and need the same
    executable/auditable fields once their reported entities are resolved.
    Unresolved class-level relations remain non-propagating but retain a
    complete audit status instead of masquerading as materialized rules.
    """
    if clean_text(row.get("reaction_arity")):
        return
    source_name = clean_text(row.get("source_entity"))
    target_name = clean_text(row.get("target_entity"))
    if not source_name or not target_name:
        return

    source_id = clean_text(row.get("source_entity_id"))
    target_id = clean_text(row.get("target_entity_id"))
    source = registry.entities.get(source_id)
    target = registry.entities.get(target_id)
    source_forms = sorted(
        form.form_id
        for form in registry.forms.values()
        if form.entity_id == source_id and form.form_type == "neutral_molecule"
    )
    target_forms = sorted(
        form.form_id
        for form in registry.forms.values()
        if form.entity_id == target_id and form.form_type == "neutral_molecule"
    )
    reactant_payload = [
        {
            "coefficient": 1,
            "entity_id": source_id,
            "role": "substrate",
        }
    ]
    product_payload = [
        {
            "coefficient": 1,
            "entity_id": target_id,
            "role": "product",
        }
    ]
    claimed_delta_mass = safe_float(row.get("delta_mass"))
    delta_mass: float | None = None
    if (
        source is not None
        and target is not None
        and source.exact_mass is not None
        and target.exact_mass is not None
    ):
        delta_mass = float(target.exact_mass) - float(source.exact_mass)
    formulas_resolved = bool(
        source is not None
        and target is not None
        and source.formula
        and target.formula
    )
    operator = re.sub(
        r"[^a-z0-9]+",
        "_",
        clean_text(row.get("transformation_name")).lower(),
    ).strip("_")

    chemical_validation_status = (
        "valid" if formulas_resolved else "unresolved_entities"
    )
    component_formula = clean_text(row.get("component_delta_formula"))
    if (
        formulas_resolved
        and delta_mass is not None
        and component_formula
        and clean_text(row.get("claim_source"))
        == "evidence_guided_component_join"
    ):
        try:
            component_mass = formula_exact_mass(component_formula)
        except (TypeError, ValueError):
            chemical_validation_status = "invalid_component_delta_formula"
        else:
            expected_delta = (
                -component_mass
                if claimed_delta_mass is not None and claimed_delta_mass < 0
                else component_mass
            )
            if abs(delta_mass - expected_delta) > 0.01:
                chemical_validation_status = "component_delta_mismatch"
                row["relation_evidence_status"] = "rejected_component_delta"
                row["propagation_policy"] = "never_propagate"
            else:
                row["relation_evidence_status"] = "component_delta_consistent"

    row.update(
        {
            "reaction_arity": 1,
            "reactant_entities": reactant_payload,
            "product_entities": product_payload,
            "anchor_reactant_index": 0,
            "network_anchor_role": "substrate",
            "resolved_network_anchor_count": 0,
            "anchor_resolution_stage": "network_preflight",
            "reaction_type": "unary_literature_transformation",
            "reaction_operator": operator,
            "formula_equation": (
                f"{source.formula} -> {target.formula}"
                if formulas_resolved
                else ""
            ),
            "reactant_form_ids": source_forms,
            "product_form_ids": target_forms,
            "fragment_evidence_contract": row.get("fragment_evidence_contract")
            or {},
            "calculated_delta_mass": format_float(delta_mass),
            "chemical_validation_status": chemical_validation_status,
            "product_resolution_status": (
                "unique_reported_entity"
                if target is not None
                else "unresolved_product_class"
            ),
            "derivation_id": clean_text(row.get("derivation_id"))
            or stable_id(
                "derivation",
                clean_text(row.get("rule_id")),
                source_id,
                target_id,
            ),
            "required_context": clean_text(row.get("required_context"))
            or (
                "real resolved network anchor"
                if source is not None and target is not None
                else "non_propagating_unresolved_entity"
            ),
        }
    )
    if delta_mass is not None:
        row["delta_mass"] = format_float(delta_mass)


REACTION_DERIVATION_AUDIT_COLUMNS = [
    "template_claim_id",
    "derivation_id",
    "status",
    "anchor_entity_id",
    "resolved_network_anchor_count",
    "total_combination_count",
    "materialized_count",
    "reactant_entities",
    "target_entity_id",
    "product_resolution_status",
    "product_formula",
    "calculated_product_mass",
    "chemical_validation_status",
    "propagation_eligible_at_compile_time",
    "detail",
]


def _entity_records(
    compound_rules: list[dict[str, Any]],
    registry: EntityRegistry | None = None,
) -> dict[str, EntityRecord]:
    records: dict[str, EntityRecord] = {}
    for row in compound_rules:
        entity_id = clean_text(row.get("entity_id"))
        if not entity_id:
            continue
        exact_mass = safe_float(row.get("exact_mass"))
        fragments = tuple(
            value
            for value in (safe_float(item) for item in split_values(row.get("reported_fragments")))
            if value is not None
        )
        records[entity_id] = EntityRecord(
            entity_id=entity_id,
            entity_name=clean_text(row.get("compound_name")),
            formula=clean_text(row.get("formula")),
            exact_mass=exact_mass,
            reported_fragments=fragments,
            evidence_ids=tuple(split_values(row.get("evidence_ids"))),
        )
    if registry is not None:
        for entity_id, entity in registry.entities.items():
            projected = records.get(entity_id)
            records[entity_id] = EntityRecord(
                entity_id=entity_id,
                entity_name=entity.canonical_name,
                formula=entity.formula or (projected.formula if projected else ""),
                exact_mass=(
                    entity.exact_mass
                    if entity.exact_mass is not None
                    else (projected.exact_mass if projected else None)
                ),
                reported_fragments=(
                    projected.reported_fragments if projected else ()
                ),
                evidence_ids=tuple(
                    dict.fromkeys(
                        (
                            *(projected.evidence_ids if projected else ()),
                            *entity.evidence_ids,
                        )
                    )
                ),
            )
    return records


def compile_reaction_templates(
    claims: list[dict[str, Any]],
    compound_rules: list[dict[str, Any]],
    warnings: list[str],
    registry=None,
    fragments: tuple[FragmentEvidence, ...] = (),
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
    tuple[EntityForm, ...],
]:
    templates: list[ReactionTemplateClaim] = []
    template_claim_rows: dict[str, dict[str, Any]] = {}
    memberships: list[EntityClassMembershipClaim] = []
    for claim in claims:
        claim_type = clean_text(claim.get("claim_type"))
        if claim_type not in {"reaction_template", "entity_class_membership"}:
            continue
        if clean_text(claim.get("structured_validation_status")) == "invalid":
            warnings.append(
                f"{evidence_id(claim)}: invalid structured claim: "
                f"{clean_text(claim.get('structured_validation_error'))}"
            )
            continue
        try:
            parsed = parse_structured_claim(claim)
        except ClaimValidationError as exc:
            warnings.append(f"{evidence_id(claim)}: invalid structured claim: {exc}")
            continue
        if isinstance(parsed, ReactionTemplateClaim):
            templates.append(parsed)
            template_claim_rows[parsed.claim_id] = claim
        elif isinstance(parsed, EntityClassMembershipClaim):
            memberships.append(parsed)

    if not templates:
        return (
            [],
            [],
            [],
            {
                "template_count": 0,
                "materialized_count": 0,
                "emitted_rule_count": 0,
                "derived_component_count": 0,
            },
            (),
        )

    entities = _entity_records(compound_rules, registry)
    if registry is not None:
        materialization = materialize_reaction_templates(
            registry=registry,
            templates=templates,
            # Real anchors are spectrum assignments and are resolved by Network V5 preflight.
            resolved_anchor_entity_ids=set(),
        )
    else:
        materialization = materialize_reaction_templates(
            templates=templates,
            entities=entities,
            memberships=memberships,
            resolved_anchor_entity_ids=set(),
        )
    derived_compounds: list[dict[str, Any]] = []
    derived_transformations: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    template_by_claim_id = {item.claim_id: item for item in templates}
    for entity in materialization.derived_entities:
        entities[entity.entity_id] = entity
        template_claim_id = next(
            (
                item.claim_id
                for item in templates
                if any(participant.entity_id == entity.entity_id for participant in item.reactants)
            ),
            "",
        )
        template = template_by_claim_id.get(template_claim_id)
        derived_compounds.append(
            {
                "rule_id": stable_id("cmp", entity.entity_id),
                "entity_id": entity.entity_id,
                "entity_origin": entity.entity_origin,
                "compound_name": entity.entity_name,
                "synonyms": "",
                "compound_class": template.compound_class if template else "",
                "subclass": "",
                "formula": entity.formula,
                "exact_mass": format_float(entity.exact_mass),
                "ion_mode": "",
                "adduct": "",
                "reported_precursor_mz": "",
                "reported_fragments": "",
                "reported_neutral_losses": "",
                "literature_status": "mechanism_derived",
                "derivation_id": entity.derivation_id,
                "evidence_ids": join_unique(list(entity.evidence_ids)),
                "review_status": "candidate",
                "review_note": (
                    "Reactant formula and mass derived by inverse stoichiometric balance from an "
                    "evidence-backed reaction template and reported product."
                ),
            }
        )
    known_derived_ids: set[str] = {item.entity_id for item in materialization.derived_entities}
    for reaction in materialization.materialized_reactions:
        canonical_target_id = resolve_fragment_supported_alias_entity_id(
            registry,
            reaction.target_entity_id,
            fragments,
        )
        if canonical_target_id != reaction.target_entity_id:
            canonical_target = registry.entities[canonical_target_id]
            reaction = replace(
                reaction,
                target_entity_id=canonical_target_id,
                target_entity_name=canonical_target.canonical_name,
                reported_product_candidates=(canonical_target_id,),
            )
        reactant_payload = [
            {"entity_id": item.entity_id, "role": item.role, "coefficient": item.coefficient}
            for item in reaction.reactants
        ]
        audit_rows.append(
            {
                "template_claim_id": reaction.template_claim_id,
                "derivation_id": reaction.derivation_id,
                "status": "materialized",
                "anchor_entity_id": reaction.anchor_entity_id,
                "resolved_network_anchor_count": 0,
                "total_combination_count": "",
                "materialized_count": 1,
                "reactant_entities": reactant_payload,
                "target_entity_id": reaction.target_entity_id,
                "product_resolution_status": reaction.product_resolution_status,
                "product_formula": reaction.product_formula,
                "calculated_product_mass": format_float(reaction.calculated_product_mass),
                "chemical_validation_status": reaction.chemical_validation_status,
                "propagation_eligible_at_compile_time": False,
                "detail": "real network anchors are resolved during Network V5 preflight",
            }
        )
        if (
            not reaction.target_entity_id
            or reaction.chemical_validation_status not in {"valid", "derived_formula_valid"}
            or reaction.product_resolution_status
            not in {"unique_reported_entity", "mechanism_derived_entity"}
        ):
            continue
        if reaction.product_resolution_status == "mechanism_derived_entity" and reaction.target_entity_id not in known_derived_ids:
            known_derived_ids.add(reaction.target_entity_id)
            derived_compounds.append(
                {
                    "rule_id": stable_id("cmp", reaction.target_entity_id),
                    "entity_id": reaction.target_entity_id,
                    "entity_origin": "mechanism_derived",
                    "compound_name": reaction.target_entity_name,
                    "synonyms": "",
                    "compound_class": next(
                        (item.compound_class for item in templates if item.claim_id == reaction.template_claim_id),
                        "",
                    ),
                    "subclass": "",
                    "formula": reaction.product_formula,
                    "exact_mass": format_float(reaction.calculated_product_mass),
                    "ion_mode": "",
                    "adduct": "",
                    "reported_precursor_mz": "",
                    "reported_fragments": "",
                    "reported_neutral_losses": "",
                    "literature_status": "mechanism_derived",
                    "derivation_id": reaction.derivation_id,
                    "evidence_ids": join_unique(list(reaction.evidence_ids)),
                    "review_status": "candidate",
                    "review_note": "Product formula and mass derived from an evidence-backed reaction template.",
                }
            )
        source = entities.get(reaction.anchor_entity_id)
        if source is None:
            continue
        source_mass = source.exact_mass
        if source_mass is None and source.formula:
            try:
                source_mass = formula_exact_mass(source.formula)
            except ValueError:
                source_mass = None
        delta_mass = (
            reaction.calculated_product_mass - source_mass
            if reaction.calculated_product_mass is not None and source_mass is not None
            else None
        )
        product_payload = [
            {"entity_id": reaction.target_entity_id, "role": "product", "coefficient": 1}
        ]
        fragment_contract = compile_fragment_evidence_contract(
            reaction,
            fragments,
        )
        target_has_reported_fragments = bool(
            fragment_contract.reported_target_fragment_ids
        )
        raw_template = template_claim_rows.get(reaction.template_claim_id, {})
        raw_evidence_type = clean_text(raw_template.get("evidence_type")).lower()
        if raw_evidence_type == "schema_propagated":
            rule_evidence_type = "literature_inferred"
        elif raw_evidence_type in TRANSFORMATION_EVIDENCE_TYPES:
            rule_evidence_type = raw_evidence_type
        else:
            rule_evidence_type = "mechanism_derived"
        template_support_level = (
            clean_text(raw_template.get("template_support_level"))
            or "not_audited"
        )
        candidate_policy = (
            clean_text(raw_template.get("candidate_policy"))
            or "literature_supported"
        )
        propagation_policy = (
            clean_text(raw_template.get("propagation_policy"))
            or (
                "allow_after_identity_gate"
                if len(reaction.reactants) == 1
                else "no_propagation"
            )
        )
        derived_transformations.append(
            {
                "rule_id": stable_id("tr", reaction.derivation_id),
                "compound_class": next(
                    (item.compound_class for item in templates if item.claim_id == reaction.template_claim_id),
                    "",
                ),
                "subclass": "",
                "source_entity": source.entity_name,
                "target_entity": reaction.target_entity_name,
                "transformation_name": reaction.reaction_name,
                "evidence_type": rule_evidence_type,
                "template_support_level": template_support_level,
                "candidate_policy": candidate_policy,
                "propagation_policy": propagation_policy,
                "reaction_arity": len(reaction.reactants),
                "reactant_entities": reactant_payload,
                "product_entities": product_payload,
                "source_entity_id": reaction.anchor_entity_id,
                "target_entity_id": reaction.target_entity_id,
                "reactant_form_ids": [
                    item.form_id for item in reaction.reactants if item.form_id
                ],
                "product_form_ids": [],
                "fragment_evidence_contract": fragment_contract.canonical_json(),
                "anchor_reactant_index": reaction.anchor_reactant_index,
                "network_anchor_role": reaction.network_anchor_role,
                "resolved_network_anchor_count": 0,
                "anchor_resolution_stage": "network_preflight",
                "reaction_type": reaction.reaction_type,
                "reaction_operator": reaction.reaction_operator,
                "formula_equation": (
                    f"{'+'.join(item.formula or entities[item.entity_id].formula for item in reaction.reactants)}"
                    f" -> {reaction.product_formula}"
                ),
                "calculated_delta_mass": format_float(delta_mass),
                "chemical_validation_status": reaction.chemical_validation_status,
                "product_resolution_status": reaction.product_resolution_status,
                "derivation_id": reaction.derivation_id,
                "delta_mass": format_float(delta_mass),
                "direction": "source_to_target",
                "repeatable": "FALSE",
                "max_repeat": "1",
                "required_context": (
                    "literature_explanation_only; no propagation"
                    if candidate_policy == "literature_explanation_only"
                    else (
                        "real resolved network anchor; target-specific reported fragments"
                        + (
                            "; no propagation"
                            if propagation_policy == "no_propagation"
                            else ""
                        )
                        if target_has_reported_fragments
                        else "exploratory_only_missing_target_fragments"
                    )
                ),
                "evidence_ids": join_unique(list(reaction.evidence_ids)),
                "review_status": "candidate",
                "review_note": "Materialized from an evidence-backed variable-arity reaction template.",
            }
        )
    for audit in materialization.template_audits:
        audit_rows.append(
            {
                "template_claim_id": audit.template_claim_id,
                "derivation_id": "",
                "status": audit.status,
                "anchor_entity_id": "",
                "resolved_network_anchor_count": audit.resolved_network_anchor_count,
                "total_combination_count": audit.total_combination_count,
                "materialized_count": audit.materialized_count,
                "reactant_entities": [],
                "target_entity_id": "",
                "product_resolution_status": "",
                "product_formula": "",
                "calculated_product_mass": "",
                "chemical_validation_status": "",
                "propagation_eligible_at_compile_time": False,
                "detail": audit.detail,
            }
        )
    return (
        derived_compounds,
        derived_transformations,
        audit_rows,
        {
            "template_count": len(templates),
            "materialized_count": len(materialization.materialized_reactions),
            "emitted_rule_count": len(derived_transformations),
            "derived_component_count": (
                len(materialization.derived_entities)
                + len(materialization.derived_forms)
            ),
        },
        materialization.derived_forms,
    )


def component_name_for_claim(claim: dict[str, Any]) -> str:
    claim_type = clean_text(claim.get("claim_type"))
    if claim_type == "precursor":
        return first_nonempty(claim.get("precursor_name"), claim.get("compound_name"), claim.get("source_entity"))
    if claim_type == "structural_component":
        return first_nonempty(claim.get("component_name"), claim.get("compound_name"), claim.get("source_entity"))
    return first_nonempty(
        claim.get("component_name"),
        claim.get("precursor_name"),
        claim.get("compound_name"),
        claim.get("source_entity"),
    )


def _is_analytical_precursor_claim(claim: dict[str, Any]) -> bool:
    name = component_name_for_claim(claim)
    return bool(
        clean_text(claim.get("adduct"))
        or re.search(r"\[\s*M[^\]]*\][+-]?", name, re.I)
    )


def _is_component_bearing_reaction_role(role: str) -> bool:
    normalized = clean_text(role).lower()
    if normalized in {"", "substrate", "parent_compound", "product", "catalyst"}:
        return False
    return any(
        token in normalized
        for token in (
            "precursor",
            "donor",
            "component",
            "moiety",
            "residue",
            "glycos",
            "sugar",
            "acyl",
        )
    )


def _expand_component_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for claim in claims:
        claim_type = clean_text(claim.get("claim_type"))
        if claim_type == "entity_component_membership":
            component_name = clean_text(claim.get("component_name"))
            if not component_name:
                continue
            parent = first_nonempty(claim.get("entity_name"), claim.get("entity_id"))
            expanded.append(
                {
                    **claim,
                    "component_type": "entity_component_membership",
                    "entity_id": clean_text(claim.get("component_entity_id")),
                    "role": first_nonempty(
                        claim.get("membership_role"), claim.get("role"), "reported_component"
                    ),
                    "reaction_logic": first_nonempty(
                        claim.get("reaction_logic"),
                        f"{parent} contains or is formed with {component_name}" if parent else "",
                    ),
                }
            )
        elif claim_type == "reaction_template":
            ev_id = evidence_id(claim)
            for participant in claim.get("reactants") or []:
                if not isinstance(participant, dict):
                    continue
                role = clean_text(participant.get("role"))
                name = clean_text(participant.get("entity_name"))
                if not name or not ev_id or not _is_component_bearing_reaction_role(role):
                    continue
                expanded.append(
                    {
                        "claim_id": ev_id,
                        "claim_type": "reaction_component",
                        "component_type": "reaction_component",
                        "compound_class": clean_text(claim.get("compound_class")),
                        "component_name": name,
                        "entity_id": clean_text(participant.get("entity_id")),
                        "entity_class_id": clean_text(participant.get("entity_class_id")),
                        "entity_scope": clean_text(participant.get("scope")) or "specific_entity",
                        "formula": clean_text(participant.get("formula")),
                        "exact_mass": clean_text(participant.get("exact_mass")),
                        "role": role,
                        "reaction_logic": first_nonempty(
                            claim.get("reaction_name"), claim.get("evidence_summary")
                        ),
                        "evidence_summary": clean_text(claim.get("evidence_summary")),
                    }
                )
    return expanded


def compile_biosynthetic_component_rules(
    claims: list[dict[str, Any]],
    concept_context: dict[str, Any],
    enrichment_statistics: dict[str, int],
) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    concept_claims = concept_context.get("component_claims", [])
    direct_claims = [
        claim
        for claim in claims
        if clean_text(claim.get("claim_type"))
        not in {"entity_component_membership", "reaction_template"}
    ]
    all_claims = (
        direct_claims
        + _expand_component_claims(claims)
        + [claim for claim in concept_claims if isinstance(claim, dict)]
    )
    for claim in all_claims:
        claim_type = clean_text(claim.get("claim_type"))
        if claim_type not in {
            "biosynthetic_component",
            "precursor",
            "structural_component",
            "entity_component_membership",
            "reaction_component",
        }:
            continue
        if claim_type == "precursor" and _is_analytical_precursor_claim(claim):
            continue
        component_name = component_name_for_claim(claim)
        ev_id = first_nonempty(
            claim.get("claim_id"), claim.get("evidence_ids"), claim.get("chunk_id")
        )
        if not component_name or not ev_id:
            continue
        if claim in concept_claims:
            enrichment_statistics["biosynthetic_component_rules_from_concepts"] += 1
        compound_class = clean_text(claim.get("compound_class"))
        component_type = clean_text(claim.get("component_type")) or claim_type
        key = (compound_class.lower(), component_type, component_name.lower())
        row = {
            "rule_id": stable_id("bio", compound_class, component_type, component_name),
            "entity_id": clean_text(claim.get("entity_id"))
            or stable_id("entity", compound_class, component_name),
            "form_id": clean_text(claim.get("form_id")),
            "entity_class_id": clean_text(claim.get("entity_class_id")),
            "entity_scope": clean_text(claim.get("entity_scope")) or "specific_entity",
            "compound_class": compound_class,
            "component_type": component_type,
            "component_name": component_name,
            "formula": clean_text(claim.get("formula")),
            "exact_mass": format_float(claim.get("exact_mass")),
            "role": clean_text(claim.get("role")) or component_type,
            "reaction_logic": clean_text(claim.get("reaction_logic")) or clean_text(claim.get("evidence_summary")),
            "delta_mass_to_product": format_float(claim.get("delta_mass_to_product")),
            "decarboxy_delta_mass": format_float(claim.get("decarboxy_delta_mass")),
            "evidence_ids": ev_id,
            "review_status": "candidate",
            "review_note": clean_text(claim.get("evidence_summary")),
        }
        add_merged(merged, key, row)
    return list(merged.values())


def count_removed_missing_evidence(claims: list[dict[str, Any]]) -> int:
    return sum(1 for claim in claims if not evidence_id(claim))


def compile_rules(
    evidence_claims_jsonl: Path | str,
    output_dir: Path | str,
    report_path: Path | str,
    concepts_json: Path | str | None = None,
    min_reported_fragment_compound_count: int = 20,
    min_reported_fragment_support_fraction: float = 0.20,
    max_class_consensus_fragments: int = 20,
    enable_literature_inference: bool = False,
    augmented_claims_jsonl: Path | str | None = None,
    inference_report_path: Path | str | None = None,
    inference_min_confidence: float = 0.65,
    allow_table_delta_only_inference: bool = False,
    enable_diagnostic_evidence_mining: bool = False,
    corpus_jsonl: Path | str = "rag/corpus/chunks.jsonl",
    diagnostic_evidence_claims_jsonl: Path | str = "rag/evidence_claims/diagnostic_evidence_claims.jsonl",
    diagnostic_evidence_report_path: Path | str = "rag/reports/diagnostic_evidence_mining_report.json",
    registry_dir: Path | str | None = None,
) -> dict[str, Any]:
    if not registry_dir:
        raise RuleCompilationError(
            "registry_dir is required; the refactored compiler emits schema 4.0 only"
        )
    claims_path = resolve(evidence_claims_jsonl)
    rules_dir = resolve(output_dir)
    report = resolve(report_path)
    inference_report: dict[str, Any] | None = None
    if enable_literature_inference:
        augmented_path = resolve(augmented_claims_jsonl) if augmented_claims_jsonl else (
            PROJECT_ROOT / "rag" / "evidence_claims" / "evidence_claims_augmented.jsonl"
        )
        inference_report_file = resolve(inference_report_path) if inference_report_path else (
            PROJECT_ROOT / "rag" / "reports" / "literature_supported_inference_report.json"
        )
        inference_report = run_literature_supported_inference(
            evidence_claims_jsonl=claims_path,
            concepts_json=concepts_json,
            out_jsonl=augmented_path,
            report_path=inference_report_file,
            min_confidence=inference_min_confidence,
            allow_table_delta_only=allow_table_delta_only_inference,
        )
        claims_path = augmented_path
    claims = load_claims(claims_path)
    concepts_path = resolve(concepts_json) if concepts_json else None
    concept_context = load_concepts(concepts_path)
    diagnostic_evidence_report: dict[str, Any] | None = None
    if enable_diagnostic_evidence_mining:
        compound_class_hint = first_nonempty(
            *(claim.get("compound_class") for claim in claims),
            concept_context.get("compound_class"),
        )
        diagnostic_evidence_report = run_diagnostic_evidence_mining(
            corpus_jsonl=corpus_jsonl,
            out_jsonl=diagnostic_evidence_claims_jsonl,
            report_path=diagnostic_evidence_report_path,
            compound_class=compound_class_hint,
        )
        diagnostic_claims = load_claims(resolve(diagnostic_evidence_claims_jsonl))
        claims = claims + diagnostic_claims
    enrichment_statistics = {
        "subclass_values_added_from_concepts": 0,
        "neutral_loss_interpretations_added": 0,
        "transformation_names_added": 0,
        "biosynthetic_component_rules_from_concepts": 0,
    }
    warnings = collect_formula_warnings(claims)

    compound_rules = compile_compound_rules(claims, concept_context, enrichment_statistics)
    diagnostic_rules, removed_fragments = compile_diagnostic_fragment_rules(
        claims,
        concept_context,
        enrichment_statistics,
        min_reported_fragment_compound_count=min_reported_fragment_compound_count,
        min_reported_fragment_support_fraction=(
            min_reported_fragment_support_fraction
        ),
        max_class_consensus_fragments=max_class_consensus_fragments,
    )
    neutral_loss_rules, removed_losses = compile_neutral_loss_rules(claims, concept_context, enrichment_statistics)
    transformation_rules = compile_transformation_rules(claims, warnings, concept_context, enrichment_statistics)
    biosynthetic_rules = compile_biosynthetic_component_rules(claims, concept_context, enrichment_statistics)
    source_registry_dir = resolve(registry_dir)
    evidence_inventory_path = source_registry_dir / "evidence_inventory.jsonl"
    fragment_evidence_path = source_registry_dir / "fragment_evidence.jsonl"
    if not evidence_inventory_path.exists() or not fragment_evidence_path.exists():
        raise RuleCompilationError(
            "schema 4.0 requires evidence_inventory.jsonl and "
            "fragment_evidence.jsonl in registry_dir"
        )
    registry = load_entity_registry(source_registry_dir)
    fragment_rows, fragment_reference_reconciliation = (
        reconcile_fragment_entity_references(
            load_fragment_evidence(fragment_evidence_path),
            claims,
            registry,
        )
    )
    fragment_rows = derive_fragment_specificity(fragment_rows)
    (
        derived_compounds,
        derived_transformations,
        derivation_audit,
        materialization_summary,
        derived_reactant_forms,
    ) = compile_reaction_templates(
        claims,
        compound_rules,
        warnings,
        registry=registry,
        fragments=fragment_rows,
    )
    compound_rules.extend(derived_compounds)
    transformation_rules.extend(derived_transformations)
    augmented_entities = dict(registry.entities)
    augmented_forms = dict(registry.forms)
    for form in derived_reactant_forms:
        augmented_forms[form.form_id] = form
    for row in derived_compounds:
        entity_id = clean_text(row.get("entity_id"))
        formula = clean_text(row.get("formula"))
        if not entity_id or not formula or entity_id in augmented_entities:
            continue
        evidence_ids = tuple(split_values(row.get("evidence_ids")))
        entity = RegistryEntityRecord(
            entity_id=entity_id,
            canonical_name=clean_text(row.get("compound_name")),
            reported_names=(clean_text(row.get("compound_name")),),
            entity_kind="molecule",
            compound_class=clean_text(row.get("compound_class")),
            formula=formula,
            exact_mass=safe_float(row.get("exact_mass")),
            ion_modes=tuple(split_values(row.get("ion_mode"))),
            evidence_ids=evidence_ids,
        )
        augmented_entities[entity_id] = entity
        form_id = stable_registry_id(
            "form",
            {
                "entity_id": entity_id,
                "form_type": "neutral_molecule",
                "formula": formula,
            },
        )
        augmented_forms[form_id] = EntityForm(
            form_id=form_id,
            entity_id=entity_id,
            form_type="neutral_molecule",
            formula=formula,
            exact_mass=entity.exact_mass,
            formula_origin="reaction_operator_derived",
            reaction_operator="",
            evidence_ids=evidence_ids,
        )
    for row in biosynthetic_rules:
        entity_id = clean_text(row.get("entity_id"))
        component_name = clean_text(row.get("component_name"))
        if not entity_id or not component_name or entity_id in augmented_entities:
            continue
        augmented_entities[entity_id] = RegistryEntityRecord(
            entity_id=entity_id,
            canonical_name=component_name,
            reported_names=(component_name,),
            entity_kind="moiety",
            compound_class=clean_text(row.get("compound_class")),
            formula=clean_text(row.get("formula")),
            exact_mass=safe_float(row.get("exact_mass")),
            ion_modes=(),
            evidence_ids=tuple(split_values(row.get("evidence_ids"))),
        )
    name_groups: dict[str, set[str]] = {}
    for entity in augmented_entities.values():
        for name in entity.reported_names:
            name_groups.setdefault(normalize_registry_name(name), set()).add(
                entity.entity_id
            )
    registry = EntityRegistry(
        entities=augmented_entities,
        forms=augmented_forms,
        classes=registry.classes,
        memberships=registry.memberships,
        name_index={
            key: tuple(sorted(values)) for key, values in name_groups.items()
        },
        audits=registry.audits,
    )
    resolved_compound_rules: list[dict[str, Any]] = []
    for row in compound_rules:
        entity_id = resolve_registry_entity_id(
            registry,
            row.get("compound_name"),
            preferred_id=row.get("entity_id"),
            formula=row.get("formula"),
            exact_mass=row.get("exact_mass"),
        )
        if not entity_id:
            warnings.append(
                f"{row.get('rule_id')}: compound entity unresolved or "
                "chemically ambiguous in Entity Registry"
            )
            continue
        entity = registry.entities[entity_id]
        row["entity_id"] = entity.entity_id
        row["formula"] = entity.formula or row.get("formula", "")
        row["exact_mass"] = (
            entity.exact_mass
            if entity.exact_mass is not None
            else row.get("exact_mass", "")
        )
        row["target_origin"] = (
            "mechanism_derived"
            if clean_text(row.get("entity_origin")).startswith("mechanism")
            else "reported"
        )
        resolved_compound_rules.append(row)
    compound_rules = resolved_compound_rules
    existing_entity_ids = {
        clean_text(row.get("entity_id")) for row in compound_rules
    }
    for entity in registry.entities.values():
        if (
            entity.entity_kind != "molecule"
            or entity.entity_id in existing_entity_ids
            or not entity.formula
            or entity.exact_mass is None
        ):
            continue
        compound_rules.append(
            {
                "rule_id": stable_id("compound", entity.entity_id),
                "entity_id": entity.entity_id,
                "entity_origin": "reported",
                "target_origin": "reported",
                "compound_name": entity.canonical_name,
                "synonyms": join_unique(list(entity.reported_names)),
                "compound_class": entity.compound_class,
                "formula": entity.formula,
                "exact_mass": format_float(entity.exact_mass),
                "ion_mode": join_unique(list(entity.ion_modes)),
                "evidence_ids": join_unique(list(entity.evidence_ids)),
                "review_status": "candidate",
            }
        )
    for row in transformation_rules:
        source_id = clean_text(row.get("source_entity_id"))
        target_id = clean_text(row.get("target_entity_id"))
        if source_id not in registry.entities:
            source_id = resolve_registry_entity_id(
                registry,
                row.get("source_entity"),
            )
        if target_id not in registry.entities:
            target_id = resolve_registry_entity_id(
                registry,
                row.get("target_entity"),
            )
        row["source_entity_id"] = source_id
        row["target_entity_id"] = target_id
        row.setdefault("reactant_form_ids", "")
        if not row.get("product_form_ids") and target_id:
            row["product_form_ids"] = [
                form.form_id
                for form in registry.forms.values()
                if form.entity_id == target_id and form.form_type == "neutral_molecule"
            ]
        row.setdefault("fragment_evidence_contract", "{}")
        materialize_direct_unary_transformation(row, registry)
    for row in biosynthetic_rules:
        entity_id = resolve_registry_entity_id(
            registry,
            row.get("component_name"),
            preferred_id=row.get("entity_id"),
            formula=row.get("formula"),
            exact_mass=row.get("exact_mass"),
        )
        if not entity_id:
            continue
        row["entity_id"] = entity_id
        form_ids = sorted(
            form.form_id
            for form in registry.forms.values()
            if form.entity_id == entity_id
        )
        row["form_id"] = form_ids[0] if len(form_ids) == 1 else ""
        class_ids = sorted(
            item.entity_class_id
            for item in registry.memberships
            if item.entity_id == entity_id
        )
        row["entity_class_id"] = class_ids[0] if len(class_ids) == 1 else ""

    tables = {
        RULE_TABLES["compound"]: RuleTable(tuple(COMPOUND_COLUMNS), tuple(compound_rules)),
        RULE_TABLES["transformation"]: RuleTable(tuple(TRANSFORMATION_COLUMNS), tuple(transformation_rules)),
        RULE_TABLES["diagnostic_fragment"]: RuleTable(tuple(DIAGNOSTIC_FRAGMENT_COLUMNS), tuple(diagnostic_rules)),
        RULE_TABLES["neutral_loss"]: RuleTable(tuple(NEUTRAL_LOSS_COLUMNS), tuple(neutral_loss_rules)),
        RULE_TABLES["biosynthetic_component"]: RuleTable(
            tuple(BIOSYNTHETIC_COMPONENT_COLUMNS), tuple(biosynthetic_rules)
        ),
    }
    write_entity_registry(registry, rules_dir)
    write_fragment_evidence_registry(
        fragment_rows,
        rules_dir / "fragment_evidence.jsonl",
    )
    registry_artifacts = {
        "entity_registry.jsonl": rules_dir / "entity_registry.jsonl",
        "entity_forms.jsonl": rules_dir / "entity_forms.jsonl",
        "entity_classes.jsonl": rules_dir / "entity_classes.jsonl",
        "entity_class_memberships.jsonl": (
            rules_dir / "entity_class_memberships.jsonl"
        ),
        "evidence_inventory.jsonl": evidence_inventory_path,
        "fragment_evidence.jsonl": rules_dir / "fragment_evidence.jsonl",
    }
    manifest = write_rule_bundle(
        rules_dir, tables, registry_artifacts=registry_artifacts
    )
    write_csv(
        rules_dir / "reaction_derivation_audit.csv",
        REACTION_DERIVATION_AUDIT_COLUMNS,
        derivation_audit,
    )

    compiler_gap_records = audit_compiler_gaps(
        claims=claims,
        derivation_rows=derivation_audit,
        registry_audits=[asdict(row) for row in registry.audits],
        validation_warnings=warnings,
        rule_rows_by_table={
            table_name: list(table.rows) for table_name, table in tables.items()
        },
        evidence_inventory_rows=load_claims(evidence_inventory_path),
    )
    compiler_gap_summary = write_compiler_gap_audit(
        compiler_gap_records, report.parent
    )

    class_candidates = [claim.get("compound_class") for claim in claims] + [concept_context.get("compound_class")]
    summary = {
        "compound_class": first_nonempty(*class_candidates),
        "input_claim_count": len(claims),
        "input_concept_count": len(concept_context.get("concepts", [])),
        "output_rule_counts": {
            "compound": len(compound_rules),
            "transformation": len(transformation_rules),
            "diagnostic_fragment": len(diagnostic_rules),
            "neutral_loss": len(neutral_loss_rules),
            "biosynthetic_component": len(biosynthetic_rules),
        },
        "enrichment_statistics": enrichment_statistics,
        "removed_claim_count": count_removed_missing_evidence(claims) + removed_fragments + removed_losses,
        "validation_warnings": warnings,
        "reaction_template_materialization": materialization_summary,
        "fragment_reference_reconciliation": fragment_reference_reconciliation,
        "compiler_gap_audit": compiler_gap_summary,
        "rules_manifest": {
            "schema_version": manifest["schema_version"],
            "path": str(rules_dir / "rules_manifest.json"),
        },
    }
    if inference_report is not None:
        summary["literature_supported_inference"] = inference_report
    if diagnostic_evidence_report is not None:
        summary["diagnostic_evidence_mining"] = diagnostic_evidence_report
    write_json(report, summary)
    return summary


def default_report_path(output_dir: Path) -> Path:
    reports_dir = PROJECT_ROOT / "rag" / "reports"
    if output_dir == PROJECT_ROOT / "rag" / "rules_candidate":
        return reports_dir / "rule_enrichment_report.json"
    return output_dir.parent / "rule_enrichment_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile Agent 2 evidence claims into generic five-table RAG rules.")
    parser.add_argument("--evidence_claims_jsonl", default="rag/evidence_claims/evidence_claims.jsonl")
    parser.add_argument("--concepts_json", default="", help="Optional discovered concepts JSON for future enrichment.")
    parser.add_argument("--output_dir", default="rag/rules_candidate")
    parser.add_argument(
        "--registry_dir",
        default="",
        help="Entity Registry directory. When supplied, emits schema 4.0.",
    )
    parser.add_argument("--report_path", default="", help="Default: rag/reports/rule_enrichment_report.json")
    parser.add_argument(
        "--min_reported_fragment_compound_count",
        type=int,
        default=20,
        help=(
            "Minimum distinct compounds supporting a reported fragment before "
            "it can enter the bounded class-consensus diagnostic set."
        ),
    )
    parser.add_argument(
        "--min_reported_fragment_support_fraction",
        type=float,
        default=0.20,
        help=(
            "Minimum fraction of fragment-bearing compounds in the same class "
            "and reported ion mode that must support a consensus fragment."
        ),
    )
    parser.add_argument(
        "--max_class_consensus_fragments",
        type=int,
        default=20,
        help="Maximum class-consensus diagnostic fragments per class and ion mode.",
    )
    parser.add_argument(
        "--enable_literature_inference",
        action="store_true",
        help="Infer review-only transformation claims from literature-reported compounds before compiling rules.",
    )
    parser.add_argument(
        "--augmented_claims_jsonl",
        default="rag/evidence_claims/evidence_claims_augmented.jsonl",
        help="Output JSONL for original claims plus inferred transformation claims.",
    )
    parser.add_argument(
        "--inference_report_path",
        default="rag/reports/literature_supported_inference_report.json",
        help="Report path for literature-supported transformation inference.",
    )
    parser.add_argument("--inference_min_confidence", type=float, default=0.65)
    parser.add_argument(
        "--allow_table_delta_only_inference",
        action="store_true",
        help=(
            "Allow inferred transformations supported only by same-table/source co-reporting plus a universal mass "
            "delta. Default keeps only name-stem, modifier-pattern, or subclass-supported inferred transformations."
        ),
    )
    parser.add_argument(
        "--enable_diagnostic_evidence_mining",
        action="store_true",
        help=(
            "Mine diagnostic_fragment claims from local corpus chunks using explicit diagnostic/fingerprint/"
            "structural-mapping evidence types. Does not use reported-fragment frequency."
        ),
    )
    parser.add_argument("--corpus_jsonl", default="rag/corpus/chunks.jsonl")
    parser.add_argument(
        "--diagnostic_evidence_claims_jsonl",
        default="rag/evidence_claims/diagnostic_evidence_claims.jsonl",
    )
    parser.add_argument(
        "--diagnostic_evidence_report_path",
        default="rag/reports/diagnostic_evidence_mining_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.min_reported_fragment_support_fraction <= 1:
        print(
            "ERROR: --min_reported_fragment_support_fraction must be in (0, 1].",
            file=sys.stderr,
        )
        return 1
    if args.max_class_consensus_fragments <= 0:
        print(
            "ERROR: --max_class_consensus_fragments must be positive.",
            file=sys.stderr,
        )
        return 1
    output_dir = resolve(args.output_dir)
    report_path = resolve(args.report_path) if args.report_path else default_report_path(output_dir)
    try:
        summary = compile_rules(
            evidence_claims_jsonl=args.evidence_claims_jsonl,
            output_dir=output_dir,
            report_path=report_path,
            concepts_json=args.concepts_json or None,
            min_reported_fragment_compound_count=max(1, args.min_reported_fragment_compound_count),
            min_reported_fragment_support_fraction=(
                args.min_reported_fragment_support_fraction
            ),
            max_class_consensus_fragments=args.max_class_consensus_fragments,
            enable_literature_inference=args.enable_literature_inference,
            augmented_claims_jsonl=args.augmented_claims_jsonl,
            inference_report_path=args.inference_report_path,
            inference_min_confidence=args.inference_min_confidence,
            allow_table_delta_only_inference=args.allow_table_delta_only_inference,
            enable_diagnostic_evidence_mining=args.enable_diagnostic_evidence_mining,
            corpus_jsonl=args.corpus_jsonl,
            diagnostic_evidence_claims_jsonl=args.diagnostic_evidence_claims_jsonl,
            diagnostic_evidence_report_path=args.diagnostic_evidence_report_path,
            registry_dir=args.registry_dir or None,
        )
    except (RuleCompilationError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
