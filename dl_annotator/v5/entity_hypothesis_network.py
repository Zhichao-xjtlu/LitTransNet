"""Registry-first, metabolite-class-agnostic evidence graph network."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rag.core.entity_registry import (
    EntityRegistry,
    EntityResolution,
    load_entity_registry,
    normalize_registry_name,
)

from .rule_loader import validate_rules_manifest


SUPPORTED_ADDUCTS: Mapping[str, tuple[str, float, int]] = {
    "[M+H]+": ("positive", 1.007276466621, 1),
    "[M+Na]+": ("positive", 22.989218, 1),
    "[M+NH4]+": ("positive", 18.033823, 1),
    "[M-H]-": ("negative", -1.007276466621, 1),
    "[M+HCOO]-": ("negative", 44.998201, 1),
}


def split_values(value: object) -> tuple[str, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return tuple(str(item).strip() for item in parsed if str(item).strip())
    return tuple(
        item.strip()
        for item in re.split(r"[;|]", text)
        if item.strip()
    )


def safe_float(value: object) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


@dataclass(frozen=True)
class NetworkEntity:
    entity_id: str
    canonical_name: str
    compound_class: str
    entity_kind: str
    formula: str
    exact_mass: float | None
    ion_modes: tuple[str, ...]
    adducts: tuple[str, ...]
    target_origin: str
    reported_fragment_ids: tuple[str, ...]
    reported_fragment_mz: tuple[float, ...]
    network_target_eligible: bool
    network_anchor_eligible: bool


@dataclass(frozen=True)
class NetworkTransformation:
    rule_id: str
    source_entity_id: str
    target_entity_id: str
    evidence_type: str
    reaction_type: str
    reaction_operator: str
    reactant_entity_ids: tuple[str, ...]
    product_entity_ids: tuple[str, ...]
    reactant_form_ids: tuple[str, ...]
    product_form_ids: tuple[str, ...]
    anchor_reactant_index: int
    network_anchor_role: str
    chemical_validation_status: str
    product_resolution_status: str
    target_origin: str
    delta_mass: float | None
    ion_modes: tuple[str, ...]
    fragment_evidence_contract: Mapping[str, tuple[str, ...]]
    evidence_ids: tuple[str, ...]
    relation_evidence_status: str = ""
    claim_source: str = ""
    template_support_level: str = ""
    candidate_policy: str = "literature_supported"
    propagation_policy: str = "allow_after_identity_gate"


@dataclass(frozen=True)
class NetworkFragmentEvidence:
    fragment_id: str
    entity_id: str
    entity_class_id: str
    fragment_mz: float
    ion_mode: str
    adduct: str
    evidence_role: str
    specificity_scope: str
    source_structure: str
    discriminative_status: str
    competitor_entity_count: int


@dataclass(frozen=True)
class SpectrumNode:
    node_id: str
    precursor_mz: float
    mz_array: np.ndarray
    intensity_array: np.ndarray
    ion_mode: str
    rt_min: float | None = None
    raw_indices: tuple[int, ...] = ()
    known_match: bool = False
    library_name: str = ""
    seed_resolution_status: str = ""
    seed_resolution_entity_ids: tuple[str, ...] = ()
    feature_id: str = ""
    source_spectrum_id: str = ""
    library_candidate_names: tuple[str, ...] = ()
    library_candidate_scores: tuple[float, ...] = ()


@dataclass(frozen=True)
class SeedAssignment:
    node_id: str
    entity_id: str
    library_name: str


@dataclass(frozen=True)
class NetworkKnowledge:
    registry: EntityRegistry
    compounds: Mapping[str, NetworkEntity]
    transformations: tuple[NetworkTransformation, ...]
    outgoing_rules: Mapping[str, tuple[NetworkTransformation, ...]]
    diagnostic_rows: tuple[Mapping[str, str], ...]
    neutral_loss_rows: tuple[Mapping[str, str], ...]
    fragment_mz_by_id: Mapping[str, float]
    fragment_evidence_by_id: Mapping[str, NetworkFragmentEvidence]


@dataclass(frozen=True)
class NetworkConfig:
    ppm_tolerance: float = 10.0
    absolute_tolerance_da: float = 0.01
    fragment_tolerance_da: float = 0.10
    neutral_loss_tolerance_da: float = 0.15
    max_depth: int = 2
    fragment_gate_mode: str = "reaction_evidence"
    min_class_diagnostic_matches: int = 2
    min_target_fragment_matches: int = 1
    min_target_product_ions: int = 2
    discriminative_statuses: frozenset[str] = frozenset(
        {"unique", "low_sharing"}
    )
    max_signature_competitor_fraction: float = 0.1
    allow_generic_adducts: bool = True
    allowed_target_fragment_ids: frozenset[str] | None = None
    fragment_mz_shift_da: float = 0.0
    shuffle_fragment_entity_labels: bool = False
    shuffle_reaction_roles: bool = False

    def __post_init__(self) -> None:
        if self.fragment_gate_mode not in {
            "evidence_bundle",
            "reaction_evidence",
            "strict",
            "high_recall",
            "mass_only",
            "required_empty",
            "theoretical_catalog_only",
        }:
            raise ValueError(
                "fragment_gate_mode must be evidence_bundle, reaction_evidence, strict, "
                "high_recall, mass_only, required_empty, or "
                "theoretical_catalog_only"
            )
        if self.min_class_diagnostic_matches not in {2, 3}:
            raise ValueError(
                "min_class_diagnostic_matches must be 2 or 3"
            )
        if self.min_target_fragment_matches <= 0:
            raise ValueError("min_target_fragment_matches must be positive")
        if self.min_target_product_ions <= 0:
            raise ValueError("min_target_product_ions must be positive")
        if not 0 < self.max_signature_competitor_fraction <= 1:
            raise ValueError(
                "max_signature_competitor_fraction must be in (0, 1]"
            )


@dataclass(frozen=True)
class NetworkRunResult:
    nodes: tuple[Mapping[str, Any], ...]
    hypotheses: tuple[Mapping[str, Any], ...]
    edges: tuple[Mapping[str, Any], ...]


MANDATORY_GATES = (
    "source_propagation",
    "ion_mode",
    "chemical_validation",
    "target_mass_available",
    "precursor_ppm",
    "precursor_da",
    "fragment_identity",
    "isomer_uniqueness",
)


@dataclass(frozen=True)
class MassGateAudit:
    ppm_pass: bool
    da_pass: bool
    error_ppm: float
    error_da: float
    matched_adduct: str

    @property
    def passed(self) -> bool:
        return self.ppm_pass and self.da_pass


@dataclass(frozen=True)
class FragmentMatch:
    evidence_id: str
    evidence_scope: str
    role: str
    evidence_role: str = "unassigned_peak"
    ion_mode: str = "not_reported"
    discriminative_status: str = "not_evaluated"
    competitor_entity_count: int = 0


@dataclass(frozen=True)
class FragmentGateAudit:
    passed: bool
    target_fragment_ids: tuple[str, ...]
    component_fragment_ids: tuple[str, ...]
    explicit_target_diagnostic_ids: tuple[str, ...] = ()
    target_product_ion_ids: tuple[str, ...] = ()
    class_diagnostic_ids: tuple[str, ...] = ()
    theoretical_catalog_ids: tuple[str, ...] = ()
    discriminative_fragment_ids: tuple[str, ...] = ()
    fragment_gate_level: str = "failed"
    signature_competitor_entity_count: int = 0
    signature_entity_universe_count: int = 0


@dataclass(frozen=True)
class GateAudit:
    gates: Mapping[str, bool]
    matched_target_fragment_ids: tuple[str, ...]
    matched_component_fragment_ids: tuple[str, ...]
    matched_neutral_loss_ids: tuple[str, ...]
    failure_reasons: tuple[str, ...]
    mass_error_ppm: float = float("inf")
    mass_error_da: float = float("inf")
    matched_adduct: str = ""
    matched_explicit_diagnostic_ids: tuple[str, ...] = ()
    matched_target_product_ion_ids: tuple[str, ...] = ()
    matched_class_diagnostic_ids: tuple[str, ...] = ()
    matched_theoretical_catalog_ids: tuple[str, ...] = ()
    discriminative_fragment_ids: tuple[str, ...] = ()
    fragment_gate_level: str = "failed"
    fragment_competitor_entity_count: int = 0
    signature_competitor_entity_count: int = 0
    signature_entity_universe_count: int = 0

    @property
    def passed(self) -> bool:
        return all(self.gates.get(name, False) for name in MANDATORY_GATES)


@dataclass(frozen=True)
class Hypothesis:
    source_node_id: str
    target_node_id: str
    source_entity_id: str
    target_entity_id: str
    rule_id: str
    evidence_type: str
    target_origin: str
    depth: int
    source_propagation: bool
    audit: GateAudit
    template_support_level: str = ""
    candidate_policy: str = "literature_supported"
    propagation_policy: str = "allow_after_identity_gate"


@dataclass(frozen=True)
class NodeDecision:
    node_id: str
    target_entity_id: str
    source_entity_id: str
    rule_id: str
    evidence_type: str
    target_origin: str
    annotation_status: str
    propagation_eligible: bool
    ambiguity_reason: str = ""
    fragment_gate_level: str = ""
    target_entity_group_id: str = ""
    ambiguous_target_entity_ids: tuple[str, ...] = ()
    ambiguous_rule_ids: tuple[str, ...] = ()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _json_contract(value: object) -> Mapping[str, tuple[str, ...]]:
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        parsed = {}
    if not isinstance(parsed, Mapping):
        return {}
    return {
        str(key): split_values(item)
        if not isinstance(item, (list, tuple))
        else tuple(str(value) for value in item)
        for key, item in parsed.items()
    }


def _entity_ids(value: object) -> tuple[str, ...]:
    text = str(value or "").strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return tuple(
                str(item.get("entity_id", "")).strip()
                for item in parsed
                if isinstance(item, Mapping)
                and str(item.get("entity_id", "")).strip()
            )
    return split_values(value)


def load_network_knowledge(rules_dir: Path) -> NetworkKnowledge:
    root = Path(rules_dir)
    validate_rules_manifest(root)
    registry = load_entity_registry(root)
    compound_rows = _read_csv(root / "compound_rules.csv")
    diagnostic_rows = tuple(_read_csv(root / "diagnostic_fragment_rules.csv"))
    neutral_loss_rows = tuple(_read_csv(root / "neutral_loss_rules.csv"))
    fragment_evidence_by_id: dict[str, NetworkFragmentEvidence] = {}
    for row in _read_jsonl(root / "fragment_evidence.jsonl"):
        fragment_id = str(row.get("fragment_id", "")).strip()
        fragment_mz = safe_float(row.get("fragment_mz"))
        if not fragment_id or fragment_mz is None:
            continue
        fragment_evidence_by_id[fragment_id] = NetworkFragmentEvidence(
            fragment_id=fragment_id,
            entity_id=str(row.get("entity_id", "")).strip(),
            entity_class_id=str(row.get("entity_class_id", "")).strip(),
            fragment_mz=fragment_mz,
            ion_mode=str(row.get("ion_mode", "")).strip() or "not_reported",
            adduct=str(row.get("adduct", "")).strip(),
            evidence_role=str(row.get("evidence_role", "")).strip(),
            specificity_scope=str(
                row.get("specificity_scope") or row.get("evidence_scope") or ""
            ).strip(),
            source_structure=str(row.get("source_structure", "")).strip(),
            discriminative_status=str(
                row.get("discriminative_status", "")
            ).strip()
            or "not_evaluated",
            competitor_entity_count=int(
                safe_float(row.get("competitor_entity_count")) or 0
            ),
        )
    fragment_mz_by_id: dict[str, float] = {
        fragment_id: row.fragment_mz
        for fragment_id, row in fragment_evidence_by_id.items()
    }
    compounds: dict[str, NetworkEntity] = {}
    for row in compound_rows:
        entity_id = row.get("entity_id", "").strip()
        entity = registry.entities.get(entity_id)
        if entity is None or entity.entity_kind != "molecule":
            continue
        fragment_values = tuple(
            value
            for item in split_values(row.get("reported_fragments"))
            if (value := safe_float(item)) is not None
        )
        fragment_ids = tuple(
            f"reported:{entity_id}:{index}:{value:.6f}"
            for index, value in enumerate(fragment_values)
        )
        fragment_mz_by_id.update(zip(fragment_ids, fragment_values))
        adducts = tuple(
            item for item in split_values(row.get("adduct")) if item in SUPPORTED_ADDUCTS
        )
        compounds[entity_id] = NetworkEntity(
            entity_id=entity_id,
            canonical_name=entity.canonical_name,
            compound_class=entity.compound_class,
            entity_kind=entity.entity_kind,
            formula=entity.formula,
            exact_mass=entity.exact_mass,
            ion_modes=tuple(
                mode for mode in entity.ion_modes if mode in {"positive", "negative"}
            ),
            adducts=adducts,
            target_origin=row.get("target_origin", "").strip() or "reported",
            reported_fragment_ids=fragment_ids,
            reported_fragment_mz=fragment_values,
            network_target_eligible=bool(entity.formula and entity.exact_mass is not None),
            network_anchor_eligible=True,
        )
    transformations: list[NetworkTransformation] = []
    for row in _read_csv(root / "transformation_rules.csv"):
        source_id = row.get("source_entity_id", "").strip()
        target_id = row.get("target_entity_id", "").strip()
        reactants = _entity_ids(row.get("reactant_entities"))
        products = _entity_ids(row.get("product_entities"))
        try:
            anchor_index = int(row.get("anchor_reactant_index", ""))
        except ValueError:
            continue
        if source_id not in compounds or target_id not in compounds:
            continue
        transformation = NetworkTransformation(
            rule_id=row.get("rule_id", "").strip(),
            source_entity_id=source_id,
            target_entity_id=target_id,
            evidence_type=row.get("evidence_type", "").strip(),
            reaction_type=row.get("reaction_type", "").strip(),
            reaction_operator=row.get("reaction_operator", "").strip(),
            reactant_entity_ids=reactants,
            product_entity_ids=products,
            reactant_form_ids=split_values(row.get("reactant_form_ids")),
            product_form_ids=split_values(row.get("product_form_ids")),
            anchor_reactant_index=anchor_index,
            network_anchor_role=row.get("network_anchor_role", "").strip(),
            chemical_validation_status=row.get("chemical_validation_status", "").strip(),
            product_resolution_status=row.get("product_resolution_status", "").strip(),
            target_origin=compounds[target_id].target_origin,
            delta_mass=safe_float(row.get("delta_mass")),
            ion_modes=tuple(
                mode
                for mode in split_values(row.get("ion_mode"))
                if mode in {"positive", "negative"}
            ),
            fragment_evidence_contract=_json_contract(
                row.get("fragment_evidence_contract")
            ),
            evidence_ids=split_values(row.get("evidence_ids")),
            relation_evidence_status=row.get(
                "relation_evidence_status", ""
            ).strip(),
            claim_source=row.get("claim_source", "").strip(),
            template_support_level=row.get(
                "template_support_level", ""
            ).strip(),
            candidate_policy=(
                row.get("candidate_policy", "").strip()
                or "literature_supported"
            ),
            propagation_policy=(
                row.get("propagation_policy", "").strip()
                or "allow_after_identity_gate"
            ),
        )
        transformations.append(transformation)
    outgoing: dict[str, list[NetworkTransformation]] = {}
    for row in transformations:
        outgoing.setdefault(row.source_entity_id, []).append(row)
    return NetworkKnowledge(
        registry=registry,
        compounds=compounds,
        transformations=tuple(transformations),
        outgoing_rules={
            key: tuple(sorted(values, key=lambda item: item.rule_id))
            for key, values in sorted(outgoing.items())
        },
        diagnostic_rows=diagnostic_rows,
        neutral_loss_rows=neutral_loss_rows,
        fragment_mz_by_id=fragment_mz_by_id,
        fragment_evidence_by_id=fragment_evidence_by_id,
    )


def resolve_seed_entity(
    knowledge: NetworkKnowledge, library_name: str
) -> EntityResolution:
    resolution = knowledge.registry.resolve_name(library_name)
    if resolution.status == "unresolved":
        normalized = normalize_registry_name(library_name)
        resolution = knowledge.registry.resolve_name(normalized)
    eligible = tuple(
        entity_id
        for entity_id in resolution.entity_ids
        if entity_id in knowledge.compounds
        and knowledge.compounds[entity_id].network_anchor_eligible
    )
    if not eligible:
        return EntityResolution("not_seed_eligible", resolution.entity_ids)
    if len(eligible) == 1:
        return EntityResolution("resolved", eligible)
    outgoing_eligible = tuple(
        entity_id
        for entity_id in eligible
        if entity_id in knowledge.outgoing_rules
    )
    if len(outgoing_eligible) == 1:
        return EntityResolution("resolved", outgoing_eligible)
    return EntityResolution("ambiguous", outgoing_eligible or eligible)


def evaluate_mass_gate(
    expected_mass: float,
    observed_mass: float,
    ppm_tolerance: float,
    da_tolerance: float,
    matched_adduct: str = "",
) -> MassGateAudit:
    error_da = abs(float(observed_mass) - float(expected_mass))
    error_ppm = error_da / max(abs(float(expected_mass)), 1e-12) * 1e6
    return MassGateAudit(
        ppm_pass=error_ppm <= ppm_tolerance,
        da_pass=error_da <= da_tolerance,
        error_ppm=error_ppm,
        error_da=error_da,
        matched_adduct=matched_adduct,
    )


def evaluate_fragment_gate(
    target_origin: str,
    matched_evidence: Sequence[FragmentMatch],
    *,
    min_class_diagnostic_matches: int = 2,
    min_target_fragment_matches: int = 1,
    min_target_product_ions: int = 2,
    discriminative_statuses: frozenset[str] = frozenset(
        {"unique", "low_sharing"}
    ),
    fragment_gate_mode: str = "strict",
    signature_competitor_entity_count: int = 0,
    signature_entity_universe_count: int = 0,
    max_signature_competitor_fraction: float = 0.1,
    relation_evidence_type: str = "",
    relation_evidence_status: str = "",
    relation_chemical_validation: str = "",
    product_resolution_status: str = "",
) -> FragmentGateAudit:
    target_rows = [
        row
        for row in matched_evidence
        if row.role == "target"
    ]
    target_ids = tuple(
        sorted(
            {
                row.evidence_id
                for row in target_rows
                if row.evidence_role != "class_diagnostic"
            }
        )
    )
    component_rows = [
        row
        for row in matched_evidence
        if row.evidence_scope in {"component_specific", "reaction_associated"}
    ]
    component_ids = tuple(sorted({row.evidence_id for row in component_rows}))
    roles = {row.role for row in component_rows}
    explicit_ids = tuple(
        sorted(
            {
                row.evidence_id
                for row in target_rows
                if row.evidence_role == "explicit_target_diagnostic"
            }
        )
    )
    product_ion_ids = tuple(
        sorted(
            {
                row.evidence_id
                for row in target_rows
                if row.evidence_role == "target_product_ion"
            }
        )
    )
    class_ids = tuple(
        sorted(
            {
                row.evidence_id
                for row in target_rows
                if row.evidence_role == "class_diagnostic"
            }
        )
    )
    catalog_ids = tuple(
        sorted(
            {
                row.evidence_id
                for row in target_rows
                if row.evidence_role == "theoretical_catalog"
            }
        )
    )
    discriminative_ids = tuple(
        sorted(
            {
                row.evidence_id
                for row in target_rows
                if row.discriminative_status in discriminative_statuses
                and row.evidence_role
                in {"explicit_target_diagnostic", "target_product_ion"}
            }
        )
    )
    signature_discriminative = (
        signature_competitor_entity_count > 0
        and signature_entity_universe_count > 0
        and (
            signature_competitor_entity_count
            / signature_entity_universe_count
        )
        <= max_signature_competitor_fraction
    )
    gate_level = "failed"
    if fragment_gate_mode == "evidence_bundle":
        relation_is_chemically_valid = relation_chemical_validation in {
            "valid",
            "derived_formula_valid",
        }
        product_is_resolved = product_resolution_status in {
            "unique_reported_entity",
            "ambiguous_reported_entities",
        }
        explicit_relation = relation_evidence_type == "explicit_report"
        balanced_component_relation = (
            relation_evidence_type == "literature_inferred"
            and relation_evidence_status == "component_delta_consistent"
        )
        target_pair = len(product_ion_ids) >= min_target_product_ions
        class_pair = len(class_ids) >= min_class_diagnostic_matches
        passed = (
            target_origin != "mechanism_derived"
            and relation_is_chemically_valid
            and product_is_resolved
            and (
                class_pair
                or (target_pair and explicit_relation)
                or (target_pair and balanced_component_relation)
            )
        )
        if passed and class_pair:
            gate_level = (
                "class_consensus_pair"
                if min_class_diagnostic_matches == 2
                else "class_consensus_triplet"
            )
        elif passed and explicit_relation:
            gate_level = "explicit_relation_target_ion_pair"
        elif passed and balanced_component_relation:
            gate_level = "balanced_relation_target_ion_pair"
    elif fragment_gate_mode == "reaction_evidence":
        passed = (
            target_origin != "mechanism_derived"
            and len(class_ids) >= min_class_diagnostic_matches
        )
        if passed:
            gate_level = (
                "class_consensus_pair"
                if min_class_diagnostic_matches == 2
                else "class_consensus_triplet"
            )
    elif fragment_gate_mode == "mass_only":
        passed = True
        gate_level = "mass_only"
    elif fragment_gate_mode == "required_empty":
        passed = False
    elif fragment_gate_mode == "theoretical_catalog_only":
        passed = bool(catalog_ids)
        gate_level = "theoretical_catalog" if passed else "failed"
    elif fragment_gate_mode == "high_recall":
        if target_origin == "mechanism_derived":
            passed = "core" in roles and bool(
                roles & {"donor", "donor_or_reaction", "reaction"}
            )
            gate_level = "mechanism_components" if passed else "failed"
        else:
            passed = len(target_ids) >= min_target_fragment_matches
            gate_level = "broad_reported_fragment" if passed else "failed"
    elif target_origin == "mechanism_derived":
        passed = False
    elif set(explicit_ids) & set(discriminative_ids):
        passed = True
        gate_level = "explicit_target_diagnostic"
    elif (
        len(product_ion_ids) >= min_target_product_ions
        and set(product_ion_ids) & set(discriminative_ids)
    ):
        passed = True
        gate_level = "target_product_ion_pair"
    elif (
        len(product_ion_ids) >= min_target_product_ions
        and signature_discriminative
    ):
        passed = True
        gate_level = "target_product_ion_signature"
    else:
        passed = False
    return FragmentGateAudit(
        passed,
        target_ids,
        component_ids,
        explicit_target_diagnostic_ids=explicit_ids,
        target_product_ion_ids=product_ion_ids,
        class_diagnostic_ids=class_ids,
        theoretical_catalog_ids=catalog_ids,
        discriminative_fragment_ids=discriminative_ids,
        fragment_gate_level=gate_level,
        signature_competitor_entity_count=(
            signature_competitor_entity_count
        ),
        signature_entity_universe_count=signature_entity_universe_count,
    )


def _matched_fragment_evidence(
    node: SpectrumNode,
    transformation: NetworkTransformation,
    target: NetworkEntity,
    knowledge: NetworkKnowledge,
    config: NetworkConfig,
) -> tuple[FragmentMatch, ...]:
    observed = np.asarray(node.mz_array, dtype=float)

    def matched(evidence_id: str) -> bool:
        expected = knowledge.fragment_mz_by_id.get(evidence_id)
        return expected is not None and bool(
            np.any(
                np.abs(
                    observed
                    - (float(expected) + float(config.fragment_mz_shift_da))
                )
                <= config.fragment_tolerance_da
            )
        )

    rows: list[FragmentMatch] = []
    diagnostic_candidates: dict[str, tuple[int, tuple[int, ...]]] = {}
    for diagnostic in knowledge.diagnostic_rows:
        if (
            str(diagnostic.get("diagnostic_origin", "")).strip()
            != "class_consensus_frequency"
        ):
            continue
        diagnostic_class = str(
            diagnostic.get("compound_class", "")
        ).strip().casefold()
        if (
            not diagnostic_class
            or diagnostic_class != target.compound_class.casefold()
        ):
            continue
        diagnostic_mode = str(
            diagnostic.get("ion_mode", "")
        ).strip().casefold()
        if diagnostic_mode != node.ion_mode:
            continue
        rule_id = str(diagnostic.get("rule_id", "")).strip()
        diagnostic_mz = safe_float(diagnostic.get("fragment_mz"))
        if not rule_id or diagnostic_mz is None:
            continue
        peak_indexes = tuple(
            sorted(
                (
                    index
                    for index, observed_mz in enumerate(observed)
                    if abs(
                        float(observed_mz)
                        - (
                            float(diagnostic_mz)
                            + float(config.fragment_mz_shift_da)
                        )
                    )
                    <= config.fragment_tolerance_da
                ),
                key=lambda index: (
                    abs(
                        float(observed[index])
                        - (
                            float(diagnostic_mz)
                            + float(config.fragment_mz_shift_da)
                        )
                    ),
                    index,
                ),
            )
        )
        if peak_indexes:
            diagnostic_candidates[rule_id] = (
                int(safe_float(diagnostic.get("support_rank")) or 10**9),
                peak_indexes,
            )

    peak_to_rule: dict[int, str] = {}
    rule_to_peak: dict[str, int] = {}

    def assign_diagnostic(rule_id: str, seen_peaks: set[int]) -> bool:
        for peak_index in diagnostic_candidates[rule_id][1]:
            if peak_index in seen_peaks:
                continue
            seen_peaks.add(peak_index)
            prior_rule = peak_to_rule.get(peak_index)
            if prior_rule is None or assign_diagnostic(
                prior_rule, seen_peaks
            ):
                peak_to_rule[peak_index] = rule_id
                rule_to_peak[rule_id] = peak_index
                return True
        return False

    for rule_id in sorted(
        diagnostic_candidates,
        key=lambda item: (
            len(diagnostic_candidates[item][1]),
            diagnostic_candidates[item][0],
            item,
        ),
    ):
        assign_diagnostic(rule_id, set())
    rows.extend(
        FragmentMatch(
            evidence_id=rule_id,
            evidence_scope="class_diagnostic",
            role="target",
            evidence_role="class_diagnostic",
            ion_mode=node.ion_mode,
            discriminative_status="class_consensus",
        )
        for rule_id in sorted(rule_to_peak)
    )
    if config.fragment_gate_mode in {
        "high_recall",
        "theoretical_catalog_only",
    }:
        rows.extend(
            FragmentMatch(
                evidence_id,
                "target_associated",
                "target",
                evidence_role="theoretical_catalog",
                ion_mode="not_reported",
                discriminative_status="not_evaluated",
            )
            for evidence_id in target.reported_fragment_ids
            if (
                config.allowed_target_fragment_ids is None
                or evidence_id in config.allowed_target_fragment_ids
            )
            if matched(evidence_id)
        )
    fragment_target_entity_id = target.entity_id
    if config.shuffle_fragment_entity_labels:
        fragment_entity_ids = sorted(
            {
                row.entity_id
                for row in knowledge.fragment_evidence_by_id.values()
                if row.entity_id
            }
        )
        if fragment_target_entity_id in fragment_entity_ids and len(
            fragment_entity_ids
        ) > 1:
            index = fragment_entity_ids.index(fragment_target_entity_id)
            fragment_target_entity_id = fragment_entity_ids[
                (index + 1) % len(fragment_entity_ids)
            ]
    for fragment in knowledge.fragment_evidence_by_id.values():
        if fragment.entity_id != fragment_target_entity_id:
            continue
        if (
            config.allowed_target_fragment_ids is not None
            and fragment.fragment_id not in config.allowed_target_fragment_ids
        ):
            continue
        if config.fragment_gate_mode == "strict" and fragment.ion_mode != node.ion_mode:
            continue
        if (
            config.fragment_gate_mode != "strict"
            and fragment.ion_mode not in {node.ion_mode, "not_reported"}
        ):
            continue
        if matched(fragment.fragment_id):
            rows.append(
                FragmentMatch(
                    fragment.fragment_id,
                    fragment.specificity_scope,
                    "target",
                    evidence_role=fragment.evidence_role,
                    ion_mode=fragment.ion_mode,
                    discriminative_status=fragment.discriminative_status,
                    competitor_entity_count=fragment.competitor_entity_count,
                )
            )
    contract = transformation.fragment_evidence_contract
    for key, scope, role, evidence_role in (
        (
            "explicit_target_diagnostic_ids",
            "target_specific",
            "target",
            "explicit_target_diagnostic",
        ),
        (
            "target_product_ion_ids",
            "target_specific",
            "target",
            "target_product_ion",
        ),
        (
            "class_diagnostic_ids",
            "class_diagnostic",
            "target",
            "class_diagnostic",
        ),
        (
            "theoretical_catalog_ids",
            "target_associated",
            "target",
            "theoretical_catalog",
        ),
        (
            "core_fragment_evidence_ids",
            "component_specific",
            "core",
            "reaction_supporting_fragment",
        ),
        (
            "donor_fragment_evidence_ids",
            "component_specific",
            "donor",
            "reaction_supporting_fragment",
        ),
        (
            "reaction_fragment_evidence_ids",
            "reaction_associated",
            "reaction",
            "reaction_supporting_fragment",
        ),
    ):
        if config.shuffle_fragment_entity_labels and role == "target":
            continue
        for evidence_id in contract.get(key, ()):
            if (
                role == "target"
                and config.allowed_target_fragment_ids is not None
                and evidence_id not in config.allowed_target_fragment_ids
            ):
                continue
            fragment = knowledge.fragment_evidence_by_id.get(evidence_id)
            if (
                config.fragment_gate_mode == "strict"
                and role == "target"
                and (fragment is None or fragment.ion_mode != node.ion_mode)
            ):
                continue
            if fragment is not None and fragment.ion_mode not in {
                node.ion_mode,
                "not_reported",
            }:
                continue
            if matched(evidence_id):
                rows.append(
                    FragmentMatch(
                        evidence_id,
                        (
                            fragment.specificity_scope
                            if fragment is not None and role == "target"
                            else scope
                        ),
                        role,
                        evidence_role=(
                            fragment.evidence_role
                            if fragment is not None and role == "target"
                            else evidence_role
                        ),
                        ion_mode=(
                            fragment.ion_mode if fragment else "not_reported"
                        ),
                        discriminative_status=(
                            fragment.discriminative_status
                            if fragment
                            else "not_evaluated"
                        ),
                        competitor_entity_count=(
                            fragment.competitor_entity_count if fragment else 0
                        ),
                    )
                )
    return tuple(
        {
            (row.evidence_id, row.role): row
            for row in rows
        }.values()
    )


def _best_mass_gate(
    node: SpectrumNode,
    target: NetworkEntity,
    config: NetworkConfig,
) -> MassGateAudit:
    if target.exact_mass is None:
        return MassGateAudit(False, False, float("inf"), float("inf"), "")
    generic_adducts = tuple(
        adduct
        for adduct, (mode, _, _) in SUPPORTED_ADDUCTS.items()
        if mode == node.ion_mode
    )
    adducts = tuple(
        dict.fromkeys(
            (
                *target.adducts,
                *(generic_adducts if config.allow_generic_adducts else ()),
            )
        )
    )
    candidates: list[MassGateAudit] = []
    for adduct in adducts:
        mode, offset, charge = SUPPORTED_ADDUCTS[adduct]
        if mode != node.ion_mode:
            continue
        expected_mz = (target.exact_mass + offset) / abs(charge)
        candidates.append(
            evaluate_mass_gate(
                expected_mz,
                node.precursor_mz,
                config.ppm_tolerance,
                config.absolute_tolerance_da,
                adduct,
            )
        )
    return min(
        candidates,
        key=lambda row: (not row.passed, row.error_da, row.error_ppm),
        default=MassGateAudit(False, False, float("inf"), float("inf"), ""),
    )


def _product_ion_signature_competitors(
    target_entity_id: str,
    node_ion_mode: str,
    matched_evidence: Sequence[FragmentMatch],
    knowledge: NetworkKnowledge,
    tolerance_da: float,
    expected_mz_shift_da: float = 0.0,
) -> tuple[int, int]:
    matched_fragments = tuple(
        knowledge.fragment_evidence_by_id[row.evidence_id]
        for row in matched_evidence
        if row.role == "target"
        and row.evidence_role == "target_product_ion"
        and row.evidence_id in knowledge.fragment_evidence_by_id
        and knowledge.fragment_evidence_by_id[row.evidence_id].entity_id
        == target_entity_id
        and knowledge.fragment_evidence_by_id[row.evidence_id].ion_mode
        == node_ion_mode
    )
    signature_values = tuple(
        sorted(
            {
                round(row.fragment_mz + expected_mz_shift_da, 9)
                for row in matched_fragments
            }
        )
    )
    entity_values: dict[str, list[float]] = {}
    for row in knowledge.fragment_evidence_by_id.values():
        if (
            row.evidence_role != "target_product_ion"
            or row.ion_mode != node_ion_mode
            or not row.entity_id
        ):
            continue
        entity_values.setdefault(row.entity_id, []).append(row.fragment_mz)
    if len(signature_values) < 2 or not entity_values:
        return 0, len(entity_values)
    competitors = sum(
        all(
            any(abs(candidate - expected) <= tolerance_da for candidate in values)
            for expected in signature_values
        )
        for values in entity_values.values()
    )
    return competitors, len(entity_values)


def evaluate_hypothesis_gates(
    hypothesis: Hypothesis | NetworkTransformation,
    node: SpectrumNode,
    knowledge: NetworkKnowledge,
    config: NetworkConfig,
    *,
    source_propagation: bool | None = None,
) -> GateAudit:
    transformation = (
        knowledge.transformations[
            next(
                index
                for index, row in enumerate(knowledge.transformations)
                if row.rule_id == hypothesis.rule_id
            )
        ]
        if isinstance(hypothesis, Hypothesis)
        else hypothesis
    )
    target = knowledge.compounds[transformation.target_entity_id]
    source_ok = (
        hypothesis.source_propagation
        if isinstance(hypothesis, Hypothesis)
        else bool(source_propagation)
    )
    modes = transformation.ion_modes or target.ion_modes
    ion_ok = not modes or node.ion_mode in modes
    chemical_ok = transformation.chemical_validation_status in {
        "valid",
        "derived_formula_valid",
    }
    mass = _best_mass_gate(node, target, config)
    matches: tuple[FragmentMatch, ...] = ()
    if mass.passed:
        matches = _matched_fragment_evidence(
            node,
            transformation,
            target,
            knowledge,
            config,
        )
        (
            signature_competitor_count,
            signature_entity_universe_count,
        ) = _product_ion_signature_competitors(
            target.entity_id,
            node.ion_mode,
            matches,
            knowledge,
            config.fragment_tolerance_da,
            config.fragment_mz_shift_da,
        )
        fragment = evaluate_fragment_gate(
            target.target_origin,
            matches,
            min_class_diagnostic_matches=(
                config.min_class_diagnostic_matches
            ),
            min_target_fragment_matches=config.min_target_fragment_matches,
            min_target_product_ions=config.min_target_product_ions,
            discriminative_statuses=config.discriminative_statuses,
            fragment_gate_mode=config.fragment_gate_mode,
            signature_competitor_entity_count=(
                signature_competitor_count
            ),
            signature_entity_universe_count=(
                signature_entity_universe_count
            ),
            max_signature_competitor_fraction=(
                config.max_signature_competitor_fraction
            ),
            relation_evidence_type=transformation.evidence_type,
            relation_evidence_status=(
                transformation.relation_evidence_status
            ),
            relation_chemical_validation=(
                transformation.chemical_validation_status
            ),
            product_resolution_status=(
                transformation.product_resolution_status
            ),
        )
    else:
        fragment = FragmentGateAudit(False, (), ())
    gates = {
        "source_propagation": source_ok,
        "ion_mode": ion_ok,
        "chemical_validation": chemical_ok,
        "target_mass_available": target.exact_mass is not None,
        "precursor_ppm": mass.ppm_pass,
        "precursor_da": mass.da_pass,
        "fragment_identity": fragment.passed,
        "isomer_uniqueness": True,
    }
    return GateAudit(
        gates=gates,
        matched_target_fragment_ids=fragment.target_fragment_ids,
        matched_component_fragment_ids=fragment.component_fragment_ids,
        matched_neutral_loss_ids=(),
        failure_reasons=tuple(key for key, value in gates.items() if not value),
        mass_error_ppm=mass.error_ppm,
        mass_error_da=mass.error_da,
        matched_adduct=mass.matched_adduct,
        matched_explicit_diagnostic_ids=(
            fragment.explicit_target_diagnostic_ids
        ),
        matched_target_product_ion_ids=fragment.target_product_ion_ids,
        matched_class_diagnostic_ids=fragment.class_diagnostic_ids,
        matched_theoretical_catalog_ids=fragment.theoretical_catalog_ids,
        discriminative_fragment_ids=fragment.discriminative_fragment_ids,
        fragment_gate_level=fragment.fragment_gate_level,
        fragment_competitor_entity_count=(
            fragment.signature_competitor_entity_count
            or max(
                (
                    row.competitor_entity_count
                    for row in matches
                    if row.evidence_id
                    in fragment.discriminative_fragment_ids
                ),
                default=0,
            )
        ),
        signature_competitor_entity_count=(
            fragment.signature_competitor_entity_count
        ),
        signature_entity_universe_count=(
            fragment.signature_entity_universe_count
        ),
    )


def resolve_depth_hypotheses(
    hypotheses: Sequence[Hypothesis],
    *,
    fragment_gate_mode: str = "strict",
) -> tuple[NodeDecision, ...]:
    grouped: dict[str, list[Hypothesis]] = {}
    for row in hypotheses:
        grouped.setdefault(row.target_node_id, []).append(row)
    decisions: list[NodeDecision] = []
    for node_id, rows in sorted(grouped.items()):
        passing = [row for row in rows if row.audit.passed and row.evidence_type != "delta_only"]
        if not passing:
            mass_supported = [
                row
                for row in rows
                if row.audit.gates.get("chemical_validation")
                and row.audit.gates.get("precursor_ppm")
                and row.audit.gates.get("precursor_da")
            ]
            if mass_supported:
                row = sorted(mass_supported, key=lambda item: (item.target_entity_id, item.rule_id))[0]
                has_fragment_support = bool(
                    row.audit.matched_target_fragment_ids
                    or row.audit.matched_component_fragment_ids
                )
                if row.target_origin == "mechanism_derived" and has_fragment_support:
                    status = "mechanism_supported_exploratory"
                elif (
                    row.candidate_policy == "literature_explanation_only"
                    and has_fragment_support
                ):
                    status = "literature_explained_candidate"
                elif has_fragment_support:
                    status = "exploratory_candidate"
                else:
                    status = "candidate"
                decisions.append(
                    NodeDecision(
                        node_id, row.target_entity_id, row.source_entity_id, row.rule_id,
                        row.evidence_type, row.target_origin, status, False,
                        "identity_gate_not_satisfied",
                        row.audit.fragment_gate_level,
                    )
                )
            continue
        by_target = {
            target_id: [row for row in passing if row.target_entity_id == target_id]
            for target_id in sorted({row.target_entity_id for row in passing})
        }
        if len(by_target) > 1:
            uniquely_target_supported = [
                target_id
                for target_id, target_rows in by_target.items()
                if any(
                    row.audit.discriminative_fragment_ids
                    for row in target_rows
                )
                and all(
                    not any(
                        row.audit.discriminative_fragment_ids
                        for row in other_rows
                    )
                    for other_id, other_rows in by_target.items()
                    if other_id != target_id
                )
            ]
            if len(uniquely_target_supported) != 1:
                first = passing[0]
                target_ids = tuple(sorted(by_target))
                rule_ids = tuple(sorted({row.rule_id for row in passing}))
                source_ids = tuple(
                    sorted({row.source_entity_id for row in passing})
                )
                reported_group = all(
                    row.target_origin == "reported"
                    and row.evidence_type
                    in {"explicit_report", "literature_inferred"}
                    for row in passing
                )
                if reported_group:
                    digest = hashlib.sha1(
                        "|".join(target_ids).encode("utf-8")
                    ).hexdigest()[:16]
                    explanation_only_group = all(
                        row.candidate_policy == "literature_explanation_only"
                        for row in passing
                    )
                    decisions.append(
                        NodeDecision(
                            node_id=node_id,
                            target_entity_id="",
                            source_entity_id=(
                                source_ids[0] if len(source_ids) == 1 else ""
                            ),
                            rule_id="",
                            evidence_type="literature_supported_group",
                            target_origin="reported_group",
                            annotation_status=(
                                "ambiguous_literature_explained_candidate"
                                if explanation_only_group
                                else "ambiguous_group_discover"
                            ),
                            propagation_eligible=False,
                            ambiguity_reason="multiple_gate_passing_targets",
                            fragment_gate_level="ambiguous_group",
                            target_entity_group_id=f"entity_group_{digest}",
                            ambiguous_target_entity_ids=target_ids,
                            ambiguous_rule_ids=rule_ids,
                        )
                    )
                else:
                    decisions.append(
                        NodeDecision(
                            node_id, "", first.source_entity_id, "", "", "",
                            "ambiguous", False,
                            "multiple_gate_passing_targets", "ambiguous",
                            ambiguous_target_entity_ids=target_ids,
                            ambiguous_rule_ids=rule_ids,
                        )
                    )
                continue
            passing = by_target[uniquely_target_supported[0]]
        row = sorted(passing, key=lambda item: (item.target_entity_id, item.rule_id))[0]
        if row.candidate_policy == "literature_explanation_only":
            annotation_status = "literature_explained_candidate"
            propagation = False
        elif (
            fragment_gate_mode in {
                "strict",
                "reaction_evidence",
                "evidence_bundle",
            }
            and row.target_origin == "reported"
        ):
            annotation_status = "fragment_supported_discover"
            propagation = (
                row.propagation_policy == "allow_after_identity_gate"
            )
        elif row.target_origin == "mechanism_derived":
            annotation_status = "mechanism_supported_exploratory"
            propagation = False
        else:
            annotation_status = "exploratory_candidate"
            propagation = False
        decisions.append(
            NodeDecision(
                node_id, row.target_entity_id, row.source_entity_id, row.rule_id,
                row.evidence_type, row.target_origin, annotation_status, propagation,
                "",
                row.audit.fragment_gate_level,
            )
        )
    return tuple(decisions)


def run_entity_hypothesis_network(
    nodes: Sequence[SpectrumNode],
    seeds: Sequence[SeedAssignment],
    knowledge: NetworkKnowledge,
    config: NetworkConfig | None = None,
) -> NetworkRunResult:
    settings = config or NetworkConfig()
    if settings.max_depth <= 0:
        raise ValueError("max_depth must be positive")
    nodes_by_id = {row.node_id: row for row in nodes}
    assignments: dict[str, NodeDecision] = {}
    frontier: list[tuple[str, str]] = []
    seed_node_ids = {seed.node_id for seed in seeds}
    result_rows: dict[str, dict[str, Any]] = {
        node.node_id: {
            "node_id": node.node_id,
            "feature_id": node.feature_id,
            "source_spectrum_id": node.source_spectrum_id,
            "precursor_mz": node.precursor_mz,
            "ion_mode": node.ion_mode,
            "rt_min": node.rt_min if node.rt_min is not None else "",
            "raw_indices": ";".join(str(index) for index in node.raw_indices),
            "library_name": node.library_name,
            "library_candidate_names": ";".join(node.library_candidate_names),
            "library_candidate_scores": ";".join(
                f"{value:.8f}" for value in node.library_candidate_scores
            ),
            "seed_resolution_status": node.seed_resolution_status,
            "seed_resolution_entity_ids": ";".join(
                node.seed_resolution_entity_ids
            ),
            "target_entity_id": "",
            "annotation_status": "known_match" if node.known_match else "unassigned",
            "propagation_eligible": False,
            "ambiguity_reason": (
                ""
                if not node.known_match
                or node.seed_resolution_status == "resolved"
                else f"library_entity_{node.seed_resolution_status or 'unresolved'}"
            ),
            "network_depth": 0 if node.known_match else "",
        }
        for node in nodes
    }
    for node in nodes:
        if node.known_match and node.node_id not in seed_node_ids:
            assignments[node.node_id] = NodeDecision(
                node.node_id,
                "",
                "",
                "",
                "library_match",
                "",
                "known_match",
                False,
                f"library_entity_{node.seed_resolution_status or 'unresolved'}",
            )
    seeds_by_node: dict[str, dict[str, set[str]]] = {}
    for seed in seeds:
        if seed.node_id not in nodes_by_id or seed.entity_id not in knowledge.compounds:
            continue
        seeds_by_node.setdefault(seed.node_id, {}).setdefault(
            seed.entity_id, set()
        ).add(seed.library_name)
    for node_id, entity_names in sorted(seeds_by_node.items()):
        entity_ids = tuple(sorted(entity_names))
        for entity_id in entity_ids:
            frontier.append((node_id, entity_id))
        library_names = tuple(
            sorted(
                {
                    name
                    for names in entity_names.values()
                    for name in names
                    if name
                }
            )
        )
        if len(entity_ids) == 1:
            entity_id = entity_ids[0]
            assignments[node_id] = NodeDecision(
                node_id, entity_id, "", "", "library_match",
                knowledge.compounds[entity_id].target_origin, "known_match", True,
            )
            result_rows[node_id].update(
                {
                    "target_entity_id": entity_id,
                    "library_name": ";".join(library_names),
                    "seed_resolution_status": "resolved",
                    "seed_resolution_entity_ids": entity_id,
                    "annotation_status": "known_match",
                    "propagation_eligible": True,
                    "ambiguity_reason": "",
                    "network_depth": 0,
                }
            )
            continue
        digest = hashlib.sha256("|".join(entity_ids).encode("utf-8")).hexdigest()[:16]
        group_id = f"entity_group_{digest}"
        assignments[node_id] = NodeDecision(
            node_id,
            "",
            "",
            "",
            "library_match",
            "reported_group",
            "ambiguous_library_seed",
            True,
            "near_tied_library_entities",
            "library_seed_group",
            group_id,
            entity_ids,
            (),
        )
        result_rows[node_id].update(
            {
                "target_entity_id": "",
                "target_entity_group_id": group_id,
                "ambiguous_target_entity_ids": ";".join(entity_ids),
                "ambiguous_target_entities": ";".join(
                    knowledge.compounds[entity_id].canonical_name
                    for entity_id in entity_ids
                ),
                "library_name": ";".join(library_names),
                "seed_resolution_status": "ambiguous_library_seed",
                "seed_resolution_entity_ids": ";".join(entity_ids),
                "annotation_status": "ambiguous_library_seed",
                "propagation_eligible": True,
                "ambiguity_reason": "near_tied_library_entities",
                "fragment_gate_level": "library_seed_group",
                "network_depth": 0,
            }
        )
    visited: set[tuple[str, str]] = set(frontier)
    all_hypotheses: list[Hypothesis] = []
    edges: list[dict[str, Any]] = []
    for depth in range(1, settings.max_depth + 1):
        depth_hypotheses: list[Hypothesis] = []
        depth_states: set[tuple[str, str]] = set()
        # Gate evaluation depends on the source entity/rule, not on which
        # replicate spectrum node supplied that same entity.  Collapse such
        # equivalent frontier states to the lexicographically first node so
        # all distinct source-entity/rule paths remain auditable without an
        # O(replicates x nodes x rules) hypothesis explosion.
        canonical_source_nodes: dict[str, str] = {}
        for source_node_id, source_entity_id in frontier:
            canonical_source_nodes[source_entity_id] = min(
                source_node_id,
                canonical_source_nodes.get(source_entity_id, source_node_id),
            )
        canonical_frontier = sorted(
            (source_node_id, source_entity_id)
            for source_entity_id, source_node_id in canonical_source_nodes.items()
        )
        for source_node_id, source_entity_id in canonical_frontier:
            rule_source_entity_id = source_entity_id
            if settings.shuffle_reaction_roles:
                source_ids = sorted(knowledge.outgoing_rules)
                if source_entity_id in source_ids and len(source_ids) > 1:
                    index = source_ids.index(source_entity_id)
                    rule_source_entity_id = source_ids[
                        (index + 1) % len(source_ids)
                    ]
            for transformation in knowledge.outgoing_rules.get(
                rule_source_entity_id, ()
            ):
                for node in sorted(nodes, key=lambda item: item.node_id):
                    if node.node_id in assignments:
                        continue
                    state = (node.node_id, transformation.target_entity_id)
                    if state in visited:
                        continue
                    depth_states.add(state)
                    target = knowledge.compounds[transformation.target_entity_id]
                    if not _best_mass_gate(node, target, settings).passed:
                        continue
                    provisional = Hypothesis(
                        source_node_id=source_node_id,
                        target_node_id=node.node_id,
                        source_entity_id=source_entity_id,
                        target_entity_id=transformation.target_entity_id,
                        rule_id=transformation.rule_id,
                        evidence_type=transformation.evidence_type,
                        target_origin=transformation.target_origin,
                        depth=depth,
                        source_propagation=True,
                        audit=GateAudit({}, (), (), (), ()),
                        template_support_level=(
                            transformation.template_support_level
                        ),
                        candidate_policy=transformation.candidate_policy,
                        propagation_policy=transformation.propagation_policy,
                    )
                    audit = evaluate_hypothesis_gates(
                        transformation,
                        node,
                        knowledge,
                        settings,
                        source_propagation=True,
                    )
                    depth_hypotheses.append(
                        Hypothesis(**{**provisional.__dict__, "audit": audit})
                    )
        # A state is closed only after every source/rule path at this BFS
        # depth has been enumerated.  This preserves auditable competing
        # evidence paths and prevents CSV/source iteration order from deciding
        # which hypothesis survives.
        visited.update(depth_states)
        all_hypotheses.extend(depth_hypotheses)
        decisions = resolve_depth_hypotheses(
            depth_hypotheses,
            fragment_gate_mode=settings.fragment_gate_mode,
        )
        next_frontier: list[tuple[str, str]] = []
        for decision in decisions:
            assignments[decision.node_id] = decision
            result_rows[decision.node_id].update(
                {
                    "source_entity_id": decision.source_entity_id,
                    "target_entity_id": decision.target_entity_id,
                    "rule_id": decision.rule_id,
                    "evidence_type": decision.evidence_type,
                    "target_origin": decision.target_origin,
                    "annotation_status": decision.annotation_status,
                    "propagation_eligible": decision.propagation_eligible,
                    "ambiguity_reason": decision.ambiguity_reason,
                    "fragment_gate_level": decision.fragment_gate_level,
                    "target_entity_group_id": (
                        decision.target_entity_group_id
                    ),
                    "ambiguous_target_entity_ids": ";".join(
                        decision.ambiguous_target_entity_ids
                    ),
                    "ambiguous_target_entities": ";".join(
                        knowledge.compounds[entity_id].canonical_name
                        for entity_id in decision.ambiguous_target_entity_ids
                        if entity_id in knowledge.compounds
                    ),
                    "ambiguous_rule_ids": ";".join(
                        decision.ambiguous_rule_ids
                    ),
                    "network_depth": depth,
                }
            )
            if decision.annotation_status == "fragment_supported_discover":
                edges.append(
                    {
                        "source_entity_id": decision.source_entity_id,
                        "target_entity_id": decision.target_entity_id,
                        "rule_id": decision.rule_id,
                        "annotation_status": decision.annotation_status,
                    }
                )
            elif decision.annotation_status == "ambiguous_group_discover":
                edges.append(
                    {
                        "source_entity_id": decision.source_entity_id,
                        "target_entity_id": decision.target_entity_group_id,
                        "rule_id": "",
                        "annotation_status": decision.annotation_status,
                        "ambiguous_target_entity_ids": ";".join(
                            decision.ambiguous_target_entity_ids
                        ),
                        "ambiguous_rule_ids": ";".join(
                            decision.ambiguous_rule_ids
                        ),
                    }
                )
            if decision.propagation_eligible:
                next_frontier.append((decision.node_id, decision.target_entity_id))
        frontier = next_frontier
        if not frontier:
            break
    hypothesis_rows = tuple(
        {
            "source_node_id": row.source_node_id,
            "target_node_id": row.target_node_id,
            "source_entity_id": row.source_entity_id,
            "target_entity_id": row.target_entity_id,
            "rule_id": row.rule_id,
            "evidence_type": row.evidence_type,
            "target_origin": row.target_origin,
            "template_support_level": row.template_support_level,
            "candidate_policy": row.candidate_policy,
            "propagation_policy": row.propagation_policy,
            "network_depth": row.depth,
            **{f"gate_{key}": value for key, value in row.audit.gates.items()},
            "matched_target_fragment_ids": ";".join(row.audit.matched_target_fragment_ids),
            "matched_component_fragment_ids": ";".join(row.audit.matched_component_fragment_ids),
            "matched_neutral_loss_ids": ";".join(row.audit.matched_neutral_loss_ids),
            "matched_explicit_diagnostic_ids": ";".join(
                row.audit.matched_explicit_diagnostic_ids
            ),
            "matched_target_product_ion_ids": ";".join(
                row.audit.matched_target_product_ion_ids
            ),
            "matched_class_diagnostic_ids": ";".join(
                row.audit.matched_class_diagnostic_ids
            ),
            "matched_theoretical_catalog_ids": ";".join(
                row.audit.matched_theoretical_catalog_ids
            ),
            "discriminative_fragment_ids": ";".join(
                row.audit.discriminative_fragment_ids
            ),
            "fragment_gate_level": row.audit.fragment_gate_level,
            "fragment_competitor_entity_count": (
                row.audit.fragment_competitor_entity_count
            ),
            "signature_competitor_entity_count": (
                row.audit.signature_competitor_entity_count
            ),
            "signature_entity_universe_count": (
                row.audit.signature_entity_universe_count
            ),
            "mass_error_ppm": row.audit.mass_error_ppm,
            "mass_error_da": row.audit.mass_error_da,
            "matched_adduct": row.audit.matched_adduct,
            "annotation_status": assignments.get(
                row.target_node_id,
                NodeDecision("", "", "", "", "", "", "rejected", False),
            ).annotation_status,
            "propagation_eligible": assignments.get(
                row.target_node_id,
                NodeDecision("", "", "", "", "", "", "rejected", False),
            ).propagation_eligible,
        }
        for row in all_hypotheses
    )
    return NetworkRunResult(
        nodes=tuple(result_rows[key] for key in sorted(result_rows)),
        hypotheses=hypothesis_rows,
        edges=tuple(edges),
    )
