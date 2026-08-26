"""Evidence-only fragment contracts for compiled reaction hypotheses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable

from .evidence_models import FragmentEvidence
from .reaction_compiler import MaterializedReaction


@dataclass(frozen=True)
class FragmentEvidenceContract:
    reported_target_fragment_ids: tuple[str, ...]
    explicit_target_diagnostic_ids: tuple[str, ...]
    target_product_ion_ids: tuple[str, ...]
    class_diagnostic_ids: tuple[str, ...]
    theoretical_catalog_ids: tuple[str, ...]
    core_fragment_evidence_ids: tuple[str, ...]
    donor_fragment_evidence_ids: tuple[str, ...]
    reaction_fragment_evidence_ids: tuple[str, ...]

    def canonical_json(self) -> str:
        return json.dumps(
            asdict(self),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def compile_fragment_evidence_contract(
    materialized: MaterializedReaction,
    fragments: Iterable[FragmentEvidence],
) -> FragmentEvidenceContract:
    """Link reported evidence; never fabricate target fragments for a product."""

    by_entity: dict[str, list[FragmentEvidence]] = {}
    reaction_rows: list[FragmentEvidence] = []
    for row in fragments:
        by_entity.setdefault(row.entity_id, []).append(row)
        if row.evidence_scope == "reaction_associated":
            reaction_rows.append(row)
    target_rows = tuple(by_entity.get(materialized.target_entity_id, ()))
    explicit = tuple(
        sorted(
            row.fragment_id
            for row in target_rows
            if row.evidence_role == "explicit_target_diagnostic"
            and row.specificity_scope == "target_specific"
        )
    )
    product_ions = tuple(
        sorted(
            row.fragment_id
            for row in target_rows
            if row.evidence_role == "target_product_ion"
            and row.specificity_scope == "target_specific"
        )
    )
    target = tuple(sorted(set(explicit) | set(product_ions)))
    class_diagnostic = tuple(
        sorted(
            row.fragment_id
            for row in target_rows
            if row.evidence_role == "class_diagnostic"
        )
    )
    catalog = tuple(
        sorted(
            row.fragment_id
            for row in target_rows
            if row.evidence_role == "theoretical_catalog"
        )
    )
    core: set[str] = set()
    donor: set[str] = set()
    for reactant in materialized.reactants:
        destination = core if reactant.entity_id == materialized.anchor_entity_id else donor
        destination.update(
            row.fragment_id
            for row in by_entity.get(reactant.entity_id, ())
            if row.evidence_scope == "component_specific"
        )
    return FragmentEvidenceContract(
        reported_target_fragment_ids=target,
        explicit_target_diagnostic_ids=explicit,
        target_product_ion_ids=product_ions,
        class_diagnostic_ids=class_diagnostic,
        theoretical_catalog_ids=catalog,
        core_fragment_evidence_ids=tuple(sorted(core)),
        donor_fragment_evidence_ids=tuple(sorted(donor)),
        reaction_fragment_evidence_ids=tuple(
            sorted(row.fragment_id for row in reaction_rows)
        ),
    )
