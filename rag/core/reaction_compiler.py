"""Materialize evidence-backed reaction templates into concrete entity rules."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Mapping

from .chemical_consistency import (
    ChemicalValidationResult,
    derive_missing_reactant_formula,
    validate_reaction_balance,
)
from .reaction_models import (
    EntityClassMembershipClaim,
    ReactionParticipant,
    ReactionTemplateClaim,
    stable_structured_id,
)
from .entity_registry import EntityForm, EntityRegistry, registry_name_keys
from .evidence_models import EvidenceRef


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    entity_name: str
    formula: str
    exact_mass: float | None
    reported_fragments: tuple[float, ...]
    evidence_ids: tuple[str, ...]
    entity_origin: str = "reported"
    derivation_id: str = ""


@dataclass(frozen=True)
class MaterializationConfig:
    max_combinations_per_template: int = 1000


@dataclass(frozen=True)
class ConcreteReactant:
    entity_id: str
    role: str
    coefficient: int
    form_id: str = ""
    formula: str = ""


@dataclass(frozen=True)
class MaterializedReaction:
    derivation_id: str
    template_claim_id: str
    reaction_name: str
    reaction_type: str
    reaction_operator: str
    reactants: tuple[ConcreteReactant, ...]
    anchor_entity_id: str
    anchor_reactant_index: int
    network_anchor_role: str
    target_entity_id: str
    target_entity_name: str
    product_resolution_status: str
    reported_product_candidates: tuple[str, ...]
    product_formula: str
    calculated_product_mass: float | None
    chemical_validation_status: str
    operator_schema_version: str
    propagation_eligible: bool
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class TemplateAudit:
    template_claim_id: str
    status: str
    resolved_network_anchor_count: int
    resolved_network_anchor_ids: tuple[str, ...]
    total_combination_count: int
    materialized_count: int
    detail: str = ""


@dataclass(frozen=True)
class MaterializationReport:
    materialized_reactions: tuple[MaterializedReaction, ...]
    template_audits: tuple[TemplateAudit, ...]
    derived_entities: tuple[EntityRecord, ...] = ()
    derived_forms: tuple[EntityForm, ...] = ()


def _normalized_name(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _name_index(entities: Mapping[str, EntityRecord]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for entity_id, item in entities.items():
        for key in registry_name_keys(item.entity_name):
            grouped.setdefault(key, []).append(entity_id)
    return {key: tuple(sorted(values)) for key, values in grouped.items()}


def _membership_index(
    memberships: Iterable[EntityClassMembershipClaim],
    entities: Mapping[str, EntityRecord],
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, set[str]] = {}
    for claim in memberships:
        if claim.entity_id in entities:
            grouped.setdefault(claim.entity_class_id, set()).add(claim.entity_id)
    return {key: tuple(sorted(values)) for key, values in grouped.items()}


def _participant_candidates(
    participant: ReactionParticipant,
    entities: Mapping[str, EntityRecord],
    names: Mapping[str, tuple[str, ...]],
    memberships: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    if participant.scope == "entity_class":
        return memberships.get(participant.entity_class_id, ())
    if participant.scope == "unresolved":
        return ()
    if participant.entity_id:
        direct = entities.get(participant.entity_id)
        if direct is None:
            return ()
        if direct.formula:
            return (participant.entity_id,)
        exact_name_candidates = tuple(
            entity_id
            for entity_id, entity in entities.items()
            if _normalized_name(entity.entity_name)
            == _normalized_name(participant.entity_name)
            and entity.formula
        )
        exact_signatures = {
            (
                entities[entity_id].formula,
                (
                    round(float(entities[entity_id].exact_mass), 6)
                    if entities[entity_id].exact_mass is not None
                    else None
                ),
            )
            for entity_id in exact_name_candidates
        }
        if exact_name_candidates and len(exact_signatures) == 1:
            return (sorted(exact_name_candidates)[0],)
        if exact_name_candidates:
            return ()
        name_candidates = tuple(
            entity_id
            for entity_id in names.get(_normalized_name(participant.entity_name), ())
            if entities[entity_id].formula
        )
        signatures = {
            (
                entities[entity_id].formula,
                (
                    round(float(entities[entity_id].exact_mass), 6)
                    if entities[entity_id].exact_mass is not None
                    else None
                ),
            )
            for entity_id in name_candidates
        }
        if name_candidates and len(signatures) == 1:
            return (sorted(name_candidates)[0],)
        if name_candidates:
            return ()
        return (participant.entity_id,)
    return names.get(_normalized_name(participant.entity_name), ())


def _complete_single_missing_reactants(
    templates: Iterable[ReactionTemplateClaim],
    entities: Mapping[str, EntityRecord],
) -> tuple[dict[str, EntityRecord], tuple[EntityRecord, ...]]:
    completed = dict(entities)
    derived: dict[str, EntityRecord] = {}
    for template in templates:
        missing = [
            participant
            for participant in template.reactants
            if participant.scope == "specific_entity"
            and participant.entity_id
            and participant.entity_id not in completed
        ]
        if len(missing) != 1 or len(template.products) != 1:
            continue
        product = template.products[0]
        if product.scope != "specific_entity" or not product.entity_id or product.entity_id not in completed:
            continue
        known_reactants: list[tuple[str, int]] = []
        valid_known = True
        for participant in template.reactants:
            if participant is missing[0]:
                continue
            entity = completed.get(participant.entity_id)
            if entity is None or not entity.formula:
                valid_known = False
                break
            known_reactants.append((entity.formula, participant.coefficient))
        product_entity = completed[product.entity_id]
        if not valid_known or not product_entity.formula:
            continue
        result = derive_missing_reactant_formula(
            known_reactants=known_reactants,
            products=[(product_entity.formula, product.coefficient)],
            reaction_operator=template.reaction_operator,
            missing_coefficient=missing[0].coefficient,
        )
        if result.status != "derived_formula_valid":
            continue
        derivation_id = stable_structured_id(
            "component_derivation",
            {
                "template_claim_id": template.claim_id,
                "entity_id": missing[0].entity_id,
                "formula": result.formula,
            },
        )
        record = EntityRecord(
            entity_id=missing[0].entity_id,
            entity_name=missing[0].entity_name,
            formula=result.formula,
            exact_mass=result.exact_mass,
            reported_fragments=(),
            evidence_ids=(template.claim_id, product_entity.entity_id),
            entity_origin="mechanism_derived_component",
            derivation_id=derivation_id,
        )
        existing = derived.get(record.entity_id)
        if existing is not None and existing.formula != record.formula:
            continue
        derived[record.entity_id] = record
        completed[record.entity_id] = record
    return completed, tuple(sorted(derived.values(), key=lambda item: item.entity_id))


def _product_candidates(
    products: tuple[ReactionParticipant, ...],
    entities: Mapping[str, EntityRecord],
    names: Mapping[str, tuple[str, ...]],
    memberships: Mapping[str, tuple[str, ...]],
) -> tuple[str, tuple[str, ...]]:
    if len(products) != 1:
        return "unresolved_product_class", ()
    product = products[0]
    candidates = _participant_candidates(product, entities, names, memberships)
    if product.scope == "specific_entity":
        if len(candidates) == 1:
            return "unique_reported_entity", candidates
        if len(candidates) > 1:
            return "ambiguous_reported_entities", candidates
        return "unresolved_product_class", ()
    if product.scope == "entity_class" and len(candidates) > 1:
        return "ambiguous_reported_entities", candidates
    return "mechanism_derived_entity", candidates


def _chemical_validation(
    concrete: tuple[ConcreteReactant, ...],
    entities: Mapping[str, EntityRecord],
    target_entity_id: str,
    reaction_operator: str,
) -> ChemicalValidationResult:
    reactants: list[tuple[str, int]] = []
    for item in concrete:
        formula = item.formula or entities[item.entity_id].formula
        if not formula:
            return ChemicalValidationResult(
                status="formula_parse_failed",
                reaction_operator=reaction_operator,
                operator_schema_version="",
                expected_product_formula="",
                calculated_product_mass=None,
                atom_balance={},
                warning=f"reactant {item.entity_id!r} has no formula",
            )
        reactants.append((formula, item.coefficient))
    products: list[tuple[str, int]] = []
    if target_entity_id:
        product_formula = entities[target_entity_id].formula
        if not product_formula:
            return ChemicalValidationResult(
                status="formula_parse_failed",
                reaction_operator=reaction_operator,
                operator_schema_version="",
                expected_product_formula="",
                calculated_product_mass=None,
                atom_balance={},
                warning=f"product {target_entity_id!r} has no formula",
            )
        products.append((product_formula, 1))
    return validate_reaction_balance(reactants, products, reaction_operator)


def _derive_missing_reactant_form(
    template: ReactionTemplateClaim,
    concrete: tuple[ConcreteReactant, ...],
    entities: Mapping[str, EntityRecord],
    target_entity_id: str,
) -> tuple[tuple[ConcreteReactant, ...], EntityForm | None]:
    if not target_entity_id:
        return concrete, None
    target = entities.get(target_entity_id)
    if target is None or not target.formula:
        return concrete, None
    missing_indexes = [
        index
        for index, item in enumerate(concrete)
        if not (item.formula or entities[item.entity_id].formula)
    ]
    if len(missing_indexes) != 1:
        return concrete, None
    missing_index = missing_indexes[0]
    known_reactants = [
        (
            item.formula or entities[item.entity_id].formula,
            item.coefficient,
        )
        for index, item in enumerate(concrete)
        if index != missing_index
    ]
    result = derive_missing_reactant_formula(
        known_reactants=known_reactants,
        products=[(target.formula, 1)],
        reaction_operator=template.reaction_operator,
        missing_coefficient=concrete[missing_index].coefficient,
    )
    if result.status != "derived_formula_valid":
        return concrete, None
    missing = concrete[missing_index]
    form_id = stable_structured_id(
        "form",
        {
            "template_claim_id": template.claim_id,
            "entity_id": missing.entity_id,
            "form_type": "incorporated_residue",
            "formula": result.formula,
            "reaction_operator": template.reaction_operator,
        },
    )
    derived_form = EntityForm(
        form_id=form_id,
        entity_id=missing.entity_id,
        form_type="incorporated_residue",
        formula=result.formula,
        exact_mass=result.exact_mass,
        formula_origin="reaction_operator_derived",
        reaction_operator=template.reaction_operator,
        evidence_ids=(template.evidence.chunk_id,),
    )
    replaced = list(concrete)
    replaced[missing_index] = ConcreteReactant(
        entity_id=missing.entity_id,
        role=missing.role,
        coefficient=missing.coefficient,
        form_id=form_id,
        formula=result.formula,
    )
    return tuple(replaced), derived_form


def _derivation_id(
    template: ReactionTemplateClaim,
    concrete: tuple[ConcreteReactant, ...],
    product_resolution_status: str,
    target_entity_id: str,
    product_formula: str,
) -> str:
    canonical_reactants = sorted(
        (
            {
                "entity_id": item.entity_id,
                "form_id": item.form_id,
                "role": item.role,
                "coefficient": item.coefficient,
            }
            for item in concrete
        ),
        key=lambda item: (item["role"], item["entity_id"], item["coefficient"]),
    )
    return stable_structured_id(
        "derivation",
        {
            "reaction_name": template.reaction_name,
            "reaction_type": template.reaction_type,
            "reaction_operator": template.reaction_operator,
            "reactants": canonical_reactants,
            "anchor_entity_id": concrete[template.anchor_reactant_index].entity_id,
            "product_resolution_status": product_resolution_status,
            "target_entity_id": target_entity_id,
            "product_formula": product_formula,
        },
    )


def materialize_reaction_templates(
    registry: EntityRegistry | None = None,
    templates: Iterable[ReactionTemplateClaim] = (),
    config: MaterializationConfig | None = None,
    *,
    entities: Mapping[str, EntityRecord] | None = None,
    memberships: Iterable[EntityClassMembershipClaim] = (),
    resolved_anchor_entity_ids: set[str] | frozenset[str] | None = None,
) -> MaterializationReport:
    if registry is not None and not isinstance(registry, EntityRegistry):
        if templates:
            raise TypeError("ambiguous positional arguments")
        templates = registry
        registry = None
    settings = config or MaterializationConfig()
    if settings.max_combinations_per_template <= 0:
        raise ValueError("max_combinations_per_template must be positive")
    template_rows = tuple(templates)
    registry_forms_by_entity: dict[str, tuple[str, str]] = {}
    if registry is not None:
        entities = {
            entity_id: EntityRecord(
                entity_id=entity.entity_id,
                entity_name=entity.canonical_name,
                formula=entity.formula,
                exact_mass=entity.exact_mass,
                reported_fragments=(),
                evidence_ids=entity.evidence_ids,
            )
            for entity_id, entity in registry.entities.items()
        }
        memberships = tuple(
            EntityClassMembershipClaim(
                claim_id=row.membership_id,
                entity_id=row.entity_id,
                entity_class_id=row.entity_class_id,
                membership_role=row.role,
                evidence=EvidenceRef(
                    chunk_id=row.evidence_ids[0] if row.evidence_ids else "registry",
                    evidence_quote="registry evidence membership",
                ),
            )
            for row in registry.memberships
        )
        for entity_id in registry.entities:
            forms = sorted(
                (
                    0 if form.form_type == "neutral_molecule" else 1,
                    form.form_id,
                    form.formula,
                )
                for form in registry.forms.values()
                if form.entity_id == entity_id
                and form.form_type in {"neutral_molecule", "reported_ion"}
            )
            if forms:
                registry_forms_by_entity[entity_id] = (forms[0][1], forms[0][2])
    if entities is None:
        raise ValueError("entities or registry is required")
    completed_entities, derived_entities = _complete_single_missing_reactants(template_rows, entities)
    entities = completed_entities
    names = _name_index(entities)
    class_members = _membership_index(memberships, entities)
    real_anchors = (
        set(resolved_anchor_entity_ids)
        if resolved_anchor_entity_ids is not None
        else {
            entity_id
            for entity_id, entity in entities.items()
            if entity.formula
        }
    )
    reactions: list[MaterializedReaction] = []
    audits: list[TemplateAudit] = []
    derived_forms: dict[str, EntityForm] = {}

    for template in template_rows:
        pools = [
            _participant_candidates(participant, entities, names, class_members)
            for participant in template.reactants
        ]
        missing_index = next((index for index, pool in enumerate(pools) if not pool), None)
        if missing_index is not None:
            participant = template.reactants[missing_index]
            status = (
                "entity_class_has_no_members"
                if participant.scope == "entity_class"
                else "reactant_unresolved"
            )
            audits.append(
                TemplateAudit(template.claim_id, status, 0, (), 0, 0, f"reactant_index={missing_index}")
            )
            continue

        anchor_pool = tuple(sorted(set(pools[template.anchor_reactant_index]) & real_anchors))
        combination_count = 1
        for pool in pools:
            combination_count *= len(pool)
        if combination_count > settings.max_combinations_per_template:
            audits.append(
                TemplateAudit(
                    template.claim_id,
                    "combination_limit_exceeded",
                    len(anchor_pool),
                    anchor_pool,
                    combination_count,
                    0,
                    f"limit={settings.max_combinations_per_template}",
                )
            )
            continue

        resolution, reported_candidates = _product_candidates(
            template.products, entities, names, class_members
        )
        start_count = len(reactions)
        for combination in itertools.product(*pools):
            concrete = tuple(
                ConcreteReactant(
                    entity_id,
                    participant.role,
                    participant.coefficient,
                    participant.form_id
                    or registry_forms_by_entity.get(entity_id, ("", ""))[0],
                    (
                        registry.forms[participant.form_id].formula
                        if registry is not None
                        and participant.form_id
                        and participant.form_id in registry.forms
                        else registry_forms_by_entity.get(entity_id, ("", ""))[1]
                    ),
                )
                for participant, entity_id in zip(template.reactants, combination)
            )
            anchor_entity_id = concrete[template.anchor_reactant_index].entity_id
            target_entity_id = reported_candidates[0] if resolution == "unique_reported_entity" else ""
            concrete, derived_form = _derive_missing_reactant_form(
                template,
                concrete,
                entities,
                target_entity_id,
            )
            if derived_form is not None:
                derived_forms[derived_form.form_id] = derived_form
            chemical = _chemical_validation(
                concrete, entities, target_entity_id, template.reaction_operator
            )
            product_formula = chemical.expected_product_formula
            calculated_mass = chemical.calculated_product_mass
            resolved_status = resolution
            target_name = entities[target_entity_id].entity_name if target_entity_id else ""
            if resolution == "mechanism_derived_entity" and chemical.status == "derived_formula_valid":
                target_entity_id = stable_structured_id(
                    "mechanism_entity",
                    {
                        "reaction_type": template.reaction_type,
                        "reaction_operator": template.reaction_operator,
                        "reactants": sorted(
                            (item.entity_id, item.role, item.coefficient) for item in concrete
                        ),
                        "formula": product_formula,
                    },
                )
                target_name = f"mechanism-derived {template.reaction_name} [{product_formula}]"
            elif resolution == "mechanism_derived_entity":
                resolved_status = "unresolved_product_class"
            chemically_valid = chemical.status in {"valid", "derived_formula_valid"}
            propagation_eligible = (
                anchor_entity_id in real_anchors
                and chemically_valid
                and resolved_status in {"unique_reported_entity", "mechanism_derived_entity"}
                and bool(target_entity_id)
            )
            derivation_id = _derivation_id(
                template, concrete, resolved_status, target_entity_id, product_formula
            )
            reactions.append(
                MaterializedReaction(
                    derivation_id=derivation_id,
                    template_claim_id=template.claim_id,
                    reaction_name=template.reaction_name,
                    reaction_type=template.reaction_type,
                    reaction_operator=template.reaction_operator,
                    reactants=concrete,
                    anchor_entity_id=anchor_entity_id,
                    anchor_reactant_index=template.anchor_reactant_index,
                    network_anchor_role=template.network_anchor_role,
                    target_entity_id=target_entity_id,
                    target_entity_name=target_name,
                    product_resolution_status=resolved_status,
                    reported_product_candidates=reported_candidates,
                    product_formula=product_formula,
                    calculated_product_mass=calculated_mass,
                    chemical_validation_status=chemical.status,
                    operator_schema_version=chemical.operator_schema_version,
                    propagation_eligible=propagation_eligible,
                    evidence_ids=(template.claim_id,),
                )
            )
        materialized_count = len(reactions) - start_count
        audit_status = "materialized" if anchor_pool else "no_real_network_anchor"
        audits.append(
            TemplateAudit(
                template.claim_id,
                audit_status,
                len(anchor_pool),
                anchor_pool,
                combination_count,
                materialized_count,
            )
        )

    return MaterializationReport(
        tuple(reactions),
        tuple(audits),
        derived_entities,
        tuple(derived_forms[key] for key in sorted(derived_forms)),
    )
