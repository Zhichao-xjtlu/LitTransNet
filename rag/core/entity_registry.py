"""Evidence-derived identity registry for the universal RAG pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .chemical_consistency import ChemicalConsistencyError, formula_exact_mass, parse_formula
from .evidence_inventory import EvidenceInventory
from .evidence_models import clean_text


ENTITY_KINDS = frozenset({"molecule", "moiety", "entity_class"})
FORM_TYPES = frozenset(
    {
        "neutral_molecule",
        "reported_ion",
        "incorporated_residue",
        "functional_group",
    }
)
FORMULA_ORIGINS = frozenset({"reported", "reaction_operator_derived"})


def stable_registry_id(prefix: str, payload: Mapping[str, object]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha1(canonical.encode('utf-8')).hexdigest()[:16]}"


def split_text_values(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        values = [clean_text(item) for item in value]
    else:
        values = [clean_text(item) for item in re.split(r"[;|]", str(value or ""))]
    return tuple(dict.fromkeys(item for item in values if item))


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    canonical_name: str
    reported_names: tuple[str, ...]
    entity_kind: str
    compound_class: str
    formula: str
    exact_mass: float | None
    ion_modes: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class EntityForm:
    form_id: str
    entity_id: str
    form_type: str
    formula: str
    exact_mass: float | None
    formula_origin: str
    reaction_operator: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class EntityClassRecord:
    entity_class_id: str
    class_label: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class EntityClassMembership:
    membership_id: str
    entity_id: str
    entity_class_id: str
    role: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class EntityResolution:
    status: str
    entity_ids: tuple[str, ...]


@dataclass(frozen=True)
class RegistryAuditRow:
    record_type: str
    record_id: str
    status: str
    detail: str


@dataclass(frozen=True)
class RegistryManifest:
    schema_version: str
    artifact_hashes: Mapping[str, str]


@dataclass(frozen=True)
class EntityRegistry:
    entities: Mapping[str, EntityRecord]
    forms: Mapping[str, EntityForm]
    classes: Mapping[str, EntityClassRecord]
    memberships: tuple[EntityClassMembership, ...]
    name_index: Mapping[str, tuple[str, ...]]
    audits: tuple[RegistryAuditRow, ...]

    def resolve_name(self, value: object) -> EntityResolution:
        key = normalize_registry_name(value)
        entity_ids = self.name_index.get(key, ())
        direct_ids = tuple(
            entity_id
            for entity_id in entity_ids
            if entity_id in self.entities
            and (
                normalize_registry_name(
                    self.entities[entity_id].canonical_name
                )
                == key
                or any(
                    normalize_registry_name(reported_name) == key
                    for reported_name in self.entities[entity_id].reported_names
                )
            )
        )
        if len(direct_ids) == 1:
            return EntityResolution("resolved", direct_ids)
        if len(direct_ids) > 1:
            return EntityResolution("ambiguous", direct_ids)
        if len(entity_ids) == 1:
            return EntityResolution("resolved", entity_ids)
        if len(entity_ids) > 1:
            return EntityResolution("ambiguous", entity_ids)
        return EntityResolution("unresolved", ())


def normalize_registry_name(value: object) -> str:
    """Normalize spacing/case while preserving chemically meaningful tokens."""

    text = unicodedata.normalize("NFKC", clean_text(value))
    text = re.sub(r"(?<=\d)-\(([RrSs])\)", r"(\1)", text)
    text = re.sub(
        r"\s+(?:CE|collision\s*energy|scan|file)\s*[:=_-]?\s*\d+(?:\.\d+)?(?:\s*eV)?$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(text.casefold().split())


def registry_name_quality_issue(
    value: object,
    compound_class: object = "",
) -> str:
    """Return a conservative, class-agnostic entity-name quality issue.

    Chemical modifiers are preserved.  A name is quarantined only when it is
    demonstrably not a uniquely specified molecule: a stereochemical/locant
    prefix followed solely by the reported class label, or mixed Latin and
    Cyrillic text typical of encoding corruption.  Formula equality is never
    used to repair or select an identity.
    """

    raw = unicodedata.normalize("NFKC", clean_text(value))
    if not raw:
        return "missing_name"
    has_latin = any(
        "LATIN" in unicodedata.name(character, "")
        for character in raw
        if character.isalpha()
    )
    has_cyrillic = any(
        "CYRILLIC" in unicodedata.name(character, "")
        for character in raw
        if character.isalpha()
    )
    if has_latin and has_cyrillic:
        return "mixed_latin_cyrillic_name"

    class_name = normalize_registry_name(compound_class)
    if not class_name:
        return ""
    normalized = normalize_registry_name(raw)
    # Examples: 20(R)-Class, 20-(S)-Class, 3,20(R)-Class.  This deliberately
    # does not strip alpha/beta/iso/neo/decarboxy or named substituents.
    stripped = re.sub(
        r"^(?:(?:\d+(?:,\d+)*)?(?:\([rs]\))[-\s]*|\d+(?:,\d+)*[-\s]*)+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    if normalized == class_name or stripped == class_name:
        return "underspecified_compound_class_label"
    return ""


def registry_name_keys(value: object) -> tuple[str, ...]:
    """Return evidence-explicit name keys without stripping chemistry tokens."""

    full = normalize_registry_name(value)
    if not full:
        return ()
    keys = [full]
    match = re.fullmatch(r"(.+?)\s+\(([^()]*)\)\s*", full)
    if not match:
        return tuple(keys)
    outer = normalize_registry_name(match.group(1))
    inner = normalize_registry_name(match.group(2))
    if len(re.findall(r"[^\W\d_]", inner, flags=re.UNICODE)) < 3:
        return tuple(keys)
    for candidate in (
        outer,
        inner,
        *(
            normalize_registry_name(item)
            for item in re.split(r"\s*/\s*|\s*;\s*", inner)
        ),
    ):
        if candidate and candidate not in keys:
            keys.append(candidate)
    return tuple(keys)


def _claim_evidence_ids(
    claim: Mapping[str, object],
    inventory: EvidenceInventory,
) -> tuple[str, ...]:
    claim_id = clean_text(claim.get("claim_id"))
    chunk_ids = {clean_text(claim.get("chunk_id"))}
    source_ids = claim.get("source_chunk_ids")
    if isinstance(source_ids, (list, tuple)):
        chunk_ids.update(clean_text(item) for item in source_ids if clean_text(item))
    return tuple(
        sorted(
            row.evidence_id
            for row in inventory.evidence
            if row.chunk_id in chunk_ids
            and (
                not claim_id
                or row.evidence_quote
                == clean_text(claim.get("evidence_quote") or claim.get("evidence_sentence"))
            )
        )
    )


def _formula_mass(formula: str) -> float | None:
    if not formula:
        return None
    try:
        parse_formula(formula)
        return formula_exact_mass(formula)
    except ChemicalConsistencyError:
        return None


def _entity_name(claim: Mapping[str, object]) -> str:
    return clean_text(
        claim.get("entity_name")
        or claim.get("compound_name")
        or claim.get("component_name")
        or claim.get("precursor_name")
    )


def _entity_id(claim: Mapping[str, object], name: str) -> str:
    supplied = clean_text(claim.get("entity_id"))
    if supplied:
        return supplied
    return stable_registry_id(
        "entity",
        {
            "compound_class": clean_text(claim.get("compound_class")).casefold(),
            "reported_name": name.casefold(),
        },
    )


def _alias_prefix_suffix(value: object) -> tuple[str, str] | None:
    """Split a simple reported series label without discarding modifiers."""

    text = normalize_registry_name(value)
    match = re.fullmatch(r"([a-z][a-z0-9]{0,30})(?:-|\s+)([a-z]+\d*[a-z]?)", text)
    if not match:
        return None
    return match.group(1), match.group(2)


def _series_signature_for_prefix(
    value: object,
    prefix: str,
) -> tuple[str, str] | None:
    """Return modifier and suffix for an evidence-learned series prefix."""

    text = normalize_registry_name(value)
    match = re.fullmatch(
        rf"(?P<modifier>.*?)(?<![a-z0-9]){re.escape(prefix)}"
        rf"(?:-|\s+)(?P<suffix>[a-z]+\d*[a-z]?)",
        text,
    )
    if not match:
        return None
    modifier = " ".join(re.findall(r"[a-z0-9]+", match.group("modifier")))
    return modifier, match.group("suffix")


def _evidence_derived_alias_id_map(
    claims: Sequence[Mapping[str, object]],
    inventory: EvidenceInventory,
    *,
    minimum_series_support: int = 2,
) -> tuple[dict[str, str], tuple[RegistryAuditRow, ...]]:
    """Learn abbreviation/full-prefix aliases from repeated reported series.

    A pair is eligible only when at least two distinct suffixes establish the
    same short/full prefix convention and each paired entity has the same
    reported formula.  Formula equality alone never creates an alias.
    """

    all_records: list[dict[str, str]] = []
    records: list[dict[str, str]] = []
    for claim in claims:
        if not _is_entity_claim(claim):
            continue
        name = _entity_name(claim)
        formula = clean_text(claim.get("formula"))
        if not name or not formula:
            continue
        if not _claim_evidence_ids(claim, inventory):
            continue
        common = {
            "entity_id": _entity_id(claim, name),
            "name": name,
            "formula": formula,
            "compound_class": clean_text(
                claim.get("compound_class")
            ).casefold(),
            "evidence_quote": clean_text(claim.get("evidence_quote")),
        }
        all_records.append(common)
        parsed = _alias_prefix_suffix(name)
        if parsed is not None:
            prefix, suffix = parsed
            records.append(
                {
                    **common,
                "prefix": prefix,
                "suffix": suffix,
                }
            )

    by_suffix: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in records:
        by_suffix.setdefault(
            (row["compound_class"], row["suffix"], row["formula"]), []
        ).append(row)
    prefix_pair_suffixes: dict[tuple[str, str, str], set[str]] = {}
    compatible_pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    for (compound_class, suffix, _), rows in by_suffix.items():
        for left in rows:
            for right in rows:
                if left["entity_id"] == right["entity_id"]:
                    continue
                short, long = sorted(
                    (left, right), key=lambda item: len(item["prefix"])
                )
                if (
                    len(short["prefix"]) > 4
                    or len(long["prefix"]) <= len(short["prefix"])
                    or long["prefix"][0] != short["prefix"][0]
                ):
                    continue
                key = (compound_class, short["prefix"], long["prefix"])
                prefix_pair_suffixes.setdefault(key, set()).add(suffix)
                compatible_pairs.append((short, long))

    aliases: dict[str, str] = {}
    audits: list[RegistryAuditRow] = []
    exact_name_groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in all_records:
        exact_name_groups.setdefault(
            (
                row["compound_class"],
                row["formula"],
                normalize_registry_name(row["name"]),
            ),
            [],
        ).append(row)
    for rows in exact_name_groups.values():
        by_id = {row["entity_id"]: row for row in rows}
        if len(by_id) <= 1:
            continue
        canonical = min(
            by_id.values(),
            key=lambda row: (
                unicodedata.normalize("NFKC", row["name"]) != row["name"],
                row["entity_id"],
            ),
        )
        for row in by_id.values():
            if row["entity_id"] == canonical["entity_id"]:
                continue
            aliases[row["entity_id"]] = canonical["entity_id"]
            audits.append(
                RegistryAuditRow(
                    "entity_alias",
                    row["entity_id"],
                    "alias_collapsed",
                    f"{row['name']} -> {canonical['name']}; compatibility-equivalent reported name",
                )
            )
    accepted_prefix_pairs = {
        key
        for key, suffixes in prefix_pair_suffixes.items()
        if len(suffixes) >= minimum_series_support
    }
    expanded_pairs = list(compatible_pairs)
    for compound_class, short_prefix, long_prefix in accepted_prefix_pairs:
        short_groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        long_groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        for row in all_records:
            if row["compound_class"] != compound_class:
                continue
            short_signature = _series_signature_for_prefix(
                row["name"], short_prefix
            )
            if short_signature is not None:
                modifier, suffix = short_signature
                short_groups.setdefault(
                    (row["formula"], modifier, suffix), []
                ).append(row)
            long_signature = _series_signature_for_prefix(
                row["name"], long_prefix
            )
            if long_signature is not None:
                modifier, suffix = long_signature
                long_groups.setdefault(
                    (row["formula"], modifier, suffix), []
                ).append(row)
        for signature in sorted(set(short_groups) & set(long_groups)):
            short_rows = {
                row["entity_id"]: row for row in short_groups[signature]
            }
            long_rows = {
                row["entity_id"]: row for row in long_groups[signature]
            }
            if len(short_rows) == 1 and len(long_rows) == 1:
                expanded_pairs.append(
                    (next(iter(short_rows.values())), next(iter(long_rows.values())))
                )

    for short, long in expanded_pairs:
        key = (
            short["compound_class"],
            _alias_prefix_suffix(short["name"])[0]
            if _alias_prefix_suffix(short["name"])
            else "",
            _alias_prefix_suffix(long["name"])[0]
            if _alias_prefix_suffix(long["name"])
            else "",
        )
        if key not in accepted_prefix_pairs and not any(
            _series_signature_for_prefix(short["name"], short_prefix)
            and _series_signature_for_prefix(long["name"], long_prefix)
            for group_class, short_prefix, long_prefix in accepted_prefix_pairs
            if group_class == short["compound_class"]
        ):
            continue
        existing = aliases.get(short["entity_id"])
        if existing and existing != long["entity_id"]:
            continue
        aliases[short["entity_id"]] = long["entity_id"]
        audits.append(
            RegistryAuditRow(
                "entity_alias",
                short["entity_id"],
                "alias_collapsed",
                f"{short['name']} -> {long['name']}; evidence-derived series prefix",
            )
        )

    grouped_records: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in all_records:
        grouped_records.setdefault(
            (row["compound_class"], row["formula"]), []
        ).append(row)
    identifier_suffix = re.compile(
        r"(?:[a-z]{1,6}\d+[a-z0-9-]*|[ivxlcdm]+)",
        re.IGNORECASE,
    )
    for rows in grouped_records.values():
        unique_rows = {
            row["entity_id"]: row for row in rows
        }
        for short in unique_rows.values():
            short_name = normalize_registry_name(short["name"])
            quote = normalize_registry_name(short.get("evidence_quote", ""))
            if not short_name or not quote:
                continue
            candidates: list[dict[str, str]] = []
            for long in unique_rows.values():
                if long["entity_id"] == short["entity_id"]:
                    continue
                long_name = normalize_registry_name(long["name"])
                if not long_name.startswith(short_name + " "):
                    continue
                suffix = long_name[len(short_name) :].strip()
                if not identifier_suffix.fullmatch(suffix):
                    continue
                if long_name not in quote:
                    continue
                candidates.append(long)
            candidate_ids = {
                aliases.get(row["entity_id"], row["entity_id"])
                for row in candidates
            }
            if len(candidate_ids) != 1:
                continue
            canonical_id = next(iter(candidate_ids))
            long = next(
                (
                    row
                    for row in candidates
                    if row["entity_id"] == canonical_id
                ),
                candidates[0],
            )
            existing = aliases.get(short["entity_id"])
            if existing and existing != canonical_id:
                continue
            aliases[short["entity_id"]] = canonical_id
            audits.append(
                RegistryAuditRow(
                    "entity_alias",
                    short["entity_id"],
                    "alias_collapsed",
                    f"{short['name']} -> {long['name']}; evidence-reported identifier suffix",
                )
            )
    for entity_id in tuple(aliases):
        target = aliases[entity_id]
        seen = {entity_id}
        while target in aliases and target not in seen:
            seen.add(target)
            target = aliases[target]
        aliases[entity_id] = target
    return aliases, tuple(
        dict.fromkeys(audits)
    )


def _formula_source_priority(claim: Mapping[str, object]) -> int:
    """Rank explicit table formula sources without using class knowledge."""

    source_structure = clean_text(claim.get("source_structure")).casefold()
    if source_structure in {"supplementary_catalog", "supplementary_table"}:
        return 4
    if source_structure in {
        "structured_table",
        "compound_identification_table",
        "fragmentation_table",
    }:
        return 3
    if source_structure:
        return 2
    return 1


def _preferred_entity_formulas(
    claims: Sequence[Mapping[str, object]],
    inventory: EvidenceInventory,
) -> tuple[dict[str, str], set[str]]:
    """Choose a deterministic evidence consensus for conflicting formulas."""

    candidates: dict[str, dict[str, dict[str, object]]] = {}
    for index, claim in enumerate(claims):
        if not _is_entity_claim(claim):
            continue
        name = _entity_name(claim)
        formula = clean_text(claim.get("formula"))
        if not name or not formula or _formula_mass(formula) is None:
            continue
        evidence_ids = _claim_evidence_ids(claim, inventory)
        if not evidence_ids:
            continue
        entity_id = _entity_id(claim, name)
        row = candidates.setdefault(entity_id, {}).setdefault(
            formula,
            {
                "evidence_ids": set(),
                "max_priority": 0,
                "priority_sum": 0,
                "first_index": index,
            },
        )
        row["evidence_ids"].update(evidence_ids)
        priority = _formula_source_priority(claim)
        row["max_priority"] = max(int(row["max_priority"]), priority)
        row["priority_sum"] = int(row["priority_sum"]) + priority
        row["first_index"] = min(int(row["first_index"]), index)

    preferred: dict[str, str] = {}
    conflicts: set[str] = set()
    for entity_id, formula_rows in candidates.items():
        if len(formula_rows) > 1:
            conflicts.add(entity_id)
        preferred[entity_id] = max(
            formula_rows,
            key=lambda formula: (
                len(formula_rows[formula]["evidence_ids"]),
                int(formula_rows[formula]["max_priority"]),
                int(formula_rows[formula]["priority_sum"]),
                -int(formula_rows[formula]["first_index"]),
            ),
        )
    return preferred, conflicts


def _is_entity_claim(claim: Mapping[str, object]) -> bool:
    return clean_text(claim.get("claim_type")).casefold() in {
        "entity",
        "compound",
        "precursor",
        "structural_component",
        "biosynthetic_component",
    }


def _reaction_participant_entity_claims(
    claims: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Project evidence-backed specific participants into registry entity claims."""

    projected: list[dict[str, object]] = []
    for claim in claims:
        if clean_text(claim.get("claim_type")).casefold() != "reaction_template":
            continue
        if clean_text(claim.get("structured_validation_status")).casefold() == "invalid":
            continue
        for field in ("reactants", "products"):
            participants = claim.get(field)
            if not isinstance(participants, (list, tuple)):
                continue
            for participant in participants:
                if not isinstance(participant, Mapping):
                    continue
                if clean_text(participant.get("scope")).casefold() != "specific_entity":
                    continue
                name = clean_text(participant.get("entity_name"))
                entity_id = clean_text(participant.get("entity_id"))
                if not name or not entity_id:
                    continue
                projected.append(
                    {
                        **claim,
                        "claim_type": "entity",
                        "entity_id": entity_id,
                        "entity_name": name,
                        "entity_kind": "molecule",
                        "formula": clean_text(participant.get("formula")),
                        "exact_mass": participant.get("exact_mass"),
                        "reported_names": [name],
                    }
                )
    return tuple(projected)


def _membership_stub_entity_claims(
    claims: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Recover a concrete entity when a membership claim reports a name, not an opaque ID."""

    existing_names = {
        normalize_registry_name(_entity_name(claim))
        for claim in claims
        if _is_entity_claim(claim) and _entity_name(claim)
    }
    projected: list[dict[str, object]] = []
    for claim in claims:
        if clean_text(claim.get("claim_type")).casefold() != "entity_class_membership":
            continue
        reported_name = clean_text(claim.get("entity_name"))
        supplied = clean_text(claim.get("entity_id"))
        if not reported_name and supplied and not re.fullmatch(
            r"(?:entity|compound|cmp|component|precursor|donor)_[A-Za-z0-9]+",
            supplied,
            flags=re.IGNORECASE,
        ):
            reported_name = supplied
        if not reported_name:
            continue
        if normalize_registry_name(reported_name) in existing_names:
            continue
        projected.append(
            {
                **claim,
                "claim_type": "entity",
                "entity_id": stable_registry_id(
                    "entity",
                    {
                        "compound_class": clean_text(
                            claim.get("compound_class")
                        ).casefold(),
                        "reported_name": reported_name.casefold(),
                    },
                ),
                "entity_name": reported_name,
                "entity_kind": "molecule",
                "formula": "",
                "reported_names": [reported_name],
            }
        )
    return tuple(projected)


def build_entity_registry(
    claims: Sequence[Mapping[str, object]],
    inventory: EvidenceInventory,
) -> EntityRegistry:
    entities: dict[str, EntityRecord] = {}
    forms: dict[str, EntityForm] = {}
    classes: dict[str, EntityClassRecord] = {}
    memberships: dict[str, EntityClassMembership] = {}
    audits: list[RegistryAuditRow] = []

    entity_claims = (
        tuple(claims)
        + _reaction_participant_entity_claims(claims)
        + _membership_stub_entity_claims(claims)
    )
    accepted_entity_claims: list[Mapping[str, object]] = []
    quarantined_records: set[tuple[str, str]] = set()
    for claim in entity_claims:
        if not _is_entity_claim(claim):
            accepted_entity_claims.append(claim)
            continue
        name = _entity_name(claim)
        issue = registry_name_quality_issue(
            name,
            claim.get("compound_class"),
        )
        if not issue:
            accepted_entity_claims.append(claim)
            continue
        record_id = _entity_id(claim, name)
        audit_key = (record_id, issue)
        if audit_key not in quarantined_records:
            quarantined_records.add(audit_key)
            audits.append(
                RegistryAuditRow(
                    "entity",
                    record_id,
                    "quarantined",
                    issue,
                )
            )
    entity_claims = tuple(accepted_entity_claims)
    alias_id_map, alias_audits = _evidence_derived_alias_id_map(
        entity_claims,
        inventory,
    )
    audits.extend(alias_audits)
    entity_claims = tuple(
        {
            **dict(claim),
            "entity_id": alias_id_map.get(
                _entity_id(claim, _entity_name(claim)),
                _entity_id(claim, _entity_name(claim)),
            ),
        }
        for claim in entity_claims
    )
    preferred_formulas, formula_conflicts = _preferred_entity_formulas(
        entity_claims,
        inventory,
    )
    for entity_id in sorted(formula_conflicts):
        audits.append(
            RegistryAuditRow(
                "entity",
                entity_id,
                "conflict",
                "entity_formula_conflict",
            )
        )
    for claim in entity_claims:
        if not _is_entity_claim(claim):
            continue
        name = _entity_name(claim)
        if not name:
            audits.append(RegistryAuditRow("entity", clean_text(claim.get("claim_id")), "rejected", "missing_name"))
            continue
        entity_id = _entity_id(claim, name)
        evidence_ids = _claim_evidence_ids(claim, inventory)
        if not evidence_ids:
            audits.append(RegistryAuditRow("entity", entity_id, "rejected", "missing_accepted_evidence"))
            continue
        entity_kind = clean_text(claim.get("entity_kind")).casefold() or (
            "moiety"
            if clean_text(claim.get("claim_type")).casefold() == "structural_component"
            else "molecule"
        )
        if entity_kind not in ENTITY_KINDS:
            audits.append(RegistryAuditRow("entity", entity_id, "rejected", "invalid_entity_kind"))
            continue
        formula = preferred_formulas.get(
            entity_id,
            clean_text(claim.get("formula")),
        )
        calculated_mass = _formula_mass(formula)
        names = tuple(
            dict.fromkeys(
                (
                    name,
                    *split_text_values(claim.get("reported_names")),
                    *split_text_values(claim.get("synonyms")),
                )
            )
        )
        modes = tuple(
            item
            for item in split_text_values(claim.get("ion_mode"))
            if item.casefold() in {"positive", "negative", "not_reported"}
        )
        incoming = EntityRecord(
            entity_id=entity_id,
            canonical_name=name,
            reported_names=names,
            entity_kind=entity_kind,
            compound_class=clean_text(claim.get("compound_class")),
            formula=formula,
            exact_mass=calculated_mass,
            ion_modes=tuple(item.casefold() for item in modes),
            evidence_ids=evidence_ids,
        )
        existing = entities.get(entity_id)
        if existing and existing.formula and incoming.formula and existing.formula != incoming.formula:
            audits.append(RegistryAuditRow("entity", entity_id, "conflict", "entity_formula_conflict"))
            continue
        if existing:
            entities[entity_id] = EntityRecord(
                entity_id=entity_id,
                canonical_name=existing.canonical_name,
                reported_names=tuple(dict.fromkeys(existing.reported_names + incoming.reported_names)),
                entity_kind=existing.entity_kind,
                compound_class=existing.compound_class or incoming.compound_class,
                formula=existing.formula or incoming.formula,
                exact_mass=existing.exact_mass if existing.exact_mass is not None else incoming.exact_mass,
                ion_modes=tuple(dict.fromkeys(existing.ion_modes + incoming.ion_modes)),
                evidence_ids=tuple(sorted(set(existing.evidence_ids + incoming.evidence_ids))),
            )
        else:
            entities[entity_id] = incoming

    name_to_entity_ids: dict[str, set[str]] = {}
    for entity in entities.values():
        for reported_name in (entity.canonical_name, *entity.reported_names):
            for normalized_name in registry_name_keys(reported_name):
                name_to_entity_ids.setdefault(normalized_name, set()).add(
                    entity.entity_id
                )

    for claim in claims:
        claim_type = clean_text(claim.get("claim_type")).casefold()
        evidence_ids = _claim_evidence_ids(claim, inventory)
        if claim_type == "entity_form":
            entity_id = alias_id_map.get(
                clean_text(claim.get("entity_id")),
                clean_text(claim.get("entity_id")),
            )
            form_type = clean_text(claim.get("form_type")).casefold()
            formula_origin = clean_text(claim.get("formula_origin")).casefold()
            formula = clean_text(claim.get("formula"))
            form_id = clean_text(claim.get("form_id")) or stable_registry_id(
                "form",
                {
                    "entity_id": entity_id,
                    "form_type": form_type,
                    "formula": formula,
                    "reaction_operator": clean_text(claim.get("reaction_operator")),
                },
            )
            if (
                entity_id not in entities
                or form_type not in FORM_TYPES
                or formula_origin not in FORMULA_ORIGINS
                or not evidence_ids
                or _formula_mass(formula) is None
            ):
                audits.append(RegistryAuditRow("form", form_id, "rejected", "invalid_entity_form"))
                continue
            forms[form_id] = EntityForm(
                form_id=form_id,
                entity_id=entity_id,
                form_type=form_type,
                formula=formula,
                exact_mass=_formula_mass(formula),
                formula_origin=formula_origin,
                reaction_operator=clean_text(claim.get("reaction_operator")),
                evidence_ids=evidence_ids,
            )
        elif claim_type == "entity_class_membership":
            entity_id = alias_id_map.get(
                clean_text(claim.get("entity_id")),
                clean_text(claim.get("entity_id")),
            )
            if entity_id not in entities:
                resolved_ids = name_to_entity_ids.get(
                    normalize_registry_name(entity_id), set()
                )
                if len(resolved_ids) == 1:
                    entity_id = next(iter(resolved_ids))
            class_id = clean_text(claim.get("entity_class_id"))
            role = clean_text(claim.get("membership_role") or claim.get("role"))
            if entity_id not in entities or not class_id or not role or not evidence_ids:
                audits.append(
                    RegistryAuditRow(
                        "membership",
                        clean_text(claim.get("claim_id")),
                        "rejected",
                        "invalid_entity_class_membership",
                    )
                )
                continue
            classes.setdefault(
                class_id,
                EntityClassRecord(
                    entity_class_id=class_id,
                    class_label=clean_text(claim.get("entity_class_name")) or class_id,
                    evidence_ids=evidence_ids,
                ),
            )
            membership_id = clean_text(claim.get("membership_id")) or stable_registry_id(
                "membership",
                {"entity_id": entity_id, "entity_class_id": class_id, "role": role},
            )
            memberships[membership_id] = EntityClassMembership(
                membership_id=membership_id,
                entity_id=entity_id,
                entity_class_id=class_id,
                role=role,
                evidence_ids=evidence_ids,
            )

    for entity in tuple(entities.values()):
        if entity.entity_kind == "molecule" and entity.formula:
            form_type = (
                "reported_ion"
                if re.search(r"(?:\^\d+[+-]|[+-])$", entity.formula)
                else "neutral_molecule"
            )
            form_id = stable_registry_id(
                "form",
                {
                    "entity_id": entity.entity_id,
                    "form_type": form_type,
                    "formula": entity.formula,
                },
            )
            forms.setdefault(
                form_id,
                EntityForm(
                    form_id=form_id,
                    entity_id=entity.entity_id,
                    form_type=form_type,
                    formula=entity.formula,
                    exact_mass=entity.exact_mass,
                    formula_origin="reported",
                    reaction_operator="",
                    evidence_ids=entity.evidence_ids,
                ),
            )

    name_groups: dict[str, set[str]] = {}
    for entity in entities.values():
        for name in entity.reported_names:
            for key in registry_name_keys(name):
                name_groups.setdefault(key, set()).add(entity.entity_id)
    name_index = {key: tuple(sorted(values)) for key, values in sorted(name_groups.items())}
    return EntityRegistry(
        entities=dict(sorted(entities.items())),
        forms=dict(sorted(forms.items())),
        classes=dict(sorted(classes.items())),
        memberships=tuple(memberships[key] for key in sorted(memberships)),
        name_index=name_index,
        audits=tuple(audits),
    )


def _jsonl_bytes(rows: Sequence[object]) -> bytes:
    return "".join(
        json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def write_entity_registry(registry: EntityRegistry, output_dir: Path) -> RegistryManifest:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "entity_registry.jsonl": _jsonl_bytes(tuple(registry.entities.values())),
        "entity_forms.jsonl": _jsonl_bytes(tuple(registry.forms.values())),
        "entity_classes.jsonl": _jsonl_bytes(tuple(registry.classes.values())),
        "entity_class_memberships.jsonl": _jsonl_bytes(registry.memberships),
    }
    hashes: dict[str, str] = {}
    for filename, payload in artifacts.items():
        (output_dir / filename).write_bytes(payload)
        hashes[filename] = hashlib.sha256(payload).hexdigest()
    with (output_dir / "entity_resolution_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["record_type", "record_id", "status", "detail"]
        )
        writer.writeheader()
        for row in registry.audits:
            writer.writerow(asdict(row))
    manifest = RegistryManifest("1.0", hashes)
    (output_dir / "registry_manifest.json").write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_entity_registry(root: Path) -> EntityRegistry:
    def read_jsonl(filename: str) -> list[dict[str, Any]]:
        path = root / filename
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    entities = {row["entity_id"]: EntityRecord(**row) for row in read_jsonl("entity_registry.jsonl")}
    forms = {row["form_id"]: EntityForm(**row) for row in read_jsonl("entity_forms.jsonl")}
    classes = {
        row["entity_class_id"]: EntityClassRecord(**row)
        for row in read_jsonl("entity_classes.jsonl")
    }
    memberships = tuple(
        EntityClassMembership(**row) for row in read_jsonl("entity_class_memberships.jsonl")
    )
    name_groups: dict[str, set[str]] = {}
    for entity in entities.values():
        for name in entity.reported_names:
            for key in registry_name_keys(name):
                name_groups.setdefault(key, set()).add(entity.entity_id)
    return EntityRegistry(
        entities=entities,
        forms=forms,
        classes=classes,
        memberships=memberships,
        name_index={key: tuple(sorted(values)) for key, values in name_groups.items()},
        audits=(),
    )
