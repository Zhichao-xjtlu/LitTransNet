#!/usr/bin/env python3
"""Infer candidate transformation claims from literature-reported compounds.

This module does not call an LLM, does not search the web, and does not invent
compounds. It only compares compounds already present in local evidence claims
and emits review-only transformation hypotheses when a universal mass delta is
chemically consistent with the reported masses/formulas and compound names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.core.knowledge_enrichment import formula_mass
from rag.core.io_utils import (
    atomic_write_json as write_json,
    clean_text,
    join_unique,
    safe_float,
    split_values,
)

PROTON_MASS = 1.007276466812
SODIUM_ADDUCT_MASS = 22.989218
AMMONIUM_ADDUCT_MASS = 18.033823

UNIVERSAL_TRANSFORMATION_DELTAS: dict[float, str] = {
    0.0000: "isomerization",
    -2.0157: "dehydrogenation_or_oxidation_like",
    2.0157: "hydrogenation_or_reduction_like",
    -43.9898: "decarboxylation_CO2_loss",
    43.9898: "carboxylation_CO2_gain",
    -18.0106: "dehydration_H2O_loss",
    18.0106: "hydration_H2O_gain",
    79.9568: "sulfation_SO3_gain",
    -79.9568: "desulfation_SO3_loss",
    162.0528: "hexosylation",
    -162.0528: "hexose_loss",
    176.0321: "glucuronidation_or_hexuronic_acid_addition",
    -176.0321: "hexuronic_acid_loss",
    14.0157: "methylation_or_CH2_gain",
    -14.0157: "demethylation_or_CH2_loss",
    42.0106: "acetylation_like",
    -42.0106: "deacetylation_like",
    86.0004: "malonylation_like",
    -86.0004: "demalonylation_like",
}

NAME_MODIFIERS = {
    "iso",
    "neo",
    "decarboxy",
    "bidecarboxy",
    "tridecarboxy",
    "dehydro",
    "hydroxy",
    "oxo",
    "glycosyl",
    "glucosyl",
    "sulfated",
    "sulfate",
    "sulfo",
    "malonyl",
    "acetyl",
    "methyl",
    "demethyl",
}


class LiteratureInferenceError(RuntimeError):
    """Raised when literature-supported inference cannot run."""


@dataclass
class CompoundCandidate:
    compound_name: str
    compound_class: str
    subclass: str = ""
    formula: str = ""
    exact_mass: float | None = None
    mass_source: str = ""
    source_file: str = ""
    chunk_id: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    evidence_summary: str = ""

    @property
    def stem(self) -> str:
        return normalized_name_stem(self.compound_name)

    @property
    def modifiers(self) -> set[str]:
        return name_modifiers(self.compound_name)


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def format_float(value: Any, digits: int = 4) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    text = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def stable_id(prefix: str, *parts: Any) -> str:
    basis = "|".join(clean_text(part).lower() for part in parts)
    return f"{prefix}_" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def claim_evidence_ids(claim: dict[str, Any]) -> list[str]:
    ids = split_values(claim.get("evidence_ids"))
    if not ids:
        ids = split_values(claim.get("claim_id"))
    if not ids:
        ids = split_values(claim.get("chunk_id"))
    return ids


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise LiteratureInferenceError(f"Evidence claims file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise LiteratureInferenceError(f"Invalid JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def neutral_mass_from_adduct(precursor_mz: Any, adduct: Any) -> tuple[float | None, str]:
    mz = safe_float(precursor_mz)
    text = clean_text(adduct).replace(" ", "")
    if mz is None or not text:
        return None, ""
    if text == "[M+H]+":
        return mz - PROTON_MASS, "precursor_adduct"
    if text == "[M-H]-":
        return mz + PROTON_MASS, "precursor_adduct"
    if text == "[M+Na]+":
        return mz - SODIUM_ADDUCT_MASS, "precursor_adduct"
    if text == "[M+NH4]+":
        return mz - AMMONIUM_ADDUCT_MASS, "precursor_adduct"
    return None, ""


def claim_mass(claim: dict[str, Any]) -> tuple[float | None, str]:
    exact = safe_float(claim.get("exact_mass"))
    if exact is not None:
        return exact, "exact_mass"
    calculated = formula_mass(clean_text(claim.get("formula")))
    if calculated is not None:
        return calculated, "formula"
    inferred, source = neutral_mass_from_adduct(
        claim.get("reported_precursor_mz") or claim.get("precursor_mz"),
        claim.get("adduct"),
    )
    return inferred, source


def normalized_name_tokens(name: str) -> list[str]:
    text = clean_text(name).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("β", "beta").replace("α", "alpha")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return [token for token in text.split() if token]


def strip_modifier_prefixes(token: str) -> str:
    changed = True
    current = token
    while changed and current:
        changed = False
        for modifier in sorted(NAME_MODIFIERS, key=len, reverse=True):
            if current == modifier:
                return ""
            if current.startswith(modifier) and len(current) > len(modifier) + 2:
                current = current[len(modifier) :]
                changed = True
                break
    return current


def normalized_name_stem(name: str) -> str:
    tokens: list[str] = []
    for token in normalized_name_tokens(name):
        stripped = strip_modifier_prefixes(token)
        if stripped and stripped not in NAME_MODIFIERS and not stripped.isdigit():
            tokens.append(stripped)
    return " ".join(tokens)


def name_modifiers(name: str) -> set[str]:
    modifiers: set[str] = set()
    for token in normalized_name_tokens(name):
        for modifier in NAME_MODIFIERS:
            if token == modifier or token.startswith(modifier):
                modifiers.add(modifier)
    return modifiers


def build_compound_candidates(claims: list[dict[str, Any]]) -> list[CompoundCandidate]:
    merged: dict[tuple[str, str], CompoundCandidate] = {}
    for claim in claims:
        if clean_text(claim.get("claim_type")) != "compound":
            continue
        name = clean_text(claim.get("compound_name"))
        compound_class = clean_text(claim.get("compound_class"))
        if not name:
            continue
        mass, mass_source = claim_mass(claim)
        key = (compound_class.lower(), name.lower())
        evidence_ids = claim_evidence_ids(claim)
        if key not in merged:
            merged[key] = CompoundCandidate(
                compound_name=name,
                compound_class=compound_class,
                subclass=clean_text(claim.get("subclass")),
                formula=clean_text(claim.get("formula")),
                exact_mass=mass,
                mass_source=mass_source,
                source_file=clean_text(claim.get("source_file")),
                chunk_id=clean_text(claim.get("chunk_id")),
                evidence_ids=evidence_ids,
                evidence_summary=clean_text(claim.get("evidence_summary")),
            )
            continue
        existing = merged[key]
        existing.evidence_ids = split_values(join_unique([";".join(existing.evidence_ids), ";".join(evidence_ids)]))
        if not existing.subclass:
            existing.subclass = clean_text(claim.get("subclass"))
        if not existing.formula:
            existing.formula = clean_text(claim.get("formula"))
        if existing.exact_mass is None and mass is not None:
            existing.exact_mass = mass
            existing.mass_source = mass_source
        if not existing.source_file:
            existing.source_file = clean_text(claim.get("source_file"))
        if not existing.chunk_id:
            existing.chunk_id = clean_text(claim.get("chunk_id"))
        existing.evidence_summary = join_unique([existing.evidence_summary, claim.get("evidence_summary")])
    return list(merged.values())


def same_table_or_source(a: CompoundCandidate, b: CompoundCandidate) -> bool:
    return bool(a.source_file and b.source_file and a.source_file == b.source_file)


def same_chunk(a: CompoundCandidate, b: CompoundCandidate) -> bool:
    return bool(a.chunk_id and b.chunk_id and a.chunk_id == b.chunk_id)


def share_subclass(a: CompoundCandidate, b: CompoundCandidate) -> bool:
    return bool(a.subclass and b.subclass and a.subclass.lower() == b.subclass.lower())


def similar_name_stem(a: CompoundCandidate, b: CompoundCandidate) -> bool:
    if not a.stem or not b.stem:
        return False
    return a.stem == b.stem or a.stem in b.stem or b.stem in a.stem


def pair_selection_reasons(a: CompoundCandidate, b: CompoundCandidate) -> list[str]:
    reasons: list[str] = []
    if same_table_or_source(a, b):
        reasons.append("same_source_file")
    if same_chunk(a, b):
        reasons.append("same_chunk")
    if share_subclass(a, b):
        reasons.append("same_subclass")
    if similar_name_stem(a, b):
        reasons.append("similar_name_stem")
    return reasons


def mass_tolerance(a: CompoundCandidate, b: CompoundCandidate) -> float:
    if a.mass_source == "precursor_adduct" or b.mass_source == "precursor_adduct":
        return 0.05
    if a.mass_source == "formula" and b.mass_source == "formula":
        return 0.01
    return 0.02


def match_universal_delta(delta: float, tolerance: float) -> tuple[float | None, str, float]:
    best_mass: float | None = None
    best_name = ""
    best_error = tolerance
    for expected, label in UNIVERSAL_TRANSFORMATION_DELTAS.items():
        error = abs(delta - expected)
        if error <= best_error:
            best_mass = expected
            best_name = label
            best_error = error
    return best_mass, best_name, best_error


def formula_delta_consistent(a: CompoundCandidate, b: CompoundCandidate, expected_delta: float) -> bool:
    mass_a = formula_mass(a.formula)
    mass_b = formula_mass(b.formula)
    if mass_a is None or mass_b is None:
        return False
    return abs((mass_b - mass_a) - expected_delta) <= 0.02


def name_pattern_supports(source: CompoundCandidate, target: CompoundCandidate, transformation_name: str) -> bool:
    source_mods = source.modifiers
    target_mods = target.modifiers
    if transformation_name == "isomerization":
        return bool(source_mods.symmetric_difference(target_mods) & {"iso"}) or similar_name_stem(source, target)
    if transformation_name == "decarboxylation_CO2_loss":
        return bool(target_mods & {"decarboxy", "bidecarboxy", "tridecarboxy"})
    if transformation_name == "carboxylation_CO2_gain":
        return bool(target_mods & {"carboxy"})
    if transformation_name == "dehydrogenation_or_oxidation_like":
        return bool(target_mods & {"neo", "dehydro", "oxo"})
    if transformation_name == "hydrogenation_or_reduction_like":
        return False
    if transformation_name == "hexosylation":
        return bool(target_mods & {"glycosyl", "glucosyl"})
    if transformation_name == "hexose_loss":
        return bool(source_mods & {"glycosyl", "glucosyl"})
    if transformation_name == "sulfation_SO3_gain":
        return bool(target_mods & {"sulfated", "sulfate", "sulfo"})
    if transformation_name == "desulfation_SO3_loss":
        return bool(source_mods & {"sulfated", "sulfate", "sulfo"})
    if transformation_name == "methylation_or_CH2_gain":
        return bool(target_mods & {"methyl"})
    if transformation_name == "demethylation_or_CH2_loss":
        return bool(source_mods & {"methyl"} or target_mods & {"demethyl"})
    if transformation_name == "acetylation_like":
        return bool(target_mods & {"acetyl"})
    if transformation_name == "deacetylation_like":
        return bool(source_mods & {"acetyl"})
    if transformation_name == "malonylation_like":
        return bool(target_mods & {"malonyl"})
    if transformation_name == "demalonylation_like":
        return bool(source_mods & {"malonyl"})
    return False


def confidence_score(
    source: CompoundCandidate,
    target: CompoundCandidate,
    expected_delta: float,
    name_supported: bool,
    pair_reasons: list[str],
) -> tuple[float, bool]:
    score = 0.5
    if "same_source_file" in pair_reasons or "same_chunk" in pair_reasons:
        score += 0.2
    score += 0.15
    if name_supported:
        score += 0.1
    if share_subclass(source, target):
        score += 0.05
    formula_ok = formula_delta_consistent(source, target, expected_delta)
    if formula_ok:
        score += 0.05
    return min(score, 0.95), formula_ok


def orientation_preference(source: CompoundCandidate, target: CompoundCandidate) -> tuple[int, int]:
    return (-len(source.modifiers), len(target.modifiers))


def delta_direction_preference(expected_delta: float) -> int:
    return 1 if expected_delta > 0 else 0


def evaluate_orientation(
    source: CompoundCandidate,
    target: CompoundCandidate,
    pair_reasons: list[str],
) -> dict[str, Any] | None:
    if source.exact_mass is None or target.exact_mass is None:
        return None
    observed_delta = target.exact_mass - source.exact_mass
    expected_delta, transformation_name, error = match_universal_delta(observed_delta, mass_tolerance(source, target))
    if expected_delta is None:
        return None
    name_supported = name_pattern_supports(source, target, transformation_name)
    confidence, formula_ok = confidence_score(source, target, expected_delta, name_supported, pair_reasons)
    return {
        "source": source,
        "target": target,
        "observed_delta": observed_delta,
        "expected_delta": expected_delta,
        "transformation_name": transformation_name,
        "delta_error": error,
        "name_supported": name_supported,
        "formula_consistent": formula_ok,
        "confidence": confidence,
        "orientation_preference": orientation_preference(source, target),
        "delta_direction_preference": delta_direction_preference(expected_delta),
    }


def best_pair_inference(a: CompoundCandidate, b: CompoundCandidate, pair_reasons: list[str]) -> dict[str, Any] | None:
    candidates = [
        item for item in (evaluate_orientation(a, b, pair_reasons), evaluate_orientation(b, a, pair_reasons)) if item
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item["confidence"],
            1 if item["name_supported"] else 0,
            item["delta_direction_preference"],
            item["orientation_preference"][0],
            item["orientation_preference"][1],
            -abs(item["delta_error"]),
        ),
    )


def inferred_claim_from_pair(inference: dict[str, Any], pair_reasons: list[str]) -> dict[str, Any]:
    source: CompoundCandidate = inference["source"]
    target: CompoundCandidate = inference["target"]
    evidence_ids = join_unique([";".join(source.evidence_ids), ";".join(target.evidence_ids)])
    delta_text = format_float(inference["expected_delta"])
    claim_id = stable_id(
        "claim_inferred",
        source.compound_class,
        source.compound_name,
        target.compound_name,
        inference["transformation_name"],
        delta_text,
    )
    summary = (
        f"Inferred from literature-reported compounds: source={source.compound_name} "
        f"(mass={format_float(source.exact_mass)}, evidence={';'.join(source.evidence_ids)}); "
        f"target={target.compound_name} (mass={format_float(target.exact_mass)}, "
        f"evidence={';'.join(target.evidence_ids)}); observed_delta="
        f"{format_float(inference['observed_delta'])}; matched_universal_delta={delta_text} "
        f"({inference['transformation_name']}); name_pattern_supported="
        f"{bool(inference['name_supported'])}; pair_reasons={','.join(pair_reasons)}."
    )
    return {
        "claim_id": claim_id,
        "claim_type": "transformation",
        "compound_class": source.compound_class or target.compound_class,
        "subclass": source.subclass if source.subclass == target.subclass else "",
        "source_entity": source.compound_name,
        "target_entity": target.compound_name,
        "transformation_name": inference["transformation_name"],
        "delta_mass": delta_text,
        "source_exact_mass": format_float(source.exact_mass),
        "target_exact_mass": format_float(target.exact_mass),
        "evidence_ids": evidence_ids,
        "chunk_id": "",
        "source_chunk_ids": [item for item in [source.chunk_id, target.chunk_id] if item],
        "evidence_quote": "",
        "evidence_summary": summary,
        "traceability_status": "inferred_from_literature_compounds",
        "confidence": round(float(inference["confidence"]), 4),
        "review_status": "candidate",
        "direction": "source_to_target",
        "inference_basis": ";".join(pair_reasons),
    }


def infer_transformation_claims(
    claims: list[dict[str, Any]],
    min_confidence: float = 0.65,
    allow_table_delta_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    compounds = build_compound_candidates(claims)
    inferred: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    skipped = Counter()
    candidate_pair_count = 0
    delta_matched_pair_count = 0
    examples: list[dict[str, Any]] = []

    for i, first in enumerate(compounds):
        for second in compounds[i + 1 :]:
            if first.compound_name.lower() == second.compound_name.lower():
                skipped["same_compound"] += 1
                continue
            pair_reasons = pair_selection_reasons(first, second)
            if not pair_reasons:
                skipped["not_related_by_source_subclass_or_name"] += 1
                continue
            candidate_pair_count += 1
            if first.exact_mass is None or second.exact_mass is None:
                skipped["missing_mass"] += 1
                continue
            inference = best_pair_inference(first, second, pair_reasons)
            if inference is None:
                skipped["no_universal_delta_match"] += 1
                continue
            delta_matched_pair_count += 1
            specific_relation = (
                bool(inference["name_supported"])
                or similar_name_stem(first, second)
                or share_subclass(first, second)
            )
            if not allow_table_delta_only and not specific_relation:
                skipped["table_delta_only_without_name_or_subclass_support"] += 1
                continue
            if inference["confidence"] < min_confidence:
                skipped["below_confidence_threshold"] += 1
                continue
            claim = inferred_claim_from_pair(inference, pair_reasons)
            key = (
                claim["compound_class"].lower(),
                claim["source_entity"].lower(),
                claim["target_entity"].lower(),
                claim["transformation_name"].lower(),
                claim["delta_mass"],
            )
            if key in inferred:
                inferred[key]["evidence_ids"] = join_unique([inferred[key]["evidence_ids"], claim["evidence_ids"]])
                continue
            inferred[key] = claim
            if len(examples) < 10:
                examples.append(
                    {
                        "source_entity": claim["source_entity"],
                        "target_entity": claim["target_entity"],
                        "transformation_name": claim["transformation_name"],
                        "delta_mass": claim["delta_mass"],
                        "confidence": claim["confidence"],
                    }
                )

    by_name = Counter(claim["transformation_name"] for claim in inferred.values())
    report = {
        "input_compound_count": len(compounds),
        "candidate_pair_count": candidate_pair_count,
        "delta_matched_pair_count": delta_matched_pair_count,
        "inferred_transformation_count": len(inferred),
        "inferred_by_transformation_name": dict(sorted(by_name.items())),
        "skipped_pair_reasons": dict(sorted(skipped.items())),
        "examples": examples,
    }
    return list(inferred.values()), report


def run_literature_supported_inference(
    evidence_claims_jsonl: Path | str,
    out_jsonl: Path | str,
    concepts_json: Path | str | None = None,
    report_path: Path | str | None = None,
    min_confidence: float = 0.65,
    allow_table_delta_only: bool = False,
) -> dict[str, Any]:
    del concepts_json  # Reserved for future concept-aware constraints.
    claims_path = resolve(evidence_claims_jsonl)
    out_path = resolve(out_jsonl)
    claims = load_jsonl(claims_path)
    inferred, report = infer_transformation_claims(
        claims,
        min_confidence=min_confidence,
        allow_table_delta_only=allow_table_delta_only,
    )
    write_jsonl(out_path, claims + inferred)
    if report_path:
        write_json(resolve(report_path), report)
    return report


def default_out_path() -> Path:
    return PROJECT_ROOT / "rag" / "evidence_claims" / "evidence_claims_augmented.jsonl"


def default_report_path() -> Path:
    return PROJECT_ROOT / "rag" / "reports" / "literature_supported_inference_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer review-only transformation claims from literature-reported compound masses."
    )
    parser.add_argument("--evidence_claims_jsonl", default="rag/evidence_claims/evidence_claims.jsonl")
    parser.add_argument("--concepts_json", default="", help="Reserved optional discovered concepts JSON.")
    parser.add_argument("--out_jsonl", default=str(default_out_path()))
    parser.add_argument("--report_path", default=str(default_report_path()))
    parser.add_argument("--min_confidence", type=float, default=0.65)
    parser.add_argument(
        "--allow_table_delta_only",
        action="store_true",
        help=(
            "Allow transformations supported only by same-table/source co-reporting plus a universal mass delta. "
            "By default, inferred transformations also require name-stem, modifier-pattern, or subclass support."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_literature_supported_inference(
            evidence_claims_jsonl=args.evidence_claims_jsonl,
            concepts_json=args.concepts_json or None,
            out_jsonl=args.out_jsonl,
            report_path=args.report_path,
            min_confidence=args.min_confidence,
            allow_table_delta_only=args.allow_table_delta_only,
        )
    except (OSError, json.JSONDecodeError, LiteratureInferenceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
