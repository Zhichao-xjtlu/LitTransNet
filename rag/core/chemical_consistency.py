"""Deterministic, compound-class-independent formula and mass validation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


ATOMIC_MASSES = {
    "H": 1.00782503223,
    "B": 11.00930536,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "Na": 22.989769282,
    "Mg": 23.985041697,
    "Si": 27.97692653465,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "K": 38.9637064864,
    "Ca": 39.962590863,
    "Fe": 55.93493633,
    "Br": 78.9183376,
    "I": 126.904468,
}

FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
FORMULA_CHARGE_SUFFIX = re.compile(r"(?:\^\d+[+-]|[+-])$")
OPERATORS_PATH = Path(__file__).with_name("reaction_operators.json")


class ChemicalConsistencyError(ValueError):
    """Raised for invalid formulas or impossible deterministic operations."""


@dataclass(frozen=True)
class MassConsistencyResult:
    passed: bool
    error_da: float
    error_ppm: float


@dataclass(frozen=True)
class ChemicalValidationResult:
    status: str
    reaction_operator: str
    operator_schema_version: str
    expected_product_formula: str
    calculated_product_mass: float | None
    atom_balance: dict[str, int]
    warning: str = ""


@dataclass(frozen=True)
class DerivedFormulaResult:
    status: str
    formula: str
    exact_mass: float | None
    atom_counts: dict[str, int]
    reaction_operator: str
    operator_schema_version: str
    warning: str = ""


def parse_formula(formula: str) -> dict[str, int]:
    text = str(formula or "").strip()
    if not text:
        raise ChemicalConsistencyError("formula is empty")
    text = FORMULA_CHARGE_SUFFIX.sub("", text)
    if not text:
        raise ChemicalConsistencyError("formula has no atoms")
    counts: dict[str, int] = {}
    position = 0
    for match in FORMULA_TOKEN.finditer(text):
        if match.start() != position:
            raise ChemicalConsistencyError(f"invalid formula syntax at position {position}: {text!r}")
        element = match.group(1)
        if element not in ATOMIC_MASSES:
            raise ChemicalConsistencyError(f"unknown element {element!r}")
        count = int(match.group(2) or "1")
        if count <= 0:
            raise ChemicalConsistencyError(f"element {element!r} must have a positive count")
        counts[element] = counts.get(element, 0) + count
        position = match.end()
    if position != len(text):
        raise ChemicalConsistencyError(f"invalid formula syntax at position {position}: {text!r}")
    return counts


def _hill_elements(counts: Mapping[str, int]) -> list[str]:
    if counts.get("C", 0) > 0:
        return [item for item in ("C", "H") if counts.get(item, 0) > 0] + sorted(
            item for item, count in counts.items() if count > 0 and item not in {"C", "H"}
        )
    return sorted(item for item, count in counts.items() if count > 0)


def format_formula(counts: Mapping[str, int]) -> str:
    parts: list[str] = []
    for element in _hill_elements(counts):
        count = int(counts[element])
        if count < 0:
            raise ChemicalConsistencyError(f"negative atom count for {element}: {count}")
        if count:
            parts.append(element if count == 1 else f"{element}{count}")
    if not parts:
        raise ChemicalConsistencyError("formula has no atoms")
    return "".join(parts)


def formula_exact_mass(formula: str | Mapping[str, int]) -> float:
    counts = parse_formula(formula) if isinstance(formula, str) else dict(formula)
    return sum(ATOMIC_MASSES[element] * count for element, count in counts.items())


def apply_formula_delta(formula: str | Mapping[str, int], delta: Mapping[str, int]) -> str:
    counts = parse_formula(formula) if isinstance(formula, str) else dict(formula)
    for element, change in delta.items():
        if element not in ATOMIC_MASSES:
            raise ChemicalConsistencyError(f"unknown element {element!r} in formula delta")
        if isinstance(change, bool) or not isinstance(change, int):
            raise ChemicalConsistencyError(f"formula delta for {element} must be an integer")
        counts[element] = counts.get(element, 0) + change
        if counts[element] < 0:
            raise ChemicalConsistencyError(f"negative atom count for {element}: {counts[element]}")
    return format_formula(counts)


def bounded_mass_consistent(
    expected_mass: float,
    observed_mass: float,
    ppm_tol: float,
    da_tol: float,
) -> MassConsistencyResult:
    expected = float(expected_mass)
    observed = float(observed_mass)
    error_da = abs(observed - expected)
    error_ppm = math.inf if expected == 0 else error_da / abs(expected) * 1_000_000.0
    return MassConsistencyResult(
        passed=error_da <= float(da_tol) and error_ppm <= float(ppm_tol),
        error_da=error_da,
        error_ppm=error_ppm,
    )


def load_reaction_operators(path: Path = OPERATORS_PATH) -> tuple[str, dict[str, dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = str(payload.get("schema_version", ""))
    operators = payload.get("operators")
    if not version or not isinstance(operators, dict):
        raise ChemicalConsistencyError("reaction operator registry is invalid")
    return version, operators


def _sum_formulas(items: Iterable[tuple[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for formula, coefficient in items:
        if isinstance(coefficient, bool) or not isinstance(coefficient, int) or coefficient <= 0:
            raise ChemicalConsistencyError("formula coefficient must be a positive integer")
        for element, count in parse_formula(formula).items():
            total[element] = total.get(element, 0) + count * coefficient
    return total


def validate_reaction_balance(
    reactants: Iterable[tuple[str, int]],
    products: Iterable[tuple[str, int]],
    reaction_operator: str,
) -> ChemicalValidationResult:
    version, operators = load_reaction_operators()
    operator_id = str(reaction_operator or "").strip()
    if not operator_id or operator_id not in operators:
        return ChemicalValidationResult(
            status="stoichiometry_unresolved",
            reaction_operator=operator_id,
            operator_schema_version=version,
            expected_product_formula="",
            calculated_product_mass=None,
            atom_balance={},
            warning="a valid explicit reaction operator is required",
        )
    try:
        expected_counts = _sum_formulas(reactants)
        delta = operators[operator_id].get("formula_delta", {})
        if not isinstance(delta, dict):
            raise ChemicalConsistencyError(f"operator {operator_id!r} has invalid formula_delta")
        expected_formula = apply_formula_delta(expected_counts, delta)
        expected_counts = parse_formula(expected_formula)
        product_rows = list(products)
        if not product_rows:
            return ChemicalValidationResult(
                status="derived_formula_valid",
                reaction_operator=operator_id,
                operator_schema_version=version,
                expected_product_formula=expected_formula,
                calculated_product_mass=formula_exact_mass(expected_counts),
                atom_balance={},
                warning="product formula was derived from explicit reactants and reaction operator",
            )
        actual_counts = _sum_formulas(product_rows)
    except ChemicalConsistencyError as exc:
        return ChemicalValidationResult(
            status="formula_parse_failed",
            reaction_operator=operator_id,
            operator_schema_version=version,
            expected_product_formula="",
            calculated_product_mass=None,
            atom_balance={},
            warning=str(exc),
        )
    elements = sorted(set(expected_counts) | set(actual_counts))
    atom_balance = {
        element: actual_counts.get(element, 0) - expected_counts.get(element, 0)
        for element in elements
        if actual_counts.get(element, 0) != expected_counts.get(element, 0)
    }
    return ChemicalValidationResult(
        status="valid" if not atom_balance else "atom_balance_failed",
        reaction_operator=operator_id,
        operator_schema_version=version,
        expected_product_formula=expected_formula,
        calculated_product_mass=formula_exact_mass(expected_counts),
        atom_balance=atom_balance,
        warning="" if not atom_balance else "reported products do not match the operator-balanced reactants",
    )


def derive_missing_reactant_formula(
    *,
    known_reactants: Iterable[tuple[str, int]],
    products: Iterable[tuple[str, int]],
    reaction_operator: str,
    missing_coefficient: int = 1,
) -> DerivedFormulaResult:
    """Solve one missing reactant by inverse atom balance.

    The equation is: products = known reactants + missing reactant + operator
    delta. No chemical identity or name is inferred here.
    """

    version, operators = load_reaction_operators()
    operator_id = str(reaction_operator or "").strip()
    if operator_id not in operators:
        return DerivedFormulaResult(
            "stoichiometry_unresolved", "", None, {}, operator_id, version, "unknown reaction operator"
        )
    if isinstance(missing_coefficient, bool) or not isinstance(missing_coefficient, int) or missing_coefficient <= 0:
        return DerivedFormulaResult(
            "inverse_balance_impossible", "", None, {}, operator_id, version, "invalid missing coefficient"
        )
    try:
        known_counts = _sum_formulas(known_reactants)
        product_counts = _sum_formulas(products)
        delta = operators[operator_id].get("formula_delta", {})
        if not isinstance(delta, dict):
            raise ChemicalConsistencyError("operator formula_delta is invalid")
    except ChemicalConsistencyError as exc:
        return DerivedFormulaResult(
            "formula_parse_failed", "", None, {}, operator_id, version, str(exc)
        )
    missing_counts: dict[str, int] = {}
    elements = set(known_counts) | set(product_counts) | set(delta)
    for element in elements:
        total = product_counts.get(element, 0) - known_counts.get(element, 0) - int(delta.get(element, 0))
        if total < 0:
            return DerivedFormulaResult(
                "inverse_balance_impossible",
                "",
                None,
                {},
                operator_id,
                version,
                f"negative inferred atom count for {element}",
            )
        if total % missing_coefficient != 0:
            return DerivedFormulaResult(
                "inverse_balance_nonintegral",
                "",
                None,
                {},
                operator_id,
                version,
                f"atom count for {element} is not divisible by coefficient {missing_coefficient}",
            )
        count = total // missing_coefficient
        if count:
            missing_counts[element] = count
    try:
        formula = format_formula(missing_counts)
    except ChemicalConsistencyError as exc:
        return DerivedFormulaResult(
            "inverse_balance_impossible", "", None, {}, operator_id, version, str(exc)
        )
    return DerivedFormulaResult(
        "derived_formula_valid",
        formula,
        formula_exact_mass(missing_counts),
        missing_counts,
        operator_id,
        version,
        "formula derived by inverse atom balance from evidence-backed product and reactants",
    )
