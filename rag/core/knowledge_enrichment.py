"""Universal chemistry enrichment utilities for Agent 3.

This module must remain metabolite-class agnostic. It contains only generic
mass-loss and formula calculations used to annotate already evidence-backed
rules.
"""

from __future__ import annotations

import re
from typing import Any


MASS_TOLERANCE_DA = 0.05

NEUTRAL_LOSS_INTERPRETATIONS = {
    18.0106: "water loss",
    17.0265: "ammonia loss",
    43.9898: "CO2 loss",
    162.0528: "hexose loss",
    146.0579: "deoxyhexose loss",
    132.0423: "pentose loss",
}

TRANSFORMATION_INTERPRETATIONS = {
    -43.9898: "decarboxylation",
    -18.0106: "dehydration",
    -162.0528: "hexose loss",
}

ATOMIC_MASSES = {
    "H": 1.00782503223,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "Br": 78.9183376,
    "Na": 22.989769282,
    "K": 38.9637064864,
}


def safe_float(value: Any) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def match_mass_dictionary(value: Any, dictionary: dict[float, str], tolerance: float = MASS_TOLERANCE_DA) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    best_label = ""
    best_error = tolerance
    for mass, label in dictionary.items():
        error = abs(number - mass)
        if error <= best_error:
            best_error = error
            best_label = label
    return best_label


def interpret_neutral_loss(value: Any) -> str:
    return match_mass_dictionary(value, NEUTRAL_LOSS_INTERPRETATIONS)


def interpret_transformation_delta(value: Any) -> str:
    return match_mass_dictionary(value, TRANSFORMATION_INTERPRETATIONS)


def formula_mass(formula: str) -> float | None:
    text = "" if formula is None else str(formula).strip()
    if not text:
        return None
    pos = 0
    total = 0.0
    for match in re.finditer(r"([A-Z][a-z]?)(\d*)", text):
        if match.start() != pos:
            return None
        element = match.group(1)
        if element not in ATOMIC_MASSES:
            return None
        count = int(match.group(2) or "1")
        total += ATOMIC_MASSES[element] * count
        pos = match.end()
    return total if pos == len(text) else None
