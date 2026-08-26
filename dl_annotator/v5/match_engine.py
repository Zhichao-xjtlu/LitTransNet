"""Generic, polarity-aware LC-MS/MS library matching primitives.

The module is metabolite-class agnostic and deliberately keeps collision
energy as audit metadata rather than an identity gate.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np


MATCH_CONTRACT_VERSION = "3.1"
VALID_ION_MODES = frozenset({"positive", "negative"})


def normalize_ion_mode(value: object) -> str:
    text = str(value or "").strip().casefold().replace(" ", "")
    if text in {"positive", "pos", "+", "p", "positiveionmode", "esi+", "es+"}:
        return "positive"
    if text in {"negative", "neg", "-", "n", "negativeionmode", "esi-", "es-"}:
        return "negative"
    return ""


def ion_mode_from_adduct(value: object) -> str:
    text = str(value or "").strip().replace(" ", "")
    if not text:
        return ""
    if text.endswith("+") or re.search(r"\[[^]]+\]\+", text):
        return "positive"
    if text.endswith("-") or re.search(r"\[[^]]+\]-", text):
        return "negative"
    return ""


def bounded_mass_match(observed: float, reference: float, ppm: float, da: float) -> tuple[bool, float, float]:
    error_da = abs(float(observed) - float(reference))
    error_ppm = error_da / max(abs(float(reference)), 1e-12) * 1e6
    return error_ppm <= float(ppm) and error_da <= float(da), error_da, error_ppm


def top_peaks(mz: Sequence[float], intensity: Sequence[float], top_n: int) -> tuple[np.ndarray, np.ndarray]:
    mz_arr = np.asarray(mz, dtype=np.float32)
    int_arr = np.asarray(intensity, dtype=np.float32)
    size = min(mz_arr.size, int_arr.size)
    mz_arr = mz_arr[:size]
    int_arr = int_arr[:size]
    keep = np.isfinite(mz_arr) & np.isfinite(int_arr) & (int_arr > 0)
    mz_arr = mz_arr[keep]
    int_arr = int_arr[keep]
    if int_arr.size == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    base = float(np.max(int_arr))
    strong = int_arr >= base * 0.05
    if np.any(strong):
        mz_arr = mz_arr[strong]
        int_arr = int_arr[strong]
    if int_arr.size > int(top_n):
        selected = np.argsort(int_arr)[-int(top_n) :]
        mz_arr = mz_arr[selected]
        int_arr = int_arr[selected]
    order = np.argsort(mz_arr)
    return mz_arr[order].astype(np.float32), int_arr[order].astype(np.float32)


def cosine_greedy(
    query_mz: Sequence[float],
    query_intensity: Sequence[float],
    reference_mz: Sequence[float],
    reference_intensity: Sequence[float],
    tolerance_da: float,
) -> tuple[float, int, tuple[float, ...]]:
    q_mz = np.asarray(query_mz, dtype=float)
    q_it = np.asarray(query_intensity, dtype=float)
    r_mz = np.asarray(reference_mz, dtype=float)
    r_it = np.asarray(reference_intensity, dtype=float)
    if q_mz.size == 0 or r_mz.size == 0:
        return 0.0, 0, ()
    q_norm = float(np.linalg.norm(q_it))
    r_norm = float(np.linalg.norm(r_it))
    if q_norm <= 0 or r_norm <= 0:
        return 0.0, 0, ()
    q_weight = q_it / q_norm
    r_weight = r_it / r_norm
    candidates: list[tuple[float, int, int]] = []
    for qi, value in enumerate(q_mz.tolist()):
        left = int(np.searchsorted(r_mz, value - float(tolerance_da), side="left"))
        right = int(np.searchsorted(r_mz, value + float(tolerance_da), side="right"))
        for ri in range(left, right):
            candidates.append((float(q_weight[qi] * r_weight[ri]), qi, ri))
    used_q: set[int] = set()
    used_r: set[int] = set()
    score = 0.0
    matched_query: list[float] = []
    for contribution, qi, ri in sorted(candidates, reverse=True):
        if qi in used_q or ri in used_r:
            continue
        used_q.add(qi)
        used_r.add(ri)
        score += contribution
        matched_query.append(float(q_mz[qi]))
    return float(min(1.0, max(0.0, score))), len(used_q), tuple(sorted(matched_query))


@dataclass(frozen=True)
class QuerySpectrum:
    file: str
    scan_id: str
    rt_min: float
    precursor_mz: float
    mz_array: np.ndarray
    intensity_array: np.ndarray
    ion_mode: str
    collision_energy_ev: float | None = None
    feature_id: str = ""
    source_spectrum_id: str = ""
    feature_table_file: str = ""
    feature_metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class LibrarySpectrum:
    source: str
    name: str
    precursor_mz: float
    rt_min: float
    mz_array: np.ndarray
    intensity_array: np.ndarray
    ion_mode: str
    adduct: str = ""
    collision_energy_ev: float | None = None


@dataclass(frozen=True)
class MatchHit:
    name: str
    name_clean: str
    source: str
    score: float
    similarity: float
    matched_n: int
    matched_query_mz: tuple[float, ...]
    ppm_error: float
    mass_error_da: float
    rt_delta_min: float
    ion_mode: str
    adduct: str
    collision_energy_ev: float | None
    ion_mode_consistent: bool
    reference_precursor_mz: float

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "name_clean": self.name_clean,
            "source": self.source,
            "score": self.score,
            "similarity": self.similarity,
            "matched_n": self.matched_n,
            "matched_query_mz": list(self.matched_query_mz),
            "ppm_error": self.ppm_error,
            "mass_error_da": self.mass_error_da,
            "rt_delta_min": self.rt_delta_min,
            "ion_mode": self.ion_mode,
            "adduct": self.adduct,
            "collision_energy_ev": self.collision_energy_ev,
            "ion_mode_consistent": int(self.ion_mode_consistent),
            "reference_precursor_mz": self.reference_precursor_mz,
        }


def clean_library_name(value: object) -> str:
    text = str(value or "").strip()
    return text.split("|", 1)[0].strip() if "|" in text else text


def _metadata_value(metadata: Mapping[str, str], *keys: str) -> str:
    normalized = {re.sub(r"[^a-z0-9]", "", key.casefold()): value for key, value in metadata.items()}
    for key in keys:
        value = normalized.get(re.sub(r"[^a-z0-9]", "", key.casefold()), "")
        if value:
            return value
    return ""


def _float_from_text(value: object) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value or ""))
    if not match:
        return None
    number = float(match.group(0))
    return number if math.isfinite(number) else None


def _mode_from_metadata(metadata: Mapping[str, str], adduct: str) -> str:
    explicit = _metadata_value(metadata, "ion_mode", "ionmode", "polarity", "ionization_mode")
    mode = normalize_ion_mode(explicit)
    if mode:
        return mode
    comment = _metadata_value(metadata, "comment", "comments")
    match = re.search(r"(?:polarity|ion[_ ]?mode)\s*=\s*([^;]+)", comment, flags=re.I)
    if match:
        mode = normalize_ion_mode(match.group(1))
        if mode:
            return mode
    return ion_mode_from_adduct(adduct)


def iter_msp(path: Path, *, top_n: int, default_ion_mode: str = "") -> Iterator[LibrarySpectrum]:
    metadata: dict[str, str] = {}
    peaks_mz: list[float] = []
    peaks_intensity: list[float] = []

    def emit() -> LibrarySpectrum | None:
        if not metadata:
            return None
        precursor = _float_from_text(
            _metadata_value(metadata, "precursor_mz", "precursormz", "pepmass", "parentmass")
        )
        name = _metadata_value(metadata, "name", "compound_name")
        if precursor is None or not name or not peaks_mz:
            return None
        rt = _float_from_text(_metadata_value(metadata, "retentiontime", "retention_time", "rt", "rtmin"))
        adduct = _metadata_value(metadata, "precursor_type", "precursortype", "adduct", "ion_type")
        mode = _mode_from_metadata(metadata, adduct) or normalize_ion_mode(default_ion_mode)
        comment = _metadata_value(metadata, "comment", "comments")
        ce_text = _metadata_value(metadata, "collisionenergy", "collision_energy", "ce")
        if not ce_text:
            ce_match = re.search(r"(?:ObservedCE|CollisionEnergy|CE)\s*=\s*([-+]?\d+(?:\.\d+)?)", comment, flags=re.I)
            ce_text = ce_match.group(1) if ce_match else ""
        mz, intensity = top_peaks(peaks_mz, peaks_intensity, top_n)
        return LibrarySpectrum(
            source=path.stem,
            name=name,
            precursor_mz=float(precursor),
            rt_min=float(rt) if rt is not None else math.nan,
            mz_array=mz,
            intensity_array=intensity,
            ion_mode=mode,
            adduct=adduct,
            collision_energy_ev=_float_from_text(ce_text),
        )

    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                item = emit()
                if item is not None:
                    yield item
                metadata = {}
                peaks_mz = []
                peaks_intensity = []
                continue
            peak = re.match(r"^([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)", line)
            if peak:
                peaks_mz.append(float(peak.group(1)))
                peaks_intensity.append(float(peak.group(2)))
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
        item = emit()
        if item is not None:
            yield item


def load_msp_libraries(
    paths: Iterable[Path],
    *,
    top_n: int,
    default_ion_mode: str = "",
    max_entries_per_file: int = 120_000,
) -> tuple[list[LibrarySpectrum], int]:
    accepted: list[LibrarySpectrum] = []
    unknown_mode_count = 0
    for path in paths:
        for index, item in enumerate(iter_msp(path, top_n=top_n, default_ion_mode=default_ion_mode)):
            if index >= int(max_entries_per_file):
                break
            if item.ion_mode not in VALID_ION_MODES:
                unknown_mode_count += 1
                continue
            accepted.append(item)
    return accepted, unknown_mode_count


def build_library_index(entries: Sequence[LibrarySpectrum]) -> dict[str, tuple[np.ndarray, tuple[LibrarySpectrum, ...]]]:
    result: dict[str, tuple[np.ndarray, tuple[LibrarySpectrum, ...]]] = {}
    for mode in sorted(VALID_ION_MODES):
        selected = sorted(
            (item for item in entries if item.ion_mode == mode),
            key=lambda item: (item.precursor_mz, item.name, item.source),
        )
        result[mode] = (
            np.asarray([item.precursor_mz for item in selected], dtype=np.float64),
            tuple(selected),
        )
    return result


def search_library(
    query: QuerySpectrum,
    index: Mapping[str, tuple[np.ndarray, tuple[LibrarySpectrum, ...]]],
    *,
    ppm_tolerance: float,
    absolute_tolerance_da: float,
    fragment_tolerance_da: float,
    rt_tolerance_min: float,
    top_k: int,
) -> list[MatchHit]:
    if query.ion_mode not in VALID_ION_MODES:
        return []
    precursor, entries = index.get(query.ion_mode, (np.asarray([], dtype=float), ()))
    search_da = min(float(absolute_tolerance_da), abs(query.precursor_mz) * float(ppm_tolerance) * 1e-6)
    left = int(np.searchsorted(precursor, query.precursor_mz - search_da, side="left"))
    right = int(np.searchsorted(precursor, query.precursor_mz + search_da, side="right"))
    hits: list[MatchHit] = []
    for reference in entries[left:right]:
        passed, error_da, error_ppm = bounded_mass_match(
            query.precursor_mz, reference.precursor_mz, ppm_tolerance, absolute_tolerance_da
        )
        if not passed:
            continue
        rt_delta = math.nan
        if math.isfinite(query.rt_min) and math.isfinite(reference.rt_min):
            rt_delta = abs(query.rt_min - reference.rt_min)
            if float(rt_tolerance_min) > 0 and rt_delta > float(rt_tolerance_min):
                continue
        similarity, matched_n, matched_mz = cosine_greedy(
            query.mz_array,
            query.intensity_array,
            reference.mz_array,
            reference.intensity_array,
            fragment_tolerance_da,
        )
        if matched_n == 0:
            continue
        hits.append(
            MatchHit(
                name=reference.name,
                name_clean=clean_library_name(reference.name),
                source=reference.source,
                score=similarity,
                similarity=similarity,
                matched_n=matched_n,
                matched_query_mz=matched_mz,
                ppm_error=error_ppm,
                mass_error_da=error_da,
                rt_delta_min=rt_delta,
                ion_mode=reference.ion_mode,
                adduct=reference.adduct,
                collision_energy_ev=reference.collision_energy_ev,
                ion_mode_consistent=reference.ion_mode == query.ion_mode,
                reference_precursor_mz=reference.precursor_mz,
            )
        )
    hits.sort(key=lambda item: (-item.score, item.ppm_error, item.name, item.source))
    # A library commonly contains many replicate spectra per chemical name.
    # Seed competition must occur between entities, not between replicate
    # spectra; otherwise one well-represented isomer can occupy every top-k
    # slot and hide a nearly tied alternative entity.
    best_by_name: dict[str, MatchHit] = {}
    for hit in hits:
        key = clean_library_name(hit.name).casefold()
        if key not in best_by_name:
            best_by_name[key] = hit
    entity_hits = sorted(
        best_by_name.values(),
        key=lambda item: (-item.score, item.ppm_error, item.name, item.source),
    )
    return entity_hits[: int(top_k)] if int(top_k) > 0 else entity_hits


def build_decoy_library(entries: Sequence[LibrarySpectrum], seed: int = 42) -> list[LibrarySpectrum]:
    rng = np.random.default_rng(seed)
    losses_by_mode: dict[str, list[float]] = {mode: [] for mode in VALID_ION_MODES}
    for item in entries:
        losses_by_mode[item.ion_mode].extend(
            item.precursor_mz - float(value)
            for value in item.mz_array
            if 5.0 <= item.precursor_mz - float(value) <= 600.0
        )
    decoys: list[LibrarySpectrum] = []
    for item in entries:
        pool = losses_by_mode[item.ion_mode] or [18.0106, 44.0095, 162.0528]
        count = len(item.mz_array)
        losses = rng.choice(np.asarray(pool, dtype=float), size=count, replace=True)
        losses = losses + rng.normal(0.0, 0.03, size=count)
        mz = np.clip(item.precursor_mz - losses, 40.0, max(40.1, item.precursor_mz - 0.5))
        intensity = np.asarray(item.intensity_array, dtype=float).copy()
        if intensity.size > 1:
            intensity = intensity[rng.permutation(intensity.size)]
        mz, intensity = top_peaks(mz, intensity, max(1, count))
        decoys.append(
            replace(
                item,
                source=f"decoy_{item.source}",
                name=f"DECOY_{item.name}",
                mz_array=mz,
                intensity_array=intensity,
            )
        )
    return decoys


def estimate_qvalues(target_scores: Sequence[float], decoy_scores: Sequence[float]) -> np.ndarray:
    target = np.asarray(target_scores, dtype=float)
    decoy = np.asarray(decoy_scores, dtype=float)
    if target.size == 0:
        return np.asarray([], dtype=np.float32)
    order = np.argsort(-target, kind="stable")
    fdr = np.ones(target.size, dtype=float)
    for rank, index in enumerate(order, start=1):
        threshold = target[index]
        fdr[rank - 1] = (1.0 + float(np.sum(decoy >= threshold))) / float(rank)
    q_sorted = np.minimum.accumulate(fdr[::-1])[::-1]
    result = np.ones(target.size, dtype=float)
    for position, index in enumerate(order):
        result[index] = q_sorted[position]
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def spectrum_tokens(query: QuerySpectrum, top_n: int) -> tuple[np.ndarray, np.ndarray]:
    mz, intensity = top_peaks(query.mz_array, query.intensity_array, top_n)
    tokens = np.zeros((int(top_n), 3), dtype=np.float32)
    mask = np.ones((int(top_n),), dtype=np.bool_)
    size = min(int(top_n), mz.size)
    if size:
        maximum = max(float(np.max(intensity)), 1e-12)
        tokens[:size, 0] = mz[:size] / 2000.0
        tokens[:size, 1] = intensity[:size] / maximum
        tokens[:size, 2] = float(query.precursor_mz) / 2000.0
        mask[:size] = False
    return tokens, mask
