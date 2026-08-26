"""MS-DIAL GNPS-export feature table + MGF input adapter.

The core matcher and network consume a class-agnostic feature contract.  This
module is the only place that knows common MS-DIAL column/header aliases.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from match_engine import QuerySpectrum, ion_mode_from_adduct, normalize_ion_mode, top_peaks


FEATURE_INPUT_CONTRACT_VERSION = "1.0"


def _key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


FEATURE_ID_KEYS = {
    "featureid",
    "alignmentid",
    "rowid",
    "peakid",
    "id",
}
MZ_KEYS = {
    "averagemz",
    "rowmz",
    "precursormz",
    "mz",
    "mass",
}
RT_MIN_KEYS = {
    "averagertmin",
    "averagert",
    "rowretentiontime",
    "retentiontimemin",
    "rtmin",
    "rt",
}
ION_MODE_KEYS = {"ionmode", "polarity", "ionizationmode"}
ADDUCT_KEYS = {"adducttype", "adduct", "precursortype"}
CHARGE_KEYS = {"charge", "precursorcharge"}


def _find_column(fieldnames: Sequence[str], aliases: set[str], label: str) -> str:
    matches = [name for name in fieldnames if _key(name) in aliases]
    if not matches:
        raise ValueError(
            f"MS-DIAL feature table is missing {label}; columns={list(fieldnames)}"
        )
    return matches[0]


def _optional_column(fieldnames: Sequence[str], aliases: set[str]) -> str:
    return next((name for name in fieldnames if _key(name) in aliases), "")


def _finite_float(value: object, *, field: str, allow_blank: bool = False) -> float:
    text = str(value or "").strip()
    if not text and allow_blank:
        return math.nan
    try:
        number = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if not math.isfinite(number):
        if allow_blank:
            return math.nan
        raise ValueError(f"non-finite {field}: {value!r}")
    return number


def canonical_feature_id(value: object) -> str:
    text = str(value or "").strip().strip('"')
    if not text:
        return ""
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def _delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
    if not sample.strip():
        raise ValueError(f"empty feature table: {path}")
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
    except csv.Error:
        return "\t" if sample.count("\t") > sample.count(",") else ","


@dataclass(frozen=True)
class FeatureRow:
    feature_id: str
    precursor_mz: float
    rt_min: float
    ion_mode: str
    adduct: str
    charge: str
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class MGFRecord:
    feature_id: str
    spectrum_id: str
    precursor_mz: float
    rt_min: float
    ion_mode: str
    collision_energy_ev: float | None
    mz_array: np.ndarray
    intensity_array: np.ndarray
    metadata: Mapping[str, str]


def read_feature_table(path: Path) -> tuple[list[FeatureRow], dict[str, str]]:
    delimiter = _delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        fieldnames: list[str] = []
        header_row_number = 0
        for row_number, values in enumerate(reader, start=1):
            candidate = [str(value or "").strip() for value in values]
            keys = {_key(value) for value in candidate}
            if keys & FEATURE_ID_KEYS and keys & MZ_KEYS and keys & RT_MIN_KEYS:
                fieldnames = candidate
                header_row_number = row_number
                break
        if not fieldnames:
            raise ValueError(
                f"feature table has no recognizable Alignment ID/mz/RT header: {path}"
            )
        id_col = _find_column(fieldnames, FEATURE_ID_KEYS, "feature/alignment ID")
        mz_col = _find_column(fieldnames, MZ_KEYS, "precursor m/z")
        rt_col = _find_column(fieldnames, RT_MIN_KEYS, "retention time in minutes")
        mode_col = _optional_column(fieldnames, ION_MODE_KEYS)
        adduct_col = _optional_column(fieldnames, ADDUCT_KEYS)
        charge_col = _optional_column(fieldnames, CHARGE_KEYS)
        rows: list[FeatureRow] = []
        seen: set[str] = set()
        for row_number, values in enumerate(reader, start=header_row_number + 1):
            if not any(str(value or "").strip() for value in values):
                continue
            row = {
                key: str(values[index] if index < len(values) else "").strip()
                for index, key in enumerate(fieldnames)
            }
            feature_id = canonical_feature_id(row.get(id_col))
            if not feature_id:
                raise ValueError(f"blank feature ID at {path}:{row_number}")
            if feature_id in seen:
                raise ValueError(f"duplicate feature ID {feature_id!r} at {path}:{row_number}")
            seen.add(feature_id)
            adduct = row.get(adduct_col, "") if adduct_col else ""
            mode = normalize_ion_mode(row.get(mode_col, "")) if mode_col else ""
            mode = mode or ion_mode_from_adduct(adduct)
            rows.append(
                FeatureRow(
                    feature_id=feature_id,
                    precursor_mz=_finite_float(row.get(mz_col), field="precursor m/z"),
                    rt_min=_finite_float(row.get(rt_col), field="retention time", allow_blank=True),
                    ion_mode=mode,
                    adduct=adduct,
                    charge=row.get(charge_col, "") if charge_col else "",
                    metadata=row,
                )
            )
    return rows, {
        "delimiter": "tab" if delimiter == "\t" else delimiter,
        "header_row_number": str(header_row_number),
        "preamble_row_count": str(header_row_number - 1),
        "feature_id_column": id_col,
        "precursor_mz_column": mz_col,
        "rt_min_column": rt_col,
        "ion_mode_column": mode_col,
        "adduct_column": adduct_col,
        "charge_column": charge_col,
    }


def _metadata_value(metadata: Mapping[str, str], *aliases: str) -> str:
    normalized = {_key(name): value for name, value in metadata.items()}
    return next((normalized.get(_key(alias), "") for alias in aliases if normalized.get(_key(alias))), "")


def _feature_id_from_mgf(metadata: Mapping[str, str]) -> str:
    direct = _metadata_value(metadata, "FEATURE_ID", "ALIGNMENT_ID", "ROW_ID", "SCANS")
    if direct:
        return canonical_feature_id(direct.split()[0])
    title = _metadata_value(metadata, "TITLE")
    patterns = (
        r"(?:feature|alignment|row)[ _-]*(?:id)?\s*[=: ]\s*([A-Za-z0-9_.-]+)",
        r"^\s*([+-]?\d+(?:\.0+)?)\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, title, flags=re.IGNORECASE)
        if match:
            return canonical_feature_id(match.group(1))
    return ""


def _parse_charge_mode(value: object) -> str:
    text = str(value or "").strip()
    if text.endswith("+"):
        return "positive"
    if text.endswith("-"):
        return "negative"
    return ""


def iter_mgf(path: Path, top_n: int) -> Iterator[MGFRecord]:
    metadata: dict[str, str] = {}
    mz_values: list[float] = []
    intensity_values: list[float] = []
    inside = False
    record_number = 0

    def emit() -> MGFRecord | None:
        nonlocal record_number
        if not inside:
            return None
        record_number += 1
        feature_id = _feature_id_from_mgf(metadata)
        precursor_text = _metadata_value(metadata, "PEPMASS", "PRECURSORMZ").split()[0]
        precursor = _finite_float(precursor_text, field="MGF PEPMASS")
        rt_seconds = _metadata_value(metadata, "RTINSECONDS")
        rt_minutes = _metadata_value(metadata, "RTINMINUTES", "RTMIN")
        if rt_seconds:
            rt_min = _finite_float(rt_seconds, field="MGF RTINSECONDS", allow_blank=True) / 60.0
        else:
            rt_min = _finite_float(rt_minutes, field="MGF RT", allow_blank=True)
        adduct = _metadata_value(metadata, "ADDUCT", "PRECURSORTYPE", "IONTYPE")
        mode = normalize_ion_mode(_metadata_value(metadata, "IONMODE", "POLARITY"))
        mode = mode or ion_mode_from_adduct(adduct) or _parse_charge_mode(
            _metadata_value(metadata, "CHARGE")
        )
        ce_text = _metadata_value(metadata, "COLLISIONENERGY", "COLLISION_ENERGY", "CE")
        ce = None
        if ce_text:
            match = re.search(r"[-+]?\d+(?:\.\d+)?", ce_text)
            ce = float(match.group(0)) if match else None
        mz, intensity = top_peaks(mz_values, intensity_values, top_n)
        spectrum_id = _metadata_value(metadata, "TITLE") or f"mgf_record_{record_number}"
        return MGFRecord(
            feature_id=feature_id,
            spectrum_id=spectrum_id,
            precursor_mz=precursor,
            rt_min=rt_min,
            ion_mode=mode,
            collision_energy_ev=ce,
            mz_array=mz,
            intensity_array=intensity,
            metadata=dict(metadata),
        )

    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper == "BEGIN IONS":
                if inside:
                    raise ValueError(f"nested BEGIN IONS in {path}")
                metadata = {}
                mz_values = []
                intensity_values = []
                inside = True
                continue
            if upper == "END IONS":
                item = emit()
                if item is not None:
                    yield item
                inside = False
                continue
            if not inside:
                continue
            if "=" in line:
                name, value = line.split("=", 1)
                metadata[name.strip()] = value.strip()
                continue
            parts = re.split(r"\s+", line)
            if len(parts) >= 2:
                try:
                    mz_values.append(float(parts[0]))
                    intensity_values.append(float(parts[1]))
                except ValueError:
                    continue
    if inside:
        raise ValueError(f"unterminated MGF block in {path}")


def load_msdial_gnps_queries(
    feature_table_path: Path,
    mgf_path: Path,
    *,
    requested_mode: str,
    top_n: int,
    max_queries: int = 0,
    link_mz_tolerance_da: float = 0.02,
    link_rt_tolerance_min: float = 0.25,
) -> tuple[list[QuerySpectrum], dict[str, object]]:
    requested_mode = normalize_ion_mode(requested_mode)
    if not requested_mode:
        raise ValueError("requested_mode must be positive or negative")
    features, column_map = read_feature_table(feature_table_path)
    spectra = list(iter_mgf(mgf_path, top_n=top_n))
    by_feature: dict[str, list[MGFRecord]] = defaultdict(list)
    audit = Counter()
    for spectrum in spectra:
        audit["mgf_spectrum_count"] += 1
        if not spectrum.feature_id:
            audit["mgf_missing_feature_id"] += 1
            continue
        by_feature[spectrum.feature_id].append(spectrum)

    feature_ids = {item.feature_id for item in features}
    unmatched_mgf = sorted(set(by_feature) - feature_ids)
    audit["mgf_feature_ids_not_in_table"] = len(unmatched_mgf)
    queries: list[QuerySpectrum] = []
    missing_ms2: list[str] = []
    polarity_conflicts: list[str] = []
    mass_conflicts: list[str] = []
    rt_conflicts: list[str] = []
    duplicate_mgf_features: list[str] = []

    for feature in features:
        audit["feature_table_row_count"] += 1
        feature_mode = feature.ion_mode or requested_mode
        if feature_mode != requested_mode:
            audit["opposite_mode_feature_skipped"] += 1
            continue
        candidates = by_feature.get(feature.feature_id, [])
        if not candidates:
            missing_ms2.append(feature.feature_id)
            audit["feature_without_ms2"] += 1
            continue
        if len(candidates) > 1:
            duplicate_mgf_features.append(feature.feature_id)
            audit["feature_with_multiple_mgf_spectra"] += 1
        # Deterministic representative: maximum total ion current, then peak
        # count, then spectrum title. MS-DIAL normally exports one spectrum.
        spectrum = sorted(
            candidates,
            key=lambda item: (
                -float(np.sum(item.intensity_array)),
                -int(item.mz_array.size),
                item.spectrum_id,
            ),
        )[0]
        spectrum_mode = spectrum.ion_mode or requested_mode
        if spectrum_mode != requested_mode:
            polarity_conflicts.append(feature.feature_id)
            audit["mgf_polarity_conflict_rejected"] += 1
            continue
        if abs(feature.precursor_mz - spectrum.precursor_mz) > float(link_mz_tolerance_da):
            mass_conflicts.append(feature.feature_id)
            audit["feature_mgf_mass_conflict_rejected"] += 1
            continue
        if (
            math.isfinite(feature.rt_min)
            and math.isfinite(spectrum.rt_min)
            and abs(feature.rt_min - spectrum.rt_min) > float(link_rt_tolerance_min)
        ):
            rt_conflicts.append(feature.feature_id)
            audit["feature_mgf_rt_conflict_audited"] += 1
        if spectrum.mz_array.size == 0:
            audit["empty_mgf_spectrum_rejected"] += 1
            continue
        metadata = {
            "feature_table": dict(feature.metadata),
            "mgf_metadata": dict(spectrum.metadata),
            "feature_adduct": feature.adduct,
            "feature_charge": feature.charge,
            "mgf_candidate_count": len(candidates),
        }
        queries.append(
            QuerySpectrum(
                file=str(mgf_path),
                scan_id=spectrum.spectrum_id,
                rt_min=feature.rt_min if math.isfinite(feature.rt_min) else spectrum.rt_min,
                precursor_mz=feature.precursor_mz,
                mz_array=spectrum.mz_array,
                intensity_array=spectrum.intensity_array,
                ion_mode=requested_mode,
                collision_energy_ev=spectrum.collision_energy_ev,
                feature_id=feature.feature_id,
                source_spectrum_id=spectrum.spectrum_id,
                feature_table_file=str(feature_table_path),
                feature_metadata=metadata,
            )
        )
        audit["accepted_feature_with_ms2"] += 1
        if max_queries > 0 and len(queries) >= max_queries:
            audit["max_queries_truncated"] = 1
            break

    details: dict[str, object] = dict(sorted(audit.items()))
    details.update(
        {
            "feature_input_contract_version": FEATURE_INPUT_CONTRACT_VERSION,
            "adapter": "msdial_gnps_export",
            "feature_table_path": str(feature_table_path),
            "mgf_path": str(mgf_path),
            "column_map": column_map,
            "missing_ms2_feature_ids": missing_ms2,
            "unmatched_mgf_feature_ids": unmatched_mgf,
            "polarity_conflict_feature_ids": polarity_conflicts,
            "mass_conflict_feature_ids": mass_conflicts,
            "rt_conflict_feature_ids": rt_conflicts,
            "multiple_mgf_feature_ids": duplicate_mgf_features,
            "link_mz_tolerance_da": float(link_mz_tolerance_da),
            "link_rt_tolerance_min": float(link_rt_tolerance_min),
        }
    )
    return queries, details


def metadata_json(value: Mapping[str, object] | None) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)
