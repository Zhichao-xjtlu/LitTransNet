"""Bounded pickle ownership and deterministic CSV output for Network V5."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dl_annotator.v5.entity_hypothesis_network import (
    NetworkKnowledge,
    SeedAssignment,
    SpectrumNode,
    resolve_seed_entity,
)
from rag.core.entity_registry import EntityResolution


def nodes_and_seeds(
    match_obj: Mapping[str, Any],
    ion_mode: str,
    knowledge: NetworkKnowledge,
) -> tuple[list[SpectrumNode], list[SeedAssignment]]:
    """Convert the required match payload without retaining its container."""

    contract_version = str(match_obj.get("match_contract_version", ""))
    if contract_version != "3.1" or match_obj.get("input_unit") != "aligned_feature":
        raise ValueError(
            "Network V5 requires match contract 3.1 with input_unit='aligned_feature'; "
            f"received version={contract_version!r}, input_unit={match_obj.get('input_unit')!r}"
        )
    spec_df = match_obj["spec_df"]
    required_columns = {
        "feature_id",
        "source_spectrum_id",
        "precursor_mz",
        "mz_array",
        "intensity_array",
        "ion_mode",
    }
    missing = sorted(required_columns - set(spec_df.columns))
    if missing:
        raise ValueError(f"feature-level match spec_df missing columns: {missing}")
    feature_ids = [str(value).strip() for value in spec_df["feature_id"]]
    if any(not value for value in feature_ids):
        raise ValueError("feature-level match contains a blank feature_id")
    if len(set(feature_ids)) != len(feature_ids):
        raise ValueError("feature-level match contains duplicate feature_id values")
    precursors = np.asarray(spec_df["precursor_mz"], dtype=float)
    known = np.asarray(
        match_obj.get("known_match_arr", np.zeros(len(precursors))), dtype=bool
    )
    all_hits = match_obj.get("all_hits", [[] for _ in precursors])
    seed_candidate_hits = match_obj.get("seed_candidate_hits")
    if not isinstance(seed_candidate_hits, Sequence) or len(seed_candidate_hits) != len(precursors):
        raise ValueError(
            "feature-level match contract requires one seed_candidate_hits entry per feature"
        )
    nodes: list[SpectrumNode] = []
    seeds: list[SeedAssignment] = []
    for index, precursor in enumerate(precursors):
        feature_id = feature_ids[index]
        node_id = f"feature::{feature_id}"
        row_mode = str(spec_df.iloc[index].get("ion_mode", "")).strip().casefold()
        if row_mode != ion_mode:
            raise ValueError(
                f"feature {feature_id!r} ion mode {row_mode!r} conflicts with network run {ion_mode!r}"
            )
        library_name = ""
        resolution = None
        candidate_names: list[str] = []
        candidate_scores: list[float] = []
        entity_library_names: dict[str, list[str]] = {}
        if known[index]:
            hit = (
                all_hits[index][0]
                if index < len(all_hits) and all_hits[index]
                else {}
            )
            library_name = str(hit.get("name_clean") or hit.get("name") or "")
            top_resolution = resolve_seed_entity(knowledge, library_name)
            for candidate in seed_candidate_hits[index]:
                candidate_name = str(
                    candidate.get("name_clean") or candidate.get("name") or ""
                ).strip()
                if not candidate_name:
                    continue
                candidate_names.append(candidate_name)
                candidate_scores.append(float(candidate.get("score", 0.0) or 0.0))
                candidate_resolution = resolve_seed_entity(knowledge, candidate_name)
                if candidate_resolution.status not in {"resolved", "ambiguous"}:
                    continue
                for entity_id in candidate_resolution.entity_ids:
                    entity_library_names.setdefault(entity_id, []).append(candidate_name)
            resolved_entity_ids = tuple(sorted(entity_library_names))
            if len(resolved_entity_ids) == 1:
                resolution_status = "resolved"
            elif len(resolved_entity_ids) > 1:
                resolution_status = "ambiguous_library_seed"
            else:
                resolution_status = top_resolution.status
                resolved_entity_ids = top_resolution.entity_ids
            resolution = EntityResolution(
                resolution_status,
                resolved_entity_ids,
            )
        nodes.append(
            SpectrumNode(
                node_id=node_id,
                precursor_mz=float(precursor),
                mz_array=np.asarray(spec_df.iloc[index]["mz_array"], dtype=float),
                intensity_array=np.asarray(
                    spec_df.iloc[index]["intensity_array"], dtype=float
                ),
                ion_mode=ion_mode,
                rt_min=float(spec_df.iloc[index].get("rt_min", 0.0)),
                raw_indices=(index,),
                known_match=bool(known[index]),
                library_name=library_name,
                seed_resolution_status=resolution.status if resolution else "",
                seed_resolution_entity_ids=(
                    resolution.entity_ids if resolution else ()
                ),
                feature_id=feature_id,
                source_spectrum_id=str(
                    spec_df.iloc[index].get("source_spectrum_id", "")
                ),
                library_candidate_names=tuple(candidate_names),
                library_candidate_scores=tuple(candidate_scores),
            )
        )
        if known[index] and resolution is not None and resolution.status in {
            "resolved",
            "ambiguous_library_seed",
        }:
            for entity_id in resolution.entity_ids:
                names = entity_library_names.get(entity_id, [library_name])
                seeds.append(
                    SeedAssignment(node_id, entity_id, ";".join(sorted(set(names))))
                )
    return nodes, seeds


def load_match_nodes_and_seeds(
    path: Path | str,
    ion_mode: str,
    knowledge: NetworkKnowledge,
) -> tuple[list[SpectrumNode], list[SeedAssignment]]:
    """Load one pickle in a narrow scope and return only network-owned data."""

    with Path(path).open("rb") as handle:
        match_obj = pickle.load(handle)
    if not isinstance(match_obj, Mapping):
        raise TypeError("match pickle must contain a mapping")
    return nodes_and_seeds(match_obj, ion_mode, knowledge)


def write_rows(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
