"""Feature-level, polarity-aware LC-MS/MS library matching CLI.

The primary input is one MS-DIAL GNPS-export alignment feature table plus its
representative MS/MS MGF.  One accepted alignment feature becomes one query
and, downstream, one network node.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from feature_input import FEATURE_INPUT_CONTRACT_VERSION, load_msdial_gnps_queries, metadata_json
from match_engine import (
    MATCH_CONTRACT_VERSION,
    QuerySpectrum,
    build_decoy_library,
    build_library_index,
    estimate_qvalues,
    load_msp_libraries,
    normalize_ion_mode,
    search_library,
    spectrum_tokens,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature-level library match from an MS-DIAL GNPS export"
    )
    parser.add_argument("--experiment_manifest", required=True)
    parser.add_argument("--ion_mode", required=True, choices=["positive", "negative"])
    parser.add_argument(
        "--feature_table",
        nargs="+",
        default=None,
        help="Override manifest match.feature_table_globs (.txt/.tsv/.csv). Exactly one file is required.",
    )
    parser.add_argument(
        "--spectra_mgf",
        nargs="+",
        default=None,
        help="Override manifest match.spectra_mgf_globs. Exactly one MGF is required.",
    )
    parser.add_argument("--library_glob", nargs="+", default=None)
    parser.add_argument(
        "--library_default_ion_mode",
        choices=["", "positive", "negative"],
        default=None,
    )
    parser.add_argument("--top_k_peaks", type=int, default=20)
    parser.add_argument("--top_n_report", type=int, default=20)
    parser.add_argument("--max_msp_per_file", type=int, default=120000)
    parser.add_argument("--precursor_ppm", type=float, default=10.0)
    parser.add_argument("--precursor_tol_da", type=float, default=0.01)
    parser.add_argument("--fragment_tol_da", type=float, default=0.10)
    parser.add_argument(
        "--rt_tol_min",
        type=float,
        default=0.0,
        help="0 disables library RT filtering; feature RT remains audit metadata.",
    )
    parser.add_argument(
        "--feature_mgf_mz_tol_da",
        type=float,
        default=0.02,
        help="Maximum table-to-MGF precursor mismatch for linking the same feature.",
    )
    parser.add_argument(
        "--feature_mgf_rt_tol_min",
        type=float,
        default=0.25,
        help="Maximum table-to-MGF RT mismatch for input-integrity checking only.",
    )
    parser.add_argument("--search_top_k", type=int, default=5)
    parser.add_argument("--match_score_min", type=float, default=0.60)
    parser.add_argument("--match_matched_n_min", type=int, default=1)
    parser.add_argument("--match_qvalue_max", type=float, default=0.25)
    parser.add_argument(
        "--seed_competitor_margin",
        type=float,
        default=0.01,
        help=(
            "Absolute cosine-score margin from the best entity-level library hit. "
            "All passing entities inside the margin become auditable seed hypotheses."
        ),
    )
    parser.add_argument("--decoy_seed", type=int, default=42)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--out_pkl", default="")
    parser.add_argument("--out_csv", default="")
    parser.add_argument("--out_summary_json", default="")
    return parser.parse_args(argv)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("experiment manifest must be a JSON object")
    required = ("experiment_id", "compound_class", "workspace")
    missing = [key for key in required if not manifest.get(key)]
    if missing:
        raise ValueError(f"experiment manifest missing fields: {', '.join(missing)}")
    return manifest, _repo_root()


def _resolve_pattern(pattern: str, repo_root: Path) -> str:
    path = Path(pattern)
    return str(path if path.is_absolute() else repo_root / path)


def _expand_patterns(
    patterns: Iterable[str], repo_root: Path, suffixes: set[str]
) -> list[Path]:
    accepted = {item.casefold() for item in suffixes}
    found: set[Path] = set()
    for pattern in patterns:
        resolved = _resolve_pattern(str(pattern), repo_root)
        for name in glob.glob(resolved, recursive=True):
            path = Path(name)
            if path.is_file() and path.suffix.casefold() in accepted:
                found.add(path.resolve())
    return sorted(found, key=lambda item: str(item).casefold())


def _require_one(paths: Sequence[Path], label: str, patterns: Sequence[str]) -> Path:
    if not paths:
        raise FileNotFoundError(f"no {label} matched: {list(patterns)}")
    if len(paths) != 1:
        raise ValueError(
            f"exactly one {label} is required per aligned ion-mode run; matched {len(paths)}: "
            + ", ".join(str(path) for path in paths)
        )
    return paths[0]


def _output_paths(args: argparse.Namespace, manifest: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    base = (
        Path(args.output_dir)
        if args.output_dir
        else Path(manifest["workspace"]["database_dir"]) / "match" / args.ion_mode
    )
    if not base.is_absolute():
        base = repo_root / base
    base.mkdir(parents=True, exist_ok=True)
    stem = f"{manifest['experiment_id']}.{args.ion_mode}.match_v5"
    return {
        "pkl": Path(args.out_pkl) if args.out_pkl else base / f"{stem}.pkl",
        "csv": Path(args.out_csv) if args.out_csv else base / f"{stem}.csv",
        "summary": Path(args.out_summary_json)
        if args.out_summary_json
        else base / f"{stem}.summary.json",
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["query_index", "feature_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_match(
    queries: Sequence[QuerySpectrum],
    library_entries: Sequence[Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if float(getattr(args, "seed_competitor_margin", 0.01)) < 0:
        raise ValueError("seed_competitor_margin must be non-negative")
    target_index = build_library_index(library_entries)
    decoy_entries = build_decoy_library(library_entries, seed=args.decoy_seed)
    decoy_index = build_library_index(decoy_entries)
    all_hits: list[list[dict[str, Any]]] = []
    all_decoy_hits: list[list[dict[str, Any]]] = []
    entity_hits_by_query: list[list[Any]] = []
    top_report: list[tuple[np.ndarray, np.ndarray]] = []
    tokens: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    target_scores: list[float] = []
    decoy_scores: list[float] = []

    for query in queries:
        hits = search_library(
            query,
            target_index,
            ppm_tolerance=args.precursor_ppm,
            absolute_tolerance_da=args.precursor_tol_da,
            fragment_tolerance_da=args.fragment_tol_da,
            rt_tolerance_min=args.rt_tol_min,
            top_k=0,
        )
        decoy_hits = search_library(
            query,
            decoy_index,
            ppm_tolerance=args.precursor_ppm,
            absolute_tolerance_da=args.precursor_tol_da,
            fragment_tolerance_da=args.fragment_tol_da,
            rt_tolerance_min=args.rt_tol_min,
            top_k=1,
        )
        entity_hits_by_query.append(hits)
        hit_rows = [hit.as_dict() for hit in hits[: int(args.search_top_k)]]
        decoy_rows = [hit.as_dict() for hit in decoy_hits]
        all_hits.append(hit_rows)
        all_decoy_hits.append(decoy_rows)
        target_scores.append(float(hits[0].score) if hits else 0.0)
        decoy_scores.append(float(decoy_hits[0].score) if decoy_hits else 0.0)
        top_report.append((query.mz_array.copy(), query.intensity_array.copy()))
        token, mask = spectrum_tokens(query, args.top_k_peaks)
        tokens.append(token)
        masks.append(mask)

    qvalues = estimate_qvalues(target_scores, decoy_scores)
    known: list[int] = []
    mode_consistent: list[int] = []
    seed_candidate_hits: list[list[dict[str, Any]]] = []
    rows: list[dict[str, Any]] = []
    for index, query in enumerate(queries):
        top = all_hits[index][0] if all_hits[index] else {}
        consistent = bool(top) and normalize_ion_mode(top.get("ion_mode")) == query.ion_mode
        accepted = bool(
            consistent
            and target_scores[index] >= args.match_score_min
            and int(top.get("matched_n", 0) or 0) >= args.match_matched_n_min
            and float(qvalues[index]) <= args.match_qvalue_max
        )
        known.append(int(accepted))
        mode_consistent.append(int(consistent))
        candidates: list[dict[str, Any]] = []
        if accepted:
            margin = float(getattr(args, "seed_competitor_margin", 0.01))
            for hit in entity_hits_by_query[index]:
                score_delta = float(target_scores[index] - hit.score)
                if score_delta > margin + 1e-12:
                    continue
                if hit.score < args.match_score_min:
                    continue
                if int(hit.matched_n) < args.match_matched_n_min:
                    continue
                candidate = hit.as_dict()
                candidate["score_delta_from_best"] = max(0.0, score_delta)
                candidates.append(candidate)
        seed_candidate_hits.append(candidates)
        rows.append(
            {
                "query_index": index,
                "feature_id": query.feature_id,
                "feature_table_file": query.feature_table_file,
                "mgf_file": query.file,
                "mgf_spectrum_id": query.source_spectrum_id or query.scan_id,
                "precursor_mz": query.precursor_mz,
                "rt_min": query.rt_min,
                "query_ion_mode": query.ion_mode,
                "query_collision_energy_ev": query.collision_energy_ev,
                "known_match": int(accepted),
                "seed_candidate_count": len(candidates),
                "seed_is_ambiguous": int(len(candidates) > 1),
                "seed_candidate_names": ";".join(
                    str(item.get("name_clean") or item.get("name") or "")
                    for item in candidates
                ),
                "seed_candidate_score_deltas": ";".join(
                    f"{float(item.get('score_delta_from_best', 0.0)):.8f}"
                    for item in candidates
                ),
                "target_score": target_scores[index],
                "decoy_score": decoy_scores[index],
                "q_value": float(qvalues[index]),
                "library_name": top.get("name", ""),
                "library_name_clean": top.get("name_clean", ""),
                "library_source": top.get("source", ""),
                "library_precursor_mz": top.get("reference_precursor_mz", ""),
                "library_ion_mode": top.get("ion_mode", ""),
                "library_adduct": top.get("adduct", ""),
                "library_collision_energy_ev": top.get("collision_energy_ev", ""),
                "ion_mode_consistent": int(consistent),
                "matched_fragment_count": top.get("matched_n", 0),
                "precursor_error_ppm": top.get("ppm_error", ""),
                "precursor_error_da": top.get("mass_error_da", ""),
                "rt_delta_min": top.get("rt_delta_min", ""),
            }
        )

    spec_df = pd.DataFrame(
        [
            {
                "feature_id": query.feature_id,
                "feature_table_file": query.feature_table_file,
                "mgf_file": query.file,
                "source_spectrum_id": query.source_spectrum_id or query.scan_id,
                "rt_min": query.rt_min,
                "precursor_mz": query.precursor_mz,
                "mz_array": query.mz_array,
                "intensity_array": query.intensity_array,
                "ion_mode": query.ion_mode,
                "collision_energy_ev": query.collision_energy_ev,
                "feature_metadata_json": metadata_json(query.feature_metadata),
            }
            for query in queries
        ]
    )
    match_obj: dict[str, Any] = {
        "match_contract_version": MATCH_CONTRACT_VERSION,
        "feature_input_contract_version": FEATURE_INPUT_CONTRACT_VERSION,
        "input_unit": "aligned_feature",
        "spec_df": spec_df,
        "top_report": top_report,
        "all_hits": all_hits,
        "all_decoy_hits": all_decoy_hits,
        "seed_candidate_hits": seed_candidate_hits,
        "seed_competitor_margin": float(
            getattr(args, "seed_competitor_margin", 0.01)
        ),
        "target_score_arr": np.asarray(target_scores, dtype=np.float32),
        "decoy_score_arr": np.asarray(decoy_scores, dtype=np.float32),
        "qvalue_arr": qvalues,
        "known_match_arr": np.asarray(known, dtype=np.int32),
        "known_match_ion_mode_consistent_arr": np.asarray(mode_consistent, dtype=np.int32),
        "query_ion_mode_arr": np.asarray([item.ion_mode for item in queries], dtype=object),
        "query_collision_energy_arr": np.asarray(
            [item.collision_energy_ev for item in queries], dtype=object
        ),
        "library_top1_ion_mode_arr": np.asarray(
            [hits[0].get("ion_mode", "") if hits else "" for hits in all_hits], dtype=object
        ),
        "library_top1_adduct_arr": np.asarray(
            [hits[0].get("adduct", "") if hits else "" for hits in all_hits], dtype=object
        ),
        "known_subclass_arr": np.asarray([""] * len(queries), dtype=object),
        "known_subclass_idx_arr": np.full(len(queries), -1, dtype=np.int32),
        "subclass_vocab": [],
        "diag_hits_list": [np.asarray([], dtype=np.float32) for _ in queries],
        "nl_list": [
            np.asarray(query.precursor_mz - query.mz_array, dtype=np.float32) for query in queries
        ],
        "tokens": np.stack(tokens)
        if tokens
        else np.empty((0, args.top_k_peaks, 3), dtype=np.float32),
        "pad_masks": np.stack(masks)
        if masks
        else np.empty((0, args.top_k_peaks), dtype=np.bool_),
    }
    return match_obj, rows


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    manifest_path = Path(args.experiment_manifest)
    manifest, repo_root = _load_manifest(manifest_path)
    match_config = manifest.get("match", {})
    input_format = str(match_config.get("input_format", "")).strip().casefold()
    if input_format != "msdial_gnps":
        raise ValueError("manifest match.input_format must be 'msdial_gnps'")
    mode_config = match_config.get("feature_inputs", {}).get(args.ion_mode, match_config)
    table_patterns = args.feature_table or mode_config.get("feature_table_globs", [])
    mgf_patterns = args.spectra_mgf or mode_config.get("spectra_mgf_globs", [])
    library_patterns = args.library_glob or match_config.get("library_globs", [])
    if not table_patterns or not mgf_patterns or not library_patterns:
        raise ValueError(
            "feature table, MGF, and library patterns must be provided by CLI or manifest.match"
        )
    feature_table_path = _require_one(
        _expand_patterns(table_patterns, repo_root, {".txt", ".tsv", ".csv"}),
        "MS-DIAL feature table",
        table_patterns,
    )
    mgf_path = _require_one(
        _expand_patterns(mgf_patterns, repo_root, {".mgf"}),
        "MS-DIAL representative spectra MGF",
        mgf_patterns,
    )
    library_paths = _expand_patterns(library_patterns, repo_root, {".msp"})
    if not library_paths:
        raise FileNotFoundError(f"no MSP files matched: {library_patterns}")

    default_mode = args.library_default_ion_mode
    if default_mode is None:
        default_mode = str(match_config.get("library_default_ion_mode", ""))
    default_mode = normalize_ion_mode(default_mode)
    if default_mode and default_mode != args.ion_mode:
        raise ValueError("library_default_ion_mode must match the requested run ion mode")

    queries, input_audit = load_msdial_gnps_queries(
        feature_table_path,
        mgf_path,
        requested_mode=args.ion_mode,
        top_n=max(args.top_k_peaks, args.top_n_report),
        max_queries=args.max_queries,
        link_mz_tolerance_da=args.feature_mgf_mz_tol_da,
        link_rt_tolerance_min=args.feature_mgf_rt_tol_min,
    )
    if not queries:
        raise RuntimeError(
            f"no linked {args.ion_mode} features with MS2; input audit={input_audit}"
        )
    library_entries, unknown_library_modes = load_msp_libraries(
        library_paths,
        top_n=args.top_k_peaks,
        default_ion_mode=default_mode,
        max_entries_per_file=args.max_msp_per_file,
    )
    library_entries = [item for item in library_entries if item.ion_mode == args.ion_mode]
    if not library_entries:
        raise RuntimeError(f"no {args.ion_mode} library spectra with resolved polarity")

    match_obj, rows = run_match(queries, library_entries, args)
    match_obj.update(
        {
            "experiment_id": manifest["experiment_id"],
            "compound_class": manifest["compound_class"],
            "params": vars(args),
            "source_feature_table": str(feature_table_path),
            "source_mgf": str(mgf_path),
            "source_library_files": [str(path) for path in library_paths],
            "feature_input_audit": input_audit,
        }
    )
    paths = _output_paths(args, manifest, repo_root)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    with paths["pkl"].open("wb") as handle:
        pickle.dump(match_obj, handle)
    _write_csv(paths["csv"], rows)
    summary = {
        "stage": "generic_feature_library_match_v5",
        "match_contract_version": MATCH_CONTRACT_VERSION,
        "feature_input_contract_version": FEATURE_INPUT_CONTRACT_VERSION,
        "input_format": "msdial_gnps",
        "input_unit": "aligned_feature",
        "experiment_id": manifest["experiment_id"],
        "compound_class": manifest["compound_class"],
        "ion_mode": args.ion_mode,
        "collision_energy_policy": "audit_metadata_only",
        "library_mass_criterion": "ppm_and_absolute_da",
        "feature_mgf_link_policy": "exact_feature_id_then_mz_and_rt_integrity_checks",
        "feature_table_file": str(feature_table_path),
        "mgf_file": str(mgf_path),
        "library_file_count": len(library_paths),
        "feature_input_audit": input_audit,
        "accepted_feature_count": len(queries),
        "accepted_library_spectrum_count": len(library_entries),
        "unknown_library_mode_rejected_count": unknown_library_modes,
        "known_match_feature_count": int(np.sum(match_obj["known_match_arr"])),
        "ambiguous_seed_feature_count": sum(
            len(items) > 1 for items in match_obj["seed_candidate_hits"]
        ),
        "seed_candidate_hypothesis_count": sum(
            len(items) for items in match_obj["seed_candidate_hits"]
        ),
        "seed_competitor_margin": float(args.seed_competitor_margin),
        "seed_competition_unit": "best_spectrum_per_clean_library_entity_name",
        "known_match_unique_library_name_count": len(
            {
                str(match_obj["all_hits"][index][0].get("name_clean", ""))
                for index, value in enumerate(match_obj["known_match_arr"])
                if value and match_obj["all_hits"][index]
            }
        ),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    with paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
