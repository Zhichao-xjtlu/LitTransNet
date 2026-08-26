"""CLI adapter for the registry-backed Network V5 engine."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dl_annotator.v5.entity_hypothesis_network import (
    NetworkConfig,
    load_network_knowledge,
    run_entity_hypothesis_network,
)
from dl_annotator.v5.preflight import run_network_preflight
from dl_annotator.v5.network_io import (
    load_match_nodes_and_seeds,
    nodes_and_seeds,
    write_rows as _write_rows,
)

_nodes_and_seeds = nodes_and_seeds


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Registry-backed evidence graph Network V5"
    )
    parser.add_argument("--match_pkl", required=True)
    parser.add_argument("--rag_rules_dir", required=True)
    parser.add_argument("--ion_mode", required=True, choices=["positive", "negative"])
    parser.add_argument("--mass_tol_ppm", type=float, default=10.0)
    parser.add_argument("--mass_tol_da", type=float, default=0.01)
    parser.add_argument("--fragment_tol_da", type=float, default=0.10)
    parser.add_argument(
        "--fragment_gate_mode",
        choices=[
            "evidence_bundle",
            "reaction_evidence",
            "strict",
            "high_recall",
            "mass_only",
            "required_empty",
            "theoretical_catalog_only",
        ],
        default="reaction_evidence",
    )
    parser.add_argument(
        "--min_class_diagnostic_matches",
        type=int,
        choices=[2, 3],
        default=2,
    )
    parser.add_argument("--min_target_product_ions", type=int, default=2)
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--result_csv", required=True)
    parser.add_argument("--hypothesis_audit_csv", required=True)
    parser.add_argument("--cytoscape_edges_csv", required=True)
    parser.add_argument("--out_pkl", required=True)
    parser.add_argument("--out_summary_json", required=True)
    return parser.parse_args(argv)


def summarize_discoveries(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Count reported entities and reported ambiguity groups as formal units."""

    exact_ids = {
        str(row.get("target_entity_id", "")).strip()
        for row in rows
        if str(row.get("annotation_status", "")).strip()
        in {
            "fragment_supported_discover",
            "exact_discover",
            "mechanism_supported_discover",
        }
        and str(row.get("target_entity_id", "")).strip()
    }
    ambiguous_groups = {
        str(row.get("target_entity_group_id", "")).strip(): str(
            row.get("ambiguous_target_entities", "")
        ).strip()
        for row in rows
        if str(row.get("annotation_status", "")).strip()
        == "ambiguous_group_discover"
        and str(row.get("target_entity_group_id", "")).strip()
    }
    ambiguous_member_ids = {
        entity_id.strip()
        for row in rows
        if str(row.get("annotation_status", "")).strip()
        == "ambiguous_group_discover"
        for entity_id in str(
            row.get("ambiguous_target_entity_ids", "")
        ).split(";")
        if entity_id.strip()
    }
    resolved_exact_ids = exact_ids - ambiguous_member_ids
    formal_statuses = {
        "fragment_supported_discover",
        "exact_discover",
        "mechanism_supported_discover",
        "ambiguous_group_discover",
    }
    return {
        "formal_discover_node_count": sum(
            str(row.get("annotation_status", "")).strip()
            in formal_statuses
            for row in rows
        ),
        "raw_exact_discover_entity_count": len(exact_ids),
        "exact_discover_entity_count": len(resolved_exact_ids),
        "ambiguous_discover_group_count": len(ambiguous_groups),
        "formal_discovery_unit_count": (
            len(resolved_exact_ids) + len(ambiguous_groups)
        ),
        "exact_discover_entity_ids": sorted(resolved_exact_ids),
        "raw_exact_discover_entity_ids": sorted(exact_ids),
        "ambiguous_discover_groups": dict(sorted(ambiguous_groups.items())),
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    knowledge = load_network_knowledge(Path(args.rag_rules_dir))
    nodes, seeds = load_match_nodes_and_seeds(
        Path(args.match_pkl), args.ion_mode, knowledge
    )
    config = NetworkConfig(
        ppm_tolerance=args.mass_tol_ppm,
        absolute_tolerance_da=args.mass_tol_da,
        fragment_tolerance_da=args.fragment_tol_da,
        fragment_gate_mode=args.fragment_gate_mode,
        min_class_diagnostic_matches=args.min_class_diagnostic_matches,
        min_target_product_ions=args.min_target_product_ions,
        max_depth=args.max_depth,
    )
    preflight = run_network_preflight(nodes, seeds, knowledge, config)
    if not preflight.ready:
        raise RuntimeError(
            "network preflight stopped enumeration: "
            + ",".join(preflight.stop_reasons)
        )
    result = run_entity_hypothesis_network(
        nodes,
        seeds,
        knowledge,
        config,
    )
    _write_rows(Path(args.result_csv), result.nodes)
    _write_rows(Path(args.hypothesis_audit_csv), result.hypotheses)
    _write_rows(Path(args.cytoscape_edges_csv), result.edges)
    with Path(args.out_pkl).open("wb") as handle:
        pickle.dump(result, handle)
    annotation_status_counts: dict[str, int] = {}
    for row in result.nodes:
        status = str(row.get("annotation_status", "")).strip()
        annotation_status_counts[status] = (
            annotation_status_counts.get(status, 0) + 1
        )
    discovery_summary = summarize_discoveries(result.nodes)
    summary = {
        "input_unit": "aligned_feature",
        "feature_count": len(result.nodes),
        "node_count": len(result.nodes),
        "hypothesis_count": len(result.hypotheses),
        "edge_count": len(result.edges),
        "annotation_status_counts": dict(
            sorted(annotation_status_counts.items())
        ),
        "exact_discover_node_count": annotation_status_counts.get(
            "fragment_supported_discover", 0
        ),
        "ambiguous_group_discover_node_count": annotation_status_counts.get(
            "ambiguous_group_discover", 0
        ),
        **discovery_summary,
        "seed_feature_count": len({seed.node_id for seed in seeds}),
        "seed_hypothesis_count": len(seeds),
        "ambiguous_library_seed_feature_count": annotation_status_counts.get(
            "ambiguous_library_seed", 0
        ),
        "resolved_seed_feature_count": sum(
            row.get("annotation_status") == "known_match"
            and bool(row.get("propagation_eligible"))
            for row in result.nodes
        ),
        "resolved_seed_count": sum(
            row.get("annotation_status") == "known_match"
            and bool(row.get("propagation_eligible"))
            for row in result.nodes
        ),
        "fragment_gate_mode": config.fragment_gate_mode,
        "min_class_diagnostic_matches": (
            config.min_class_diagnostic_matches
        ),
        "min_target_product_ions": config.min_target_product_ions,
        "preflight": preflight.__dict__,
    }
    Path(args.out_summary_json).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
