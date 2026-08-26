"""Run the generic match -> entity-network chain for one experiment/mode."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run feature-table/MGF library matching and the entity-reaction network"
    )
    parser.add_argument("--experiment_manifest", required=True)
    parser.add_argument("--ion_mode", required=True, choices=["positive", "negative"])
    parser.add_argument("--stage", choices=["match", "network", "all"], default="all")
    parser.add_argument("--rules_dir", default="", help="Override manifest.network.rules_dir")
    parser.add_argument("--match_pkl", default="", help="Override the canonical match pickle for network")
    parser.add_argument(
        "--feature_table",
        default="",
        help="Optional single MS-DIAL GNPS-export feature table override.",
    )
    parser.add_argument(
        "--spectra_mgf",
        default="",
        help="Optional single MS-DIAL GNPS-export MGF override.",
    )
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--match_qvalue_max", type=float, default=0.25)
    parser.add_argument("--seed_competitor_margin", type=float, default=0.01)
    parser.add_argument("--mass_tol_ppm", type=float, default=10.0)
    parser.add_argument("--mass_tol_da", type=float, default=0.01)
    parser.add_argument("--max_depth", type=int, default=2)
    return parser.parse_args(argv)


def _repo_root() -> Path:
    """Resolve the repository root independently of manifest nesting."""
    return Path(__file__).resolve().parents[2]


def _manifest(path: Path) -> tuple[dict[str, Any], Path]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value, _repo_root()


def _absolute(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _run(command: list[str], root: Path) -> None:
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main(argv: Sequence[str] | None = None) -> dict[str, str]:
    args = parse_args(argv)
    manifest_path = Path(args.experiment_manifest).resolve()
    manifest, repo_root = _manifest(manifest_path)
    experiment_id = str(manifest["experiment_id"])
    database_dir = _absolute(manifest["workspace"]["database_dir"], repo_root)
    match_dir = database_dir / "match" / args.ion_mode
    result_dir = database_dir / "results" / args.ion_mode
    match_path = (
        _absolute(args.match_pkl, repo_root)
        if args.match_pkl
        else match_dir / f"{experiment_id}.{args.ion_mode}.match_v5.pkl"
    )

    if args.stage in {"match", "all"}:
        command = [
            sys.executable,
            str(repo_root / "dl_annotator" / "v5" / "match_branch_v5.py"),
            "--experiment_manifest",
            str(manifest_path),
            "--ion_mode",
            args.ion_mode,
            "--match_qvalue_max",
            str(args.match_qvalue_max),
            "--seed_competitor_margin",
            str(args.seed_competitor_margin),
        ]
        if args.feature_table:
            command.extend(["--feature_table", args.feature_table])
        if args.spectra_mgf:
            command.extend(["--spectra_mgf", args.spectra_mgf])
        if args.max_queries > 0:
            command.extend(["--max_queries", str(args.max_queries)])
        _run(command, repo_root)

    if args.stage in {"network", "all"}:
        if not match_path.exists():
            raise FileNotFoundError(f"match pickle not found: {match_path}")
        configured_rules = manifest.get("network", {}).get("rules_dir", "")
        rules_dir = _absolute(args.rules_dir or configured_rules, repo_root)
        if not (rules_dir / "rules_manifest.json").exists():
            raise FileNotFoundError(
                f"versioned rules are not ready: {rules_dir / 'rules_manifest.json'}"
            )
        result_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(repo_root / "dl_annotator" / "v5" / "network_branch_v5.py"),
            "--match_pkl",
            str(match_path),
            "--rag_rules_dir",
            str(rules_dir),
            "--ion_mode",
            args.ion_mode,
            "--mass_tol_ppm",
            str(args.mass_tol_ppm),
            "--mass_tol_da",
            str(args.mass_tol_da),
            "--max_depth",
            str(args.max_depth),
            "--result_csv",
            str(result_dir / "network_result.csv"),
            "--hypothesis_audit_csv",
            str(result_dir / "hypothesis_audit.csv"),
            "--cytoscape_edges_csv",
            str(result_dir / "network_edges.csv"),
            "--out_pkl",
            str(result_dir / "network_v5.pkl"),
            "--out_summary_json",
            str(result_dir / "network_summary.json"),
        ]
        _run(command, repo_root)

    outputs = {
        "match_pkl": str(match_path),
        "network_result_csv": str(result_dir / "network_result.csv"),
        "hypothesis_audit_csv": str(result_dir / "hypothesis_audit.csv"),
        "network_summary_json": str(result_dir / "network_summary.json"),
    }
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return outputs


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
