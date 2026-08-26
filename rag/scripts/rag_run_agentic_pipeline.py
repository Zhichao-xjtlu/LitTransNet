#!/usr/bin/env python3
"""Run the canonical BM25S literature-to-rules pipeline exactly once."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from rag.core.entity_registry import build_entity_registry, write_entity_registry
from rag.core.evidence_inventory import (
    build_evidence_inventory,
    derive_fragment_specificity,
    write_evidence_inventory,
)
from rag.core.io_utils import (
    read_jsonl as _read_jsonl,
    resolve_project_path,
    sha256_file,
)
from rag.core.stage_cache import StageCache
from rag.core.literature_mining import (
    LiteratureMiningError,
    chat_completion_openai_compatible,
    default_query_plan_path,
    load_chunks,
    load_json,
    merge_claims,
    run_literature_mining,
    safe_stem,
    write_json,
    write_jsonl,
)
from rag.core.rule_compilation import RuleCompilationError, compile_rules


METHOD_VERSION = "bm25s-deterministic-compiler/1.0"


class AgenticPipelineError(RuntimeError):
    """Raised when the one-pass literature-to-rules workflow cannot continue."""


def resolve(path: Path | str) -> Path:
    return resolve_project_path(path, PROJECT_ROOT)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return _read_jsonl(path)
    except (OSError, ValueError) as exc:
        raise AgenticPipelineError(f"Unable to read JSONL {path}: {exc}") from exc


def _concepts_path(root: Path, compound_class: str) -> Path:
    return root / "discovered_concepts" / f"{safe_stem(compound_class)}_concepts.json"


def _compiler_cache_outputs(
    rules_dir: Path,
    compiler_report_path: Path,
    gap_summary_path: Path,
) -> dict[str, Path]:
    outputs = {
        "compiler_report": compiler_report_path,
        "gap_summary": gap_summary_path,
        "rules_manifest": rules_dir / "rules_manifest.json",
    }
    for name in (
        "compound_rules.csv",
        "transformation_rules.csv",
        "diagnostic_fragment_rules.csv",
        "neutral_loss_rules.csv",
        "biosynthetic_component_rules.csv",
    ):
        outputs[f"rules.{name}"] = rules_dir / name
    return outputs


def _freeze_claim_set(
    root: Path,
    base_claims: Sequence[Mapping[str, Any]],
    reaction_recovery_claims_jsonl: Path | str | None,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    claims = [dict(row) for row in base_claims]
    recovery_path = (
        resolve(reaction_recovery_claims_jsonl)
        if reaction_recovery_claims_jsonl
        else None
    )
    recovery_claims: list[dict[str, Any]] = []
    recovery_added_count = 0
    if recovery_path is not None:
        recovery_claims = read_jsonl(recovery_path)
        before_count = len(claims)
        claims = merge_claims(claims, recovery_claims)
        recovery_added_count = len(claims) - before_count
    if not claims:
        raise AgenticPipelineError("The final pre-compiler claim set is empty")

    frozen_path = root / "evidence_claims" / "frozen_claim_set.jsonl"
    write_jsonl(frozen_path, claims)
    frozen_utc = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "frozen-claim-set/1.0",
        "frozen_utc": frozen_utc,
        "path": str(frozen_path),
        "sha256": sha256_file(frozen_path),
        "claim_count": len(claims),
        "agent2_claim_count": len(base_claims),
        "reaction_recovery_input_claim_count": len(recovery_claims),
        "reaction_recovery_added_claim_count": recovery_added_count,
        "reaction_recovery_claims_path": str(recovery_path or ""),
        "mutable_after_freeze": False,
    }
    write_json(root / "reports" / "frozen_claim_set_manifest.json", manifest)
    return frozen_path, claims, manifest


def run_agentic_pipeline(
    *,
    query_plan_path: Path | str,
    corpus_jsonl: Path | str,
    index_path: Path | str,
    output_root: Path | str,
    compound_class: str,
    chat_completion: Callable[
        [list[dict[str, str]], str, str, str], str
    ] = chat_completion_openai_compatible,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    reuse_mining_results: bool = False,
    top_k: int = 12,
    max_iterations: int = 2,
    reaction_recovery_claims_jsonl: Path | str | None = None,
    registry_output_dir: Path | str | None = None,
    rules_output_dir: Path | str | None = None,
    min_reported_fragment_compound_count: int = 20,
    min_reported_fragment_support_fraction: float = 0.20,
    max_class_consensus_fragments: int = 20,
    force: bool = False,
    mining_runner: Callable[..., dict[str, Any]] = run_literature_mining,
    compiler_runner: Callable[..., dict[str, Any]] = compile_rules,
) -> dict[str, Any]:
    """Mine or reuse claims, freeze them, compile once, and stop."""

    root = resolve(output_root)
    root.mkdir(parents=True, exist_ok=True)
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    claims_path = root / "evidence_claims" / "evidence_claims.jsonl"

    if reuse_mining_results:
        mining_summary_path = reports_dir / "literature_mining_summary.json"
        if not mining_summary_path.exists() or not claims_path.exists():
            raise AgenticPipelineError(
                "--reuse_mining_results requires the Agent 2 summary and claims under output_root"
            )
        mining_summary = load_json(mining_summary_path)
    else:
        mining_summary = mining_runner(
            query_plan_path=query_plan_path,
            corpus_jsonl=corpus_jsonl,
            index_path=index_path,
            output_root=output_root,
            chat_completion=chat_completion,
            model=model,
            base_url=base_url,
            api_key=api_key,
            top_k=top_k,
            max_iterations=max_iterations,
        )
    if not claims_path.exists():
        raise AgenticPipelineError(f"Agent 2 did not write claims: {claims_path}")

    base_claims = read_jsonl(claims_path)
    frozen_path, frozen_claims, frozen_manifest = _freeze_claim_set(
        root, base_claims, reaction_recovery_claims_jsonl
    )

    registry_dir = (
        resolve(registry_output_dir) if registry_output_dir else root / "entity_registry"
    )
    rules_dir = resolve(rules_output_dir) if rules_output_dir else root / "rules_candidate"
    compiler_report_path = reports_dir / "compiler_report.json"
    gap_summary_path = reports_dir / "compiler_gap_audit_summary.json"
    concepts_path = _concepts_path(root, compound_class)
    cache_inputs: dict[str, Path] = {
        "frozen_claims": frozen_path,
        "corpus": resolve(corpus_jsonl),
        "implementation.controller": Path(__file__).resolve(),
    }
    for source_path in sorted((PROJECT_ROOT / "rag" / "core").glob("*.py")):
        cache_inputs[f"implementation.{source_path.name}"] = source_path
    if concepts_path.exists():
        cache_inputs["concepts"] = concepts_path
    cache_params = {
        "compound_class": compound_class,
        "min_reported_fragment_compound_count": min_reported_fragment_compound_count,
        "min_reported_fragment_support_fraction": min_reported_fragment_support_fraction,
        "max_class_consensus_fragments": max_class_consensus_fragments,
    }
    cache_outputs = _compiler_cache_outputs(
        rules_dir, compiler_report_path, gap_summary_path
    )
    cache_path = root / "cache" / "deterministic_compiler.json"
    cache = StageCache()
    cache_decision = cache.validate(
        cache_path,
        stage="inventory-registry-compiler",
        version=METHOD_VERSION,
        params=cache_params,
        inputs=cache_inputs,
        outputs=cache_outputs,
    )

    if cache_decision.hit and not force:
        compiler_summary = load_json(compiler_report_path)
    else:
        # LLM boundary: everything below this line is deterministic and local.
        chunks_by_id = load_chunks(resolve(corpus_jsonl))
        inventory = build_evidence_inventory(frozen_claims, chunks_by_id)
        inventory = type(inventory)(
            evidence=inventory.evidence,
            fragments=derive_fragment_specificity(inventory.fragments),
            rejected=inventory.rejected,
        )
        write_evidence_inventory(inventory, registry_dir)
        registry = build_entity_registry(frozen_claims, inventory)
        write_entity_registry(registry, registry_dir)

        compiler_summary = compiler_runner(
            evidence_claims_jsonl=frozen_path,
            concepts_json=concepts_path if concepts_path.exists() else None,
            output_dir=rules_dir,
            report_path=compiler_report_path,
            registry_dir=registry_dir,
            min_reported_fragment_compound_count=min_reported_fragment_compound_count,
            min_reported_fragment_support_fraction=min_reported_fragment_support_fraction,
            max_class_consensus_fragments=max_class_consensus_fragments,
        )
        if all(path.is_file() for path in cache_outputs.values()):
            cache.commit(
                cache_path,
                stage="inventory-registry-compiler",
                version=METHOD_VERSION,
                params=cache_params,
                inputs=cache_inputs,
                outputs=cache_outputs,
            )
    gap_summary = load_json(gap_summary_path) if gap_summary_path.exists() else {}

    summary = {
        "method_version": METHOD_VERSION,
        "compound_class": compound_class,
        "retrieval": {
            "engine": "bm25s",
            "mode": "sparse_only",
            "tokenizer": "scientific_regex_v1",
            "index_path": str(resolve(index_path)),
            "top_k": top_k,
        },
        "literature_mining": mining_summary,
        "claim_set": frozen_manifest,
        "compilation_pass_count": 1,
        "post_freeze_llm_call_count": 0,
        "compiler": compiler_summary,
        "terminal_gap_audit": gap_summary,
        "registry_output_dir": str(registry_dir),
        "rules_output_dir": str(rules_dir),
    }
    write_json(reports_dir / "agentic_pipeline_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run BM25S literature mining, freeze the evidence claims, compile "
            "once, and emit a deterministic terminal gap audit."
        )
    )
    parser.add_argument("--compound_class", required=True)
    parser.add_argument("--query_plan", default=None)
    parser.add_argument("--corpus_jsonl", default="rag/corpus/chunks.jsonl")
    parser.add_argument("--index_path", default="rag/index/retrieval_index")
    parser.add_argument("--output_root", default="rag")
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--max_iterations", type=int, default=2)
    parser.add_argument("--reuse_mining_results", action="store_true")
    parser.add_argument(
        "--reaction_recovery_claims_jsonl",
        default=None,
        help="Optional evidence-linked claims produced before the claim-set freeze.",
    )
    parser.add_argument("--registry_output_dir", default=None)
    parser.add_argument("--rules_output_dir", default=None)
    parser.add_argument("--min_reported_fragment_compound_count", type=int, default=20)
    parser.add_argument(
        "--min_reported_fragment_support_fraction", type=float, default=0.20
    )
    parser.add_argument("--max_class_consensus_fragments", type=int, default=20)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass valid deterministic stage-cache entries.",
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top_k must be > 0")
    if args.max_iterations <= 0:
        parser.error("--max_iterations must be > 0")
    if args.min_reported_fragment_compound_count < 1:
        parser.error("--min_reported_fragment_compound_count must be >= 1")
    if not 0.0 <= args.min_reported_fragment_support_fraction <= 1.0:
        parser.error("--min_reported_fragment_support_fraction must be in [0, 1]")
    if args.max_class_consensus_fragments < 1:
        parser.error("--max_class_consensus_fragments must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    query_plan = args.query_plan or default_query_plan_path(args.compound_class)
    try:
        summary = run_agentic_pipeline(
            query_plan_path=query_plan,
            corpus_jsonl=args.corpus_jsonl,
            index_path=args.index_path,
            output_root=args.output_root,
            compound_class=args.compound_class,
            reuse_mining_results=args.reuse_mining_results,
            top_k=args.top_k,
            max_iterations=args.max_iterations,
            reaction_recovery_claims_jsonl=args.reaction_recovery_claims_jsonl,
            registry_output_dir=args.registry_output_dir,
            rules_output_dir=args.rules_output_dir,
            min_reported_fragment_compound_count=args.min_reported_fragment_compound_count,
            min_reported_fragment_support_fraction=(
                args.min_reported_fragment_support_fraction
            ),
            max_class_consensus_fragments=args.max_class_consensus_fragments,
            force=args.force,
        )
    except (
        AgenticPipelineError,
        LiteratureMiningError,
        RuleCompilationError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
