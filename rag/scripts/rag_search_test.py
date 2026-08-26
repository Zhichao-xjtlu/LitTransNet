#!/usr/bin/env python3
"""Inspect BM25S-only literature retrieval results."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.core.retrieval import (
    BM25SRetrievalIndex,
    RetrievalError,
    load_retrieval_index,
    scientific_tokenize,
)

DEFAULT_REPORT = PROJECT_ROOT / "rag" / "reports" / "search_test_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a top-k query against a local BM25S-only retrieval index."
    )
    parser.add_argument("--index_path", default="rag/index/retrieval_index")
    parser.add_argument("--query", required=True)
    parser.add_argument("--query_group", default="compound_queries")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_csv", default=str(DEFAULT_REPORT))
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top_k must be greater than zero")
    return args


def resolve_from_project(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def tokenize(text: str) -> list[str]:
    return scientific_tokenize(text)


def load_index(path: Path) -> BM25SRetrievalIndex:
    return load_retrieval_index(path)


def score_documents(index: BM25SRetrievalIndex, query: str) -> list[float]:
    """Return raw BM25S scores for callers needing a full sparse vector."""

    tokens = scientific_tokenize(query)
    scores = [0.0] * len(index.documents)
    if not tokens or not scores:
        return scores
    documents, values = index.bm25.retrieve(
        [tokens], k=len(scores), show_progress=False
    )
    for document_index, value in zip(documents[0], values[0]):
        scores[int(document_index)] = float(value)
    return scores


def locator(metadata: dict[str, Any]) -> str:
    if metadata.get("page") is not None:
        return f"page {metadata['page']}"
    parts = []
    if metadata.get("sheet_name") is not None:
        parts.append(f"sheet {metadata['sheet_name']}")
    if metadata.get("row_index") is not None:
        parts.append(f"row {metadata['row_index']}")
    return ", ".join(parts) if parts else "-"


def preview(text: str, limit: int = 360) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "score",
        "bm25s_score",
        "evidence_alignment",
        "chunk_id",
        "source_file",
        "page",
        "sheet_name",
        "row_index",
        "section",
        "text_preview",
        "text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args()
    index_path = resolve_from_project(args.index_path)
    output_path = resolve_from_project(args.output_csv)
    try:
        index = load_index(index_path)
        hits = index.search(
            args.query,
            query_group=args.query_group,
            top_k=args.top_k,
        )
    except (OSError, ValueError, RetrievalError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    for hit in hits:
        metadata = {key: value for key, value in hit.document.items() if key != "text"}
        text = str(hit.document.get("text") or "")
        row = {
            "rank": hit.rank,
            "score": f"{hit.score:.6f}",
            "bm25s_score": f"{hit.sparse_score:.6f}",
            "evidence_alignment": f"{hit.evidence_alignment:.6f}",
            "chunk_id": metadata.get("chunk_id", ""),
            "source_file": metadata.get("source_file", ""),
            "page": metadata.get("page"),
            "sheet_name": metadata.get("sheet_name"),
            "row_index": metadata.get("row_index"),
            "section": metadata.get("section"),
            "text_preview": preview(text),
            "text": text,
        }
        rows.append(row)
        print(f"\nrank: {hit.rank}")
        print(f"score: {row['score']}")
        print(
            f"components: BM25S={row['bm25s_score']} "
            f"evidence_audit={row['evidence_alignment']}"
        )
        print(f"chunk_id: {row['chunk_id']}")
        print(f"source_file: {row['source_file']}")
        print(f"location: {locator(metadata)}")
        print(f"text preview: {row['text_preview']}")

    write_results(output_path, rows)
    print(f"\nResults CSV: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
