#!/usr/bin/env python3
"""Build a BM25S-only index from local corpus chunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.core.retrieval import (
    BuiltRetrievalIndex,
    RetrievalError,
    build_retrieval_index,
    save_retrieval_index,
    scientific_tokenize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a BM25S-only retrieval index with the project scientific tokenizer."
        )
    )
    parser.add_argument(
        "--corpus_jsonl",
        default="rag/corpus/chunks.jsonl",
        help="Input corpus (default: rag/corpus/chunks.jsonl)",
    )
    parser.add_argument(
        "--index_path",
        default="rag/index/retrieval_index",
        help="Output index directory (default: rag/index/retrieval_index)",
    )
    parser.add_argument(
        "--summary_json",
        default=None,
        help="Default: rag/reports/retrieval_index_summary.json beside the index root.",
    )
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()
    if args.k1 <= 0:
        parser.error("--k1 must be greater than zero")
    if not 0 <= args.b <= 1:
        parser.error("--b must be between 0 and 1")
    return args


def resolve_from_project(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def tokenize(text: str) -> list[str]:
    """Backward-compatible public name for the shared scientific tokenizer."""

    return scientific_tokenize(text)


def load_chunks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Chunk at {path}:{line_number} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"Corpus contains no chunks: {path}")
    return rows


def build_index(
    chunks: Sequence[dict[str, Any]],
    k1: float = 1.5,
    b: float = 0.75,
) -> BuiltRetrievalIndex:
    """Build an in-memory BM25S index."""

    return build_retrieval_index(chunks, k1=k1, b=b)


def save_index(index: BuiltRetrievalIndex, path: Path | str) -> dict[str, Any]:
    return save_retrieval_index(index, path)


def main() -> int:
    args = parse_args()
    corpus_path = resolve_from_project(args.corpus_jsonl)
    index_path = resolve_from_project(args.index_path)
    summary_path = (
        resolve_from_project(args.summary_json)
        if args.summary_json
        else index_path.parent.parent / "reports" / "retrieval_index_summary.json"
    )
    try:
        chunks = load_chunks(corpus_path)
        index = build_index(chunks, k1=args.k1, b=args.b)
        manifest = save_index(index, index_path)
        summary = {
            "corpus_jsonl": str(corpus_path),
            "index_path": str(index_path),
            "engine": manifest["engine"],
            "bm25s_version": manifest["bm25s_version"],
            "tokenizer": manifest["tokenizer"],
            "document_count": manifest["document_count"],
            "k1": args.k1,
            "b": args.b,
            "retrieval_mode": manifest["retrieval_mode"],
            "corpus_sha256": manifest["corpus_sha256"],
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, RetrievalError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Built BM25S-only index for {manifest['document_count']} chunks: {index_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
