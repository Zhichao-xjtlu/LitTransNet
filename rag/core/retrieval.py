"""BM25S-only local retrieval for scientific metabolomics literature.

The project tokenizer deliberately preserves scientific punctuation, ions,
decimal masses, and positional chemical names.  BM25S is the only production
indexing and ranking engine; evidence-family alignment is retained as an audit
field and never changes rank order.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from rag.core.io_utils import iter_jsonl


INDEX_FORMAT_VERSION = 3

TOKEN_PATTERN = re.compile(
    r"\[[^\]\r\n]{1,40}\][+-]?|\bm\s*/\s*z\b|"
    r"\d+(?:\.\d+)?(?:-[a-z0-9]+)+|\d+(?:\.\d+)?|"
    r"[a-z]+(?:[-'][a-z0-9]+)*",
    re.IGNORECASE,
)


class RetrievalError(RuntimeError):
    """Raised when a BM25S index or query is unusable."""


def normalize_symbols(text: str) -> str:
    """Normalize typography without deleting scientific punctuation/numbers."""

    return (
        str(text or "")
        .lower()
        .replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\uff0f", "/")
        .replace("\uff0b", "+")
    )


def scientific_tokenize(text: str) -> list[str]:
    """Retain ions, ``m/z``, decimal masses, and hyphenated chemical terms."""

    return [
        re.sub(r"\s+", "", match.group(0))
        for match in TOKEN_PATTERN.finditer(normalize_symbols(text))
    ]


def infer_document_role(metadata: dict[str, Any]) -> str:
    explicit = str(metadata.get("document_role") or "").strip()
    if explicit:
        return explicit
    source = str(metadata.get("source_file") or "").casefold()
    section = str(metadata.get("section") or "").casefold()
    file_type = str(metadata.get("file_type") or "").casefold()
    if "supplement" in source or "supplement" in section:
        return "supplementary_table"
    if metadata.get("row_index") is not None or file_type in {"csv", "xlsx"}:
        return "compound_catalog"
    if "table" in section:
        return "experimental_table"
    if "review" in source or "review" in section:
        return "review"
    return "article_text"


EVIDENCE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "compound": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bcompound(?:s)?\b",
            r"\bidentif(?:y|ied|ication)\b",
            r"\bmolecular formula\b",
            r"\bexact mass\b",
            r"\b(?:lc|uhplc|uplc)[-\s]?(?:ms|ms/ms)\b",
        )
    ),
    "fragment": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bfragment(?:ation|s)?\b",
            r"\bproduct ion(?:s)?\b",
            r"\bdiagnostic ion(?:s)?\b",
            r"\bms\s*/\s*ms\b",
            r"\bcid\b",
        )
    ),
    "neutral_loss": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bneutral loss(?:es)?\b",
            r"\bloss of\b",
            r"\bfragment(?:ation)?\b",
            r"\bproduct ion(?:s)?\b",
        )
    ),
    "transformation": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\btransformation(?:s)?\b",
            r"\bconvert(?:s|ed|ing)?\b",
            r"\bderived from\b",
            r"\bformed (?:by|from)\b",
            r"\breact(?:s|ed|ion)?\b",
            r"\b(?:hydrolysis|oxidation|reduction|conjugation|glycosylation)\b",
        )
    ),
    "biosynthesis": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bbiosynth(?:esis|etic|esized)\b",
            r"\bpathway\b",
            r"\bprecursor(?:s)?\b",
            r"\bcomponent(?:s)?\b",
            r"\bsubstituent(?:s)?\b",
            r"\bmoiet(?:y|ies)\b",
        )
    ),
    "supplementary": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bsupplement(?:ary)?\b",
            r"\btable\s+s?\d+\b",
            r"\bretention time\b",
            r"\bobserved m\s*/\s*z\b",
            r"\bcalculated m\s*/\s*z\b",
        )
    ),
    "review": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (r"\breview\b", r"\boverview\b", r"\bsummar(?:y|izes|ised)\b")
    ),
}


def evidence_family(query_group: str) -> str:
    value = str(query_group or "").casefold()
    if "neutral_loss" in value:
        return "neutral_loss"
    if "fragment" in value or "product_ion" in value:
        return "fragment"
    if "transformation" in value or "reaction" in value or "source_target" in value:
        return "transformation"
    if "biosynth" in value or "component" in value or "formula" in value:
        return "biosynthesis"
    if "supplement" in value or "catalog" in value:
        return "supplementary"
    if "review" in value:
        return "review"
    return "compound"


def evidence_alignment_score(text: str, metadata: dict[str, Any], query_group: str) -> float:
    """Return an auditable evidence-family indicator without reranking hits."""

    family = evidence_family(query_group)
    patterns = EVIDENCE_PATTERNS[family]
    matched = sum(bool(pattern.search(text)) for pattern in patterns)
    lexical = matched / max(1, len(patterns))
    role = infer_document_role(metadata)
    role_bonus = 0.0
    if family == "supplementary" and role in {
        "supplementary_table",
        "compound_catalog",
        "experimental_table",
    }:
        role_bonus = 0.35
    elif family in {"compound", "fragment", "neutral_loss"} and role in {
        "supplementary_table",
        "compound_catalog",
        "experimental_table",
    }:
        role_bonus = 0.15
    elif family == "review" and role == "review":
        role_bonus = 0.25
    return min(1.0, lexical + role_bonus)


def _require_bm25s() -> Any:
    try:
        import bm25s
    except ImportError as exc:  # pragma: no cover - clean-install behavior
        raise RetrievalError(
            "BM25S is required. Install RAG dependencies with: "
            "python -m pip install -r rag/requirements-rag.txt"
        ) from exc
    return bm25s


def _corpus_digest(documents: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(
            json.dumps(
                document,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass
class BuiltRetrievalIndex:
    bm25: Any
    documents: list[dict[str, Any]]
    tokenized_corpus: list[list[str]]
    k1: float
    b: float


def build_retrieval_index(
    chunks: Sequence[dict[str, Any]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> BuiltRetrievalIndex:
    if not chunks:
        raise RetrievalError("Corpus contains no chunks")
    documents: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in chunks:
        row = dict(raw)
        chunk_id = str(row.get("chunk_id") or "")
        text = row.get("text")
        if not chunk_id or chunk_id in seen_ids:
            raise RetrievalError(f"Missing or duplicate chunk_id: {chunk_id!r}")
        if not isinstance(text, str):
            raise RetrievalError(f"Chunk {chunk_id!r} has no text string")
        seen_ids.add(chunk_id)
        documents.append(row)

    tokens = [scientific_tokenize(document["text"]) for document in documents]
    bm25s = _require_bm25s()
    retriever = bm25s.BM25(k1=k1, b=b, method="lucene")
    retriever.index(tokens, show_progress=False)
    return BuiltRetrievalIndex(
        bm25=retriever,
        documents=documents,
        tokenized_corpus=tokens,
        k1=k1,
        b=b,
    )


def save_retrieval_index(index: BuiltRetrievalIndex, path: Path | str) -> dict[str, Any]:
    root = Path(path)
    if root.exists() and root.is_file():
        raise RetrievalError(
            f"Index path is a file, but BM25S requires a directory: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    stale_dense = root / "dense_embeddings.npy"
    if stale_dense.exists():
        stale_dense.unlink()

    bm25_dir = root / "bm25s"
    index.bm25.save(bm25_dir, show_progress=False)

    documents_path = root / "documents.jsonl"
    with documents_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in index.documents:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tokens_path = root / "scientific_tokens.jsonl.gz"
    with gzip.open(tokens_path, "wt", encoding="utf-8", newline="\n") as handle:
        for row, tokens in zip(index.documents, index.tokenized_corpus):
            handle.write(
                json.dumps(
                    {"chunk_id": row.get("chunk_id", ""), "tokens": tokens},
                    ensure_ascii=False,
                )
                + "\n"
            )

    bm25s = _require_bm25s()
    manifest = {
        "format_version": INDEX_FORMAT_VERSION,
        "engine": "bm25s",
        "retrieval_mode": "sparse_only",
        "bm25s_version": str(getattr(bm25s, "__version__", "unknown")),
        "bm25_method": "lucene",
        "tokenizer": "scientific_regex_v1",
        "params": {"k1": index.k1, "b": index.b},
        "document_count": len(index.documents),
        "corpus_sha256": _corpus_digest(index.documents),
        "documents_file": documents_path.name,
        "tokens_file": tokens_path.name,
    }
    (root / "retrieval_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return list(iter_jsonl(path))
    except (OSError, ValueError) as exc:
        raise RetrievalError(f"Unable to read retrieval documents: {path}: {exc}") from exc


@dataclass(frozen=True)
class RetrievalHit:
    rank: int
    document_index: int
    document: dict[str, Any]
    score: float
    sparse_score: float
    evidence_alignment: float

    @property
    def final_score(self) -> float:
        """Alias retained for neutral downstream serialization."""

        return self.score


class BM25SRetrievalIndex:
    def __init__(
        self,
        *,
        root: Path,
        manifest: dict[str, Any],
        bm25: Any,
        documents: list[dict[str, Any]],
    ):
        self.root = root
        self.manifest = manifest
        self.bm25 = bm25
        self.documents = documents

    @property
    def chunk_metadata(self) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in row.items() if key != "text"}
            for row in self.documents
        ]

    @property
    def original_texts(self) -> list[str]:
        return [str(row.get("text") or "") for row in self.documents]

    def search(
        self,
        query: str,
        *,
        query_group: str = "compound_queries",
        top_k: int = 12,
    ) -> list[RetrievalHit]:
        if top_k <= 0:
            raise RetrievalError("top_k must be greater than zero")
        query_tokens = scientific_tokenize(query)
        if not query_tokens or not self.documents:
            return []
        limit = min(top_k, len(self.documents))
        document_ids, values = self.bm25.retrieve(
            [query_tokens], k=limit, show_progress=False
        )
        candidates: list[tuple[int, float]] = []
        for document_index, value in zip(document_ids[0], values[0]):
            score = float(value)
            if score > 0.0:
                candidates.append((int(document_index), score))
        candidates.sort(key=lambda item: (-item[1], item[0]))

        hits: list[RetrievalHit] = []
        for rank, (document_index, score) in enumerate(candidates, start=1):
            document = self.documents[document_index]
            metadata = {key: value for key, value in document.items() if key != "text"}
            hits.append(
                RetrievalHit(
                    rank=rank,
                    document_index=document_index,
                    document=document,
                    score=score,
                    sparse_score=score,
                    evidence_alignment=evidence_alignment_score(
                        str(document.get("text") or ""), metadata, query_group
                    ),
                )
            )
        return hits


def load_retrieval_index(path: Path | str) -> BM25SRetrievalIndex:
    root = Path(path)
    if not root.is_dir():
        raise RetrievalError(
            f"Retrieval index directory not found: {root}. "
            "Rebuild it with rag_build_bm25_index.py."
        )
    manifest_path = root / "retrieval_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalError(f"Invalid retrieval manifest: {manifest_path}: {exc}") from exc
    if manifest.get("format_version") != INDEX_FORMAT_VERSION:
        raise RetrievalError(
            f"Unsupported retrieval index format: {manifest.get('format_version')!r}; "
            f"expected {INDEX_FORMAT_VERSION} (BM25S-only)"
        )
    if manifest.get("engine") != "bm25s" or manifest.get("retrieval_mode") != "sparse_only":
        raise RetrievalError("Retrieval manifest is not a BM25S-only index")
    documents = _load_jsonl(
        root / str(manifest.get("documents_file") or "documents.jsonl")
    )
    if len(documents) != int(manifest.get("document_count") or -1):
        raise RetrievalError("Retrieval manifest document_count does not match documents.jsonl")
    if _corpus_digest(documents) != manifest.get("corpus_sha256"):
        raise RetrievalError("Retrieval corpus checksum mismatch")
    bm25s = _require_bm25s()
    bm25 = bm25s.BM25.load(root / "bm25s", load_corpus=False, show_progress=False)
    return BM25SRetrievalIndex(
        root=root,
        manifest=manifest,
        bm25=bm25,
        documents=documents,
    )


def write_retrieval_results_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
