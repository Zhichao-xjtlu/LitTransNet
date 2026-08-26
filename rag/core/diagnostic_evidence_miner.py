#!/usr/bin/env python3
"""Mine diagnostic-fragment evidence claims from local literature chunks.

Diagnostic fragments are evidence-type driven here. A fragment is emitted only
when its local text context indicates diagnostic/fingerprint/structural-mapping
evidence. Plain reported fragment tables are not enough.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.core.io_utils import (
    atomic_write_json as write_json,
    clean_text,
    join_unique,
    safe_float,
    split_values,
)

EXPLICIT_DIAGNOSTIC_TERMS = [
    "diagnostic",
    "distinctive",
    "fingerprint",
    "essential fragment",
    "essential fragments",
    "identifying ion",
    "identifying ions",
    "identifying fragment",
    "identifying fragments",
]

STRUCTURAL_MAPPING_TERMS = [
    "structural section",
    "structural sections",
    "structurally mapped",
    "matched with exclusive",
    "mapped with exclusive",
    "mapped to",
    "matched with",
    "core ions",
    "aglycone",
    "pyrimidine ring",
]

REPORT_ONLY_TERMS = [
    "fragments",
    "product ion mass list",
    "ms fragments",
]


class DiagnosticEvidenceMiningError(RuntimeError):
    """Raised when diagnostic evidence mining cannot run."""


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def format_mz(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    text = f"{number:.2f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def stable_id(prefix: str, *parts: Any) -> str:
    basis = "|".join(clean_text(part).lower() for part in parts)
    return f"{prefix}_" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def load_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DiagnosticEvidenceMiningError(f"Corpus chunks JSONL does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise DiagnosticEvidenceMiningError(f"Invalid chunk at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def contains_any(text_lower: str, terms: list[str]) -> bool:
    return any(term in text_lower for term in terms)


def diagnostic_evidence_type(text: str) -> str:
    lower = text.lower()
    if contains_any(lower, EXPLICIT_DIAGNOSTIC_TERMS):
        return "explicit_diagnostic"
    if contains_any(lower, STRUCTURAL_MAPPING_TERMS):
        return "structural_mapping"
    return ""


def infer_subclass(text: str) -> str:
    lower = text.lower()
    has_betacyanin = "betacyanin" in lower or "betanin-type" in lower or "betanidin" in lower
    has_betaxanthin = "betaxanthin" in lower or re.search(r"\b[a-z0-9-]+-bx\b", lower) is not None
    if has_betacyanin and has_betaxanthin:
        return "betacyanin; betaxanthin"
    if has_betacyanin:
        return "betacyanin"
    if has_betaxanthin:
        return "betaxanthin"
    if "betalain" in lower:
        return ""
    return ""


def evidence_windows(text: str, window_chars: int = 700) -> list[str]:
    lower = text.lower()
    terms = EXPLICIT_DIAGNOSTIC_TERMS + STRUCTURAL_MAPPING_TERMS
    spans: list[tuple[int, int]] = []
    for term in terms:
        start = 0
        while True:
            idx = lower.find(term, start)
            if idx < 0:
                break
            spans.append((max(0, idx - window_chars // 2), min(len(text), idx + len(term) + window_chars // 2)))
            start = idx + len(term)
    if not spans:
        return []
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return [text[start:end] for start, end in merged]


def mz_values_from_text(text: str, min_mz: float = 100.0, max_mz: float = 450.0) -> list[str]:
    values: list[str] = []
    patterns = [
        r"m/z\s*(?:with\s*)?([0-9]{2,4}\.[0-9]{1,4})",
        r"\b([0-9]{2,4}\.[0-9]{2})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = safe_float(match.group(1))
            if value is None:
                continue
            if min_mz <= value <= max_mz:
                values.append(format_mz(value))
    return sorted(set(values), key=lambda x: float(x))


def evidence_quote(text: str, mz: str, max_chars: int = 500) -> str:
    idx = text.find(mz)
    if idx < 0:
        idx = 0
    start = max(0, idx - max_chars // 2)
    end = min(len(text), idx + max_chars // 2)
    return clean_text(text[start:end])


def should_mine_chunk(text: str) -> bool:
    evidence_type = diagnostic_evidence_type(text)
    if not evidence_type:
        return False
    # A plain fragment table that happens to mention "fragments" is not enough
    # unless it also has explicit diagnostic or structural-mapping language.
    return True


def claim_from_chunk(
    chunk: dict[str, Any],
    mz: str,
    subclass: str,
    evidence_type: str,
    compound_class: str,
    context_text: str | None = None,
) -> dict[str, Any]:
    chunk_id = clean_text(chunk.get("chunk_id"))
    source_file = clean_text(chunk.get("source_file"))
    quote = evidence_quote(clean_text(context_text if context_text is not None else chunk.get("text")), mz)
    claim_id = stable_id("claim_diag", compound_class, subclass, mz, evidence_type, chunk_id)
    assignment = {
        "explicit_diagnostic": "diagnostic/fingerprint fragment supported by literature statement",
        "structural_mapping": "fragment structurally mapped in literature",
    }.get(evidence_type, "diagnostic fragment supported by literature")
    return {
        "claim_id": claim_id,
        "claim_type": "diagnostic_fragment",
        "compound_class": compound_class,
        "subclass": subclass,
        "fragment_mz": mz,
        "ion_mode": "",
        "fragment_assignment": assignment,
        "diagnostic_evidence_type": evidence_type,
        "evidence_ids": chunk_id,
        "chunk_id": chunk_id,
        "source_chunk_ids": [chunk_id] if chunk_id else [],
        "source_file": source_file,
        "evidence_quote": quote,
        "evidence_summary": (
            f"Evidence-type driven diagnostic fragment: m/z {mz}; "
            f"evidence_type={evidence_type}; subclass={subclass or 'class-wide'}; source={source_file}."
        ),
        "traceability_status": "diagnostic_evidence_type_mined",
        "confidence": 0.85 if evidence_type == "explicit_diagnostic" else 0.75,
        "review_status": "candidate",
    }


def merge_claim(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    merged["evidence_ids"] = join_unique([existing.get("evidence_ids", ""), incoming.get("evidence_ids", "")])
    merged["chunk_id"] = join_unique([existing.get("chunk_id", ""), incoming.get("chunk_id", "")])
    merged["source_chunk_ids"] = split_values(merged["chunk_id"])
    merged["source_file"] = join_unique([existing.get("source_file", ""), incoming.get("source_file", "")])
    merged["evidence_quote"] = join_unique([existing.get("evidence_quote", ""), incoming.get("evidence_quote", "")])
    merged["evidence_summary"] = join_unique([existing.get("evidence_summary", ""), incoming.get("evidence_summary", "")])
    try:
        merged["confidence"] = max(float(existing.get("confidence", 0)), float(incoming.get("confidence", 0)))
    except (TypeError, ValueError):
        merged["confidence"] = existing.get("confidence", incoming.get("confidence", ""))
    return merged


def mine_diagnostic_evidence_claims(
    chunks: list[dict[str, Any]],
    compound_class: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    evidence_type_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for chunk in chunks:
        text = clean_text(chunk.get("text"))
        if not text:
            skipped_counts["empty_text"] += 1
            continue
        if not should_mine_chunk(text):
            skipped_counts["no_diagnostic_evidence_type"] += 1
            continue
        evidence_type = diagnostic_evidence_type(text)
        windows = evidence_windows(text)
        if not windows:
            skipped_counts["no_mz_values"] += 1
            continue
        window_hits = 0
        for window in windows:
            subclass = infer_subclass(window) or infer_subclass(text)
            mz_values = mz_values_from_text(window)
            if not mz_values:
                continue
            window_hits += len(mz_values)
            for mz in mz_values:
                claim = claim_from_chunk(chunk, mz, subclass, evidence_type, compound_class, context_text=window)
                key = (claim["subclass"].lower(), claim["fragment_mz"], claim["diagnostic_evidence_type"])
                if key in merged:
                    merged[key] = merge_claim(merged[key], claim)
                else:
                    merged[key] = claim
                    evidence_type_counts[evidence_type] += 1
                    if len(examples) < 10:
                        examples.append(
                            {
                                "fragment_mz": claim["fragment_mz"],
                                "subclass": claim["subclass"],
                                "diagnostic_evidence_type": claim["diagnostic_evidence_type"],
                                "chunk_id": claim["chunk_id"],
                            }
                        )
        if window_hits == 0:
            skipped_counts["no_mz_values"] += 1

    claims = list(merged.values())
    report = {
        "compound_class": compound_class,
        "input_chunk_count": len(chunks),
        "diagnostic_claim_count": len(claims),
        "diagnostic_claims_by_evidence_type": dict(sorted(evidence_type_counts.items())),
        "skipped_chunk_reasons": dict(sorted(skipped_counts.items())),
        "examples": examples,
    }
    return claims, report


def run_diagnostic_evidence_mining(
    corpus_jsonl: Path | str,
    out_jsonl: Path | str,
    report_path: Path | str | None = None,
    compound_class: str = "",
) -> dict[str, Any]:
    chunks = load_chunks(resolve(corpus_jsonl))
    claims, report = mine_diagnostic_evidence_claims(chunks, compound_class=compound_class)
    write_jsonl(resolve(out_jsonl), claims)
    if report_path:
        write_json(resolve(report_path), report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine evidence-type driven diagnostic fragment claims from local RAG corpus chunks."
    )
    parser.add_argument("--corpus_jsonl", default="rag/corpus/chunks.jsonl")
    parser.add_argument("--out_jsonl", default="rag/evidence_claims/diagnostic_evidence_claims.jsonl")
    parser.add_argument("--report_path", default="rag/reports/diagnostic_evidence_mining_report.json")
    parser.add_argument("--compound_class", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_diagnostic_evidence_mining(
            corpus_jsonl=args.corpus_jsonl,
            out_jsonl=args.out_jsonl,
            report_path=args.report_path,
            compound_class=args.compound_class,
        )
    except (OSError, json.JSONDecodeError, DiagnosticEvidenceMiningError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
