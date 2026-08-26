#!/usr/bin/env python3
"""Agent 2: mine local literature for concepts and evidence claims.

This agent consumes a Query Planner output and local corpus/index files. It
does not validate chemistry, compile rule tables, or annotate spectra.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from rag.core.reaction_models import (
    ClaimValidationError,
    EntityClassMembershipClaim,
    ReactionTemplateClaim,
    parse_structured_claim,
)
from rag.core.chemical_consistency import (
    ATOMIC_MASSES,
    ChemicalConsistencyError,
    formula_exact_mass,
)
from rag.core.evidence_inventory import FRAGMENT_EVIDENCE_ROLES
from rag.core.retrieval import (
    BM25SRetrievalIndex,
    RetrievalError,
    load_retrieval_index,
)
from rag.core.llm_transport import (
    ChatCompletionResult,
    LLMResponseCache,
    TransportConfig,
    request_chat_completion,
)
from rag.core.io_utils import (
    atomic_write_json as write_json,
    atomic_write_jsonl as write_jsonl,
)
from rag.core.document_relation_extraction import (
    accept_prevalidated_structured_claims,
    apply_domain_semantic_guards,
    apply_structured_domain_scope_guard,
    apply_critic_decisions,
    build_document_windows,
    build_gap_queries,
    recover_name_encoded_modification_claims,
    recover_product_ion_cascade_neutral_losses,
    select_relation_retrieval_rows,
)


QUERY_GROUPS = [
    "compound_queries",
    "fragment_queries",
    "neutral_loss_queries",
    "transformation_queries",
    "biosynthesis_queries",
    "supplementary_table_queries",
    "review_queries",
]
CLAIM_EXTRACTION_BATCH_SIZE = 20
CONCEPT_DISCOVERY_BATCH_SIZE = 20
DOCUMENT_RELATION_BATCH_SIZE = 1
EVIDENCE_CRITIC_BATCH_SIZE = 8

CONCEPT_TYPES = {
    "compound",
    "synonym",
    "subclass",
    "precursor",
    "structural_component",
    "transformation",
    "fragment",
    "neutral_loss",
    "biosynthetic_component",
}

CLAIM_TYPES = {
    "compound",
    "precursor",
    "structural_component",
    "transformation",
    "diagnostic_fragment",
    "neutral_loss",
    "biosynthetic_component",
    "reaction_template",
    "entity_class_membership",
    "entity_component_membership",
}

class LiteratureMiningError(RuntimeError):
    """Raised when the Literature Mining Agent cannot complete."""


class LLMOutputLimitError(LiteratureMiningError):
    """Raised when a structured response is truncated by its output budget."""


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def safe_stem(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip().lower()).strip("_")
    return stem or "compound_class"


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", "" if text is None else str(text)).strip()


def normalize_for_traceability(text: Any) -> str:
    normalized = normalize_space(text).lower()
    normalized = normalized.translate(str.maketrans({"–": "-", "—": "-", "−": "-", "‐": "-"}))
    return re.sub(r"\s+", "", normalized)


def traceability_tokens(text: Any) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z]+(?:[-'][A-Za-z0-9]+)*|\d+(?:\.\d+)?", str(text))
    ]


def assess_traceability(evidence_quote: str, chunk_ids: list[str], chunks_by_id: dict[str, str]) -> dict[str, float | str]:
    quote = normalize_space(evidence_quote)
    if not quote:
        return {"status": "failed", "score": 0.0}

    normalized_quote = normalize_for_traceability(quote)
    quote_tokens = traceability_tokens(quote)
    best_score = 0.0
    for chunk_id in chunk_ids:
        raw_text = str(chunks_by_id.get(chunk_id, ""))
        if quote in raw_text:
            return {"status": "exact", "score": 1.0}

        normalized_text = normalize_for_traceability(raw_text)
        if normalized_quote and normalized_quote in normalized_text:
            return {"status": "normalized", "score": 0.95}

        if quote_tokens:
            chunk_token_set = set(traceability_tokens(raw_text))
            overlap = sum(1 for token in quote_tokens if token in chunk_token_set) / len(quote_tokens)
            best_score = max(best_score, overlap)
            if overlap > 0.75:
                return {"status": "fuzzy", "score": round(overlap, 4)}

    return {"status": "failed", "score": round(best_score, 4)}


def rare_term_score(text: Any) -> float:
    text_value = str(text)
    score = 0.0
    score += 3.0 * len(re.findall(r"\bC\d+H\d+(?:[A-Z][a-z]?\d*)*\b", text_value))
    score += 2.0 * len(re.findall(r"\bm\s*/\s*z\b|\b\d+\.\d{3,}\b", text_value, flags=re.IGNORECASE))
    score += 1.0 * len(re.findall(r"\b[A-Z][A-Za-z]+(?:[- ][A-Za-z0-9]+){1,}\b", text_value))
    score += 0.5 * len(
        re.findall(
            r"\b[A-Za-z]+(?:ine|ol|one|acid|oside|glycoside|glucoside|aglycone|ester|amide)s?\b",
            text_value,
            flags=re.IGNORECASE,
        )
    )
    return score


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise LiteratureMiningError(f"JSON file must contain an object: {path}")
    return payload


def load_chunks(path: Path) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("chunk_id") or not isinstance(row.get("text"), str):
                raise LiteratureMiningError(f"Invalid chunk at {path}:{line_number}")
            chunks[str(row["chunk_id"])] = row
    return chunks


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("LLM output must be a JSON object.")
    return payload


def llm_timeout_seconds() -> float:
    try:
        value = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "120"))
    except ValueError:
        return 120.0
    return max(15.0, min(600.0, value))


def llm_max_output_tokens() -> int | None:
    raw = os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return max(256, min(32768, value))


def llm_thinking_mode() -> str:
    """Select reasoning behavior for deterministic structured extraction."""

    value = os.environ.get("OPENAI_THINKING_MODE", "disabled").strip().lower()
    return value if value in {"disabled", "enabled", "auto"} else "disabled"


def llm_transport_retries() -> int:
    """Bound retries for transient DNS/socket failures inside one LLM call."""

    try:
        value = int(os.environ.get("OPENAI_TRANSPORT_RETRIES", "3"))
    except ValueError:
        return 3
    return max(1, min(5, value))


def llm_retry_backoff_seconds() -> float:
    try:
        value = float(os.environ.get("OPENAI_RETRY_BACKOFF_SECONDS", "2"))
    except ValueError:
        return 2.0
    return max(0.0, min(30.0, value))


def run_with_wall_clock_deadline(
    operation: Callable[[], Any],
    *,
    timeout_seconds: float,
) -> Any:
    """Run a blocking API operation under a real elapsed-time deadline.

    ``urllib`` socket timeouts apply to individual blocking socket operations;
    they are not a guaranteed total request duration when a server keeps the
    connection active.  A daemon worker prevents a stalled request from
    blocking pipeline shutdown after the configured elapsed-time budget.
    """

    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result_queue.put(("ok", operation()))
        except BaseException as exc:  # preserve the API wrapper's exact error
            result_queue.put(("error", exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=max(0.001, float(timeout_seconds)))
    if thread.is_alive():
        raise LiteratureMiningError(
            "OpenAI-compatible API exceeded the wall-clock deadline of "
            f"{float(timeout_seconds):.1f} seconds."
        )
    status, value = result_queue.get_nowait()
    if status == "error":
        raise value
    return value


def chat_completion_openai_compatible(
    messages: list[dict[str, str]],
    model: str,
    base_url: str,
    api_key: str,
) -> str:
    thinking_mode = llm_thinking_mode()
    result = request_chat_completion(
        messages,
        model=model,
        base_url=base_url,
        api_key=api_key,
        config=TransportConfig(
            timeout_seconds=llm_timeout_seconds(),
            retries=llm_transport_retries(),
            retry_backoff_seconds=llm_retry_backoff_seconds(),
            thinking_mode=None if thinking_mode == "auto" else thinking_mode,
            max_output_tokens=llm_max_output_tokens(),
            return_metadata=True,
        ),
        error_type=LiteratureMiningError,
        missing_context=" for LLM calls.",
        response_error_message=(
            "API response did not contain choices[0].message.content"
        ),
    )
    if isinstance(result, ChatCompletionResult):
        result.response_metadata["thinking_mode"] = thinking_mode
    return result


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def split_semicolon_values(value: Any) -> list[str]:
    text = normalize_space(value)
    if not text:
        return []
    return [item.strip() for item in re.split(r"[;|]+", text) if item.strip()]


def join_unique_text(values: list[Any]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        for item in split_semicolon_values(value):
            key = item.lower()
            if key not in seen:
                seen.add(key)
                out.append(item)
    return ";".join(out)


def format_numeric_text(value: Any) -> str:
    text = normalize_space(value)
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return ""
    return f"{number:.6f}".rstrip("0").rstrip(".")


def extract_first(pattern: str, text: str, flags: int = re.IGNORECASE) -> str:
    match = re.search(pattern, text, flags=flags)
    return normalize_space(match.group(1)) if match else ""


def clean_structured_name(name: str) -> str:
    cleaned = normalize_space(name)
    cleaned = re.sub(r"^(?:chemical structure|structure)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" :;,.")
    return cleaned


def repair_structured_name_from_context(name: str, document_text: str) -> str:
    """Repair PDF token splits only when the intact token occurs in evidence.

    This is deliberately vocabulary-free: a whitespace boundary is removed
    only when the resulting token is present elsewhere in the same document.
    Chemical modifiers and stereochemical locants are otherwise untouched.
    """

    repaired = normalize_space(name).strip()
    context = str(document_text or "")
    boundary_pattern = re.compile(
        r"([A-Za-z]{2,})\s+([A-Za-z]{1,})(?=[^A-Za-z]|$)"
    )
    for _ in range(8):
        changed = False

        def join_if_supported(match: re.Match[str]) -> str:
            nonlocal changed
            right = match.group(2)
            if re.fullmatch(
                r"(?:[A-Z]{1,6}\d+[A-Za-z0-9-]*|[IVXLCDM]+)",
                right,
                re.IGNORECASE,
            ) or (
                len(right) <= 2
                and re.match(r"\d", match.string[match.end() :])
            ):
                return match.group(0)
            joined = match.group(1) + match.group(2)
            if re.search(
                rf"(?<![A-Za-z]){re.escape(joined)}(?![A-Za-z])",
                context,
                re.IGNORECASE,
            ):
                changed = True
                return joined
            return match.group(0)

        repaired = boundary_pattern.sub(join_if_supported, repaired)
        if not changed:
            break
    repaired = re.sub(
        r"(?<=[A-Za-z0-9])(?=isomer(?:\b|/))",
        " ",
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = repaired.rstrip("/ ")
    return normalize_space(repaired)


def parse_structured_name(text: str) -> str:
    name = extract_first(
        r"\bName\s*:\s*(.+?)(?:\bChemical\s+Formula\b|\bMolecular\s+weight\b|\bMonoisotopic\s+mass\b|\bm\s*/\s*z\b|\bfragments?\b|$)",
        text,
    )
    return clean_structured_name(name)


def parse_structured_fragments(text: str) -> str:
    marker = re.search(r"\bfragments?\s*(?:\(\s*m\s*/\s*z\s*\)|m\s*/\s*z)?", text, flags=re.IGNORECASE)
    if not marker:
        return ""
    tail = text[marker.end() :]
    tail = re.split(r"\b(?:references|chemical structure|name\s*:|compound\s+\d+)\b", tail, maxsplit=1, flags=re.IGNORECASE)[0]
    values: list[str] = []
    for match in re.finditer(r"(?<![A-Za-z])\d{2,4}(?:\.\d+)?(?![A-Za-z])", tail):
        number = float(match.group(0))
        if 40.0 <= number <= 2000.0:
            values.append(format_numeric_text(match.group(0)))
    return join_unique_text(values)


def parse_structured_compound_record(text: str) -> dict[str, str]:
    flat = normalize_space(text)
    name = parse_structured_name(flat)
    formula = extract_first(r"\bChemical\s+Formula\s+([A-Z][A-Za-z0-9]*)\b", flat)
    exact_mass = format_numeric_text(extract_first(r"\bMonoisotopic\s+mass\s+(\d+(?:\.\d+)?)\b", flat))
    precursor_mz = format_numeric_text(
        extract_first(r"\bm\s*/\s*z\s*(?:\[[^\]]+\])?\s*(\d+(?:\.\d+)?)\b", flat)
    )
    fragments = parse_structured_fragments(flat)
    if not name or not (formula or exact_mass or precursor_mz or fragments):
        return {}
    return {
        "compound_name": name,
        "formula": formula,
        "exact_mass": exact_mass,
        "reported_precursor_mz": precursor_mz,
        "reported_fragments": fragments,
        "evidence_role": "theoretical_catalog" if fragments else "",
        "specificity_scope": "target_associated" if fragments else "",
        "source_structure": "supplementary_catalog",
    }


_TABLE_ELEMENT_PATTERN = "|".join(
    re.escape(element)
    for element in sorted(ATOMIC_MASSES, key=lambda item: (-len(item), item))
    if element not in {"C", "H"}
)
TABLE_FORMULA_PATTERN = re.compile(
    rf"\bC\s?\d{{1,3}}\s*H\s?\d{{1,3}}"
    rf"(?:\s*(?:{_TABLE_ELEMENT_PATTERN})(?:\s?\d{{1,3}})?)+\b"
)
TABLE_NUMBER_PATTERN = re.compile(r"^[\s\d.+\-\u2212\u2013\u2014]+$")


def infer_table_source_profile(texts: list[str]) -> dict[str, str]:
    combined = "\n".join(texts)
    compact = normalize_space(combined)
    negative = bool(re.search(r"\[M\s*-\s*H\]\s*[-\u2212\u2013\u02c9]|negative\s+ion\s+mode", compact, re.IGNORECASE))
    positive = bool(re.search(r"\[M\s*\+\s*H\]\s*\+|positive\s+ion\s+mode", compact, re.IGNORECASE))
    ion_mode = "negative" if negative and not positive else "positive" if positive and not negative else ""
    adduct = "[M-H]-" if ion_mode == "negative" else "[M+H]+" if ion_mode == "positive" else ""
    theoretical_after_formula = bool(
        re.search(r"(?:molecular\s+)?formula.{0,80}theoretical\s+(?:exact\s+)?mass", compact, re.IGNORECASE)
    )
    precursor_after_formula = bool(
        re.search(r"(?:molecular\s+)?formula.{0,100}(?:measured\s+(?:value\s*)?\(m\s*/\s*z\)|\[M[^\]]+\]\s*[-+\u2212\u2013\u02c9]?\s*ion)", compact, re.IGNORECASE)
    ) and not theoretical_after_formula
    explicit_product_ion_column = bool(
        re.search(
            r"(?:molecular\s+)?formula.{0,300}"
            r"(?:product\s+ions?|ms\s*2\s+ions?|fragmentation\s+ions?)",
            combined,
            re.IGNORECASE | re.DOTALL,
        )
        or any(
            re.search(r"\bformula\b", text, re.IGNORECASE)
            and re.search(
                r"\b(?:measured|observed)\b", text, re.IGNORECASE
            )
            and re.search(
                r"\b(?:product\s+ions?|ms\s*2\s+ions?|fragmentation\s+ions?)\b",
                text,
                re.IGNORECASE,
            )
            and re.search(r"\bcompound\s+name\b", text, re.IGNORECASE)
            for text in texts
        )
    )
    return {
        "ion_mode": ion_mode,
        "adduct": adduct,
        "mass_after_formula": "precursor" if precursor_after_formula else "exact" if theoretical_after_formula else "",
        "explicit_product_ion_column": "yes" if explicit_product_ion_column else "",
    }


def clean_fixed_width_name(
    value: str,
    *,
    allow_continuation: bool = False,
) -> str:
    name = normalize_space(value).strip(" :;,.|*")
    name = re.sub(r"^\(\d+\)\s*", "", name)
    name = re.sub(r"^[A-Z]?\d+\s+(?=[A-Za-z])", "", name)
    name = re.sub(r"(?:\s*\d+\)\s*,?)+$", "", name).strip(" :;,.|*")
    name = re.sub(r"\s+[ab]$", "", name).strip(" :;,.|*")
    if allow_continuation and re.fullmatch(r"\d+(?:,\d+)*-", name):
        return name
    if not name or len(name) > 140 or len(re.findall(r"[A-Za-z]", name)) < 2:
        return ""
    if TABLE_NUMBER_PATTERN.fullmatch(name) or re.fullmatch(r"(?:No\.?|[RS]|[\u03b1\u03b2])", name, re.IGNORECASE):
        return ""
    if name.startswith(("-", "/", "\u2212", "\u2013", "\u2014")) or (
        name.endswith(("-", "/")) and not allow_continuation
    ):
        return ""
    if "\ufffd" in name:
        return ""
    if any(marker in name for marker in ("[", "]", "、", "銆")) or (
        TABLE_FORMULA_PATTERN.search(name)
    ):
        return ""
    if not allow_continuation and name.lower() in {"isomer", "derivative"}:
        return ""
    return name


def is_identifier_name_suffix(value: str) -> bool:
    """Return whether a wrapped cell is an evidence-explicit name suffix."""

    return bool(
        re.fullmatch(
            r"(?:[A-Z]{1,6}\d+[A-Za-z0-9-]*|[IVXLCDM]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]+)"
            r"(?:\s+(?:isomer|derivative))?",
            normalize_space(value),
            re.IGNORECASE,
        )
    )


def fixed_width_name_before_formula(parts: list[str], formula_index: int) -> str:
    candidates: list[str] = []
    for raw_value in parts[:formula_index]:
        candidate = clean_fixed_width_name(
            raw_value,
            allow_continuation=True,
        )
        if not candidate:
            continue
        if not candidates:
            candidates.append(candidate)
            continue
        if (
            is_identifier_name_suffix(candidate)
            or candidates[-1].endswith(
                ("-", "\u2010", "\u2011", "\u2012", "\u2013")
            )
        ):
            candidates.append(candidate)
            continue
        break
    return join_wrapped_name_parts(candidates)


def join_wrapped_name_parts(parts: list[str]) -> str:
    if not parts:
        return ""
    joined = parts[0]
    for part in parts[1:]:
        last_word = re.split(r"\s+", joined)[-1].rstrip("-")
        if joined.endswith(("-", "\u2010", "\u2011", "\u2012", "\u2013")):
            joined += part
        elif (
            joined
            and joined[-1].isalpha()
            and part
            and part[0].islower()
            and len(last_word) <= 3
        ):
            joined += part
        else:
            joined += " " + part
    return normalize_space(joined)


def fixed_width_name_after_formula(
    lines: list[str],
    *,
    max_candidates: int | None = None,
) -> tuple[str, int]:
    candidates: list[str] = []
    consumed_line_count = 0
    for line_index, line in enumerate(lines):
        parts = [part.strip() for part in re.split(r"\s{2,}|\t+", line.strip()) if part.strip()]
        if line_index == 0:
            formula_index = next(
                (index for index, part in enumerate(parts) if TABLE_FORMULA_PATTERN.search(part)),
                -1,
            )
            parts = parts[formula_index + 1 :] if formula_index >= 0 else []
        line_candidates: list[str] = []
        for raw_value in parts:
            raw_candidate = normalize_space(raw_value).strip(" :;,.|*")
            candidate = (
                raw_candidate
                if is_identifier_name_suffix(raw_candidate)
                else clean_fixed_width_name(
                    raw_value,
                    allow_continuation=True,
                )
            )
            if candidate:
                line_candidates.append(candidate)
        if not line_candidates:
            if candidates:
                consumed_line_count = line_index + 1
            continue
        candidate = line_candidates[0]
        if not candidates:
            candidates.append(candidate)
            consumed_line_count = line_index + 1
            if max_candidates == 1:
                break
            continue
        if candidate[0].islower():
            candidates.append(candidate)
            consumed_line_count = line_index + 1
            if max_candidates is not None and len(candidates) >= max_candidates:
                break
            continue
        if is_identifier_name_suffix(candidate):
            candidates.append(candidate)
            consumed_line_count = line_index + 1
            break
        break
    return join_wrapped_name_parts(candidates), consumed_line_count


def fixed_width_name_on_previous_line(
    lines: list[str],
    line_index: int,
) -> tuple[str, str]:
    """Return a wrapped name prefix and its source line.

    PDF table extractors commonly place a compound-name cell one line above
    the numeric/formula row. Only the immediately preceding physical line is
    considered so unrelated prose or an earlier table row cannot be attached.
    """

    if line_index <= 0:
        return "", ""
    for previous_index in range(line_index - 1, max(-1, line_index - 3), -1):
        raw_line = lines[previous_index]
        if TABLE_FORMULA_PATTERN.search(raw_line) or re.search(
            r"\b(?:compound\s+name|chemical\s+formula|molecular\s+formula|"
            r"product\s+ions?|measured\s+value|observed\s+m\s*/\s*z)\b",
            raw_line,
            re.IGNORECASE,
        ):
            return "", ""
        parts = [
            part.strip()
            for part in re.split(r"\s{2,}|\t+", raw_line.strip())
            if part.strip()
        ]
        for raw_value in parts:
            candidate = clean_fixed_width_name(
                raw_value,
                allow_continuation=True,
            )
            if candidate:
                return candidate, "\n".join(
                    lines[previous_index:line_index]
                )
    return "", ""


def annotated_product_ions(text: str) -> str:
    values: list[str] = []
    intact_annotations = {
        "m-h",
        "m+h",
        "m+na",
        "m+nh4",
        "m+hcoo",
        "m+fa-h",
    }
    for match in re.finditer(r"(?<![A-Za-z])([0-9]{2,4}(?:\.[0-9]+)?)\s*\[([^\]]+)\]", text):
        annotation = re.sub(r"\s+", "", match.group(2)).lower().replace("\u2212", "-").replace("\u2013", "-")
        if annotation in intact_annotations or annotation.isdigit():
            continue
        value = float(match.group(1))
        if 40.0 <= value <= 2000.0:
            values.append(format_numeric_text(match.group(1)))
    return join_unique_text(values)


def delimited_product_ions(text: str) -> str:
    """Extract mass lists only when punctuation establishes a table cell/list."""

    values: list[str] = []
    text = re.sub(r"\[[^\]]*\]", " ", text)
    sequence_pattern = re.compile(
        r"(?<![A-Za-z0-9])"
        r"(\d{2,4}(?:\.\d+)?(?:\s*[,;]\s*\d{2,4}(?:\.\d+)?)+)"
        r"(?![A-Za-z0-9])"
    )
    for line in text.splitlines():
        for sequence in sequence_pattern.findall(line):
            for value_text in re.findall(r"\d{2,4}(?:\.\d+)?", sequence):
                value = float(value_text)
                if 40.0 <= value <= 2000.0:
                    values.append(format_numeric_text(value_text))
    return join_unique_text(values)


_NON_COMPONENT_LOSS_TOKENS = {
    "m",
    "h",
    "h2o",
    "co",
    "co2",
    "nh3",
    "ch2o",
    "ch3oh",
}


def _component_tokens(value: str) -> list[str]:
    """Parse reported component labels without assigning chemical meaning."""

    normalized = (
        str(value or "")
        .replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    tokens: list[str] = []
    for match in re.finditer(
        r"(?<![A-Za-z])(?:\d+)?([A-Z][a-z]{1,15}(?:\([A-Za-z]+\))?|[a-z]{3,15})(?![A-Za-z])",
        normalized,
    ):
        token = match.group(1)
        if token.casefold() not in _NON_COMPONENT_LOSS_TOKENS:
            tokens.append(token)
    return tokens


def annotated_component_profile(text: str) -> dict[str, int]:
    """Return maximum reported component counts across annotated product ions."""

    profile: dict[str, int] = {}
    for annotation in re.findall(r"\[([^\]]+)\]", str(text or "")):
        if not re.search(r"\bM\b|M\s*[-+]", annotation, re.IGNORECASE):
            continue
        local: dict[str, int] = {}
        for segment in re.split(r"[-\u2212\u2013\u2014]", annotation):
            segment = segment.strip(" ()+−–—")
            count_match = re.fullmatch(
                r"(\d+)?([A-Za-z][A-Za-z]*(?:\([A-Za-z]+\))?)",
                segment,
            )
            if not count_match:
                continue
            token = count_match.group(2)
            if token.casefold() in _NON_COMPONENT_LOSS_TOKENS:
                continue
            count = int(count_match.group(1) or 1)
            local[token] = local.get(token, 0) + count
        for token, count in local.items():
            profile[token] = max(profile.get(token, 0), count)
    return profile


def annotated_component_loss_masses(
    text: str, precursor_mz: str
) -> dict[str, float]:
    """Estimate per-component neutral-loss masses from explicit ion labels."""

    try:
        precursor = float(precursor_mz)
    except (TypeError, ValueError):
        return {}
    observations: dict[str, list[float]] = {}
    pattern = re.compile(r"(\d{2,4}(?:\.\d+)?)\s*\[([^\]]+)\]")
    for match in pattern.finditer(str(text or "")):
        product_mz = float(match.group(1))
        annotation = match.group(2)
        local: dict[str, int] = {}
        for segment in re.split(r"[-\u2212\u2013\u2014]", annotation):
            count_match = re.fullmatch(
                r"\s*(\d+)?([A-Za-z][A-Za-z]*(?:\([A-Za-z]+\))?)\s*",
                segment,
            )
            if not count_match:
                continue
            label = count_match.group(2)
            if label.casefold() in _NON_COMPONENT_LOSS_TOKENS:
                continue
            local[label] = local.get(label, 0) + int(count_match.group(1) or 1)
        if len(local) != 1:
            continue
        label, count = next(iter(local.items()))
        loss = (precursor - product_mz) / count
        if 20.0 <= loss <= 250.0:
            observations.setdefault(label, []).append(loss)
    return {
        label: sorted(values)[len(values) // 2]
        for label, values in observations.items()
        if values
    }


def parse_substituent_component_records(text: str) -> list[dict[str, Any]]:
    """Parse generic entity/R-site tables into evidence-preserving profiles."""

    records: list[dict[str, Any]] = []
    if not re.search(r"\bR\s*1\b|\bR1\b", text, re.IGNORECASE):
        return records
    lines = text.splitlines()
    for line_index, raw_line in enumerate(lines):
        parts = [part.strip() for part in re.split(r"\s{2,}|\t+", raw_line) if part.strip()]
        component_index = next(
            (
                index
                for index, part in enumerate(parts)
                if part.startswith(("-", "\u2212", "\u2013", "\u2014"))
                and _component_tokens(part)
            ),
            -1,
        )
        if component_index <= 0:
            continue
        entity_name = fixed_width_name_before_formula(parts, component_index)
        if not entity_name or re.fullmatch(r"R\s*\d+", entity_name, re.IGNORECASE):
            continue
        for continuation in lines[line_index + 1 : line_index + 4]:
            suffix = normalize_space(continuation).strip(" :;,.|*")
            if not suffix:
                continue
            if is_identifier_name_suffix(suffix):
                entity_name = join_wrapped_name_parts([entity_name, suffix])
            break
        profile: dict[str, int] = {}
        for cell in parts[component_index:]:
            if not cell.startswith(("-", "\u2212", "\u2013", "\u2014")):
                break
            for token in _component_tokens(cell):
                profile[token] = profile.get(token, 0) + 1
        if profile:
            records.append(
                {
                    "entity_name": entity_name,
                    "component_profile": profile,
                    "component_profile_source": "structure_substituent_table",
                    "evidence_quote": normalize_space(raw_line),
                    "_anchor_text": raw_line,
                }
            )
    return records


def parse_fixed_width_table_records(text: str, source_profile: dict[str, str]) -> list[dict[str, Any]]:
    records: list[dict[str, str]] = []
    lines = text.splitlines()
    local_explicit_product_ion_column = bool(
        re.search(
            r"\b(?:product\s+ions?|ms\s*2\s+ions?|fragmentation\s+ions?)\b",
            text,
            re.IGNORECASE,
        )
    )
    local_fragment_column = local_explicit_product_ion_column or bool(
        re.search(r"\bfragments?\b", text, re.IGNORECASE)
    )
    explicit_product_ion_context = bool(
        local_explicit_product_ion_column
        or source_profile.get("explicit_product_ion_column") == "yes"
    )
    for line_index, raw_line in enumerate(lines):
        formula_match = TABLE_FORMULA_PATTERN.search(raw_line)
        if not formula_match:
            continue
        parts = [part.strip() for part in re.split(r"\s{2,}|\t+", raw_line.strip()) if part.strip()]
        if len(parts) < 3:
            continue
        formula_index = next((index for index, part in enumerate(parts) if TABLE_FORMULA_PATTERN.search(part)), -1)
        if formula_index <= 0:
            continue
        direct_name = fixed_width_name_before_formula(parts, formula_index)
        prefix_name, prefix_line = fixed_width_name_on_previous_line(
            lines,
            line_index,
        )
        if direct_name and not prefix_name.endswith(
            ("-", "\u2010", "\u2011", "\u2012", "\u2013")
        ):
            prefix_name = ""
            prefix_line = ""
        name = join_wrapped_name_parts(
            [part for part in (prefix_name, direct_name) if part]
        )
        record_lines = [raw_line]
        for continuation in lines[line_index + 1 : line_index + 9]:
            if TABLE_FORMULA_PATTERN.search(continuation):
                break
            record_lines.append(continuation)
        if direct_name:
            for continuation in record_lines[1:]:
                standalone_suffix = normalize_space(continuation).strip(
                    " :;,.|*"
                )
                if is_identifier_name_suffix(standalone_suffix):
                    direct_name = join_wrapped_name_parts(
                        [direct_name, standalone_suffix]
                    )
                    name = join_wrapped_name_parts(
                        [
                            part
                            for part in (prefix_name, direct_name)
                            if part
                        ]
                    )
                    break
        needs_suffix = not direct_name or direct_name.endswith(
            ("-", "\u2010", "\u2011", "\u2012", "\u2013")
        )
        if needs_suffix:
            suffix_name, consumed_line_count = fixed_width_name_after_formula(
                record_lines,
                max_candidates=1 if prefix_name else None,
            )
            if consumed_line_count:
                record_lines = record_lines[:consumed_line_count]
            if (
                suffix_name
                and suffix_name[0].isupper()
                and not prefix_name.endswith(
                    ("-", "\u2010", "\u2011", "\u2012", "\u2013")
                )
            ):
                prefix_name = ""
                prefix_line = ""
            name = join_wrapped_name_parts(
                [
                    part
                    for part in (prefix_name, direct_name, suffix_name)
                    if part
                ]
            )
        if prefix_line:
            record_lines.insert(0, prefix_line)
        if not name or name.endswith(
            ("-", "\u2010", "\u2011", "\u2012", "\u2013")
        ):
            continue
        formula = re.sub(r"\s+", "", formula_match.group(0))
        try:
            calculated_mass = formula_exact_mass(formula)
        except ChemicalConsistencyError:
            continue
        precursor_mz = ""
        if source_profile.get("mass_after_formula") == "precursor":
            after_formula = raw_line[formula_match.end() :]
            candidates = [
                format_numeric_text(match.group(0))
                for match in re.finditer(r"(?<![A-Za-z0-9])\d{2,4}(?:\.\d+)?(?![A-Za-z])", after_formula)
                if 40.0 <= float(match.group(0)) <= 2000.0 and "." in match.group(0)
            ]
            if candidates:
                precursor_mz = candidates[0]
        record_text = "\n".join(record_lines)
        component_profile = annotated_component_profile(record_text)
        component_loss_masses = annotated_component_loss_masses(
            record_text, precursor_mz
        )
        fragments = annotated_product_ions(record_text)
        if local_fragment_column:
            fragments = join_unique_text(
                [fragments, delimited_product_ions(record_text)]
            )
        records.append(
            {
                "compound_name": name,
                "formula": formula,
                "exact_mass": format_numeric_text(calculated_mass),
                "reported_precursor_mz": precursor_mz,
                "reported_fragments": fragments,
                "ion_mode": source_profile.get("ion_mode", ""),
                "adduct": source_profile.get("adduct", ""),
                "mass_derivation": "formula_calculated",
                "evidence_quote": normalize_space(record_text),
                "evidence_role": (
                    "target_product_ion"
                    if fragments and explicit_product_ion_context
                    else "theoretical_catalog" if fragments else ""
                ),
                "specificity_scope": (
                    "target_specific"
                    if fragments and explicit_product_ion_context
                    else "target_associated" if fragments else ""
                ),
                "source_structure": (
                    "structured_identification_table"
                    if explicit_product_ion_context
                    else "structured_table"
                ),
                "component_profile": component_profile,
                "component_profile_source": (
                    "annotated_fragment_loss" if component_profile else ""
                ),
                "component_loss_masses": component_loss_masses,
                "_anchor_text": raw_line,
            }
        )
    return records


def structured_mining_contexts(
    chunks_by_id: dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, Any], str, list[tuple[str, dict[str, Any]]]]]:
    """Reconstruct PDF pages while retaining chunk-level ownership.

    Corpus chunks overlap by character offset. Parsing each overlap separately
    creates truncated duplicate entities when a table row crosses a boundary.
    Page reconstruction is deterministic and does not alter the corpus.
    """

    page_groups: dict[
        tuple[str, int],
        list[tuple[str, dict[str, Any]]],
    ] = {}
    standalone: list[
        tuple[str, dict[str, Any], str, list[tuple[str, dict[str, Any]]]]
    ] = []
    for chunk_id, chunk in chunks_by_id.items():
        page = chunk.get("page")
        start = chunk.get("char_start")
        end = chunk.get("char_end")
        text = str(chunk.get("text", ""))
        if (
            str(chunk.get("file_type", "")).lower() == "pdf"
            and isinstance(page, int)
            and isinstance(start, int)
            and isinstance(end, int)
            and end >= start
            and end - start == len(text)
        ):
            page_groups.setdefault(
                (str(chunk.get("source_file", "")), page),
                [],
            ).append((chunk_id, chunk))
        else:
            standalone.append((chunk_id, chunk, text, [(chunk_id, chunk)]))

    contexts = list(standalone)
    for _, members in sorted(page_groups.items()):
        ordered = sorted(
            members,
            key=lambda item: (
                int(item[1].get("char_start", 0)),
                item[0],
            ),
        )
        base_start = int(ordered[0][1].get("char_start", 0))
        page_text = ""
        for _, chunk in ordered:
            text = str(chunk.get("text", ""))
            offset = int(chunk.get("char_start", 0)) - base_start
            if offset < 0:
                continue
            if offset > len(page_text):
                page_text += "\n" * (offset - len(page_text))
                page_text += text
            else:
                page_text += text[max(0, len(page_text) - offset) :]
        representative_id, representative = ordered[0]
        contexts.append(
            (
                representative_id,
                representative,
                page_text,
                ordered,
            )
        )
    return contexts


def record_owner_chunk(
    record: dict[str, str],
    members: list[tuple[str, dict[str, Any]]],
    default_chunk_id: str,
) -> tuple[str, dict[str, Any]]:
    anchor = normalize_space(record.get("_anchor_text", ""))
    if anchor:
        for chunk_id, chunk in members:
            if anchor in normalize_space(chunk.get("text", "")):
                return chunk_id, chunk
    return default_chunk_id, dict(members[0][1])


def _component_membership_claims(
    *,
    compound_class: str,
    entity_name: str,
    profile: dict[str, int],
    profile_source: str,
    chunk_id: str,
    chunk: dict[str, Any],
    evidence_quote: str,
    component_loss_masses: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for component_name, component_count in sorted(profile.items()):
        if not component_name or component_count <= 0:
            continue
        basis = (
            compound_class.casefold(),
            entity_name.casefold(),
            component_name.casefold(),
            str(component_count),
            profile_source,
            chunk_id,
        )
        claims.append(
            {
                "claim_id": "claim_component_membership_"
                + hashlib.sha1("|".join(basis).encode("utf-8")).hexdigest()[:16],
                "claim_type": "entity_component_membership",
                "claim_source": "structured_table_mining",
                "compound_class": compound_class,
                "entity_name": entity_name,
                "component_name": component_name,
                "component_count": component_count,
                "component_profile_source": profile_source,
                "component_loss_mass": (
                    (component_loss_masses or {}).get(component_name, "")
                ),
                "source_chunk_ids": [chunk_id],
                "chunk_id": chunk_id,
                "source_file": chunk.get("source_file", ""),
                "page": chunk.get("page"),
                "source_structure": "structured_composition_table",
                "evidence_quote": evidence_quote,
                "evidence_summary": (
                    "Entity-component membership preserved from a local structured record."
                ),
                "traceability_status": "structured",
                "traceability_score": 1.0,
                "review_status": "candidate",
                "confidence": 0.9,
                "evidence_ids": chunk_id,
            }
        )
    return claims


def mine_structured_claims_from_chunks(compound_class: str, chunks_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    component_memberships: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    source_texts: dict[str, list[str]] = {}
    for chunk in chunks_by_id.values():
        source_texts.setdefault(str(chunk.get("source_file", "")), []).append(str(chunk.get("text", "")))
    source_profiles = {
        source_file: infer_table_source_profile(texts)
        for source_file, texts in source_texts.items()
    }
    for default_chunk_id, context_chunk, text, members in structured_mining_contexts(
        chunks_by_id
    ):
        source_file = str(context_chunk.get("source_file", ""))
        name_context = "\n".join(source_texts.get(source_file, ()))
        for profile_record in parse_substituent_component_records(text):
            profile_record["entity_name"] = repair_structured_name_from_context(
                profile_record["entity_name"], name_context
            )
            chunk_id, chunk = record_owner_chunk(
                profile_record,
                members,
                default_chunk_id,
            )
            for membership in _component_membership_claims(
                compound_class=compound_class,
                entity_name=profile_record["entity_name"],
                profile=profile_record["component_profile"],
                profile_source=profile_record["component_profile_source"],
                chunk_id=chunk_id,
                chunk=chunk,
                evidence_quote=profile_record["evidence_quote"],
                component_loss_masses={},
            ):
                membership_key = (
                    membership["entity_name"].casefold(),
                    membership["component_name"].casefold(),
                    str(membership["component_count"]),
                    str(membership.get("source_file", "")).casefold(),
                    membership["component_profile_source"],
                )
                component_memberships[membership_key] = membership
        records: list[dict[str, str]] = []
        if re.search(r"\b(?:Name\s*:|Chemical\s+Formula|Monoisotopic\s+mass|fragments?\s*\(\s*m\s*/\s*z)", text, re.IGNORECASE):
            labelled_record = parse_structured_compound_record(text)
            if labelled_record:
                records.append(labelled_record)
        records.extend(
            parse_fixed_width_table_records(
                text,
                source_profiles.get(
                    str(context_chunk.get("source_file", "")),
                    {},
                ),
            )
        )
        for record in records:
            record["compound_name"] = repair_structured_name_from_context(
                record["compound_name"], name_context
            )
            if not clean_fixed_width_name(record["compound_name"]):
                continue
            chunk_id, chunk = record_owner_chunk(
                record,
                members,
                default_chunk_id,
            )
            key = (
                compound_class.lower(),
                record["compound_name"].lower(),
                record.get("formula", "").lower(),
                record.get("exact_mass", ""),
                record.get("reported_precursor_mz", ""),
                record.get("evidence_role", ""),
                record.get("source_structure", ""),
            )
            evidence_quote = record.get("evidence_quote") or normalize_space(
                " ".join(
                    item
                    for item in [
                        record["compound_name"],
                        record.get("formula", ""),
                        record.get("exact_mass", ""),
                        record.get("reported_precursor_mz", ""),
                    ]
                    if item
                )
            )
            claim = {
            "claim_id": "claim_struct_" + hashlib.sha1("|".join(key).encode("utf-8")).hexdigest()[:16],
            "claim_type": "compound",
            "compound_class": compound_class,
            "compound_name": record["compound_name"],
            "source_chunk_ids": [chunk_id],
            "chunk_id": chunk_id,
            "source_file": chunk.get("source_file", ""),
            "page": chunk.get("page"),
            "formula": record.get("formula", ""),
            "exact_mass": record.get("exact_mass", ""),
            "reported_precursor_mz": record.get("reported_precursor_mz", ""),
            "reported_fragments": record.get("reported_fragments", ""),
            "fragment_evidence_scope": record.get("specificity_scope", ""),
            "specificity_scope": record.get("specificity_scope", ""),
            "evidence_role": record.get("evidence_role", ""),
            "source_structure": record.get("source_structure", "structured_table"),
            "ion_mode": record.get("ion_mode", ""),
            "adduct": record.get("adduct", ""),
            "mass_derivation": record.get("mass_derivation", "reported"),
            "evidence_quote": evidence_quote,
            "evidence_summary": "Structured compound record mined from local corpus text.",
            "traceability_status": "structured",
            "traceability_score": 1.0,
            "review_status": "candidate",
            "confidence": 0.85,
            "claim_source": "structured_table_mining",
            "evidence_ids": chunk_id,
            "component_profile": record.get("component_profile", {}),
            "component_profile_source": record.get("component_profile_source", ""),
            }
            if record.get("component_profile"):
                for membership in _component_membership_claims(
                    compound_class=compound_class,
                    entity_name=record["compound_name"],
                    profile=record["component_profile"],
                    profile_source=record.get(
                        "component_profile_source", "annotated_fragment_loss"
                    ),
                    chunk_id=chunk_id,
                    chunk=chunk,
                    evidence_quote=evidence_quote,
                    component_loss_masses=record.get("component_loss_masses", {}),
                ):
                    membership_key = (
                        membership["entity_name"].casefold(),
                        membership["component_name"].casefold(),
                        str(membership["component_count"]),
                        str(membership.get("source_file", "")).casefold(),
                        membership["component_profile_source"],
                    )
                    component_memberships[membership_key] = membership
            if key in merged:
                existing = merged[key]
                existing["source_chunk_ids"] = sorted(set(existing["source_chunk_ids"] + [chunk_id]))
                existing["evidence_ids"] = join_unique_text([existing.get("evidence_ids", ""), chunk_id])
                existing["reported_fragments"] = join_unique_text(
                    [existing.get("reported_fragments", ""), claim.get("reported_fragments", "")]
                )
            else:
                merged[key] = claim
    return [*merged.values(), *component_memberships.values()]


def refresh_structured_claims(
    compound_class: str,
    corpus_jsonl: Path | str,
    output_root: Path | str,
    claims_jsonl: Path | str | None = None,
) -> dict[str, Any]:
    """Refresh deterministic table claims while preserving prior LLM claims."""

    corpus_path = resolve(corpus_jsonl)
    root = resolve(output_root)
    claims_path = (
        resolve(claims_jsonl)
        if claims_jsonl
        else root / "evidence_claims" / "evidence_claims.jsonl"
    )
    if not claims_path.exists():
        raise LiteratureMiningError(
            f"structured-only refresh requires existing Agent 2 claims: {claims_path}"
        )
    chunks_by_id = load_chunks(corpus_path)
    existing_claims: list[dict[str, Any]] = []
    with claims_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LiteratureMiningError(
                    f"invalid evidence claim JSON at {claims_path}:{line_number}: {exc}"
                ) from exc
            if isinstance(row, dict) and row.get("claim_source") != "structured_table_mining":
                existing_claims.append(row)
    concept_path = (
        root
        / "discovered_concepts"
        / f"{safe_stem(compound_class)}_concepts.json"
    )
    domain_concepts: list[dict[str, Any]] = []
    if concept_path.exists():
        concept_payload = load_json(concept_path)
        for field in ("accepted_concepts", "review_concepts", "concepts"):
            values = concept_payload.get(field, [])
            if isinstance(values, list):
                domain_concepts.extend(
                    row for row in values if isinstance(row, dict)
                )
    structured_candidates = mine_structured_claims_from_chunks(
        compound_class, chunks_by_id
    )
    (
        guarded_structured_claims,
        rejected_structured_claims,
        domain_scope_audit,
    ) = apply_structured_domain_scope_guard(
        structured_candidates,
        chunks_by_id,
        compound_class=compound_class,
        domain_concepts=domain_concepts,
    )
    accepted_structured, review_structured, prevalidation_audit = (
        accept_prevalidated_structured_claims(guarded_structured_claims)
    )
    structured_claims = accepted_structured + review_structured
    merged_claims = merge_claims(existing_claims, structured_claims)
    write_jsonl(claims_path, merged_claims)
    write_jsonl(
        root / "evidence_claims" / "rejected_structured_claims.jsonl",
        rejected_structured_claims,
    )
    write_jsonl(
        root / "evidence_claims" / "structured_refresh_audit.jsonl",
        domain_scope_audit + prevalidation_audit,
    )
    summary = {
        "compound_class": compound_class,
        "refresh_mode": "structured_only",
        "claims_path": str(claims_path),
        "preserved_non_structured_claim_count": len(existing_claims),
        "structured_candidate_claim_count": len(structured_candidates),
        "structured_claim_count": len(structured_claims),
        "structured_domain_scope_rejected_count": len(
            rejected_structured_claims
        ),
        "structured_compound_claim_count": sum(
            claim.get("claim_type") == "compound" for claim in structured_claims
        ),
        "merged_claim_count": len(merged_claims),
    }
    write_json(root / "reports" / "structured_table_mining_summary.json", summary)
    return summary


def iter_plan_queries(query_plan: dict[str, Any]) -> list[dict[str, str]]:
    strategy = query_plan.get("retrieval_strategy", {})
    if not isinstance(strategy, dict):
        raise LiteratureMiningError("query_plan.retrieval_strategy must be an object.")
    queries: list[dict[str, str]] = []
    for group_name in QUERY_GROUPS:
        for query in normalize_string_list(strategy.get(group_name)):
            queries.append({"query_group": group_name, "query": query})
    if not queries:
        raise LiteratureMiningError("query_plan contains no retrieval queries.")
    return queries


def infer_document_role(metadata: dict[str, Any]) -> str:
    explicit = metadata.get("document_role")
    if explicit:
        return str(explicit)
    source = str(metadata.get("source_file", "")).lower()
    section = str(metadata.get("section", "")).lower()
    file_type = str(metadata.get("file_type", "")).lower()
    if "supplement" in source or "supplement" in section:
        return "supplementary_table"
    if metadata.get("row_index") is not None or file_type in {"csv", "xlsx"}:
        return "compound_catalog"
    if "table" in section:
        return "experimental_table"
    return "article_text"


def retrieve_for_queries(
    index: BM25SRetrievalIndex,
    queries: list[dict[str, str]],
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_info in queries:
        hits = index.search(
            query_info["query"],
            query_group=query_info["query_group"],
            top_k=top_k,
        )
        for hit in hits:
            metadata = {
                key: value for key, value in hit.document.items() if key != "text"
            }
            document_role = infer_document_role(metadata)
            row = {
                "chunk_id": metadata.get("chunk_id", ""),
                "source_file": metadata.get("source_file", ""),
                "file_type": metadata.get("file_type"),
                "page": metadata.get("page"),
                "sheet_name": metadata.get("sheet_name"),
                "row_index": metadata.get("row_index"),
                "section": metadata.get("section"),
                "char_start": metadata.get("char_start"),
                "char_end": metadata.get("char_end"),
                "document_role": document_role,
                "query_group": query_info["query_group"],
                "query": query_info["query"],
                "rank": hit.rank,
                "bm25s_score": hit.sparse_score,
                "evidence_alignment": hit.evidence_alignment,
                "retrieval_score": hit.score,
                "score": hit.score,
                "text": hit.document.get("text", ""),
            }
            rows.append(row)
    return rows


def expand_retrieval_with_adjacent_context(
    rows: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    radius: int = 1,
) -> list[dict[str, Any]]:
    """Add same-page neighboring chunks while preserving direct-hit identity.

    PDF prose and tables commonly cross the fixed character boundaries used by
    the corpus builder.  Context rows inherit the query label for routing, but
    are explicitly marked as context and never masquerade as independent BM25
    hits.  No compound vocabulary is used here.
    """

    if radius < 0:
        raise ValueError("radius must be non-negative")
    page_groups: dict[tuple[str, object], list[dict[str, Any]]] = {}
    positions: dict[str, tuple[tuple[str, object], int]] = {}
    for chunk in chunks_by_id.values():
        source = str(chunk.get("source_file", ""))
        page = chunk.get("page")
        if not source or page is None:
            continue
        page_groups.setdefault((source, page), []).append(chunk)
    for key, members in page_groups.items():
        members.sort(
            key=lambda item: (
                int(item.get("char_start", 0))
                if isinstance(item.get("char_start"), int)
                else 0,
                str(item.get("chunk_id", "")),
            )
        )
        for index, chunk in enumerate(members):
            chunk_id = str(chunk.get("chunk_id", ""))
            if chunk_id:
                positions[chunk_id] = (key, index)

    expanded: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        direct = dict(row)
        direct["retrieval_origin"] = "direct_hit"
        direct["context_anchor_chunk_id"] = ""
        query_key = str(row.get("query", ""))
        group_key = str(row.get("query_group", ""))
        chunk_id = str(row.get("chunk_id", ""))
        expanded[(group_key, query_key, chunk_id)] = direct
        position = positions.get(chunk_id)
        if position is None or radius == 0:
            continue
        page_key, anchor_index = position
        members = page_groups[page_key]
        lower = max(0, anchor_index - radius)
        upper = min(len(members), anchor_index + radius + 1)
        for neighbor_index in range(lower, upper):
            if neighbor_index == anchor_index:
                continue
            neighbor = members[neighbor_index]
            neighbor_id = str(neighbor.get("chunk_id", ""))
            key = (group_key, query_key, neighbor_id)
            candidate = {
                **neighbor,
                "document_role": infer_document_role(neighbor),
                "query_group": group_key,
                "query": query_key,
                "rank": row.get("rank"),
                "bm25s_score": 0.0,
                "evidence_alignment": 0.0,
                "retrieval_score": 0.0,
                "score": 0.0,
                "retrieval_origin": "adjacent_context",
                "context_anchor_chunk_id": chunk_id,
                "context_distance": abs(neighbor_index - anchor_index),
            }
            existing = expanded.get(key)
            if existing is None or existing.get("retrieval_origin") != "direct_hit":
                expanded[key] = candidate
    return list(expanded.values())


def chunk_text_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        chunk_id = str(row.get("chunk_id", ""))
        if chunk_id and chunk_id not in result:
            result[chunk_id] = str(row.get("text", ""))
    return result


def retrieval_row_score(row: dict[str, Any]) -> float:
    """Read current scores while accepting historical rows in unit fixtures."""

    for field in ("retrieval_score", "adjusted_retrieval_score", "score"):
        try:
            return float(row.get(field, 0.0))
        except (TypeError, ValueError):
            continue
    return 0.0


def group_chunks_by_evidence_type(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {group_name: [] for group_name in QUERY_GROUPS}
    for row in rows:
        group_name = str(row.get("query_group", ""))
        if group_name in grouped:
            grouped[group_name].append(row)
    for group_name in grouped:
        grouped[group_name] = sorted(
            grouped[group_name],
            key=lambda item: (
                -retrieval_row_score(item),
                str(item.get("chunk_id", "")),
            ),
        )
    return grouped


def _add_prompt_candidate(
    selected: dict[str, dict[str, Any]],
    row: dict[str, Any],
    max_chunks: int,
) -> bool:
    chunk_id = str(row.get("chunk_id", ""))
    if chunk_id and chunk_id not in selected:
        selected[chunk_id] = row
    return len(selected) >= max_chunks


def compact_chunks_for_prompt(
    rows: list[dict[str, Any]],
    max_chunks: int,
    max_chars: int,
    min_chunks_per_group: int = 5,
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    grouped = group_chunks_by_evidence_type(rows)
    for group_name in QUERY_GROUPS:
        for row in grouped[group_name][:min_chunks_per_group]:
            if _add_prompt_candidate(unique, row, max_chunks):
                break
        if len(unique) >= max_chunks:
            break
    if len(unique) < max_chunks:
        rare_limit = max(1, min(10, max_chunks // 5))
        rare_rows = sorted(
            rows,
            key=lambda item: (
                -rare_term_score(item.get("text", "")),
                -retrieval_row_score(item),
                str(item.get("chunk_id", "")),
            ),
        )
        added_rare = 0
        for row in rare_rows:
            if rare_term_score(row.get("text", "")) <= 0:
                break
            before_count = len(unique)
            if _add_prompt_candidate(unique, row, max_chunks):
                break
            if len(unique) > before_count:
                added_rare += 1
            if added_rare >= rare_limit:
                break
    if len(unique) < max_chunks:
        for row in sorted(
            rows,
            key=lambda item: (
                -retrieval_row_score(item),
                str(item.get("chunk_id", "")),
            ),
        ):
            if _add_prompt_candidate(unique, row, max_chunks):
                break
    context_by_anchor: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if normalize_space(row.get("retrieval_origin")) != "adjacent_context":
            continue
        anchor_id = normalize_space(row.get("context_anchor_chunk_id"))
        if anchor_id:
            context_by_anchor.setdefault(anchor_id, []).append(row)
    if context_by_anchor:
        context_ids = {
            normalize_space(context.get("chunk_id"))
            for contexts in context_by_anchor.values()
            for context in contexts
            if normalize_space(context.get("chunk_id"))
        }
        context_reserve = min(
            len(context_ids),
            max(1, max_chunks // 3) if max_chunks > 1 else 0,
        )
        direct_limit = max(1, max_chunks - context_reserve)
        context_preserving: dict[str, dict[str, Any]] = {}
        base_rows = list(unique.values())
        for row in base_rows[:direct_limit]:
            chunk_id = normalize_space(row.get("chunk_id"))
            if chunk_id:
                context_preserving.setdefault(chunk_id, row)
            for context in sorted(
                context_by_anchor.get(chunk_id, []),
                key=lambda item: (
                    int(item.get("context_distance", 0) or 0),
                    -rare_term_score(item.get("text", "")),
                    normalize_space(item.get("chunk_id")),
                ),
            ):
                context_id = normalize_space(context.get("chunk_id"))
                if context_id and len(context_preserving) < max_chunks:
                    context_preserving.setdefault(context_id, context)
                if len(context_preserving) >= max_chunks:
                    break
            if len(context_preserving) >= max_chunks:
                break
        if len(context_preserving) < max_chunks:
            for row in base_rows[direct_limit:]:
                chunk_id = normalize_space(row.get("chunk_id"))
                if chunk_id:
                    context_preserving.setdefault(chunk_id, row)
                if len(context_preserving) >= max_chunks:
                    break
        unique = context_preserving
    prompt_chunks: list[dict[str, Any]] = []
    for row in unique.values():
        text = normalize_space(row.get("text", ""))
        prompt_chunks.append(
            {
                "chunk_id": row.get("chunk_id"),
                "source_file": row.get("source_file"),
                "page": row.get("page"),
                "document_role": row.get("document_role"),
                "query_group": row.get("query_group"),
                "text": text[:max_chars],
            }
        )
    return prompt_chunks


def call_llm_json(
    call_type: str,
    messages: list[dict[str, str]],
    chat_completion: Callable[[list[dict[str, str]], str, str, str], str],
    model: str,
    base_url: str,
    api_key: str,
    raw_calls_path: Path,
    max_attempts: int = 3,
    fail_on_output_limit: bool = False,
) -> dict[str, Any]:
    cache = LLMResponseCache(raw_calls_path.parent / "llm_cache")
    cached = cache.lookup(
        call_type=call_type,
        messages=messages,
        model=model,
        base_url=base_url,
    )
    if cached is not None:
        append_jsonl(
            raw_calls_path,
            {
                "call_type": call_type,
                "attempt": 0,
                "cache_hit": True,
                "cache_key": cached.cache_key,
            },
        )
        return cached.payload
    last_error: Exception | None = None
    current_messages = list(messages)
    for attempt in range(1, max_attempts + 1):
        request_chars = sum(
            len(str(message.get("content") or ""))
            for message in current_messages
        )
        started = time.perf_counter()
        try:
            raw_response = run_with_wall_clock_deadline(
                lambda: chat_completion(
                    current_messages, model, base_url, api_key
                ),
                timeout_seconds=llm_timeout_seconds(),
            )
        except Exception as exc:
            append_jsonl(
                raw_calls_path,
                {
                    "call_type": call_type,
                    "attempt": attempt,
                    "status": "request_failed",
                    "request_chars": request_chars,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
        elapsed_seconds = round(time.perf_counter() - started, 3)
        response_metadata = getattr(raw_response, "response_metadata", {})
        if not isinstance(response_metadata, dict):
            response_metadata = {}
        if not str(raw_response or "").strip():
            append_jsonl(
                raw_calls_path,
                {
                    "call_type": call_type,
                    "attempt": attempt,
                    "status": "empty_response",
                    "request_chars": request_chars,
                    "response_chars": 0,
                    "elapsed_seconds": elapsed_seconds,
                    **response_metadata,
                },
            )
            if str(response_metadata.get("finish_reason") or "") == "length":
                raise LLMOutputLimitError(
                    f"{call_type} returned empty content because the output token "
                    "limit was exhausted; increase OPENAI_MAX_OUTPUT_TOKENS or "
                    "keep OPENAI_THINKING_MODE=disabled."
                )
            raise LiteratureMiningError(
                f"{call_type} returned empty content; not retrying to avoid "
                "duplicate token usage."
            )
        append_jsonl(
            raw_calls_path,
            {
                "call_type": call_type,
                "attempt": attempt,
                "status": "response_received",
                "request_chars": request_chars,
                "response_chars": len(raw_response),
                "elapsed_seconds": elapsed_seconds,
                "raw_response": raw_response,
                **response_metadata,
            },
        )
        if (
            fail_on_output_limit
            and str(response_metadata.get("finish_reason") or "") == "length"
        ):
            raise LLMOutputLimitError(
                f"{call_type} reached the output token limit and must be "
                "split into smaller evidence windows."
            )
        try:
            payload = parse_json_object(raw_response)
            cache.store_validated(
                call_type=call_type,
                messages=messages,
                model=model,
                base_url=base_url,
                payload=payload,
            )
            return payload
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            current_messages = current_messages + [
                {
                    "role": "user",
                    "content": "Return exactly one valid JSON object. Do not wrap it in markdown.",
                }
            ]
    raise LiteratureMiningError(f"{call_type} failed to return valid JSON after {max_attempts} attempts: {last_error}")


def discover_concepts(
    compound_class: str,
    retrieved_rows: list[dict[str, Any]],
    chat_completion: Callable[[list[dict[str, str]], str, str, str], str],
    model: str,
    base_url: str,
    api_key: str,
    raw_calls_path: Path,
    max_prompt_chunks: int,
    max_chunk_chars: int,
) -> dict[str, list[dict[str, Any]]]:
    prompt_chunks = compact_chunks_for_prompt(
        retrieved_rows, max_prompt_chunks, max_chunk_chars
    )
    raw_concepts: list[Any] = []
    call_number = 0

    def extract_batch(batch: list[dict[str, Any]]) -> None:
        nonlocal call_number
        call_number += 1
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a metabolomics literature mining agent. Perform domain concept discovery only. "
                    "The literature may describe metabolite knowledge at multiple hierarchy levels. "
                    "Distinguish compound, subclass, precursor, structural_component, transformation, "
                    "fragment, neutral_loss, and biosynthetic_component concepts. "
                    "A precursor is a molecule or intermediate that contributes to formation of the target class. "
                    "A structural_component is a chemical moiety incorporated into metabolites, such as sugars, "
                    "amino acids, aglycones, side chains, or acyl groups. "
                    "A biosynthetic_component is a pathway-related entity such as an enzyme, pathway molecule, "
                    "or biosynthetic intermediate. "
                    "Do not force every category to exist. "
                    "Use only the supplied chunks. Do not add model prior knowledge. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "domain concept discovery",
                        "compound_class": compound_class,
                        "allowed_concept_types": sorted(CONCEPT_TYPES),
                        "output_schema": {
                            "concepts": [
                                {
                                    "type": "",
                                    "name": "",
                                    "source_chunk_ids": [],
                                    "evidence_quote": "",
                                    "evidence_summary": "",
                                }
                            ]
                        },
                        "evidence_rules": [
                            "evidence_quote should be a short locating quote from the supplied chunk when possible.",
                            "evidence_summary may summarize the evidence in natural language.",
                            "Do not omit a concept only because the quote is incomplete.",
                            "Subclass discovery is optional.",
                            "Do not force hierarchical classification when literature does not explicitly support it.",
                            "Never infer subclass from model knowledge alone.",
                            "Precursor and structural_component concepts are optional.",
                            "Never infer precursor or structural_component concepts from model knowledge alone.",
                            "Prioritize compound, subclass, precursor, structural_component, transformation, fragment, neutral_loss, and biosynthetic_component concepts when explicitly supported.",
                        ],
                        "chunks": batch,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            payload = call_llm_json(
                f"concept_discovery_batch_{call_number}",
                messages,
                chat_completion,
                model,
                base_url,
                api_key,
                raw_calls_path,
                fail_on_output_limit=True,
            )
        except LLMOutputLimitError:
            if len(batch) <= 1:
                raise
            midpoint = len(batch) // 2
            extract_batch(batch[:midpoint])
            extract_batch(batch[midpoint:])
            return
        concepts = payload.get("concepts")
        if isinstance(concepts, list):
            raw_concepts.extend(concepts)

    for start in range(0, len(prompt_chunks), CONCEPT_DISCOVERY_BATCH_SIZE):
        extract_batch(prompt_chunks[start : start + CONCEPT_DISCOVERY_BATCH_SIZE])
    return normalize_concepts(raw_concepts, chunk_text_map(retrieved_rows))


def is_sentence_traceable(sentence: str, chunk_ids: list[str], chunks_by_id: dict[str, str]) -> dict[str, float | str]:
    return assess_traceability(sentence, chunk_ids, chunks_by_id)


def normalize_concepts(value: Any, chunks_by_id: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {
        "accepted_concepts": [],
        "review_concepts": [],
        "rejected_concepts": [],
    }
    if not isinstance(value, list):
        return result
    seen: set[tuple[str, str, str]] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        concept_type = str(item.get("type", "")).strip()
        name = str(item.get("name", "")).strip()
        source_chunk_ids = normalize_string_list(item.get("source_chunk_ids"))
        evidence_quote = normalize_space(item.get("evidence_quote") or item.get("evidence_sentence", ""))
        evidence_summary = normalize_space(item.get("evidence_summary") or item.get("evidence_sentence", ""))
        if concept_type not in CONCEPT_TYPES or not name or not source_chunk_ids or not evidence_quote:
            rejected = dict(item)
            rejected["rejection_reason"] = "missing_or_invalid_type_name_source_or_quote"
            result["rejected_concepts"].append(rejected)
            continue
        traceability = assess_traceability(evidence_quote, source_chunk_ids, chunks_by_id)
        key = (concept_type, name.lower(), evidence_quote.lower())
        if key in seen:
            continue
        seen.add(key)
        concept = {
            "type": concept_type,
            "name": name,
            "source_chunk_ids": source_chunk_ids,
            "evidence_quote": evidence_quote,
            "evidence_summary": evidence_summary,
            "traceability_status": traceability["status"],
            "traceability_score": traceability["score"],
            "review_status": "candidate",
        }
        if traceability["status"] in {"exact", "normalized", "fuzzy"}:
            result["accepted_concepts"].append(concept)
        else:
            concept["review_status"] = "needs_review"
            result["review_concepts"].append(concept)
    return result


def expand_queries_from_concepts(concepts: list[dict[str, Any]], max_queries: int) -> list[dict[str, str]]:
    queries: list[dict[str, str]] = []
    for concept in concepts:
        name = normalize_space(concept.get("name", ""))
        concept_type = str(concept.get("type", ""))
        if not name:
            continue
        if concept_type in {"compound", "synonym", "subclass"}:
            templates = [
                ("compound_queries", "{name} LC-MS/MS"),
                ("fragment_queries", "{name} fragmentation"),
                ("fragment_queries", "{name} diagnostic fragment"),
            ]
        elif concept_type == "transformation":
            templates = [
                ("transformation_queries", "{name} mass spectrometry"),
                ("transformation_queries", "{name} transformation"),
            ]
        elif concept_type == "neutral_loss":
            templates = [
                ("neutral_loss_queries", "{name} neutral loss"),
            ]
        elif concept_type == "biosynthetic_component":
            templates = [
                ("biosynthesis_queries", "{name} biosynthesis"),
                ("biosynthesis_queries", "{name} component"),
            ]
        elif concept_type == "precursor":
            templates = [
                ("biosynthesis_queries", "{name} precursor pathway"),
                ("biosynthesis_queries", "{name} biosynthesis conversion"),
                ("transformation_queries", "{name} formation from precursor"),
                ("compound_queries", "{name} LC-MS/MS"),
            ]
        elif concept_type == "structural_component":
            templates = [
                ("biosynthesis_queries", "{name} incorporation into metabolite"),
                ("transformation_queries", "{name} conjugation modification"),
                ("compound_queries", "{name} modified metabolites"),
            ]
        else:
            templates = [
                ("fragment_queries", "{name} product ion"),
            ]
        for group_name, template in templates:
            queries.append({"query_group": group_name, "query": template.format(name=name)})
            if len(queries) >= max_queries:
                return deduplicate_queries(queries)
    return deduplicate_queries(queries)


def deduplicate_queries(queries: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for row in queries:
        key = (row["query_group"], row["query"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _extract_claim_batch(
    compound_class: str,
    retrieved_rows: list[dict[str, Any]],
    chat_completion: Callable[[list[dict[str, str]], str, str, str], str],
    model: str,
    base_url: str,
    api_key: str,
    raw_calls_path: Path,
    max_prompt_chunks: int,
    max_chunk_chars: int,
    call_type: str = "evidence_claim_extraction",
) -> list[dict[str, Any]]:
    prompt_chunks = compact_chunks_for_prompt(retrieved_rows, max_prompt_chunks, max_chunk_chars)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a metabolomics literature mining agent. Extract evidence claims only. "
                "Claims may describe compounds, precursors, structural components, entity-component "
                "memberships, transformations, diagnostic fragments, neutral losses, biosynthetic components, "
                "structured reaction templates, or evidence-derived entity class memberships. "
                "Exhaustively enumerate every distinct relationship supported by the supplied chunks; "
                "do not return only representative examples. Preserve explicit reactant-product, composition, "
                "substituent, conjugation, hydrolysis, loss, and biosynthetic relationships. "
                "Precursor and structural_component claims are optional and must come from supplied chunks. "
                "Do not validate chemistry and do not compile rule tables. Use only supplied chunks. "
                "Classify fragment evidence by its reported semantic role, not by peak frequency. "
                "Do not promote a supplementary or structured peak catalog to diagnostic evidence. "
                "Return a JSON object only."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "exhaustive evidence relationship extraction",
                    "compound_class": compound_class,
                    "allowed_claim_types": sorted(CLAIM_TYPES),
                    "allowed_fragment_evidence_roles": sorted(
                        FRAGMENT_EVIDENCE_ROLES
                    ),
                    "output_schema": {
                        "claims": [
                            {
                                "claim_id": "",
                                "claim_type": "",
                                "compound_class": "",
                                "compound_name": "",
                                "precursor_name": "",
                                "component_name": "",
                                "source_entity": "",
                                "target_entity": "",
                                "transformation_name": "",
                                "reaction_name": "",
                                "reaction_type": "",
                                "network_anchor_role": "",
                                "anchor_reactant_index": 0,
                                "reactants": [],
                                "products": [],
                                "reaction_operator": "",
                                "formula_delta": "",
                                "stoichiometry_status": "",
                                "entity_id": "",
                                "entity_class_id": "",
                                "membership_role": "",
                                "entity_name": "",
                                "component_count": 0,
                                "component_profile_source": "",
                                "fragment_mz": "",
                                "evidence_role": "",
                                "specificity_scope": "",
                                "source_structure": "",
                                "assignment": "",
                                "ion_mode": "",
                                "adduct": "",
                                "neutral_loss_mass": "",
                                "formula": "",
                                "exact_mass": "",
                                "chunk_id": "",
                                "evidence_quote": "",
                                "evidence_summary": "",
                                "confidence": 0.0,
                            }
                        ]
                    },
                    "traceability_requirement": (
                        "Each evidence_quote should be a short locating quote from the referenced chunk when possible. "
                        "The evidence_summary may summarize the evidence and is not used for traceability."
                    ),
                    "relationship_extraction_rules": [
                        "Emit one claim per distinct reported relationship, including every named endpoint.",
                        "For composition evidence, retain entity_name, component_name, component_count, and chunk_id.",
                        "For narrative mechanisms, retain source_entity, target_entity, transformation_name, and chunk_id when stated.",
                        "Do not invent an endpoint or component that is absent from the supplied chunks.",
                    ],
                    "fragment_evidence_rules": [
                        (
                            "Use explicit_target_diagnostic only when the quote "
                            "explicitly describes a diagnostic, characteristic, "
                            "marker, or signature ion for a named target entity."
                        ),
                        (
                            "Use target_product_ion for an entity-bound product "
                            "ion that is reported but not explicitly diagnostic."
                        ),
                        (
                            "Use class_diagnostic only for evidence explicitly "
                            "supporting a class or subclass."
                        ),
                        (
                            "Use reaction_supporting_fragment for a fragment "
                            "assigned to a component or reaction relationship."
                        ),
                        (
                            "Use theoretical_catalog for bulk product-ion or "
                            "peak lists in structured and supplementary tables "
                            "unless the source explicitly labels an ion diagnostic."
                        ),
                        (
                            "Use unassigned_peak when neither a target nor a "
                            "class or reaction assignment is supported."
                        ),
                        (
                            "Preserve ion_mode and adduct only when reported in "
                            "the row, table/caption, section, or document context."
                        ),
                    ],
                    "chunks": prompt_chunks,
                },
                ensure_ascii=False,
            ),
        },
    ]
    payload = call_llm_json(
        call_type,
        messages,
        chat_completion,
        model,
        base_url,
        api_key,
        raw_calls_path,
    )
    return normalize_claims(payload.get("claims"), compound_class, chunk_text_map(retrieved_rows))


def extract_claims(
    compound_class: str,
    retrieved_rows: list[dict[str, Any]],
    chat_completion: Callable[[list[dict[str, str]], str, str, str], str],
    model: str,
    base_url: str,
    api_key: str,
    raw_calls_path: Path,
    max_prompt_chunks: int,
    max_chunk_chars: int,
) -> list[dict[str, Any]]:
    """Extract claims in bounded batches so long literature sets are not truncated.

    The content budget remains capped by ``max_prompt_chunks``. Batching mainly
    increases response capacity and preserves exhaustive relationship output;
    it does not silently send the entire corpus to the LLM.
    """

    selected = compact_chunks_for_prompt(
        retrieved_rows,
        max_prompt_chunks,
        max_chunk_chars,
    )
    if not selected:
        return []
    batch_count = (
        len(selected) + CLAIM_EXTRACTION_BATCH_SIZE - 1
    ) // CLAIM_EXTRACTION_BATCH_SIZE
    claims: list[dict[str, Any]] = []
    for batch_index in range(batch_count):
        start = batch_index * CLAIM_EXTRACTION_BATCH_SIZE
        batch = selected[start : start + CLAIM_EXTRACTION_BATCH_SIZE]
        claims.extend(
            _extract_claim_batch(
                compound_class,
                batch,
                chat_completion,
                model,
                base_url,
                api_key,
                raw_calls_path,
                len(batch),
                max_chunk_chars,
                call_type=f"evidence_claim_extraction_batch_{batch_index + 1}",
            )
        )
    return merge_claims(claims, [])


def _document_claim_prompt(
    compound_class: str,
    windows: list[dict[str, Any]],
    domain_concepts: list[dict[str, Any]] | None = None,
    evidence_gaps: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    # Window-ranking diagnostics are local orchestration metadata. Keeping
    # them out of the prompt prevents harmless ranking changes from invalidating
    # the content-addressed LLM cache or consuming model tokens.
    llm_windows = [
        {
            key: value
            for key, value in window.items()
            if key != "relation_signal_score" and not key.startswith("_")
        }
        for window in windows
    ]
    compact_domain_concepts = [
        {
            "type": str(concept.get("type") or ""),
            "name": str(concept.get("name") or ""),
        }
        for concept in (domain_concepts or [])
        if concept.get("name")
    ]
    compact_evidence_gaps = [
        {
            key: gap.get(key)
            for key in (
                "gap_id",
                "gap_type",
                "entity_terms",
                "expected_claim_types",
                "missing_evidence",
                "compiler_status",
            )
            if gap.get(key) not in (None, "", [], {})
        }
        for gap in (evidence_gaps or [])
    ]
    return [
        {
            "role": "system",
            "content": (
                "You are a metabolomics document-level relation extraction agent. "
                "Extract every relation supported by the supplied reconstructed document windows. "
                "Resolve relations across headings, captions, table rows, adjacent pages, and narrative "
                "sentences, but use no model prior knowledge. Exhaustively enumerate every distinct "
                "relationship instead of selecting representative examples. Preserve compound, precursor, "
                "structural component, entity-component membership, source-target transformation, fragment, "
                "neutral-loss, biosynthetic, reaction-template, and entity-class-membership evidence. "
                "Classify fragment evidence by semantic role: explicit_target_diagnostic, target_product_ion, "
                "class_diagnostic, reaction_supporting_fragment, neutral_loss, theoretical_catalog, or "
                "unassigned_peak. When a compound row merely lists product-ion peaks, preserve the complete "
                "list once in compound.reported_fragments; do not emit one diagnostic_fragment claim per "
                "unassigned peak. Emit a separate diagnostic_fragment claim only when the literature gives "
                "that ion an explicit target, class, structural, reaction-supporting, neutral-loss, or "
                "theoretical-catalog role. Do not promote a supplementary or structured peak catalog to "
                "diagnostic evidence. A transformation is a chemical relationship between stable molecular entities, "
                "not a tandem-MS precursor/product-ion step. Extract only the requested compound class, its "
                "evidence-derived concepts, and entities explicitly linked to them in the supplied text; "
                "ignore unrelated metabolites that merely co-occur in a paper. Return a JSON object only."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "document-level relation extraction",
                    "compound_class": compound_class,
                    "evidence_derived_domain_concepts": compact_domain_concepts,
                    "compiler_evidence_gaps": compact_evidence_gaps,
                    "allowed_claim_types": sorted(CLAIM_TYPES),
                    "allowed_fragment_evidence_roles": sorted(
                        FRAGMENT_EVIDENCE_ROLES
                    ),
                    "output_schema": {
                        "claims": [
                            {
                                "claim_type": "one allowed claim type",
                                "evidence_unit_id": "one supplied window ID",
                                "chunk_id": "one locating member chunk ID",
                                "source_chunk_ids": ["supporting member chunk IDs"],
                                "evidence_quote": "verbatim locating quote",
                                "evidence_summary": "brief evidence-bound summary",
                                "confidence": 0.0,
                                "claim_specific_fields": "include only supported fields listed below",
                            }
                        ]
                    },
                    "claim_fields_by_type": {
                        "compound": [
                            "compound_name", "formula", "exact_mass",
                            "reported_precursor_mz", "reported_fragments",
                            "ion_mode", "adduct",
                        ],
                        "precursor": [
                            "precursor_name", "formula", "exact_mass",
                            "ion_mode", "adduct",
                        ],
                        "structural_component": [
                            "component_name", "formula", "exact_mass",
                        ],
                        "entity_component_membership": [
                            "entity_name", "entity_id", "component_name",
                            "component_count", "membership_role",
                        ],
                        "transformation": [
                            "source_entity", "target_entity",
                            "transformation_name", "formula_delta",
                            "evidence_type", "ion_mode",
                        ],
                        "diagnostic_fragment": [
                            "compound_name", "fragment_mz", "evidence_role",
                            "specificity_scope", "source_structure",
                            "assignment", "ion_mode", "adduct",
                        ],
                        "neutral_loss": [
                            "compound_name", "neutral_loss_mass", "assignment",
                            "ion_mode", "adduct",
                        ],
                        "biosynthetic_component": [
                            "component_name", "precursor_name", "entity_name",
                            "membership_role", "formula", "exact_mass",
                        ],
                        "reaction_template": [
                            "reaction_name", "reaction_type",
                            "network_anchor_role", "anchor_reactant_index",
                            "reactants", "products", "reaction_operator",
                            "formula_delta", "stoichiometry_status",
                        ],
                        "entity_class_membership": [
                            "entity_id", "entity_name", "entity_class_id",
                            "membership_role",
                        ],
                    },
                    "serialization_rules": [
                        "Omit every optional field whose value is empty, null, an empty list, or an empty object.",
                        "Do not copy the full field catalog into each claim; include only common provenance fields and fields relevant to that claim_type.",
                    ],
                    "rules": [
                        "Each claim must name one supplied evidence_unit_id.",
                        "source_chunk_ids must be copied from that document window, never invented.",
                        "chunk_id should be the member chunk containing the locating quote when possible.",
                        "The evidence_quote must be a locating quote found in the reconstructed window.",
                        "Do not infer a concrete entity, endpoint, component, ion mode, adduct, or mass absent from the window.",
                        "When compiler_evidence_gaps are supplied, prioritize claims that directly address them, but emit a claim only when the supplied window supports it.",
                        "A compiler gap is a search instruction, never evidence. Do not copy a missing value from the gap into a claim.",
                        "A relationship spanning adjacent chunks may cite all supporting source_chunk_ids.",
                        "Stable-entity transformations must not use MS/MS ion expressions such as [M+H]+, [M-H]-, m/z values, or product-ion formulas as source_entity or target_entity.",
                        "Encode CID fragmentation and precursor-to-product-ion losses only as diagnostic_fragment or neutral_loss claims, never as transformation claims.",
                        "For an unassigned product-ion list, store all peaks in the owning compound claim's reported_fragments field and do not duplicate them as individual diagnostic_fragment claims.",
                        "Do not emit both a neutral_loss claim and a diagnostic_fragment claim for the same peak unless the window explicitly reports both semantic roles.",
                        "Exclude unrelated co-reported metabolite classes unless the window explicitly links the entity to compound_class or an evidence-derived domain concept.",
                        "An entity_class_membership claim must name the member entity; a class label alone is not a claim.",
                    ],
                    "document_windows": llm_windows,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _owner_chunk_for_document_claim(
    item: dict[str, Any],
    window: dict[str, Any],
    chunks_by_id: dict[str, str],
) -> str:
    allowed = [str(value) for value in window.get("source_chunk_ids", [])]
    requested = str(item.get("chunk_id") or "").strip()
    quote = normalize_space(item.get("evidence_quote") or item.get("evidence_sentence", ""))
    if requested in allowed and assess_traceability(quote, [requested], chunks_by_id)["status"] != "failed":
        return requested
    for chunk_id in allowed:
        if assess_traceability(quote, [chunk_id], chunks_by_id)["status"] != "failed":
            return chunk_id
    return requested if requested in allowed else (allowed[0] if allowed else "")


def split_document_window_for_output_limit(
    window: dict[str, Any],
    *,
    overlap_chars: int = 120,
    minimum_part_chars: int = 300,
) -> list[dict[str, Any]]:
    """Split one overflowing window at a nearby line boundary.

    The overlap preserves relations crossing the boundary; downstream semantic
    claim deduplication removes duplicates.  Provenance chunk IDs are retained
    unchanged, while each segment gets a stable evidence-unit ID.
    """

    text = str(window.get("text") or "")
    depth = int(window.get("extraction_split_depth") or 0)
    if len(text) < minimum_part_chars * 2 or depth >= 8:
        return []
    midpoint = len(text) // 2
    candidates = [
        position
        for position in (
            text.rfind("\n", max(0, midpoint - 600), midpoint + 1),
            text.find("\n", midpoint, min(len(text), midpoint + 600)),
        )
        if position >= minimum_part_chars
        and len(text) - position >= minimum_part_chars
    ]
    boundary = min(candidates, key=lambda value: abs(value - midpoint)) if candidates else midpoint
    ranges = (
        (0, min(len(text), boundary + overlap_chars)),
        (max(0, boundary - overlap_chars), len(text)),
    )
    original_id = str(window.get("window_id") or "document_window")
    parts: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(ranges, start=1):
        part_text = text[start:end]
        digest = hashlib.sha1(
            f"{original_id}|{start}|{end}|{part_text}".encode("utf-8")
        ).hexdigest()[:16]
        parts.append(
            {
                **window,
                "window_id": f"docwin_{digest}",
                "text": part_text,
                "extraction_split_depth": depth + 1,
                "parent_window_id": original_id,
                "segment_char_start": start,
                "segment_char_end": end,
                "segment_index": index,
            }
        )
    return parts


def call_was_output_limited(raw_calls_path: Path, call_type: str) -> bool:
    """Return whether a prior resumable run requires a smaller evidence window.

    A length-limited response is deliberately not cached as valid JSON.  On a
    resumed run we therefore split that evidence window before making another
    API request, avoiding payment for the same known-overflowing parent window.
    A request that exceeded the explicit wall-clock deadline is handled the
    same way: the parent request is never blindly retried, while smaller child
    windows remain eligible for a fresh call.
    """

    if not raw_calls_path.exists():
        return False
    try:
        with raw_calls_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("call_type") or "") != call_type:
                    continue
                output_limited = str(row.get("finish_reason") or "") == "length"
                deadline_exceeded = (
                    str(row.get("status") or "") == "request_failed"
                    and "wall-clock deadline" in str(row.get("error") or "").lower()
                )
                if output_limited or deadline_exceeded:
                    return True
    except (OSError, json.JSONDecodeError):
        return False
    return False


def extract_document_relations(
    compound_class: str,
    document_windows: list[dict[str, Any]],
    chunks_by_id: dict[str, str],
    chat_completion: Callable[[list[dict[str, str]], str, str, str], str],
    model: str,
    base_url: str,
    api_key: str,
    raw_calls_path: Path,
    *,
    batch_size: int = DOCUMENT_RELATION_BATCH_SIZE,
    call_type_prefix: str = "document_relation_extraction",
    domain_concepts: list[dict[str, Any]] | None = None,
    evidence_gaps: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Extract relations from reconstructed document windows in bounded batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    windows_by_id = {
        str(window.get("window_id") or ""): window
        for window in document_windows
        if window.get("window_id")
    }
    claims: list[dict[str, Any]] = []
    for batch_index, start in enumerate(
        range(0, len(document_windows), batch_size), start=1
    ):
        batch = document_windows[start : start + batch_size]
        batch_label = (
            str(batch[0].get("window_id") or f"batch_{batch_index}")
            if len(batch) == 1
            else f"batch_{batch_index}"
        )
        call_type = f"{call_type_prefix}_{batch_label}"
        if len(batch) == 1 and call_was_output_limited(raw_calls_path, call_type):
            split_windows = split_document_window_for_output_limit(batch[0])
            if not split_windows:
                raise LLMOutputLimitError(
                    f"{call_type} previously reached the output token limit and "
                    "cannot be split safely any further."
                )
            claims.extend(
                extract_document_relations(
                    compound_class,
                    split_windows,
                    chunks_by_id,
                    chat_completion,
                    model,
                    base_url,
                    api_key,
                    raw_calls_path,
                    batch_size=1,
                    call_type_prefix=call_type_prefix,
                    domain_concepts=domain_concepts,
                    evidence_gaps=evidence_gaps,
                )
            )
            continue
        try:
            payload = call_llm_json(
                call_type,
                _document_claim_prompt(
                    compound_class,
                    batch,
                    domain_concepts,
                    evidence_gaps,
                ),
                chat_completion,
                model,
                base_url,
                api_key,
                raw_calls_path,
                fail_on_output_limit=True,
            )
        except LLMOutputLimitError:
            if len(batch) != 1:
                raise
            split_windows = split_document_window_for_output_limit(batch[0])
            if not split_windows:
                raise
            claims.extend(
                extract_document_relations(
                    compound_class,
                    split_windows,
                    chunks_by_id,
                    chat_completion,
                    model,
                    base_url,
                    api_key,
                    raw_calls_path,
                    batch_size=1,
                    call_type_prefix=call_type_prefix,
                    domain_concepts=domain_concepts,
                    evidence_gaps=evidence_gaps,
                )
            )
            continue
        raw_claims = payload.get("claims")
        if not isinstance(raw_claims, list):
            continue
        for raw_item in raw_claims:
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            unit_id = str(item.get("evidence_unit_id") or "").strip()
            if not unit_id:
                requested_chunk_id = str(item.get("chunk_id") or "").strip()
                matching_units = [
                    window_id
                    for window_id, candidate_window in windows_by_id.items()
                    if requested_chunk_id
                    in {
                        str(value)
                        for value in candidate_window.get("source_chunk_ids", [])
                    }
                ]
                if len(matching_units) == 1:
                    unit_id = matching_units[0]
            window = windows_by_id.get(unit_id)
            if window is None:
                continue
            allowed_ids = [
                str(value) for value in window.get("source_chunk_ids", [])
            ]
            requested_ids = normalize_string_list(item.get("source_chunk_ids"))
            source_ids = [value for value in requested_ids if value in allowed_ids]
            if not source_ids:
                source_ids = allowed_ids
            owner = _owner_chunk_for_document_claim(item, window, chunks_by_id)
            item["chunk_id"] = unit_id
            window_text_map = {unit_id: str(window.get("text") or "")}
            normalized = normalize_claims(
                [item], compound_class, window_text_map
            )
            for claim in normalized:
                claim["evidence_unit_id"] = unit_id
                claim["chunk_id"] = owner
                claim["source_chunk_ids"] = source_ids
                claim["source_file"] = window.get("source_file", "")
                claim["source_structure"] = (
                    claim.get("source_structure")
                    or (
                        "structured_table"
                        if "table" in str(window.get("document_role") or "")
                        else "narrative"
                    )
                )
                if claim.get("traceability_status") in {
                    "exact",
                    "normalized",
                    "fuzzy",
                }:
                    claim["traceability_status"] = "document_window"
                    claim["traceability_score"] = 1.0
                    claim["review_status"] = "candidate"
                claim["claim_id"] = stable_claim_id(claim)
                claims.append(claim)
    return merge_claims(claims, [])


def _critic_prompt(
    compound_class: str,
    claims: list[dict[str, Any]],
    domain_concepts: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    compact_claims = [
        {
            key: claim.get(key)
            for key in (
                "claim_id",
                "claim_type",
                "compound_name",
                "entity_name",
                "component_name",
                "component_count",
                "source_entity",
                "target_entity",
                "transformation_name",
                "fragment_mz",
                "neutral_loss_mass",
                "formula",
                "exact_mass",
                "ion_mode",
                "adduct",
                "source_chunk_ids",
                "evidence_quote",
                "evidence_summary",
            )
            if claim.get(key) not in (None, "", [], {})
        }
        for claim in claims
    ]
    return [
        {
            "role": "system",
            "content": (
                "You are an evidence critic for metabolomics literature relations. "
                "Judge whether each proposed claim is supported by its supplied quote. "
                "You may accept, send to review, or reject existing claim IDs only; never create, "
                "rewrite, merge, or expand claims. Distinguish explicit_report, literature_inferred, "
                "and unsupported. Formula or mass plausibility alone is not literature support. "
                "Reject a transformation whose endpoints are tandem-MS ion expressions rather than stable "
                "molecular entities. Reject a claim whose relationship to the requested compound class or "
                "evidence-derived domain concepts is not established by the supplied claim evidence. "
                "Return a JSON object only."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "evidence claim criticism",
                    "compound_class": compound_class,
                    "evidence_derived_domain_concepts": [
                        {
                            "type": str(concept.get("type") or ""),
                            "name": str(concept.get("name") or ""),
                        }
                        for concept in (domain_concepts or [])
                        if concept.get("name")
                    ],
                    "allowed_verdicts": ["accept", "review", "reject"],
                    "allowed_support_types": [
                        "explicit_report",
                        "literature_inferred",
                        "unsupported",
                    ],
                    "output_schema": {
                        "decisions": [
                            {
                                "claim_id": "",
                                "verdict": "",
                                "support_type": "",
                                "reason_codes": [],
                                "missing_fields": [],
                            }
                        ]
                    },
                    "claims": compact_claims,
                },
                ensure_ascii=False,
            ),
        },
    ]


def critic_claims(
    compound_class: str,
    claims: list[dict[str, Any]],
    chunks_by_id: dict[str, str],
    chat_completion: Callable[[list[dict[str, str]], str, str, str], str],
    model: str,
    base_url: str,
    api_key: str,
    raw_calls_path: Path,
    *,
    batch_size: int = EVIDENCE_CRITIC_BATCH_SIZE,
    call_type_prefix: str = "evidence_critic",
    domain_concepts: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Criticize LLM claims without permitting the critic to create evidence."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    accepted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(claims), batch_size), start=1):
        batch = claims[start : start + batch_size]
        payload = call_llm_json(
            f"{call_type_prefix}_batch_{batch_index}",
            _critic_prompt(compound_class, batch, domain_concepts),
            chat_completion,
            model,
            base_url,
            api_key,
            raw_calls_path,
        )
        decisions = payload.get("decisions")
        if not isinstance(decisions, list):
            decisions = []
        valid_ids = {str(claim.get("claim_id") or "") for claim in batch}
        decisions = [
            decision
            for decision in decisions
            if isinstance(decision, dict)
            and str(decision.get("claim_id") or "") in valid_ids
        ]
        batch_accepted, batch_review, batch_rejected, batch_audit = (
            apply_critic_decisions(batch, decisions, chunks_by_id)
        )
        accepted.extend(batch_accepted)
        review.extend(batch_review)
        rejected.extend(batch_rejected)
        audit.extend(batch_audit)
    return accepted, review, rejected, audit


def stable_claim_id(claim: dict[str, Any]) -> str:
    # One evidence sentence commonly supports several distinct relations (for
    # example, several named substituents). Provenance alone therefore cannot
    # identify a claim. Include the semantic payload while excluding mutable
    # critic/review annotations so IDs remain deterministic across reruns.
    identity_fields = (
        "claim_type",
        "compound_class",
        "compound_name",
        "precursor_name",
        "component_name",
        "source_entity",
        "target_entity",
        "transformation_name",
        "fragment_mz",
        "reported_fragments",
        "evidence_role",
        "specificity_scope",
        "source_structure",
        "assignment",
        "ion_mode",
        "adduct",
        "entity_id",
        "entity_name",
        "entity_class_id",
        "component_count",
        "membership_role",
        "neutral_loss_mass",
        "formula",
        "exact_mass",
        "reaction_name",
        "reaction_type",
        "network_anchor_role",
        "anchor_reactant_index",
        "reactants",
        "products",
        "reaction_operator",
        "formula_delta",
        "stoichiometry_status",
        "chunk_id",
        "evidence_quote",
    )

    def canonical(value: Any) -> Any:
        if isinstance(value, str):
            return normalize_space(value).casefold()
        if isinstance(value, dict):
            return {
                str(key): canonical(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        return value

    payload = {
        field: canonical(claim.get(field))
        for field in identity_fields
        if claim.get(field) not in (None, "", [], {})
    }
    basis = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "claim_" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def _structured_claim_row(
    item: dict[str, Any],
    compound_class: str,
    chunks_by_id: dict[str, str],
) -> dict[str, Any]:
    """Preserve structured evidence fields and attach validation diagnostics."""

    chunk_id = str(item.get("chunk_id", "")).strip()
    evidence_quote = normalize_space(item.get("evidence_quote") or item.get("evidence_sentence", ""))
    evidence_summary = normalize_space(item.get("evidence_summary") or item.get("evidence_sentence", ""))
    traceability = assess_traceability(evidence_quote, [chunk_id], chunks_by_id)
    common = {
        "claim_type": str(item.get("claim_type", "")).strip(),
        "compound_class": compound_class,
        "chunk_id": chunk_id,
        "source_chunk_ids": [chunk_id],
        "evidence_quote": evidence_quote,
        "evidence_summary": evidence_summary,
        "traceability_status": traceability["status"],
        "traceability_score": traceability["score"],
        "confidence": parse_confidence(item.get("confidence")),
    }
    try:
        parsed = parse_structured_claim({**item, **common})
    except ClaimValidationError as exc:
        row = {
            **common,
            "claim_id": str(item.get("claim_id") or "").strip(),
            "structured_validation_status": "invalid",
            "structured_validation_error": str(exc),
            "review_status": "needs_review",
        }
        for field in (
            "reaction_name",
            "reaction_type",
            "network_anchor_role",
            "anchor_reactant_index",
            "reactants",
            "products",
            "reaction_operator",
            "formula_delta",
            "stoichiometry_status",
            "entity_id",
            "entity_class_id",
            "membership_role",
        ):
            if field in item:
                row[field] = item[field]
        row["claim_id"] = stable_claim_id(row)
        return row

    row = {
        **common,
        "claim_id": parsed.claim_id,
        "structured_validation_status": "valid",
        "structured_validation_error": "",
        "review_status": "candidate"
        if traceability["status"] in {"exact", "normalized", "fuzzy"}
        else "needs_review",
    }
    if isinstance(parsed, ReactionTemplateClaim):
        row.update(
            {
                "reaction_name": parsed.reaction_name,
                "reaction_type": parsed.reaction_type,
                "network_anchor_role": parsed.network_anchor_role,
                "anchor_reactant_index": parsed.anchor_reactant_index,
                "reactants": [asdict(participant) for participant in parsed.reactants],
                "products": [asdict(participant) for participant in parsed.products],
                "reaction_operator": parsed.reaction_operator,
                "formula_delta": parsed.formula_delta,
                "stoichiometry_status": parsed.stoichiometry_status,
            }
        )
    elif isinstance(parsed, EntityClassMembershipClaim):
        row.update(
            {
                "entity_id": parsed.entity_id,
                "entity_class_id": parsed.entity_class_id,
                "membership_role": parsed.membership_role,
            }
        )
    return row


def normalize_claims(value: Any, compound_class: str, chunks_by_id: dict[str, str]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return claims
    for item in value:
        if not isinstance(item, dict):
            continue
        claim_type = str(item.get("claim_type", "")).strip()
        if claim_type not in CLAIM_TYPES:
            continue
        item_class = str(item.get("compound_class") or compound_class).strip()
        if item_class != compound_class:
            continue
        if claim_type in {"reaction_template", "entity_class_membership"}:
            chunk_id = str(item.get("chunk_id", "")).strip()
            evidence_quote = normalize_space(item.get("evidence_quote") or item.get("evidence_sentence", ""))
            if not chunk_id or not evidence_quote:
                continue
            claims.append(_structured_claim_row(item, compound_class, chunks_by_id))
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        evidence_quote = normalize_space(item.get("evidence_quote") or item.get("evidence_sentence", ""))
        evidence_summary = normalize_space(item.get("evidence_summary") or item.get("evidence_sentence", ""))
        if not chunk_id or not evidence_quote:
            continue
        traceability = assess_traceability(evidence_quote, [chunk_id], chunks_by_id)
        claim = {
            "claim_id": str(item.get("claim_id") or "").strip(),
            "claim_type": claim_type,
            "compound_class": compound_class,
            "compound_name": str(item.get("compound_name", "")).strip(),
            "precursor_name": str(item.get("precursor_name", "")).strip(),
            "component_name": str(item.get("component_name", "")).strip(),
            "source_entity": str(item.get("source_entity", "")).strip(),
            "target_entity": str(item.get("target_entity", "")).strip(),
            "transformation_name": str(item.get("transformation_name", "")).strip(),
            "fragment_mz": str(item.get("fragment_mz", "")).strip(),
            "reported_fragments": str(item.get("reported_fragments", "")).strip(),
            "evidence_role": (
                str(item.get("evidence_role", "")).strip()
                if str(item.get("evidence_role", "")).strip()
                in FRAGMENT_EVIDENCE_ROLES
                else ""
            ),
            "specificity_scope": str(
                item.get("specificity_scope", "")
            ).strip(),
            "source_structure": str(item.get("source_structure", "")).strip(),
            "evidence_scope": str(item.get("evidence_scope", "")).strip(),
            "fragment_evidence_scope": str(item.get("fragment_evidence_scope", "")).strip(),
            "assignment": str(item.get("assignment", "")).strip(),
            "ion_mode": str(item.get("ion_mode", "")).strip(),
            "adduct": str(item.get("adduct", "")).strip(),
            "entity_id": str(item.get("entity_id", "")).strip(),
            "entity_class_id": str(item.get("entity_class_id", "")).strip(),
            "entity_name": str(item.get("entity_name", "")).strip(),
            "component_count": item.get("component_count", 0),
            "component_profile_source": str(
                item.get("component_profile_source", "")
            ).strip(),
            "neutral_loss_mass": str(item.get("neutral_loss_mass", "")).strip(),
            "formula": str(item.get("formula", "")).strip(),
            "exact_mass": str(item.get("exact_mass", "")).strip(),
            "chunk_id": chunk_id,
            "source_chunk_ids": [chunk_id],
            "evidence_quote": evidence_quote,
            "evidence_summary": evidence_summary,
            "traceability_status": traceability["status"],
            "traceability_score": traceability["score"],
            "review_status": "candidate" if traceability["status"] in {"exact", "normalized", "fuzzy"} else "needs_review",
            "confidence": parse_confidence(item.get("confidence")),
        }
        claim["claim_id"] = stable_claim_id(claim)
        claims.append(claim)
    return claims


def parse_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def top_sources(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source_file", "") or "")
        if source:
            counts[source] = counts.get(source, 0) + 1
    return [
        {"source_file": source, "count": count}
        for source, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def count_by_type(rows: list[dict[str, Any]], field_name: str, allowed_types: set[str]) -> dict[str, int]:
    counts = {type_name: 0 for type_name in sorted(allowed_types)}
    for row in rows:
        type_name = str(row.get(field_name, ""))
        if type_name in counts:
            counts[type_name] += 1
    return counts


def build_concept_coverage_report(compound_class: str, accepted: list[dict[str, Any]], review: list[dict[str, Any]]) -> dict[str, Any]:
    all_preserved = accepted + review
    counts = count_by_type(all_preserved, "type", CONCEPT_TYPES)
    return {
        "compound_class": compound_class,
        "subclass_status": build_subclass_status(accepted, review),
        "accepted_concept_count": len(accepted),
        "review_concept_count": len(review),
        "preserved_concept_count": len(all_preserved),
        "concept_type_counts": counts,
        "warnings": [
            f"zero_{concept_type}_concepts"
            for concept_type, count in counts.items()
            if count == 0
        ],
    }


def build_subclass_status(accepted: list[dict[str, Any]], review: list[dict[str, Any]]) -> str:
    if any(str(concept.get("type", "")) == "subclass" for concept in accepted):
        return "reported"
    if any(str(concept.get("type", "")) == "subclass" for concept in review):
        return "needs_review"
    return "not_reported"


def unique_chunk_count(rows: list[dict[str, Any]]) -> int:
    return len({str(row.get("chunk_id", "")) for row in rows if row.get("chunk_id")})


def claim_dedupe_key(claim: dict[str, Any]) -> tuple[str, ...]:
    claim_type = normalize_space(claim.get("claim_type", "")).lower()
    compound_class = normalize_space(claim.get("compound_class", "")).lower()
    if claim_type == "compound":
        return (
            claim_type,
            compound_class,
            normalize_space(claim.get("compound_name", "")).lower(),
            normalize_space(claim.get("formula", "")).lower(),
            normalize_space(claim.get("exact_mass", "")),
            normalize_space(claim.get("evidence_role", "")).lower(),
            normalize_space(claim.get("source_structure", "")).lower(),
        )
    if claim_type == "diagnostic_fragment":
        return (
            claim_type,
            compound_class,
            normalize_space(claim.get("subclass", "")).lower(),
            normalize_space(claim.get("fragment_mz", "")),
            normalize_space(claim.get("ion_mode", "")).lower(),
        )
    if claim_type == "neutral_loss":
        return (
            claim_type,
            compound_class,
            normalize_space(claim.get("subclass", "")).lower(),
            normalize_space(claim.get("neutral_loss_mass", "") or claim.get("loss_mass", "")),
            normalize_space(claim.get("ion_mode", "")).lower(),
        )
    if claim_type == "transformation":
        return (
            claim_type,
            compound_class,
            normalize_space(claim.get("source_entity", "")).lower(),
            normalize_space(claim.get("target_entity", "")).lower(),
            normalize_space(claim.get("transformation_name", "")).lower(),
        )
    if claim_type in {"biosynthetic_component", "precursor", "structural_component"}:
        return (
            claim_type,
            compound_class,
            normalize_space(claim.get("component_name", "") or claim.get("precursor_name", "")).lower(),
            normalize_space(claim.get("formula", "")).lower(),
            normalize_space(claim.get("exact_mass", "")),
        )
    if claim_type == "entity_component_membership":
        return (
            claim_type,
            compound_class,
            normalize_space(
                claim.get("entity_id", "") or claim.get("entity_name", "")
            ).lower(),
            normalize_space(claim.get("component_name", "")).lower(),
            normalize_space(claim.get("component_count", "")),
            normalize_space(claim.get("component_profile_source", "")).lower(),
        )
    return (
        claim_type,
        compound_class,
        normalize_space(claim.get("claim_id", "")).lower(),
        normalize_space(claim.get("chunk_id", "")).lower(),
        normalize_space(claim.get("evidence_quote", "")).lower(),
    )


def merge_claims(primary_claims: list[dict[str, Any]], secondary_claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, ...], dict[str, Any]] = {}
    for claim in primary_claims + secondary_claims:
        key = claim_dedupe_key(claim)
        if key in merged:
            existing = merged[key]
            existing["source_chunk_ids"] = sorted(
                set(normalize_string_list(existing.get("source_chunk_ids")) + normalize_string_list(claim.get("source_chunk_ids")))
            )
            existing["evidence_ids"] = join_unique_text([existing.get("evidence_ids", ""), claim.get("evidence_ids", ""), claim.get("claim_id", "")])
            existing["reported_fragments"] = join_unique_text(
                [existing.get("reported_fragments", ""), claim.get("reported_fragments", "")]
            )
            if not normalize_space(existing.get("reported_precursor_mz", "")):
                existing["reported_precursor_mz"] = normalize_space(claim.get("reported_precursor_mz", ""))
        else:
            merged[key] = dict(claim)
    return list(merged.values())


def initialize_agent_state(compound_class: str, initial_queries: list[dict[str, str]], query_plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "compound_class": compound_class,
        "iteration_count": 0,
        "query_plan": {
            "initial_queries": initial_queries,
            "query_groups": [group_name for group_name in QUERY_GROUPS if query_plan.get("retrieval_strategy", {}).get(group_name)],
        },
        "retrieval_state": {
            "iteration_1": {
                "query_count": 0,
                "chunk_count": 0,
                "unique_chunk_count": 0,
                "top_sources": [],
            },
            "iteration_2": {
                "expanded_query_count": 0,
                "chunk_count": 0,
                "unique_chunk_count": 0,
                "top_sources": [],
            },
        },
        "concept_state": {
            "concept_count": 0,
            "subclass_status": "not_reported",
            "accepted_concept_count": 0,
            "review_concept_count": 0,
            "rejected_concept_count": 0,
            "concept_types": count_by_type([], "type", CONCEPT_TYPES),
            "concepts": [],
        },
        "claim_state": {
            "claim_count": 0,
            "claim_types": count_by_type([], "claim_type", CLAIM_TYPES),
        },
    }


def save_agent_state(path: Path, state: dict[str, Any]) -> None:
    write_json(path, state)


def run_literature_mining(
    query_plan_path: Path | str,
    corpus_jsonl: Path | str,
    index_path: Path | str,
    output_root: Path | str,
    chat_completion: Callable[[list[dict[str, str]], str, str, str], str] = chat_completion_openai_compatible,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    top_k: int = 12,
    max_iterations: int = 2,
    max_prompt_chunks: int = 60,
    max_chunk_chars: int = 1600,
    max_expanded_queries: int = 80,
    max_document_windows: int = 24,
    document_window_chars: int = 6000,
    max_gap_queries: int = 24,
    max_gap_document_windows: int = 12,
) -> dict[str, Any]:
    plan_path = resolve(query_plan_path)
    corpus_path = resolve(corpus_jsonl)
    bm25_path = resolve(index_path)
    root = resolve(output_root)
    model = model if model is not None else os.environ.get("OPENAI_MODEL", "")
    base_url = base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL", "")
    api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")

    query_plan = load_json(plan_path)
    compound_class = str(query_plan.get("compound_class", "")).strip()
    if not compound_class:
        raise LiteratureMiningError("query_plan.compound_class is required.")

    chunks_by_id = load_chunks(corpus_path)
    if not chunks_by_id:
        raise LiteratureMiningError("corpus contains no chunks.")
    try:
        index = load_retrieval_index(bm25_path)
    except (OSError, ValueError, RetrievalError) as exc:
        raise LiteratureMiningError(f"failed to load retrieval index: {exc}") from exc

    def retrieve(queries: list[dict[str, str]]) -> list[dict[str, Any]]:
        return retrieve_for_queries(index, queries, top_k)

    mining_dir = root / "mining_results"
    concepts_dir = root / "discovered_concepts"
    claims_dir = root / "evidence_claims"
    reports_dir = root / "reports"
    raw_calls_path = root / "raw_outputs" / "literature_mining_calls.jsonl"
    raw_calls_path.parent.mkdir(parents=True, exist_ok=True)
    raw_calls_path.write_text("", encoding="utf-8")

    initial_queries = iter_plan_queries(query_plan)
    agent_state_path = mining_dir / "agent_state.json"
    agent_state = initialize_agent_state(compound_class, initial_queries, query_plan)
    initial_rows = expand_retrieval_with_adjacent_context(
        retrieve(initial_queries),
        chunks_by_id,
    )
    write_jsonl(mining_dir / "initial_retrieved_chunks.jsonl", initial_rows)
    agent_state["iteration_count"] = 1
    agent_state["retrieval_state"]["iteration_1"] = {
        "query_count": len(initial_queries),
        "chunk_count": len(initial_rows),
        "unique_chunk_count": unique_chunk_count(initial_rows),
        "top_sources": top_sources(initial_rows),
    }
    save_agent_state(agent_state_path, agent_state)

    concept_payload = discover_concepts(
        compound_class,
        initial_rows,
        chat_completion,
        model,
        base_url,
        api_key,
        raw_calls_path,
        max_prompt_chunks,
        max_chunk_chars,
    )
    accepted_concepts = concept_payload["accepted_concepts"]
    review_concepts = concept_payload["review_concepts"]
    rejected_concepts = concept_payload["rejected_concepts"]
    concepts = accepted_concepts + review_concepts
    subclass_status = build_subclass_status(accepted_concepts, review_concepts)
    write_json(
        concepts_dir / f"{safe_stem(compound_class)}_concepts.json",
        {
            "compound_class": compound_class,
            "subclass_status": subclass_status,
            "concept_count": len(concepts),
            "accepted_concepts": accepted_concepts,
            "review_concepts": review_concepts,
            "rejected_concepts": rejected_concepts,
        },
    )
    write_json(
        reports_dir / "concept_discovery_coverage.json",
        build_concept_coverage_report(compound_class, accepted_concepts, review_concepts),
    )
    agent_state["concept_state"] = {
        "concept_count": len(concepts),
        "subclass_status": subclass_status,
        "accepted_concept_count": len(accepted_concepts),
        "review_concept_count": len(review_concepts),
        "rejected_concept_count": len(rejected_concepts),
        "concept_types": count_by_type(concepts, "type", CONCEPT_TYPES),
        "concepts": concepts,
    }
    save_agent_state(agent_state_path, agent_state)

    expanded_queries: list[dict[str, str]] = []
    iterations_used = 1
    if max_iterations >= 2:
        expanded_queries = expand_queries_from_concepts(concepts, max_expanded_queries)
        iterations_used = 2

    deep_queries = deduplicate_queries(initial_queries + expanded_queries)
    deep_rows = expand_retrieval_with_adjacent_context(
        retrieve(deep_queries),
        chunks_by_id,
    )
    write_jsonl(mining_dir / "deep_retrieved_chunks.jsonl", deep_rows)
    agent_state["iteration_count"] = iterations_used
    agent_state["retrieval_state"]["iteration_2"] = {
        "expanded_query_count": len(expanded_queries),
        "chunk_count": len(deep_rows),
        "unique_chunk_count": unique_chunk_count(deep_rows),
        "top_sources": top_sources(deep_rows),
    }
    save_agent_state(agent_state_path, agent_state)

    chunks_text = {
        chunk_id: str(chunk.get("text") or "")
        for chunk_id, chunk in chunks_by_id.items()
    }
    document_retrieval_rows = select_relation_retrieval_rows(deep_rows)
    document_windows = build_document_windows(
        chunks_by_id,
        document_retrieval_rows,
        max_windows=max_document_windows,
        max_window_chars=document_window_chars,
        adjacent_pages=1,
    )
    write_jsonl(mining_dir / "document_relation_windows.jsonl", document_windows)
    document_llm_claims = extract_document_relations(
        compound_class,
        document_windows,
        chunks_text,
        chat_completion,
        model,
        base_url,
        api_key,
        raw_calls_path,
        domain_concepts=concepts,
    )
    (
        guarded_document_claims,
        semantic_rejected_document_claims,
        document_semantic_guard_audit,
    ) = apply_domain_semantic_guards(
        document_llm_claims,
        compound_class=compound_class,
        domain_concepts=concepts,
    )
    (
        accepted_document_claims,
        review_document_claims,
        critic_rejected_document_claims,
        semantic_document_critic_audit,
    ) = critic_claims(
        compound_class,
        guarded_document_claims,
        chunks_text,
        chat_completion,
        model,
        base_url,
        api_key,
        raw_calls_path,
        domain_concepts=concepts,
    )
    (
        recovered_document_claims,
        critic_rejected_document_claims,
        document_recovery_audit,
    ) = recover_name_encoded_modification_claims(
        critic_rejected_document_claims,
        domain_concepts=concepts,
    )
    (
        recovered_document_losses,
        critic_rejected_document_claims,
        document_loss_recovery_audit,
    ) = recover_product_ion_cascade_neutral_losses(
        critic_rejected_document_claims
    )
    accepted_document_claims.extend(recovered_document_losses)
    review_document_claims.extend(recovered_document_claims)
    rejected_document_claims = (
        semantic_rejected_document_claims + critic_rejected_document_claims
    )
    document_critic_audit = (
        document_semantic_guard_audit
        + semantic_document_critic_audit
        + document_recovery_audit
        + document_loss_recovery_audit
    )
    claim_prompt_chunk_count = sum(
        len(window.get("source_chunk_ids", [])) for window in document_windows
    )
    claim_extraction_batch_count = (
        len(document_windows) + DOCUMENT_RELATION_BATCH_SIZE - 1
    ) // DOCUMENT_RELATION_BATCH_SIZE
    structured_candidate_claims = mine_structured_claims_from_chunks(
        compound_class, chunks_by_id
    )
    (
        structured_claims,
        rejected_structured_claims,
        structured_domain_scope_audit,
    ) = apply_structured_domain_scope_guard(
        structured_candidate_claims,
        chunks_by_id,
        compound_class=compound_class,
        domain_concepts=concepts,
    )
    (
        accepted_structured_claims,
        review_structured_claims,
        structured_prevalidation_audit,
    ) = accept_prevalidated_structured_claims(
        structured_claims,
    )
    structured_critic_audit = (
        structured_domain_scope_audit + structured_prevalidation_audit
    )

    first_pass_claims = merge_claims(
        accepted_document_claims + review_document_claims,
        accepted_structured_claims + review_structured_claims,
    )
    gap_queries = build_gap_queries(first_pass_claims, max_queries=max_gap_queries)
    gap_rows: list[dict[str, Any]] = []
    gap_windows: list[dict[str, Any]] = []
    gap_llm_claims: list[dict[str, Any]] = []
    accepted_gap_claims: list[dict[str, Any]] = []
    review_gap_claims: list[dict[str, Any]] = []
    rejected_gap_claims: list[dict[str, Any]] = []
    gap_critic_audit: list[dict[str, Any]] = []
    recovered_gap_claims: list[dict[str, Any]] = []
    recovered_gap_losses: list[dict[str, Any]] = []
    if gap_queries and max_gap_document_windows > 0:
        gap_rows = expand_retrieval_with_adjacent_context(
            retrieve(gap_queries),
            chunks_by_id,
        )
        write_jsonl(mining_dir / "gap_directed_retrieved_chunks.jsonl", gap_rows)
        gap_windows = build_document_windows(
            chunks_by_id,
            gap_rows,
            max_windows=max_gap_document_windows,
            max_window_chars=document_window_chars,
            adjacent_pages=1,
        )
        write_jsonl(mining_dir / "gap_relation_windows.jsonl", gap_windows)
        gap_llm_claims = extract_document_relations(
            compound_class,
            gap_windows,
            chunks_text,
            chat_completion,
            model,
            base_url,
            api_key,
            raw_calls_path,
            call_type_prefix="gap_document_relation_extraction",
            domain_concepts=concepts,
        )
        (
            guarded_gap_claims,
            semantic_rejected_gap_claims,
            gap_semantic_guard_audit,
        ) = apply_domain_semantic_guards(
            gap_llm_claims,
            compound_class=compound_class,
            domain_concepts=concepts,
        )
        (
            accepted_gap_claims,
            review_gap_claims,
            critic_rejected_gap_claims,
            semantic_gap_critic_audit,
        ) = critic_claims(
            compound_class,
            guarded_gap_claims,
            chunks_text,
            chat_completion,
            model,
            base_url,
            api_key,
            raw_calls_path,
            call_type_prefix="gap_evidence_critic",
            domain_concepts=concepts,
        )
        (
            recovered_gap_claims,
            critic_rejected_gap_claims,
            gap_recovery_audit,
        ) = recover_name_encoded_modification_claims(
            critic_rejected_gap_claims,
            domain_concepts=concepts,
        )
        (
            recovered_gap_losses,
            critic_rejected_gap_claims,
            gap_loss_recovery_audit,
        ) = recover_product_ion_cascade_neutral_losses(
            critic_rejected_gap_claims
        )
        accepted_gap_claims.extend(recovered_gap_losses)
        review_gap_claims.extend(recovered_gap_claims)
        rejected_gap_claims = (
            semantic_rejected_gap_claims + critic_rejected_gap_claims
        )
        gap_critic_audit = (
            gap_semantic_guard_audit
            + semantic_gap_critic_audit
            + gap_recovery_audit
            + gap_loss_recovery_audit
        )

    llm_claims = merge_claims(
        accepted_document_claims + review_document_claims,
        accepted_gap_claims + review_gap_claims,
    )
    claims = merge_claims(
        llm_claims,
        accepted_structured_claims + review_structured_claims,
    )
    write_jsonl(claims_dir / "evidence_claims.jsonl", claims)
    write_jsonl(
        claims_dir / "evidence_claims_critic_audit.jsonl",
        document_critic_audit + gap_critic_audit + structured_critic_audit,
    )
    write_jsonl(
        claims_dir / "rejected_evidence_claims.jsonl",
        rejected_document_claims
        + rejected_gap_claims
        + rejected_structured_claims,
    )
    write_json(
        reports_dir / "document_relation_extraction_summary.json",
        {
            "compound_class": compound_class,
            "document_window_count": len(document_windows),
            "document_retrieval_row_count": len(document_retrieval_rows),
            "document_window_source_count": len(
                {window.get("source_file") for window in document_windows}
            ),
            "document_llm_claim_count": len(document_llm_claims),
            "document_semantic_guard_rejected": len(
                semantic_rejected_document_claims
            ),
            "document_claims_accepted": len(accepted_document_claims),
            "document_claims_review": len(review_document_claims),
            "document_name_encoded_inferences": len(
                recovered_document_claims
            ),
            "document_claims_rejected": len(rejected_document_claims),
            "gap_query_count": len(gap_queries),
            "gap_retrieved_chunk_count": len(gap_rows),
            "gap_document_window_count": len(gap_windows),
            "gap_llm_claim_count": len(gap_llm_claims),
            "gap_claims_accepted": len(accepted_gap_claims),
            "gap_claims_review": len(review_gap_claims),
            "gap_name_encoded_inferences": len(recovered_gap_claims),
            "gap_claims_rejected": len(rejected_gap_claims),
        },
    )
    write_json(
        reports_dir / "structured_table_mining_summary.json",
        {
            "compound_class": compound_class,
            "structured_candidate_claim_count": len(
                structured_candidate_claims
            ),
            "structured_claim_count": len(structured_claims),
            "structured_domain_scope_rejected_count": len(
                rejected_structured_claims
            ),
            "llm_claim_count": len(llm_claims),
            "merged_claim_count": len(claims),
            "structured_compound_claim_count": len(
                [claim for claim in structured_claims if claim.get("claim_type") == "compound"]
            ),
            "structured_component_membership_claim_count": len(
                [
                    claim
                    for claim in structured_claims
                    if claim.get("claim_type") == "entity_component_membership"
                ]
            ),
        },
    )
    agent_state["claim_state"] = {
        "claim_count": len(claims),
        "llm_claim_count": len(llm_claims),
        "document_llm_claim_count": len(document_llm_claims),
        "gap_llm_claim_count": len(gap_llm_claims),
        "structured_claim_count": len(structured_claims),
        "claim_types": count_by_type(claims, "claim_type", CLAIM_TYPES),
    }
    save_agent_state(agent_state_path, agent_state)

    summary = {
        "compound_class": compound_class,
        "initial_query_count": len(initial_queries),
        "initial_chunk_count": len(initial_rows),
        "discovered_concept_count": len(concepts),
        "subclass_status": subclass_status,
        "accepted_concept_count": len(accepted_concepts),
        "review_concept_count": len(review_concepts),
        "rejected_concept_count": len(rejected_concepts),
        "expanded_query_count": len(expanded_queries),
        "deep_chunk_count": len(deep_rows),
        "claim_count": len(claims),
        "llm_claim_count": len(llm_claims),
        "document_window_count": len(document_windows),
        "document_llm_claim_count": len(document_llm_claims),
        "critic_accepted_claim_count": len(accepted_document_claims)
        + len(accepted_gap_claims),
        "critic_review_claim_count": len(review_document_claims)
        + len(review_gap_claims),
        "name_encoded_inference_count": len(recovered_document_claims)
        + len(recovered_gap_claims),
        "critic_rejected_claim_count": len(rejected_document_claims)
        + len(rejected_gap_claims),
        "gap_query_count": len(gap_queries),
        "gap_document_window_count": len(gap_windows),
        "gap_llm_claim_count": len(gap_llm_claims),
        "structured_claim_count": len(structured_claims),
        "claim_prompt_chunk_count": claim_prompt_chunk_count,
        "claim_extraction_batch_count": claim_extraction_batch_count,
        "iterations_used": iterations_used,
        "retrieval": {
            "engine": "bm25s",
            "mode": "sparse_only",
            "tokenizer": "scientific_regex_v1",
            "top_k": top_k,
            "index_path": str(bm25_path),
        },
    }
    write_json(reports_dir / "literature_mining_summary.json", summary)
    return summary


def default_query_plan_path(compound_class: str) -> Path:
    return PROJECT_ROOT / "rag" / "query_plans" / f"{safe_stem(compound_class)}_query_plan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Agent 2 Literature Mining over local corpus/index files.")
    parser.add_argument("--compound_class", required=True)
    parser.add_argument("--query_plan", default=None, help="Default: rag/query_plans/{compound_class}_query_plan.json")
    parser.add_argument("--corpus_jsonl", default="rag/corpus/chunks.jsonl")
    parser.add_argument("--index_path", default="rag/index/retrieval_index")
    parser.add_argument("--output_root", default="rag")
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--max_iterations", type=int, default=2)
    parser.add_argument("--max_prompt_chunks", type=int, default=50)
    parser.add_argument("--max_chunk_chars", type=int, default=1600)
    parser.add_argument(
        "--max_document_windows",
        type=int,
        default=24,
        help="Maximum reconstructed document windows in the first relation pass.",
    )
    parser.add_argument(
        "--document_window_chars",
        type=int,
        default=6000,
        help="Maximum characters per reconstructed document window.",
    )
    parser.add_argument(
        "--max_gap_queries",
        type=int,
        default=24,
        help="Maximum generic entity/relation gap-directed local queries.",
    )
    parser.add_argument(
        "--max_gap_document_windows",
        type=int,
        default=12,
        help="Maximum document windows in the gap-directed extraction pass.",
    )
    parser.add_argument(
        "--refresh_structured_only",
        action="store_true",
        help="Re-mine deterministic table claims and preserve existing LLM claims without API calls.",
    )
    parser.add_argument(
        "--refresh_claims_jsonl",
        default=None,
        help=(
            "Optional claims JSONL to refresh in place when using "
            "--refresh_structured_only. Defaults to "
            "{output_root}/evidence_claims/evidence_claims.jsonl."
        ),
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top_k must be > 0")
    if args.max_iterations <= 0:
        parser.error("--max_iterations must be > 0")
    if args.max_document_windows <= 0:
        parser.error("--max_document_windows must be > 0")
    if args.document_window_chars <= 0:
        parser.error("--document_window_chars must be > 0")
    if args.max_gap_queries < 0:
        parser.error("--max_gap_queries must be >= 0")
    if args.max_gap_document_windows < 0:
        parser.error("--max_gap_document_windows must be >= 0")
    return args


def main() -> int:
    args = parse_args()
    query_plan = args.query_plan or default_query_plan_path(args.compound_class)
    try:
        if args.refresh_structured_only:
            summary = refresh_structured_claims(
                compound_class=args.compound_class,
                corpus_jsonl=args.corpus_jsonl,
                output_root=args.output_root,
                claims_jsonl=args.refresh_claims_jsonl,
            )
        else:
            summary = run_literature_mining(
                query_plan_path=query_plan,
                corpus_jsonl=args.corpus_jsonl,
                index_path=args.index_path,
                output_root=args.output_root,
                top_k=args.top_k,
                max_iterations=args.max_iterations,
                max_prompt_chunks=args.max_prompt_chunks,
                max_chunk_chars=args.max_chunk_chars,
                max_document_windows=args.max_document_windows,
                document_window_chars=args.document_window_chars,
                max_gap_queries=args.max_gap_queries,
                max_gap_document_windows=args.max_gap_document_windows,
            )
    except LiteratureMiningError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
