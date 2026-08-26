#!/usr/bin/env python3
"""Plan literature retrieval queries for a user-specified metabolite class.

This script is Phase 1 of the universal metabolomics agentic RAG workflow.
It does not search local indexes, read rule CSVs, or annotate spectra. It only
asks an OpenAI-compatible chat-completions model to produce a retrieval plan.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.core.llm_transport import TransportConfig, request_chat_completion

REQUIRED_QUERY_GROUPS = [
    "compound_queries",
    "fragment_queries",
    "neutral_loss_queries",
    "transformation_queries",
    "biosynthesis_queries",
    "supplementary_table_queries",
    "review_queries",
]

EXPECTED_RULE_TYPES = [
    "compound_rules",
    "transformation_rules",
    "diagnostic_fragment_rules",
    "neutral_loss_rules",
    "biosynthetic_component_rules",
]

DOCUMENT_PRIORITY_KEYS = [
    "supplementary_table",
    "experimental_article",
    "review",
    "database",
]

SYSTEM_PROMPT = """You are a metabolomics literature retrieval planner.

Your task is not to identify metabolites.

Your task is to design retrieval queries that maximize recovery of literature evidence required for building metabolite annotation rules.

You must generate queries for:

1. compound references
2. MS/MS diagnostic fragments
3. neutral losses
4. chemical transformations
5. biosynthetic components
6. supplementary tables and catalogs

You must consider that scientific evidence is often stored in:
- supplementary tables
- compound identification tables
- LC-MS/MS tables
- fragmentation tables
- metabolomics catalogs

Do not invent exact masses unless they are strongly associated with the compound class.
Prefer semantic search terms.
"""


class QueryPlanError(RuntimeError):
    """Raised when the query planner cannot produce a valid plan."""


def parse_multi_value(values: list[str] | None, default: list[str]) -> list[str]:
    if not values:
        return list(default)
    parsed: list[str] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                parsed.append(token)
    return parsed or list(default)


def safe_plan_stem(compound_class: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", compound_class.strip().lower()).strip("_")
    return stem or "compound_class"


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_user_prompt(
    compound_class: str,
    purposes: list[str],
    ion_modes: list[str],
    seed_compounds: list[str],
) -> str:
    request = {
        "compound_class": compound_class,
        "purpose": purposes,
        "ion_mode": ion_modes,
        "seed_compounds": seed_compounds,
    }
    output_shape = {
        "compound_class": "",
        "initial_domain_terms": [],
        "domain_keywords": [],
        "retrieval_strategy": {group_name: [] for group_name in REQUIRED_QUERY_GROUPS},
        "expected_rule_types": EXPECTED_RULE_TYPES,
        "document_priority": {key: 0 for key in DOCUMENT_PRIORITY_KEYS},
        "confidence_notes": "",
    }
    constraints = [
        "Return JSON only, using the output shape below as the top-level object.",
        "Do not repeat the labels INPUT or OUTPUT JSON SHAPE in the response.",
        "Do not identify metabolites.",
        "Do not produce rule tables.",
        "Do not invent specific metabolite rules.",
        "Do not generate alternate-name expansion or specific compound names; those must come from later retrieved literature evidence.",
        "initial_domain_terms must contain only broad class-level search terms, assay terms, and literature-table terms.",
        "Do not rely on hardcoded class-specific knowledge outside the requested planning task.",
        "Every retrieval_strategy group must be present and non-empty.",
        "Queries should be suitable for BM25S retrieval over local scientific-literature chunks.",
    ]
    return (
        "INPUT:\n"
        + json.dumps(request, ensure_ascii=False, indent=2)
        + "\n\nOUTPUT JSON SHAPE (return this object directly):\n"
        + json.dumps(output_shape, ensure_ascii=False, indent=2)
        + "\n\nCONSTRAINTS:\n- "
        + "\n- ".join(constraints)
    )


def build_messages(
    compound_class: str,
    purposes: list[str],
    ion_modes: list[str],
    seed_compounds: list[str],
    retry_note: str | None = None,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                compound_class=compound_class,
                purposes=purposes,
                ion_modes=ion_modes,
                seed_compounds=seed_compounds,
            ),
        },
    ]
    if retry_note:
        messages.append({"role": "user", "content": retry_note})
    return messages


def chat_completion_openai_compatible(
    messages: list[dict[str, str]],
    model: str,
    base_url: str,
    api_key: str,
) -> str:
    return request_chat_completion(
        messages,
        model=model,
        base_url=base_url,
        api_key=api_key,
        config=TransportConfig(
            timeout_seconds=120.0,
            retries=1,
            return_metadata=False,
        ),
        error_type=QueryPlanError,
        missing_context=" for Query Planner LLM calls.",
        response_error_message=(
            "OpenAI-compatible API response did not contain "
            "choices[0].message.content"
        ),
    )


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("LLM output JSON must be an object.")
    return payload


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


QUERY_GROUP_ALIASES = {
    "compound": "compound_queries",
    "compounds": "compound_queries",
    "compound_references": "compound_queries",
    "fragment": "fragment_queries",
    "fragments": "fragment_queries",
    "diagnostic_fragments": "fragment_queries",
    "neutral_loss": "neutral_loss_queries",
    "neutral_losses": "neutral_loss_queries",
    "transformation": "transformation_queries",
    "transformations": "transformation_queries",
    "biosynthesis": "biosynthesis_queries",
    "biosynthetic": "biosynthesis_queries",
    "supplementary_table": "supplementary_table_queries",
    "supplementary_tables": "supplementary_table_queries",
    "supplementary": "supplementary_table_queries",
    "review": "review_queries",
    "reviews": "review_queries",
}

GENERIC_INITIAL_DOMAIN_TERMS = (
    "LC-MS/MS",
    "mass spectrometry",
    "tandem mass spectrometry",
    "compound identification",
    "fragmentation",
    "neutral loss",
    "chemical transformation",
    "biosynthesis",
    "supplementary table",
)

GENERIC_QUERY_SUFFIXES = {
    "compound_queries": (
        "LC-MS/MS compound identification",
        "mass spectrometry reference compounds",
    ),
    "fragment_queries": (
        "MS/MS fragmentation product ions",
        "diagnostic fragment tandem mass spectrometry",
    ),
    "neutral_loss_queries": (
        "neutral loss tandem mass spectrometry",
        "fragmentation neutral losses",
    ),
    "transformation_queries": (
        "chemical transformation mass shift",
        "derivatives degradation products LC-MS",
    ),
    "biosynthesis_queries": (
        "biosynthesis precursors structural components",
        "biosynthetic pathway literature",
    ),
    "supplementary_table_queries": (
        "supplementary table LC-MS compound identification",
        "supplementary compound catalog mass spectrometry",
    ),
    "review_queries": (
        "review mass spectrometry annotation",
        "review fragmentation biosynthesis",
    ),
}


def normalize_query_group_name(value: Any) -> str:
    text = re.sub(r"[\s-]+", "_", str(value or "").strip().lower())
    return QUERY_GROUP_ALIASES.get(text, text)


def _query_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        for key in ("queries", "query", "search_terms", "terms", "items"):
            if key in value:
                return normalize_string_list(value.get(key))
        return []
    return normalize_string_list(value)


def normalize_retrieval_strategy(value: Any) -> dict[str, list[str]]:
    """Normalize common LLM retrieval_strategy shapes into the required schema."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if isinstance(value, dict):
        return {
            normalize_query_group_name(key): _query_values(item)
            for key, item in value.items()
            if normalize_query_group_name(key)
        }

    normalized: dict[str, list[str]] = {}
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                continue
            if not isinstance(item, dict):
                continue
            raw_group = (
                item.get("query_type")
                or item.get("query_group")
                or item.get("group")
                or item.get("category")
                or item.get("name")
                or item.get("type")
            )
            if not raw_group:
                continue
            group_name = normalize_query_group_name(raw_group)
            queries = (
                item.get("queries")
                or item.get("query")
                or item.get("search_terms")
                or item.get("terms")
            )
            normalized.setdefault(group_name, []).extend(normalize_string_list(queries))
    return normalized


def unwrap_query_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Unwrap common provider envelopes without inventing missing query content."""

    current = plan
    for _ in range(3):
        nested: Any = None
        for key in (
            "query_plan",
            "retrieval_plan",
            "plan",
            "result",
            "output",
            "required_output_schema",
        ):
            if key in current:
                nested = current.get(key)
                break
        if isinstance(nested, str):
            try:
                nested = json.loads(nested)
            except json.JSONDecodeError:
                nested = None
        if not isinstance(nested, dict):
            break
        current = nested

    if "retrieval_strategy" not in current:
        grouped = {
            group_name: current[group_name]
            for group_name in REQUIRED_QUERY_GROUPS
            if group_name in current
        }
        if grouped:
            current = {**current, "retrieval_strategy": grouped}
        else:
            for key in ("queries", "search_queries", "query_sections"):
                if key in current:
                    current = {**current, "retrieval_strategy": current[key]}
                    break
    return current


def generic_retrieval_content(compound_class: str) -> tuple[list[str], list[str], dict[str, list[str]]]:
    class_name = str(compound_class).strip()
    initial_terms = [class_name, *GENERIC_INITIAL_DOMAIN_TERMS]
    domain_keywords = list(GENERIC_INITIAL_DOMAIN_TERMS)
    strategy = {
        group_name: [f"{class_name} {suffix}" for suffix in GENERIC_QUERY_SUFFIXES[group_name]]
        for group_name in REQUIRED_QUERY_GROUPS
    }
    return initial_terms, domain_keywords, strategy


def validate_and_normalize_plan(plan: dict[str, Any], compound_class: str) -> dict[str, Any]:
    plan = unwrap_query_plan(plan)
    normalized: dict[str, Any] = {}
    normalized["compound_class"] = str(plan.get("compound_class") or compound_class).strip()
    normalized["initial_domain_terms"] = normalize_string_list(plan.get("initial_domain_terms"))
    normalized["domain_keywords"] = normalize_string_list(plan.get("domain_keywords"))

    retrieval_strategy = normalize_retrieval_strategy(plan.get("retrieval_strategy"))
    if not retrieval_strategy:
        raise ValueError("retrieval_strategy must be an object or a list of query sections.")
    normalized_strategy: dict[str, list[str]] = {}
    missing_or_empty: list[str] = []
    for group_name in REQUIRED_QUERY_GROUPS:
        queries = normalize_string_list(retrieval_strategy.get(group_name))
        if not queries:
            missing_or_empty.append(group_name)
        normalized_strategy[group_name] = queries
    if missing_or_empty:
        raise ValueError(f"retrieval_strategy groups are missing or empty: {', '.join(missing_or_empty)}")
    normalized["retrieval_strategy"] = normalized_strategy

    # LLM output is used for planning metadata, but executable retrieval text is
    # constrained to the user-provided class plus class-independent vocabulary.
    # This prevents hidden synonym, subclass, compound, mass, or fragment priors.
    generic_terms, generic_keywords, generic_strategy = generic_retrieval_content(
        normalized["compound_class"]
    )
    normalized["initial_domain_terms"] = generic_terms
    normalized["domain_keywords"] = generic_keywords
    normalized["retrieval_strategy"] = generic_strategy

    rule_types = normalize_string_list(plan.get("expected_rule_types"))
    normalized["expected_rule_types"] = rule_types or list(EXPECTED_RULE_TYPES)
    missing_rule_types = [rule_type for rule_type in EXPECTED_RULE_TYPES if rule_type not in normalized["expected_rule_types"]]
    if missing_rule_types:
        normalized["expected_rule_types"].extend(missing_rule_types)

    document_priority = plan.get("document_priority")
    if not isinstance(document_priority, dict):
        document_priority = {}
    normalized_priority: dict[str, int] = {}
    for key in DOCUMENT_PRIORITY_KEYS:
        value = document_priority.get(key, 0)
        try:
            normalized_priority[key] = int(value)
        except (TypeError, ValueError):
            normalized_priority[key] = 0
    normalized["document_priority"] = normalized_priority
    normalized["confidence_notes"] = str(plan.get("confidence_notes") or "").strip()
    return normalized


def _append_attempt_log(
    path: Path | str | None,
    attempt: int,
    raw_output: str,
    validation_error: str,
    status: str,
) -> None:
    if path is None:
        return
    log_path = Path(path)
    if not log_path.is_absolute():
        log_path = PROJECT_ROOT / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": int(attempt),
        "status": status,
        "validation_error": validation_error,
        "raw_output": raw_output,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_query_plan(
    compound_class: str,
    purposes: list[str],
    ion_modes: list[str],
    seed_compounds: list[str],
    chat_completion: Callable[[list[dict[str, str]], str, str, str], str] = chat_completion_openai_compatible,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    attempt_log_path: Path | str | None = None,
) -> dict[str, Any]:
    compound_class = compound_class.strip()
    if not compound_class:
        raise QueryPlanError("compound_class must not be empty.")

    model = model if model is not None else os.environ.get("OPENAI_MODEL", "")
    base_url = base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL", "")
    api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")

    last_error: Exception | None = None
    retry_note: str | None = None
    for attempt_index in range(3):
        messages = build_messages(
            compound_class=compound_class,
            purposes=purposes,
            ion_modes=ion_modes,
            seed_compounds=seed_compounds,
            retry_note=retry_note,
        )
        raw_output = chat_completion(messages, model, base_url, api_key)
        try:
            parsed = parse_json_object(raw_output)
            normalized = validate_and_normalize_plan(parsed, compound_class)
            _append_attempt_log(attempt_log_path, attempt_index + 1, raw_output, "", "valid")
            return normalized
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            _append_attempt_log(
                attempt_log_path,
                attempt_index + 1,
                raw_output,
                str(exc),
                "invalid",
            )
            retry_note = (
                f"Your previous response failed validation: {exc}. "
                "Return exactly one JSON object with a retrieval_strategy object containing these non-empty keys: "
                + ", ".join(REQUIRED_QUERY_GROUPS)
                + ". Do not wrap the object in another key."
            )
            if attempt_index >= 2:
                break
    raise QueryPlanError(f"Failed to obtain valid query plan JSON after 3 attempts: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a literature retrieval query plan for a metabolite class.")
    parser.add_argument("--compound_class", required=True, help="Target metabolite or compound class.")
    parser.add_argument(
        "--purpose",
        action="append",
        default=[],
        help=(
            "Planning purpose. May be repeated or comma-separated. "
            "Default: compound_annotation,network_expansion,diagnostic_rule_extraction"
        ),
    )
    parser.add_argument(
        "--ion_mode",
        action="append",
        default=[],
        help="Ion mode. May be repeated or comma-separated. Default: positive,negative",
    )
    parser.add_argument(
        "--seed_compounds",
        action="append",
        default=[],
        help="Optional seed compound name. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Output JSON path. Default: rag/query_plans/{compound_class}_query_plan.json",
    )
    parser.add_argument(
        "--attempt_log",
        default="rag/raw_outputs/query_planner_attempts.jsonl",
        help="Append raw LLM responses and validation errors for audit/debugging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    purposes = parse_multi_value(
        args.purpose,
        ["compound_annotation", "network_expansion", "diagnostic_rule_extraction"],
    )
    ion_modes = parse_multi_value(args.ion_mode, ["positive", "negative"])
    seed_compounds = parse_multi_value(args.seed_compounds, [])
    output_path = (
        resolve(args.output_json)
        if args.output_json
        else PROJECT_ROOT / "rag" / "query_plans" / f"{safe_plan_stem(args.compound_class)}_query_plan.json"
    )

    try:
        plan = build_query_plan(
            compound_class=args.compound_class,
            purposes=purposes,
            ion_modes=ion_modes,
            seed_compounds=seed_compounds,
            attempt_log_path=args.attempt_log,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except QueryPlanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote query plan: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
