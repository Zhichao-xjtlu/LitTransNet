"""Document-level evidence units and critic helpers for Agent 2.

The functions in this module are domain agnostic.  They operate on corpus
provenance and generic relation types; no metabolite-class vocabulary is
embedded here.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence


RELATION_GAPS: tuple[tuple[str, str, str], ...] = (
    ("transformation", "transformation", "reaction conversion formation product precursor"),
    ("fragment", "fragment", "MS/MS product ion fragment diagnostic"),
    ("neutral_loss", "neutral_loss", "neutral loss fragmentation"),
    (
        "component_membership",
        "entity_component_membership",
        "composition substituent residue component moiety",
    ),
)

RELATION_QUERY_GROUPS = frozenset(
    {
        "transformation_queries",
        "biosynthesis_queries",
        "supplementary_table_queries",
    }
)

DOMAIN_CONCEPT_TYPES = frozenset(
    {
        "compound",
        "synonym",
        "subclass",
        "precursor",
        "structural_component",
        "biosynthetic_component",
    }
)

# Generic linguistic evidence that a passage states a relationship instead of
# merely cataloguing entities or masses. These patterns deliberately contain
# no metabolite-class or compound vocabulary.
RELATION_SIGNAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bcondens(?:e[sd]?|ation)\b",
        r"\bform(?:s|ed)?\s+by\b",
        r"\b(?:reacts?|reacted)\s+with\b",
        r"\bconvert(?:s|ed)?\s+(?:to|into)\b",
        r"\bderived\s+from\b",
        r"\b(?:union|combination|conjugation|coupling)\s+(?:of|with)\b",
        r"\blink(?:s|ed)?\s+to\b",
        r"\b(?:biosynthesi[sz]ed|generated|produced)\s+from\b",
    )
)


def _clean(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _traceability_text(value: object) -> str:
    return _clean(value).casefold().replace("−", "-").replace("–", "-").replace("—", "-")


def select_relation_retrieval_rows(
    retrieved_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select direct hits from relation-bearing query groups.

    Catalog and fragment queries remain useful to deterministic parsers, but
    they should not consume the limited document-level relation-extraction
    budget when transformation, biosynthesis, or supplementary relation hits
    are available.  If a planner produced no such hits, direct hits from all
    query groups are retained as a generic fallback.
    """

    direct = [
        dict(row)
        for row in retrieved_rows
        if str(row.get("retrieval_origin") or "direct_hit") == "direct_hit"
    ]
    relation_rows = [
        row
        for row in direct
        if str(row.get("query_group") or "") in RELATION_QUERY_GROUPS
    ]
    return relation_rows or direct


def _semantic_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _clean(value).casefold()).strip()


def _relation_signal_score(value: object) -> int:
    """Count distinct generic relation predicates present in a passage."""

    text = _clean(value)
    return sum(1 for pattern in RELATION_SIGNAL_PATTERNS if pattern.search(text))


def _domain_terms(
    compound_class: str,
    domain_concepts: Sequence[Mapping[str, Any]],
) -> list[str]:
    values = [compound_class]
    values.extend(
        str(concept.get("name") or "")
        for concept in domain_concepts
        if str(concept.get("type") or "") in DOMAIN_CONCEPT_TYPES
    )
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        term = _semantic_text(value)
        if len(term) >= 3 and term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def _claim_domain_haystack(claim: Mapping[str, Any]) -> str:
    fields = (
        "compound_name",
        "precursor_name",
        "component_name",
        "entity_name",
        "entity_id",
        "entity_class_id",
        "source_entity",
        "target_entity",
        "reaction_name",
        "evidence_quote",
        "evidence_summary",
    )
    return _semantic_text(" ".join(str(claim.get(field) or "") for field in fields))


def _looks_like_msms_ion_endpoint(value: object) -> bool:
    text = str(value or "")
    return bool(
        re.search(r"\[\s*M[^\]]*\][+-]?", text, re.IGNORECASE)
        or re.search(r"\bm\s*/\s*z\b", text, re.IGNORECASE)
        or re.fullmatch(r"\s*\d+(?:\.\d+)?\s*(?:Da)?\s*", text)
    )


def apply_domain_semantic_guards(
    claims: Sequence[Mapping[str, Any]],
    *,
    compound_class: str,
    domain_concepts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Reject structurally invalid or out-of-domain document claims.

    The guard is evidence-derived and class agnostic.  It uses only the user
    supplied class name and concepts discovered from retrieved literature; it
    contains no metabolite-specific entity, fragment, or reaction whitelist.
    """

    terms = _domain_terms(compound_class, domain_concepts)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for source_claim in claims:
        claim = dict(source_claim)
        claim_type = str(claim.get("claim_type") or "")
        reason = ""
        if claim_type == "transformation":
            source = claim.get("source_entity")
            target = claim.get("target_entity")
            if not _clean(source) or not _clean(target):
                reason = "missing_transformation_endpoint"
            elif _looks_like_msms_ion_endpoint(source) or _looks_like_msms_ion_endpoint(target):
                reason = "msms_fragmentation_not_entity_transformation"
        elif claim_type == "entity_class_membership":
            if not _clean(claim.get("entity_id")) and not _clean(claim.get("entity_name")):
                reason = "missing_membership_entity"
        elif claim_type == "entity_component_membership":
            if not _clean(claim.get("entity_id")) and not _clean(claim.get("entity_name")):
                reason = "missing_membership_entity"
            elif not _clean(claim.get("component_name")):
                reason = "missing_component_entity"

        matched_terms: list[str] = []
        if not reason:
            haystack = _claim_domain_haystack(claim)
            matched_terms = [term for term in terms if term in haystack]
            if not matched_terms:
                reason = "domain_relevance_not_established"

        if reason:
            existing_reasons = [
                str(value)
                for value in claim.get("critic_reason_codes") or []
                if str(value)
            ]
            claim.update(
                {
                    "critic_status": "reject",
                    "critic_support_type": "unsupported",
                    "critic_reason_codes": sorted(set(existing_reasons + [reason])),
                    "review_status": "needs_review",
                    "domain_relevance_pass": False,
                }
            )
            rejected.append(claim)
            verdict = "reject"
        else:
            claim["domain_relevance_pass"] = True
            claim["domain_relevance_terms"] = matched_terms
            kept.append(claim)
            verdict = "pass"
        audit.append(
            {
                "claim_id": claim.get("claim_id", ""),
                "claim_type": claim_type,
                "semantic_guard": "domain_and_entity_semantics",
                "verdict": verdict,
                "reason_code": reason,
                "domain_relevance_terms": matched_terms,
            }
        )
    return kept, rejected, audit


def recover_name_encoded_modification_claims(
    rejected_claims: Sequence[Mapping[str, Any]],
    *,
    domain_concepts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Recover traceable name-encoded relations as literature inference.

    Scientific catalogs often report a base entity and modifier-bearing target
    names without a prose sentence that says "A transforms to B".  This
    validator may recover such a relation only when the source is an
    evidence-discovered entity, the target literally contains that source plus
    a modifier consistent with the proposed transformation, and the target
    tokens occur in a traceable quote.  It never promotes the result to an
    explicit report.
    """

    source_terms = {
        _semantic_text(concept.get("name"))
        for concept in domain_concepts
        if str(concept.get("type") or "") in {"compound", "synonym"}
        and _semantic_text(concept.get("name"))
    }
    recovered: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for source_claim in rejected_claims:
        claim = dict(source_claim)
        source = _semantic_text(claim.get("source_entity"))
        target = _semantic_text(claim.get("target_entity"))
        transformation = _semantic_text(claim.get("transformation_name"))
        quote = _semantic_text(claim.get("evidence_quote"))
        quote_tokens = set(quote.split())
        target_tokens = [token for token in target.split() if len(token) >= 2]
        modifier_text = target.replace(source, " ", 1).strip() if source else ""
        modifier_similarity = (
            SequenceMatcher(None, transformation, modifier_text)
            .find_longest_match(
                0,
                len(transformation),
                0,
                len(modifier_text),
            )
            .size
            if transformation and modifier_text
            else 0
        )
        source_evidence_basis = (
            "discovered_concept"
            if source in source_terms
            else "target_name_substring"
        )
        recoverable = bool(
            claim.get("claim_type") == "transformation"
            and claim.get("critic_traceability_pass") is True
            and source in target
            and source != target
            and target_tokens
            and all(token in quote_tokens for token in target_tokens)
            and modifier_similarity >= 6
            and not _looks_like_msms_ion_endpoint(claim.get("source_entity"))
            and not _looks_like_msms_ion_endpoint(claim.get("target_entity"))
        )
        if not recoverable:
            remaining.append(claim)
            continue
        reasons = [
            str(value)
            for value in claim.get("critic_reason_codes") or []
            if str(value)
        ]
        claim.update(
            {
                "critic_status": "review",
                "critic_support_type": "literature_inferred",
                "critic_reason_codes": sorted(
                    set(reasons + ["name_encoded_modification_relation"])
                ),
                "review_status": "needs_review",
                "evidence_type": "literature_inferred",
                "name_encoded_relation_pass": True,
            }
        )
        recovered.append(claim)
        audit.append(
            {
                "claim_id": claim.get("claim_id", ""),
                "claim_type": "transformation",
                "semantic_guard": "name_encoded_modification_relation",
                "verdict": "recover",
                "reason_code": "name_encoded_modification_relation",
                "modifier_similarity": modifier_similarity,
                "source_evidence_basis": source_evidence_basis,
            }
        )
    return recovered, remaining, audit


_TABLE_LABEL_PATTERN = re.compile(
    r"\b(?:continue\s+)?table\s+([A-Za-z]*\d+[A-Za-z]*)\s*(?:\.|:)",
    re.IGNORECASE,
)


def apply_structured_domain_scope_guard(
    claims: Sequence[Mapping[str, Any]],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    *,
    compound_class: str,
    domain_concepts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep structured rows only inside evidence-supported domain tables.

    A class-focused paper may contain unrelated formula tables.  Formula and
    mass validity alone therefore cannot assign every parsed row to the target
    class.  This guard identifies explicit ``Table N`` scopes, carries their
    labels across adjacent continuation pages, and requires the table, page,
    or entity name to contain the requested class or an evidence-discovered
    domain term.  It uses no metabolite-class whitelist.
    """

    scope_terms = {_semantic_text(compound_class)}
    scope_terms.update(
        _semantic_text(concept.get("name"))
        for concept in domain_concepts
        if str(concept.get("type") or "") == "subclass"
    )
    scope_terms = {term for term in scope_terms if len(term) >= 4}
    entity_terms = set(scope_terms)
    entity_terms.update(
        _semantic_text(concept.get("name"))
        for concept in domain_concepts
        if str(concept.get("type") or "") in {"compound", "synonym"}
    )
    entity_terms = {term for term in entity_terms if len(term) >= 4}

    page_parts: dict[tuple[str, int], list[tuple[int, str]]] = {}
    for chunk in chunks_by_id.values():
        source = str(chunk.get("source_file") or "")
        page = chunk.get("page")
        if not source or not isinstance(page, int):
            continue
        page_parts.setdefault((source, page), []).append(
            (int(chunk.get("char_start") or 0), str(chunk.get("text") or ""))
        )
    page_text = {
        key: "\n".join(text for _, text in sorted(parts))
        for key, parts in page_parts.items()
    }

    claim_sources = {
        str(claim.get("source_file") or "")
        for claim in claims
        if str(claim.get("source_file") or "")
    }
    source_pages: dict[str, set[int]] = {}
    for source, page in page_text:
        if source in claim_sources:
            source_pages.setdefault(source, set()).add(page)

    page_table: dict[tuple[str, int], str] = {}
    for source, pages in source_pages.items():
        previous_page: int | None = None
        previous_label = ""
        for page in sorted(pages):
            match = _TABLE_LABEL_PATTERN.search(page_text.get((source, page), ""))
            if match:
                label = match.group(1).casefold()
            elif previous_page is not None and page == previous_page + 1:
                label = previous_label
            else:
                label = ""
            page_table[(source, page)] = label
            previous_page = page
            previous_label = label

    def has_term(value: object, terms: set[str]) -> bool:
        normalized = _semantic_text(value)
        return bool(normalized) and any(term in normalized for term in terms)

    relevant_tables: set[tuple[str, str]] = set()
    for (source, _page), text in page_text.items():
        matches = list(_TABLE_LABEL_PATTERN.finditer(text))
        for index, match in enumerate(matches):
            section_end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else min(len(text), match.start() + 1000)
            )
            caption_or_header = text[match.start() : section_end]
            if has_term(caption_or_header, scope_terms):
                relevant_tables.add((source, match.group(1).casefold()))
        unique_labels = {match.group(1).casefold() for match in matches}
        if (
            len(unique_labels) == 1
            and has_term(text, scope_terms)
        ):
            relevant_tables.add((source, next(iter(unique_labels))))

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for source_claim in claims:
        claim = dict(source_claim)
        source = str(claim.get("source_file") or "")
        page = claim.get("page")
        table_label = (
            page_table.get((source, page), "") if isinstance(page, int) else ""
        )
        entity_text = " ".join(
            str(claim.get(field) or "")
            for field in (
                "compound_name",
                "entity_name",
                "component_name",
                "evidence_quote",
            )
        )
        page_relevant = bool(
            not table_label
            and isinstance(page, int)
            and has_term(page_text.get((source, page), ""), scope_terms)
        )
        table_relevant = bool(
            table_label and (source, table_label) in relevant_tables
        )
        entity_relevant = has_term(entity_text, entity_terms)
        relevant = page_relevant or table_relevant or entity_relevant
        if relevant:
            claim["domain_relevance_pass"] = True
            claim["structured_table_scope"] = table_label
            kept.append(claim)
            verdict = "pass"
            reason = ""
        else:
            reasons = [
                str(value)
                for value in claim.get("critic_reason_codes") or []
                if str(value)
            ]
            reason = "structured_domain_scope_not_established"
            claim.update(
                {
                    "critic_status": "reject",
                    "critic_support_type": "unsupported",
                    "critic_reason_codes": sorted(set(reasons + [reason])),
                    "critic_missing_fields": [
                        "target-domain table, page, or entity context"
                    ],
                    "critic_traceability_pass": True,
                    "review_status": "needs_review",
                    "domain_relevance_pass": False,
                    "structured_table_scope": table_label,
                }
            )
            rejected.append(claim)
            verdict = "reject"
        audit.append(
            {
                "claim_id": claim.get("claim_id", ""),
                "claim_type": claim.get("claim_type", ""),
                "semantic_guard": "structured_domain_scope",
                "verdict": verdict,
                "reason_code": reason,
                "source_file": source,
                "page": page,
                "table_scope": table_label,
                "page_relevant": page_relevant,
                "table_relevant": table_relevant,
                "entity_relevant": entity_relevant,
            }
        )
    return kept, rejected, audit


def _quote_is_traceable(
    quote: object,
    chunk_ids: Sequence[str],
    chunks_by_id: Mapping[str, object],
) -> bool:
    needle = _traceability_text(quote)
    if not needle:
        return False
    for chunk_id in chunk_ids:
        raw = chunks_by_id.get(chunk_id, "")
        if isinstance(raw, Mapping):
            raw = raw.get("text", "")
        if needle in _traceability_text(raw):
            return True
    combined_parts: list[str] = []
    for chunk_id in chunk_ids:
        raw = chunks_by_id.get(chunk_id, "")
        if isinstance(raw, Mapping):
            raw = raw.get("text", "")
        combined_parts.append(str(raw or ""))
    return needle in _traceability_text(" ".join(combined_parts))


def _reconstruct_page(
    members: Sequence[tuple[str, Mapping[str, Any]]],
) -> tuple[str, list[str], list[dict[str, Any]]]:
    ordered = sorted(
        members,
        key=lambda item: (int(item[1].get("char_start") or 0), item[0]),
    )
    if not ordered:
        return "", [], []
    base = int(ordered[0][1].get("char_start") or 0)
    text = ""
    ids: list[str] = []
    chunk_spans: list[dict[str, Any]] = []
    for chunk_id, chunk in ordered:
        value = str(chunk.get("text") or "")
        offset = max(0, int(chunk.get("char_start") or 0) - base)
        if offset > len(text):
            text += "\n" * (offset - len(text)) + value
        else:
            text += value[max(0, len(text) - offset) :]
        ids.append(chunk_id)
        chunk_spans.append(
            {
                "chunk_id": chunk_id,
                "start": offset,
                "end": offset + len(value),
            }
        )
    return text, ids, chunk_spans


def _document_pages(
    chunks_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    pdf_pages: dict[tuple[str, int], list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    standalone: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk_id, chunk in chunks_by_id.items():
        source = str(chunk.get("source_file") or "")
        page = chunk.get("page")
        if str(chunk.get("file_type") or "").casefold() == "pdf" and isinstance(page, int):
            pdf_pages[(source, page)].append((chunk_id, chunk))
        else:
            standalone[source].append(
                {
                    "position": int(chunk.get("row_index") or chunk.get("char_start") or 0),
                    "page": page,
                    "text": str(chunk.get("text") or ""),
                    "chunk_ids": [chunk_id],
                    "chunk_spans": [
                        {
                            "chunk_id": chunk_id,
                            "start": 0,
                            "end": len(str(chunk.get("text") or "")),
                        }
                    ],
                }
            )
    documents: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (source, page), members in pdf_pages.items():
        text, chunk_ids, chunk_spans = _reconstruct_page(members)
        documents[source].append(
            {
                "position": page,
                "page": page,
                "text": text,
                "chunk_ids": chunk_ids,
                "chunk_spans": chunk_spans,
            }
        )
    for source, rows in standalone.items():
        documents[source].extend(rows)
    for source in documents:
        documents[source].sort(key=lambda row: (int(row["position"]), row["chunk_ids"][0]))
    return dict(documents)


def build_document_windows(
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    retrieved_rows: Sequence[Mapping[str, Any]],
    *,
    max_windows: int,
    max_window_chars: int,
    adjacent_pages: int = 1,
) -> list[dict[str, Any]]:
    """Build ranked, contiguous document windows around retrieved evidence.

    A window contains the complete reconstructed retrieved page/row plus
    adjacent units from the same source.  It therefore preserves relationships
    cut by overlapping corpus chunks while retaining the original chunk IDs.
    """

    if max_windows <= 0 or max_window_chars <= 0:
        return []
    documents = _document_pages(chunks_by_id)
    hits_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in retrieved_rows:
        source = str(row.get("source_file") or "")
        if source in documents:
            hits_by_source[source].append(row)

    candidates: list[dict[str, Any]] = []
    for source, hits in hits_by_source.items():
        units = documents[source]
        chunk_to_position = {
            chunk_id: index
            for index, unit in enumerate(units)
            for chunk_id in unit["chunk_ids"]
        }
        selected_positions = {
            chunk_to_position[str(hit.get("chunk_id") or "")]
            for hit in hits
            if str(hit.get("chunk_id") or "") in chunk_to_position
        }
        expanded_positions: set[int] = set()
        for position in selected_positions:
            expanded_positions.update(
                range(
                    max(0, position - adjacent_pages),
                    min(len(units), position + adjacent_pages + 1),
                )
            )
        if not expanded_positions:
            continue
        ordered_positions = sorted(expanded_positions)
        groups: list[list[int]] = []
        for position in ordered_positions:
            if not groups or position != groups[-1][-1] + 1:
                groups.append([position])
            else:
                groups[-1].append(position)
        for group in groups:
            current: list[int] = []
            current_chars = 0
            for position in group:
                unit_chars = len(units[position]["text"])
                if current and current_chars + unit_chars > max_window_chars:
                    candidates.extend(
                        _bounded_window_rows(
                            _window_row(source, units, current, hits),
                            max_window_chars,
                        )
                    )
                    current = []
                    current_chars = 0
                current.append(position)
                current_chars += unit_chars
            if current:
                candidates.extend(
                    _bounded_window_rows(
                        _window_row(source, units, current, hits),
                        max_window_chars,
                    )
                )

    ranked = sorted(
        candidates,
        key=lambda row: (-float(row["retrieval_score"]), row["source_file"], row["window_id"]),
    )
    selected: list[dict[str, Any]] = []
    seen_sources: set[str] = set()

    # Reserve a bounded fraction of the LLM budget for passages that state an
    # explicit relationship. Pure BM25 ranking otherwise lets large catalogs
    # and supplementary mass tables displace lower-scoring mechanism prose.
    # Source diversity is retained so one verbose paper cannot consume the
    # entire reserved budget.
    relation_budget = min(max_windows, max(1, max_windows // 3))
    relation_ranked = sorted(
        (row for row in candidates if int(row.get("relation_signal_score") or 0) > 0),
        key=lambda row: (
            -int(row.get("relation_signal_score") or 0),
            -float(row["retrieval_score"]),
            row["source_file"],
            row["window_id"],
        ),
    )
    for row in relation_ranked:
        source = str(row["source_file"])
        if source in seen_sources:
            continue
        selected.append(row)
        seen_sources.add(source)
        if len(selected) >= relation_budget:
            break

    selected_ids = {row["window_id"] for row in selected}
    for row in ranked:
        source = str(row["source_file"])
        if source in seen_sources or row["window_id"] in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(row["window_id"])
        seen_sources.add(source)
        if len(selected) >= max_windows:
            return selected
    for row in ranked:
        if row["window_id"] in selected_ids:
            continue
        selected.append(row)
        if len(selected) >= max_windows:
            break
    return sorted(
        selected,
        key=lambda row: (-float(row["retrieval_score"]), row["source_file"], row["window_id"]),
    )


def _bounded_window_rows(
    row: Mapping[str, Any],
    max_window_chars: int,
) -> list[dict[str, Any]]:
    """Split an oversized reconstructed unit into bounded overlapping windows.

    A PDF page can be much longer than ``max_window_chars``.  The previous
    grouping logic only limited the sum of multiple pages, so one large table
    page silently bypassed the configured API bound.  Small overlap preserves
    relations that cross a character boundary while every emitted request unit
    remains within the hard limit.
    """

    base = dict(row)
    text = str(base.get("text") or "")
    if len(text) <= max_window_chars:
        base.pop("_chunk_spans", None)
        return [base]
    overlap = min(240, max(1, max_window_chars // 5))
    stride = max(1, max_window_chars - overlap)
    parent_window_id = str(base.get("window_id") or "")
    parts: list[dict[str, Any]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_window_chars)
        part = dict(base)
        part["parent_window_id"] = parent_window_id
        part["window_char_start"] = start
        part["window_char_end"] = end
        part["text"] = text[start:end]
        part["relation_signal_score"] = _relation_signal_score(part["text"])
        spans = [
            span
            for span in base.get("_chunk_spans", [])
            if int(span.get("start") or 0) < end
            and int(span.get("end") or 0) > start
        ]
        if spans:
            overlapping_ids = {
                str(span.get("chunk_id") or "") for span in spans
            }
            part["source_chunk_ids"] = [
                chunk_id
                for chunk_id in base.get("source_chunk_ids", [])
                if chunk_id in overlapping_ids
            ]
            part["retrieved_chunk_ids"] = [
                chunk_id
                for chunk_id in base.get("retrieved_chunk_ids", [])
                if chunk_id in overlapping_ids
            ]
            if not part["retrieved_chunk_ids"]:
                part["retrieval_score"] = 0.0
                part["query_groups"] = []
                part["queries"] = []
        part.pop("_chunk_spans", None)
        identity = f"{parent_window_id}|{start}|{end}"
        part["window_id"] = (
            "docwin_" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
        )
        parts.append(part)
        if end >= len(text):
            break
        start += stride
    return parts


def _window_row(
    source: str,
    units: Sequence[Mapping[str, Any]],
    positions: Sequence[int],
    hits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    chunk_ids = [
        chunk_id
        for position in positions
        for chunk_id in units[position]["chunk_ids"]
    ]
    hit_ids = {str(hit.get("chunk_id") or "") for hit in hits}
    relevant_hits = [hit for hit in hits if str(hit.get("chunk_id") or "") in set(chunk_ids)]
    score = max(
        (
            float(hit.get("adjusted_score") or hit.get("retrieval_score") or hit.get("score") or 0.0)
            for hit in relevant_hits
        ),
        default=0.0,
    )
    query_groups = sorted(
        {
            str(hit.get("query_group") or "")
            for hit in relevant_hits
            if hit.get("query_group")
        }
    )
    pages = [units[position].get("page") for position in positions]
    text_parts: list[str] = []
    chunk_spans: list[dict[str, Any]] = []
    cursor = 0
    for position in positions:
        if text_parts:
            cursor += 1
        unit = units[position]
        unit_text = str(unit["text"])
        text_parts.append(unit_text)
        for span in unit.get("chunk_spans", []):
            chunk_spans.append(
                {
                    "chunk_id": span.get("chunk_id", ""),
                    "start": cursor + int(span.get("start") or 0),
                    "end": cursor + int(span.get("end") or 0),
                }
            )
        cursor += len(unit_text)
    payload = "|".join([source, *chunk_ids])
    return {
        "window_id": "docwin_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16],
        "source_file": source,
        "document_role": str(relevant_hits[0].get("document_role") or "unknown") if relevant_hits else "unknown",
        "query_groups": query_groups,
        "queries": sorted({_clean(hit.get("query")) for hit in relevant_hits if _clean(hit.get("query"))}),
        "retrieval_score": score,
        "relation_signal_score": _relation_signal_score("\n".join(text_parts)),
        "source_chunk_ids": chunk_ids,
        "retrieved_chunk_ids": sorted(hit_ids.intersection(chunk_ids)),
        "page_start": min((page for page in pages if isinstance(page, int)), default=None),
        "page_end": max((page for page in pages if isinstance(page, int)), default=None),
        "text": "\n".join(text_parts),
        "_chunk_spans": chunk_spans,
    }


def apply_critic_decisions(
    claims: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    chunks_by_id: Mapping[str, object],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply semantic critic decisions with a non-overridable traceability gate."""

    by_id = {str(row.get("claim_id") or ""): row for row in decisions}
    accepted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for source_claim in claims:
        claim = dict(source_claim)
        claim_id = str(claim.get("claim_id") or "")
        decision = by_id.get(claim_id, {})
        semantic_verdict = str(decision.get("verdict") or "review").casefold()
        if semantic_verdict not in {"accept", "review", "reject"}:
            semantic_verdict = "review"
        chunk_ids = [str(value) for value in claim.get("source_chunk_ids") or [] if str(value)]
        if not chunk_ids and claim.get("chunk_id"):
            chunk_ids = [str(claim["chunk_id"])]
        traceable = _quote_is_traceable(claim.get("evidence_quote"), chunk_ids, chunks_by_id)
        reasons = [str(value) for value in decision.get("reason_codes") or [] if str(value)]
        final_verdict = semantic_verdict
        if not traceable:
            final_verdict = "reject" if decision else "review"
            reasons.append("untraceable_evidence_quote")
        support_type = str(decision.get("support_type") or "not_assessed")
        claim.update(
            {
                "critic_status": final_verdict,
                "critic_support_type": support_type,
                "critic_reason_codes": sorted(set(reasons)),
                "critic_missing_fields": sorted(
                    {str(value) for value in decision.get("missing_fields") or [] if str(value)}
                ),
                "critic_traceability_pass": traceable,
                "review_status": "candidate" if final_verdict == "accept" else "needs_review",
            }
        )
        if (
            claim.get("claim_type") == "transformation"
            and final_verdict in {"accept", "review"}
            and support_type
            in {"explicit_report", "literature_inferred", "delta_only"}
        ):
            claim["evidence_type"] = support_type
        audit.append(
            {
                "claim_id": claim_id,
                "claim_type": claim.get("claim_type", ""),
                "semantic_verdict": semantic_verdict,
                "final_verdict": final_verdict,
                "traceability_pass": traceable,
                "support_type": claim["critic_support_type"],
                "reason_codes": claim["critic_reason_codes"],
            }
        )
        if final_verdict == "accept":
            accepted.append(claim)
        elif final_verdict == "review":
            review.append(claim)
        else:
            rejected.append(claim)
    return accepted, review, rejected, audit


_PRODUCT_ION_LOSS_PATTERN = re.compile(
    r"m\s*/\s*z\s*(?P<product>\d+(?:\.\d+)?)\s*=\s*"
    r"(?P<source>\d+(?:\.\d+)?)\s*[-\N{MINUS SIGN}\N{EN DASH}]\s*"
    r"(?P<loss>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def recover_product_ion_cascade_neutral_losses(
    rejected_claims: Sequence[Mapping[str, Any]],
    *,
    absolute_tolerance: float = 0.05,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Recover traceable MSn neutral losses with explicit numeric balance.

    A neutral loss remains experimental evidence when the source is a reported
    product ion rather than the intact precursor.  Recovery is deliberately
    limited to claims whose quote contains ``product = source - loss`` and
    whose three numbers balance within the configured tolerance.
    """

    recovered: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for source_claim in rejected_claims:
        claim = dict(source_claim)
        if (
            str(claim.get("claim_type") or "") != "neutral_loss"
            or claim.get("critic_traceability_pass") is not True
        ):
            remaining.append(claim)
            continue
        match = _PRODUCT_ION_LOSS_PATTERN.search(
            str(claim.get("evidence_quote") or "")
        )
        try:
            reported_loss = float(
                claim.get("neutral_loss_mass") or claim.get("loss_mass")
            )
        except (TypeError, ValueError):
            match = None
            reported_loss = 0.0
        if match is None:
            remaining.append(claim)
            continue
        product_mz = float(match.group("product"))
        source_mz = float(match.group("source"))
        quoted_loss = float(match.group("loss"))
        if (
            abs((source_mz - product_mz) - quoted_loss) > absolute_tolerance
            or abs(reported_loss - quoted_loss) > absolute_tolerance
        ):
            remaining.append(claim)
            continue
        claim.update(
            {
                "critic_status": "accept",
                "critic_support_type": "explicit_report",
                "critic_reason_codes": ["explicit_product_ion_mass_balance"],
                "review_status": "candidate",
                "source_ion_type": "product_ion",
                "source_ion_mz": match.group("source"),
                "product_ion_mz": match.group("product"),
                "required_context": "product_ion_cascade",
            }
        )
        recovered.append(claim)
        audit.append(
            {
                "claim_id": claim.get("claim_id", ""),
                "claim_type": "neutral_loss",
                "verdict": "recover",
                "reason_code": "explicit_product_ion_mass_balance",
                "source_ion_mz": match.group("source"),
                "product_ion_mz": match.group("product"),
                "loss_mass": match.group("loss"),
            }
        )
    return recovered, remaining, audit


def accept_prevalidated_structured_claims(
    claims: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Preserve claims already located by the deterministic table parser.

    Structured rows may span overlapping corpus chunks.  Their parser-level
    ``structured`` traceability is therefore authoritative; re-checking the
    full reconstructed row against one owner chunk would create false rejects.
    """

    accepted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for source in claims:
        claim = dict(source)
        valid = str(claim.get("traceability_status") or "") == "structured"
        claim.update(
            {
                "critic_status": "accept" if valid else "review",
                "critic_support_type": "explicit_report" if valid else "not_assessed",
                "critic_reason_codes": [
                    "deterministic_structured_parser"
                    if valid
                    else "structured_traceability_missing"
                ],
                "critic_missing_fields": [],
                "critic_traceability_pass": valid,
                "review_status": "candidate" if valid else "needs_review",
            }
        )
        audit.append(
            {
                "claim_id": claim.get("claim_id", ""),
                "claim_type": claim.get("claim_type", ""),
                "semantic_verdict": "accept" if valid else "review",
                "final_verdict": "accept" if valid else "review",
                "traceability_pass": valid,
                "support_type": claim["critic_support_type"],
                "reason_codes": claim["critic_reason_codes"],
            }
        )
        (accepted if valid else review).append(claim)
    return accepted, review, audit


def _claim_entities(claim: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for field in (
        "compound_name",
        "entity_name",
        "source_entity",
        "target_entity",
        "precursor_name",
    ):
        value = _clean(claim.get(field))
        if value:
            values.add(value)
    return values


def build_gap_queries(
    claims: Sequence[Mapping[str, Any]], *, max_queries: int
) -> list[dict[str, str]]:
    """Generate entity-specific local queries only for missing relation types."""

    if max_queries <= 0:
        return []
    entities = sorted(
        {
            entity
            for claim in claims
            if str(claim.get("claim_type") or "") == "compound"
            for entity in _claim_entities(claim)
        },
        key=str.casefold,
    )
    covered: dict[str, set[str]] = defaultdict(set)
    for claim in claims:
        claim_type = str(claim.get("claim_type") or "")
        if claim_type == "transformation":
            relation_entities = {_clean(claim.get("source_entity"))}
        elif claim_type in {"diagnostic_fragment", "neutral_loss"}:
            relation_entities = {
                _clean(claim.get("compound_name") or claim.get("entity_name"))
            }
        elif claim_type == "entity_component_membership":
            relation_entities = {
                _clean(claim.get("entity_name") or claim.get("entity_id"))
            }
        else:
            relation_entities = _claim_entities(claim)
        relation_entities.discard("")
        for entity in relation_entities:
            covered[entity.casefold()].add(claim_type)
    queries: list[dict[str, str]] = []
    for entity in entities:
        for relation_type, claim_type, terms in RELATION_GAPS:
            if claim_type in covered[entity.casefold()]:
                continue
            queries.append(
                {
                    "entity_name": entity,
                    "relation_type": relation_type,
                    "query_group": "gap_directed_" + relation_type,
                    "query": f'"{entity}" {terms}',
                }
            )
            if len(queries) >= max_queries:
                return queries
    return queries
