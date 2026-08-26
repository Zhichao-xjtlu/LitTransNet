"""Stop-before-enumeration checks for registry-backed Network V5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .entity_hypothesis_network import (
    NetworkConfig,
    NetworkKnowledge,
    SeedAssignment,
    SpectrumNode,
)


@dataclass(frozen=True)
class PreflightReport:
    ready: bool
    stop_reasons: tuple[str, ...]
    resolved_seed_count: int
    unresolved_seed_node_ids: tuple[str, ...]
    outgoing_rule_count: int
    real_anchor_count: int
    polarity_compatible_rule_count: int
    formula_covered_rule_count: int
    fragment_contract_covered_rule_count: int
    reachable_entity_ids_by_depth: tuple[tuple[str, ...], ...]


def run_network_preflight(
    nodes: Sequence[SpectrumNode],
    seeds: Sequence[SeedAssignment],
    knowledge: NetworkKnowledge,
    config: NetworkConfig,
) -> PreflightReport:
    node_ids = {node.node_id for node in nodes}
    valid_seeds = [
        seed
        for seed in seeds
        if seed.node_id in node_ids and seed.entity_id in knowledge.compounds
    ]
    unresolved = tuple(
        sorted(
            (
                {seed.node_id for seed in seeds}
                - {seed.node_id for seed in valid_seeds}
            )
            | {
                node.node_id
                for node in nodes
                if node.known_match
                and node.seed_resolution_status
                not in {"resolved", "ambiguous_library_seed"}
            }
        )
    )
    frontier = {seed.entity_id for seed in valid_seeds}
    reachable: list[tuple[str, ...]] = [tuple(sorted(frontier))]
    rules = []
    for _ in range(config.max_depth):
        depth_rules = [
            rule
            for entity_id in frontier
            for rule in knowledge.outgoing_rules.get(entity_id, ())
        ]
        rules.extend(depth_rules)
        frontier = {rule.target_entity_id for rule in depth_rules}
        reachable.append(tuple(sorted(frontier)))
        if not frontier:
            break
    experiment_modes = {node.ion_mode for node in nodes}
    polarity_compatible = [
        rule
        for rule in rules
        if not rule.ion_modes or bool(set(rule.ion_modes) & experiment_modes)
    ]
    formula_covered = [
        rule
        for rule in rules
        if knowledge.compounds[rule.target_entity_id].exact_mass is not None
    ]
    fragment_covered = [
        rule
        for rule in rules
        if knowledge.compounds[rule.target_entity_id].reported_fragment_ids
        or any(rule.fragment_evidence_contract.values())
    ]
    reasons = []
    if not valid_seeds:
        reasons.append("no_resolved_seed")
    if valid_seeds and not rules:
        reasons.append("no_reachable_rule")
    if valid_seeds and not polarity_compatible:
        reasons.append("no_polarity_compatible_rule")
    return PreflightReport(
        ready=not reasons,
        stop_reasons=tuple(reasons),
        resolved_seed_count=len(valid_seeds),
        unresolved_seed_node_ids=unresolved,
        outgoing_rule_count=len(rules),
        real_anchor_count=len({seed.entity_id for seed in valid_seeds}),
        polarity_compatible_rule_count=len(polarity_compatible),
        formula_covered_rule_count=len(formula_covered),
        fragment_contract_covered_rule_count=len(fragment_covered),
        reachable_entity_ids_by_depth=tuple(reachable),
    )
