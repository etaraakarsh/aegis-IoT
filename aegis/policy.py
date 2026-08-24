from __future__ import annotations

import itertools
from typing import Dict, List, Sequence


def group_of(tool_name: str) -> str:
    return tool_name.rsplit("_clf_", 1)[0] if "_clf_" in tool_name \
        else tool_name.rsplit("_", 1)[0]


def tier_of(tool_name: str) -> str:
    return tool_name.rsplit("_clf_", 1)[1] if "_clf_" in tool_name else ""


def measure_marginal_values(evaluate_subset, all_tools: Sequence[str],
                            core: Sequence[str] = (), verbose=True
                            ) -> Dict[str, float]:
    """Marginal contribution of each non-core tool ON TOP OF the core.

    Measured as F1(core + t) - F1(core), within the tool's own group. This is
    the question the policy actually faces -- it always has the core, and is
    deciding what to add. Leave-one-out over the full registry answers a
    different question and answers it badly here, because redundant siblings
    mask each tool's contribution and everything measures near zero.
    """
    core = list(core)
    out: Dict[str, float] = {}
    if verbose:
        print("\n  marginal value on top of the core (within group)")
        print(f"  {'tool':<28}{'core':>9}{'core+t':>9}{'d':>10}")
        print("  " + "-" * 58)
    base_by_group: Dict[str, float] = {}
    for t in all_tools:
        if t in core:
            out[t] = float("inf")            # already mandatory
            continue
        g = group_of(t)
        if g not in base_by_group:
            base_by_group[g] = evaluate_subset(core, group=g)
        b = base_by_group[g]
        v = evaluate_subset(core + [t], group=g)
        out[t] = v - b
        if verbose:
            print(f"  {t:<28}{b:>9.4f}{v:>9.4f}{v - b:>+10.4f}")
    return out


def select_core_by_tier(evaluate_subset, all_tools: Sequence[str],
                        costs=None, epsilon: float = 0.05,
                        verbose=True) -> List[str]:
    """evaluate_subset must accept (tools, group=None). Tiers are compared
    WITHIN their own group: a group's classifier fires only on that group's
    incidents, so scoring it against the whole validation set measures mostly
    rows it never saw."""
    """Choose, per group, the cheapest classifier tier that is not clearly
    worse than the best tier. Ties go to the cheaper tier, which is the whole
    point of having tiers."""
    groups = sorted({group_of(t) for t in all_tools if "_clf_" in t})
    core: List[str] = []
    if verbose:
        print("\n  per-group tier selection (accuracy of each tier alone)")
    for g in groups:
        tiers = {tier_of(t): t for t in all_tools
                 if "_clf_" in t and group_of(t) == g}
        if not tiers:
            continue
        scored = {name: evaluate_subset([tool], group=g)
                  for name, tool in tiers.items()}
        best = max(scored.values())
        floor = best * (1.0 - epsilon)
        # Cheapest tier satisfying the accuracy floor. The threshold is the
        # SAME epsilon as the constrained formulation -- it was previously
        # hardcoded at 1%, which silently overrode the stated constraint and
        # bought the widest tier every time.
        order = sorted(tiers, key=lambda t: (costs.tool_cost.get(tiers[t], 0.0)
                                             if costs else 0.0))
        for name in order:
            if scored[name] >= floor:
                core.append(tiers[name])
                chosen = name
                break
        else:
            chosen = max(scored, key=scored.get)
            core.append(tiers[chosen])
        if verbose:
            detail = "  ".join(
                f"{k}={v:.4f}" + (f"@{costs.tool_cost.get(tiers[k], 0):.0f}"
                                  if costs else "")
                for k, v in scored.items())
            print(f"    {g:<16} {detail}  floor={floor:.4f} -> core: {chosen}")
    return core


def prune_harmful_core(core: Sequence[str], evaluate_subset,
                       verbose=True) -> List[str]:
    """Validate each core member against the CORE, not against the full
    registry.

    This distinction is easy to get wrong and silently destroys the core.
    Leave-one-out over the whole registry asks "what is lost if this tool goes
    while everything else stays". When the registry contains redundant
    alternatives -- here, three classifier tiers per group -- the answer is
    always "almost nothing", because a sibling tier covers the gap. Every tier
    then measures at or below zero and the entire core is pruned away.

    The question that actually matters is what is lost if this tool goes and no
    sibling replaces it. So each member is scored against the core alone, and
    within its own group, since a group's classifier never fires elsewhere.
    """
    core = list(core)
    keep, removed = [], []
    if verbose:
        print("\n  core validation (within-group, against the core only)")
    for t in core:
        g = group_of(t)
        with_t = evaluate_subset(core, group=g)
        without = evaluate_subset([x for x in core if x != t], group=g)
        d = with_t - without
        (keep if d > 0 else removed).append(t)
        if verbose:
            print(f"    {t:<26} with={with_t:.4f} without={without:.4f} "
                  f"d={d:+.4f} {'keep' if d > 0 else 'DROP'}")
    if removed and verbose:
        print(f"  pruned from core (non-positive contribution): {removed}")
    return keep or core                # never return an empty core


class ConstrainedPolicy:
    """Greedy selection by marginal value per unit cost, over a mandatory core."""

    def __init__(self, core: Sequence[str], values: Dict[str, float],
                 costs, budget: float, epsilon: float = 0.05):
        self.core = list(core)
        self.values = dict(values)
        self.costs = costs
        self.budget = float(budget)
        self.epsilon = float(epsilon)

    def select(self, available: Sequence[str]) -> List[str]:
        """Minimise cost subject to the accuracy floor.

        The core is chosen so that it already satisfies the floor. The
        objective is MINIMISATION, so once the floor is met the correct action
        is to stop, not to keep spending until the budget is exhausted.

        An earlier version spent the remaining budget on any tool with positive
        marginal value. On TON_IoT that re-added the expensive full-tier
        classifier the accuracy floor had just replaced with mid, so the
        constrained policy cost 36% MORE than the core alone (4379.8 vs 3230.7
        NCU) for 0.001 LESS macro-F1 -- and the epsilon sweep came out
        perfectly flat, because whatever epsilon removed, the greedy step put
        back. Buying accuracy you are not required to buy is not what the
        formulation says.

        Tools outside the core are added only when the core FAILS the floor,
        which happens when a group has no viable classifier.
        """
        chosen = [t for t in self.core if t in available]
        if chosen:
            return chosen

        # No core tool applies to this incident: fall back to the best value
        # per unit cost that the budget allows.
        spent = 0.0
        ranked = sorted(
            (t for t in available if self.values.get(t, 0.0) > 0),
            key=lambda t: -(self.values[t] / max(self.costs.tool_cost.get(t, 1.0), 1e-9)),
        )
        for t in ranked:
            c = self.costs.tool_cost.get(t, 1.0)
            if spent + c <= self.budget:
                chosen.append(t)
                spent += c
        return chosen or list(available)[:1]

    def describe(self) -> dict:
        return {"mandatory_core": self.core, "budget_ncu": self.budget,
                "epsilon": self.epsilon,
                "values": {k: (None if v == float("inf") else round(v, 6))
                           for k, v in self.values.items()}}


class UnconstrainedPolicy:
    """Cost-only minimisation. Retained deliberately: this is the formulation
    that collapses, and the ablation depends on being able to reproduce it."""

    def __init__(self, costs, budget: float):
        self.costs = costs
        self.budget = float(budget)

    def select(self, available: Sequence[str]) -> List[str]:
        chosen, spent = [], 0.0
        for t in sorted(available, key=lambda x: self.costs.tool_cost.get(x, 1.0)):
            c = self.costs.tool_cost.get(t, 1.0)
            if spent + c <= self.budget:
                chosen.append(t)
                spent += c
        return chosen


def fixed_subset(available: Sequence[str]) -> List[str]:
    """One full-tier classifier per group -- what a careful engineer picks
    without measuring anything."""
    out, seen = [], set()
    for t in sorted(available):
        if "_clf_full" in t and group_of(t) not in seen:
            out.append(t)
            seen.add(group_of(t))
    return out or list(available)[:3]
