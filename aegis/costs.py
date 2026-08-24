"""
costs.py — one total-cost number, from measurement.

CHANGE FROM v5. The old cost table was hand-written placeholder constants
(classifiers 3.0, isolation forests 1.5, everything else 1.0) with a comment
reading "REPLACE with measured values before publishing". They were never
replaced. Every cost figure in the v5 results -- including the headline "5.8x
cheaper" -- was arithmetic on numbers that had been invented, not measured.

This module removes that possibility. Costs are loaded from a JSON file written
by scripts/03_measure_costs.py, which times every tool on real incidents. If the
file is absent, load_costs() raises rather than falling back to placeholders,
because a silent fallback is exactly how the v5 problem survived to a draft.

Everything is expressed in normalized cost units (NCU), defined so that the
cheapest measured tool equals 1.0. Tool compute and LLM tokens share the unit so
the policy cannot reduce one by inflating the other.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable

COSTS_FILENAME = "measured_costs.json"


@dataclass
class CostModel:
    tool_cost: Dict[str, float]
    token_cost_per_1k: float
    latency_cost_per_s: float = 0.0
    provenance: str = "measured"

    def tools(self, called: Iterable[str]) -> float:
        # An unknown tool is charged the registry mean rather than a silent 1.0,
        # so a typo inflates cost visibly instead of hiding as the cheapest tool.
        mean = (sum(self.tool_cost.values()) / len(self.tool_cost)
                if self.tool_cost else 1.0)
        return float(sum(self.tool_cost.get(t, mean) for t in called))

    def tokens(self, n_tokens: int) -> float:
        return float(n_tokens) / 1000.0 * self.token_cost_per_1k

    def total(self, called: Iterable[str], n_tokens: int = 0,
              latency_s: float = 0.0) -> float:
        return (self.tools(called) + self.tokens(n_tokens)
                + latency_s * self.latency_cost_per_s)

    def breakdown(self, called: Iterable[str], n_tokens: int = 0,
                  latency_s: float = 0.0) -> Dict[str, float]:
        t = self.tools(called)
        k = self.tokens(n_tokens)
        return {"tool": t, "token": k,
                "latency": latency_s * self.latency_cost_per_s,
                "total": t + k + latency_s * self.latency_cost_per_s,
                "n_tokens": int(n_tokens), "latency_s": float(latency_s)}


def load_costs(data_dir: str) -> CostModel:
    path = os.path.join(data_dir, COSTS_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n{path} not found.\n\n"
            "Tool costs must be MEASURED before any cost result is reported.\n"
            "Run:  python scripts/03_measure_costs.py --data-dir "
            f"{data_dir}\n\n"
            "There is deliberately no placeholder fallback: the previous\n"
            "version of this project shipped invented costs into a manuscript\n"
            "draft because a fallback existed.\n")
    with open(path) as f:
        d = json.load(f)
    return CostModel(
        tool_cost={k: float(v) for k, v in d["tool_cost_ncu"].items()},
        token_cost_per_1k=float(d.get("token_cost_per_1k_ncu", 0.0)),
        latency_cost_per_s=float(d.get("latency_cost_per_s_ncu", 0.0)),
        provenance=d.get("provenance", "measured"),
    )


def save_costs(data_dir: str, tool_seconds: Dict[str, float],
               token_seconds_per_1k: float, meta: dict) -> str:
    """Normalize measured wall-times so the cheapest tool == 1.0 NCU."""
    if not tool_seconds:
        raise ValueError("no tools were timed")
    floor = min(v for v in tool_seconds.values() if v > 0)
    payload = {
        "provenance": "measured",
        "normalization": "cheapest measured tool = 1.0 NCU",
        "floor_seconds": floor,
        "tool_cost_ncu": {k: round(v / floor, 4) for k, v in tool_seconds.items()},
        "tool_seconds_raw": {k: round(v, 8) for k, v in tool_seconds.items()},
        "token_cost_per_1k_ncu": round(token_seconds_per_1k / floor, 6),
        "token_seconds_per_1k_raw": token_seconds_per_1k,
        "latency_cost_per_s_ncu": 0.0,
        "meta": meta,
    }
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, COSTS_FILENAME)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
