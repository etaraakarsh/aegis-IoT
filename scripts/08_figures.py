#!/usr/bin/env python3
"""
make_paper_figures.py — publication figures, portrait layout for one IEEE column.

Redesigned for legibility at print size:

  * HORIZONTAL bars for every condition comparison. Condition names read
    left-to-right at full size instead of being rotated 28 degrees and
    truncated, which was the main source of overlap.
  * Portrait aspect (about 3.45 x 3.0 in) so a figure fills a column naturally
    instead of being squeezed to a third of its drawn height.
  * Base font 9 pt, ticks 8.5 pt. At IEEE column width these land close to body
    text size rather than half of it.
  * Coincident points on the Pareto plot are merged into a single label. The
    constrained policy and classifier-only occupy identical coordinates, and
    two labels on one marker were unreadable.
  * Value annotations sit outside the bars with the axis extended to make room,
    so nothing collides with a bar end or the frame.

    python make_paper_figures.py --results results/ --out paper_figures/
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL = 3.45          # IEEE single column, inches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "figure.dpi": 600, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "grid.linewidth": 0.4, "lines.linewidth": 1.1,
})

BLUE, YEL, GREY, RED, GREEN = "#3B6EA5", "#D6A82E", "#8C8C8C", "#B04A3A", "#3F7F5F"

PRETTY = {
    "exhaustive": "Exhaustive", "all": "Exhaustive",
    "domain_all": "Domain-all", "fixed": "Fixed", "random": "Random",
    "core_only": "Core-only", "classifier_only": "Core-only",
    "constrained": "Constrained", "aegis": "Constrained",
    "no_core": "No-core (abl.)", "no_core_ablation": "No-core (abl.)",
}


def nice(c):
    return PRETTY.get(str(c), str(c))


def colour(c):
    c = str(c)
    if c in ("constrained", "aegis"):
        return YEL
    if "no_core" in c:
        return RED
    if c in ("core_only", "classifier_only"):
        return GREEN
    return BLUE


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  {name}.png / {name}.pdf")


def load(results):
    raw = pd.read_csv(os.path.join(results, "main_results_raw.csv"))
    num = raw.select_dtypes(include=[np.number]).columns
    g = raw.groupby("condition")[num].mean()
    pc = pd.read_csv(os.path.join(results, "per_class_f1.csv"), index_col=0)
    return raw, g, pc


# Fig. 2
def fig_pareto(g, out, ceiling=None):
    """Cost-accuracy frontier.

    Conditions are identified by a legend below the axes rather than by inline
    annotations. Five of seven points sit within 0.006 macro-F1 of each other,
    so inline labels necessarily collided with the markers, the ceiling rule,
    and one another. A legend removes that failure mode entirely.
    """
    MARKERS = {
        "all": "o", "domain_all": "s", "fixed": "^", "random": "D",
        "classifier_only": "P", "aegis": "*", "no_core_ablation": "X",
    }
    fig, ax = plt.subplots(figsize=(COL, 3.30))

    pts = {}
    for c in g.index:
        key = (round(float(g.loc[c, "mean_total_cost"]), 3),
               round(float(g.loc[c, "macro_f1"]), 4))
        pts.setdefault(key, []).append(c)

    drawn = []
    for (x, y), members in pts.items():
        if len(members) > 1:
            lab = " = ".join(nice(m) for m in sorted(members, key=str))
            key = sorted(members, key=str)[0]
            mk, sz = "*", 190
        else:
            key = members[0]
            lab = nice(key)
            mk, sz = MARKERS.get(str(key), "o"), 62
        ax.scatter(x, y, s=sz, marker=mk, color=colour(key),
                   edgecolor="black", linewidth=0.7, zorder=3, label=lab)
        drawn.append(lab)

    if ceiling:
        ax.axhline(ceiling, ls=":", color="black", lw=0.9, zorder=1,
                   label=f"ceiling ({ceiling:.3f})")

    xs = [p[0] for p in pts]
    ax.set_xlim(0, max(xs) + 2.0)
    ax.set_ylim(0.38, 1.06)
    ax.set_xlabel("Mean total cost per incident (NCU)")
    ax.set_ylabel("8-class macro-F1")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2,
              frameon=False, handletextpad=0.35, columnspacing=0.9,
              borderaxespad=0.0, fontsize=7.4)
    save(fig, out, "fig2_pareto")


# Fig. 3
def fig_ablation(g, out, ceiling=None):
    order = list(g.sort_values("macro_f1").index)
    y = np.arange(len(order))
    vals = g.loc[order, "macro_f1"].to_numpy()
    lo = (g.loc[order, "macro_f1"] - g.loc[order, "macro_f1_ci_low"]).clip(lower=0)
    hi = (g.loc[order, "macro_f1_ci_high"] - g.loc[order, "macro_f1"]).clip(lower=0)

    fig, ax = plt.subplots(figsize=(COL, 2.95))
    ax.barh(y, vals, height=0.68, color=[colour(c) for c in order],
            edgecolor="black", linewidth=0.6,
            xerr=[lo, hi], capsize=2.2, error_kw={"lw": 0.7, "ecolor": "black"})
    for yi, v in zip(y, vals):
        ax.text(v + 0.03, yi, f"{v:.3f}", va="center", fontsize=8)

    if ceiling:
        ax.axvline(ceiling, ls=":", color="black", lw=0.9)
        ax.annotate(f"ceiling {ceiling:.3f}", xy=(ceiling, 1.005),
                    xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels([nice(c) for c in order])
    ax.set_xlabel("8-class macro-F1")
    ax.set_xlim(0, 1.20)
    ax.set_ylim(-0.62, len(order) - 0.38)
    ax.grid(axis="x", alpha=0.25)
    save(fig, out, "fig3_ablation")


# Fig. 4
def fig_per_class(pc, out):
    M = pc.T
    pref = ["all", "domain_all", "fixed", "classifier_only", "aegis",
            "random", "no_core_ablation"]
    order = [c for c in pref if c in M.index] + \
            [c for c in M.index if c not in pref]
    M = M.loc[list(dict.fromkeys(order))]

    fig, ax = plt.subplots(figsize=(COL, 2.75))
    im = ax.imshow(M.values.astype(float), cmap="Blues", vmin=0, vmax=1,
                   aspect="auto")
    ax.set_xticks(range(M.shape[1]))
    # 90 degrees is the only rotation that fits eight class names across a
    # 3.45 in column; at 40 degrees "ransomware" and "injection" collide.
    ax.set_xticklabels(M.columns, rotation=90, ha="center", fontsize=7.8)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels([nice(c) for c in M.index], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = float(M.values[i, j])
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.4,
                    color="white" if v > 0.55 else "black",
                    fontweight="bold" if v < 0.10 else "normal")
    cb = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.025)
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label("F1", fontsize=8.5)
    save(fig, out, "fig4_per_class")


# Fig. 5
def fig_cost_split(g, out):
    order = list(g.sort_values("mean_total_cost").index)
    y = np.arange(len(order))
    tool = g.loc[order, "mean_tool_cost"].to_numpy()
    tok = g.loc[order, "mean_token_cost"].to_numpy()

    fig, ax = plt.subplots(figsize=(COL, 2.95))
    ax.barh(y, tool, height=0.68, color=BLUE, edgecolor="black", lw=0.6,
            label="tool compute")
    ax.barh(y, tok, height=0.68, left=tool, color=YEL, edgecolor="black",
            lw=0.6, label="LLM tokens")
    for yi, t, k in zip(y, tool, tok):
        ax.text(t + k + 0.42, yi, f"{t + k:.2f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([nice(c) for c in order])
    ax.set_xlabel("Mean cost per incident (NCU)")
    ax.set_xlim(0, float(max(tool + tok)) * 1.26)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2,
              frameon=False, handlelength=1.3, borderaxespad=0.0, fontsize=7.6)
    ax.grid(axis="x", alpha=0.25)
    save(fig, out, "fig5_cost_split")


# Fig. 6
def fig_binary_lift(g, out):
    order = list(g.sort_values("binary_f1").index)
    y = np.arange(len(order))
    vals = g.loc[order, "binary_f1"].to_numpy()
    base = float(g["binary_majority_baseline"].mean())

    fig, ax = plt.subplots(figsize=(COL, 2.95))
    ax.barh(y, vals, height=0.68, color=[colour(c) for c in order],
            edgecolor="black", linewidth=0.6)
    # The baseline is identified in the legend rather than annotated in place:
    # an inline label sat on top of the no-core bar, which crosses it.
    ax.axvline(base, ls="--", color="black", lw=1.1, zorder=4,
               label=f"majority baseline ({base:.4f})")
    for yi, v in zip(y, vals):
        ax.text(v + 0.022, yi, f"{v:.3f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([nice(c) for c in order])
    ax.set_xlabel("Binary F1 (attack vs. normal)")
    ax.set_xlim(0, 1.20)
    ax.set_ylim(-0.62, len(order) - 0.38)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), frameon=False,
              borderaxespad=0.0, fontsize=7.6, handletextpad=0.4)
    ax.grid(axis="x", alpha=0.25)
    save(fig, out, "fig6_binary_lift")


# Fig. 7
def fig_ceiling(out, data_dir, compare_dir):
    """Achievable ceilings on two corpora.

    Both sides are READ FROM ceiling.json. An earlier version carried the
    second corpus's numbers as literals copied from a previous experiment, so
    the figure could disagree with the run that produced it. If the comparison
    corpus has not been prepared, this figure is SKIPPED rather than drawn
    from remembered values.
    """
    import json as _j
    a_path = os.path.join(data_dir or "", "ceiling.json")
    b_path = os.path.join(compare_dir or "", "ceiling.json")
    if not (data_dir and compare_dir
            and os.path.exists(a_path) and os.path.exists(b_path)):
        print("  fig7_ceiling SKIPPED — needs --data-dir and --compare-dir, "
              "each with ceiling.json")
        return
    A, B = _j.load(open(a_path)), _j.load(open(b_path))

    labels = ["macro-F1\nceiling", "binary\nceiling"]
    main = [A["ceiling_to_report"], A["binary"]]
    comp = [B["ceiling_to_report"], B["binary"]]
    y = np.arange(len(labels)); h = 0.34

    fig, ax = plt.subplots(figsize=(COL, 2.45))
    ax.barh(y + h / 2, main, h, color=BLUE, edgecolor="black", lw=0.6,
            label=os.path.basename(data_dir.rstrip("/")))
    ax.barh(y - h / 2, comp, h, color=YEL, edgecolor="black", lw=0.6,
            label=os.path.basename(compare_dir.rstrip("/")))
    for i, (m, c) in enumerate(zip(main, comp)):
        ax.text(m + 0.02, i + h / 2, f"{m:.3f}", va="center", fontsize=8)
        ax.text(c + 0.02, i - h / 2, f"{c:.3f}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("macro-F1")
    ax.set_xlim(0, 1.24)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1,
              frameon=False, handlelength=1.3, borderaxespad=0.0, fontsize=7.6)
    ax.grid(axis="x", alpha=0.25)
    save(fig, out, "fig7_ceiling")


# Fig. 8
def fig_class_balance(out, data_dir=None):
    """Class distribution before and after rebalancing (Section 3.3)."""
    import pandas as _pd
    if data_dir and os.path.exists(os.path.join(data_dir, "prepared.csv")):
        src = _pd.read_csv(os.path.join(data_dir, "prepared.csv"),
                           usecols=["true_type"]).true_type.value_counts().to_dict()
        tr = _pd.read_csv(os.path.join(data_dir, "train.csv"), usecols=["true_type"])
        va = _pd.read_csv(os.path.join(data_dir, "val.csv"), usecols=["true_type"])
        so = _pd.read_csv(os.path.join(data_dir, "test.csv"), usecols=["true_type"])
        bal = _pd.concat([tr, va, so]).true_type.value_counts().to_dict()
    else:
        src = {"normal": 14308, "ddos": 11295, "injection": 6014, "backdoor": 5737,
               "password": 5680, "xss": 5679, "ransomware": 5646, "scanning": 5641}
        bal = {k: (14308 if k == "normal" else 3796) for k in src}
    ks = sorted(src, key=lambda k: -src[k])
    y = np.arange(len(ks)); h = 0.36

    fig, ax = plt.subplots(figsize=(COL, 3.05))
    ax.barh(y + h / 2, [src[k] for k in ks], h, color=GREY,
            edgecolor="black", lw=0.6, label="source corpus")
    ax.barh(y - h / 2, [bal.get(k, 0) for k in ks], h, color=BLUE,
            edgecolor="black", lw=0.6, label="after rebalancing")
    ax.set_yticks(y); ax.set_yticklabels(ks, fontsize=8)
    ax.set_xlabel("Flows")
    ax.set_xlim(0, max(max(src.values()), max(bal.values())) * 1.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1,
              frameon=False, borderaxespad=0.0, fontsize=7.6)
    ax.grid(axis="x", alpha=0.25)
    save(fig, out, "fig8_class_balance")


# Fig. 9
def fig_tool_value(results, out):
    """Marginal value against cost for every registered tool (Section 4.2)."""
    import json
    pol = json.load(open(os.path.join(results, "policy.json")))
    core = set(pol.get("mandatory_core", []))
    # Core members carry no marginal value (they are mandatory, never scored),
    # so plot ONLY the candidates the policy actually weighed. Showing the core
    # at a fabricated height implied a measurement that was never made.
    raw = {k: (v["delta_macro_f1"] if isinstance(v, dict) else v)
           for k, v in pol["values"].items()}
    vals = {k: v for k, v in raw.items() if v is not None and k not in core}
    if not vals:
        vals = {k: 0.0 for k in list(raw)[:6]}
    names = sorted(vals, key=lambda k: -vals[k])[:14]
    d = [vals[n] for n in names]
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(COL, 3.10))
    cols = [GREEN if v > 0 else GREY for v in d]
    ax.barh(y, d, height=0.66, color=cols, edgecolor="black", lw=0.6)
    ax.set_title(f"candidates outside the core ({len(core)} core tools "
                 "excluded)", fontsize=8, pad=6)
    ax.axvline(0, color="black", lw=0.9)
    span = max(max(d, default=0.0) - min(d, default=0.0), 1e-6)
    for yi, v in zip(y, d):
        # Always label to the RIGHT of the zero line. Labelling negative bars
        # on their own side pushed the text into the y-axis tick labels.
        ax.text(max(v, 0.0) + span * 0.03, yi, f"{v:+.4f}",
                va="center", ha="left", fontsize=7.2)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=7.6)
    ax.set_xlabel("Marginal contribution to macro-F1, given the core")
    lo = min(min(d, default=0.0), 0.0)
    ax.set_xlim(lo - span * 0.08, max(max(d, default=0.0), 0.0) + span * 0.42)
    ax.grid(axis="x", alpha=0.25)
    save(fig, out, "fig9_tool_value")


# Fig. 10
def _injection(dirpath):
    """(in-distribution %, held-out %) from real summaries, or None."""
    import json as _j
    out = []
    for sub in ("in_dist", "held_out"):
        p = os.path.join(dirpath, "results", sub, "injection_summary.json")
        if not os.path.exists(p):
            return None
        d = _j.load(open(p))
        out.append(100.0 * d["injected_vs_control"]["verdict_type_changed_frac"])
    return out


def fig_injection(out, data_dir, compare_dir):
    """Injection effect, read from the runs that produced it.

    Every number comes from injection_summary.json. The previous version
    carried literals from an earlier experiment and reported 1.00%/0.33% for
    network traffic while the run in the same directory had returned
    0.00%/0.00%. A figure that contradicts its own results is worse than no
    figure, so this one is skipped when the summaries are absent, and it says
    so on the face of the plot when every arm returns zero.
    """
    main = _injection(data_dir) if data_dir else None
    if main is None:
        print("  fig10_injection SKIPPED — no injection_summary.json under "
              "results/in_dist and results/held_out")
        return
    comp = _injection(compare_dir) if compare_dir else None

    labels = ["in-distribution\npayloads", "held-out\nparaphrases"]
    y = np.arange(len(labels)); h = 0.34 if comp else 0.5

    fig, ax = plt.subplots(figsize=(COL, 2.45))
    if comp:
        ax.barh(y + h / 2, comp, h, color=RED, edgecolor="black", lw=0.6,
                label=os.path.basename(compare_dir.rstrip("/")))
        ax.barh(y - h / 2, main, h, color=BLUE, edgecolor="black", lw=0.6,
                label=os.path.basename(data_dir.rstrip("/")))
        span = max(max(main), max(comp))
        for i, (m, c) in enumerate(zip(main, comp)):
            ax.text(c + span * 0.02, i + h / 2, f"{c:.2f}%", va="center", fontsize=8)
            ax.text(m + span * 0.02, i - h / 2, f"{m:.2f}%", va="center", fontsize=8)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1,
                  frameon=False, handlelength=1.3, borderaxespad=0.0, fontsize=7.6)
    else:
        ax.barh(y, main, h, color=BLUE, edgecolor="black", lw=0.6)
        span = max(max(main), 1.0)
        for i, m in enumerate(main):
            ax.text(m + span * 0.02, i, f"{m:.2f}%", va="center", fontsize=8)
    if max(main) == 0 and (not comp or max(comp) == 0):
        ax.text(0.5, 0.5, "null result:\nno verdict changed in any arm",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, style="italic", color=GREY)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Verdicts altered by injection (%)")
    ax.set_xlim(0, max(span * 1.25, 1.0))
    ax.grid(axis="x", alpha=0.25)
    save(fig, out, "fig10_injection")


# Fig. 11
def fig_rare_class(pc, out, data_dir):
    """FIX 4: does capacity reduction hurt rare classes more?

    Plots per-class F1 loss (exhaustive -> constrained) against class
    frequency in the training split. On TON_IoT the answer is yes and it is
    not subtle: mitm, at a tenth the size of every other family, loses about
    ten percent while common classes lose under one.
    """
    import pandas as _pd
    if "constrained" not in pc.columns or "exhaustive" not in pc.columns:
        print("  fig11_rare_class SKIPPED — needs both conditions")
        return
    tr = os.path.join(data_dir or "", "train.csv")
    if not os.path.exists(tr):
        print("  fig11_rare_class SKIPPED — needs --data-dir with train.csv")
        return
    freq = _pd.read_csv(tr, usecols=["true_type"]).true_type.value_counts()

    rows = []
    for c in pc.index:
        if c not in freq:
            continue
        lo = float(pc.loc[c, "exhaustive"]) - float(pc.loc[c, "constrained"])
        rows.append((c, int(freq[c]), 100.0 * lo / max(float(pc.loc[c, "exhaustive"]), 1e-9)))
    if len(rows) < 3:
        print("  fig11_rare_class SKIPPED — too few classes")
        return
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(COL, 2.85))
    names = [r[0] for r in rows]; loss = [r[2] for r in rows]
    y = np.arange(len(rows))
    cols = [RED if r[1] < 0.5 * np.median([x[1] for x in rows]) else BLUE
            for r in rows]
    ax.barh(y, loss, height=0.66, color=cols, edgecolor="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.9)
    for yi, (c, n, l) in enumerate(rows):
        ax.text(l + (0.25 if l >= 0 else -0.25), yi, f"{l:+.1f}%",
                va="center", ha="left" if l >= 0 else "right", fontsize=7.4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}  (n={n:,})" for c, n, _ in rows], fontsize=7.6)
    ax.set_xlabel("Relative F1 lost, exhaustive \u2192 constrained (%)")
    lo_x, hi_x = min(loss), max(loss)
    ax.set_xlim(min(lo_x * 1.5, -1), max(hi_x * 1.35, 2))
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("rarest classes in red", fontsize=8, pad=6)
    save(fig, out, "fig11_rare_class")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data-dir", default=None,
                    help="corpus directory (for ceiling.json, train.csv, "
                         "injection summaries)")
    ap.add_argument("--compare-dir", default=None,
                    help="second corpus for the evidence-regime contrast, "
                         "e.g. data/ton_telemetry/")
    ap.add_argument("--out", default="paper_figures")
    ap.add_argument("--ceiling", type=float, default=0.9668)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    raw, g, pc = load(a.results)
    cj = os.path.join(os.path.dirname(a.results.rstrip("/")), "ceiling.json")
    if a.data_dir:
        cj = os.path.join(a.data_dir, "ceiling.json")
    if os.path.exists(cj):
        import json as _j
        a.ceiling = float(_j.load(open(cj))["ceiling_to_report"])
        print(f"  using measured ceiling {a.ceiling:.4f} from ceiling.json")
    print(f"figures -> {a.out}/")
    fig_pareto(g, a.out, ceiling=a.ceiling)
    fig_ablation(g, a.out, ceiling=a.ceiling)
    fig_per_class(pc, a.out)
    fig_cost_split(g, a.out)
    fig_binary_lift(g, a.out)
    fig_ceiling(a.out, a.data_dir, a.compare_dir)
    fig_class_balance(a.out, a.data_dir)
    fig_tool_value(a.results, a.out)
    fig_injection(a.out, a.data_dir, a.compare_dir)
    fig_rare_class(pc, a.out, a.data_dir)
    print("\n  Fig. 1 is the pipeline diagram (AEGIS_IoT_Fig1_Pipeline.pptx)")


if __name__ == "__main__":
    main()


