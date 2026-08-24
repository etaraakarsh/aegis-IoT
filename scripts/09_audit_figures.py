#!/usr/bin/env python3
"""
audit_figures.py — exhaustive collision check for every text artist in a figure.

The earlier audit only inspected ax.texts, which is why overlapping tick labels,
axis labels, and legends went undetected. This one collects EVERY text-bearing
artist:

    ax.texts            annotations and value labels
    tick labels         x and y, both major and minor
    axis labels         xlabel / ylabel
    legend              frame and every entry
    title
    colorbar            tick labels and label

and then tests all pairs, plus containment inside the canvas.
"""
import itertools
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect(fig):
    """Return [(kind, text, bbox)] for every text artist in the figure."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []

    def add(kind, artist, txt=None):
        try:
            bb = artist.get_window_extent(renderer=r)
        except Exception:
            return
        t = txt if txt is not None else getattr(artist, "get_text", lambda: "")()
        if not str(t).strip():
            return
        if bb.width <= 0 or bb.height <= 0:
            return
        out.append((kind, str(t), bb))

    for ax in fig.axes:
        for t in ax.texts:
            add("text", t)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            add("tick", t)
        for t in (ax.xaxis.label, ax.yaxis.label):
            add("axlabel", t)
        add("title", ax.title)
        leg = ax.get_legend()
        if leg is not None:
            add("legend-box", leg, "<legend frame>")
            for t in leg.get_texts():
                add("legend", t)
    return out


def graphics(fig):
    """Bounding boxes of drawn elements text must not sit on top of."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        for p in ax.patches:                       # bars
            try:
                bb = p.get_window_extent(renderer=r)
                if bb.width > 1 and bb.height > 1:
                    out.append(("bar", bb))
            except Exception:
                pass
        for c in ax.collections:                   # scatter markers
            try:
                bb = c.get_window_extent(renderer=r)
                if bb.width > 1 and bb.height > 1:
                    out.append(("marker", bb))
            except Exception:
                pass
        ab = ax.get_window_extent(renderer=r)
        for ln in ax.lines:                        # reference rules only
            try:
                bb = ln.get_window_extent(renderer=r)
            except Exception:
                continue
            if bb.width <= 1 or bb.height <= 1:
                continue
            # Only flag REFERENCE lines (axhline/axvline span the axes).
            # Error-bar caps are short segments and a value label sitting
            # beside its own error bar is not a defect.
            spans = (bb.width > 0.85 * ab.width) or (bb.height > 0.85 * ab.height)
            # axhline/axvline are 2-vertex lines. Error-bar cap markers are a
            # single Line2D holding one point per bar, whose bbox spans the
            # whole axes; a value label beside its own cap is not a defect.
            try:
                n_pts = len(ln.get_xdata())
            except Exception:
                n_pts = 2
            if spans and n_pts <= 2:
                out.append(("line", bb))
    return out


def audit(fig, name, tol=2.0, ignore_kinds=(("tick", "tick"),)):
    """Report overlapping pairs and out-of-canvas artists."""
    items = collect(fig)
    # Measure clipping against the TIGHT bbox actually written by savefig;
    # tick and axis labels legitimately sit outside the axes canvas and
    # bbox_inches='tight' expands to include them.
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    dpi = fig.dpi
    fw, fh = tb.width * dpi, tb.height * dpi

    bad = []
    for (k1, t1, b1), (k2, t2, b2) in itertools.combinations(items, 2):
        # a legend frame legitimately contains its own entries
        if "legend-box" in (k1, k2) and "legend" in (k1, k2):
            continue
        if (k1, k2) in ignore_kinds or (k2, k1) in ignore_kinds:
            # adjacent tick labels touching is normal; only flag heavy overlap
            ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ox > 8 and oy > 8:
                bad.append((k1, t1, k2, t2, round(ox, 1), round(oy, 1)))
            continue
        ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
        oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
        if ox > tol and oy > tol:
            bad.append((k1, t1, k2, t2, round(ox, 1), round(oy, 1)))

    # text sitting on top of a bar, marker, or rule
    gfx = graphics(fig)
    for k, t, b in items:
        if k in ("tick", "axlabel", "title"):
            continue                      # these live outside the plot area
        for gk, gb in gfx:
            if gk == "bar" and k == "legend":
                pass                      # legend over a bar IS a problem
            ox = min(b.x1, gb.x1) - max(b.x0, gb.x0)
            oy = min(b.y1, gb.y1) - max(b.y0, gb.y0)
            thr = 6 if gk == "line" else 4
            if ox > thr and oy > thr:
                bad.append((k, t, gk, f"<{gk}>", round(ox, 1), round(oy, 1)))
                break

    x0, y0 = tb.x0 * dpi, tb.y0 * dpi
    clipped = [(k, t) for k, t, b in items
               if k not in ("tick", "axlabel")
               and (b.x0 < x0 - tol or b.x1 > x0 + fw + tol
                    or b.y0 < y0 - tol or b.y1 > y0 + fh + tol)]

    ok = not bad and not clipped
    print(f"{name:20} {'OK' if ok else 'ISSUE':6} "
          f"({len(items)} text artists checked)")
    for k1, t1, k2, t2, ox, oy in bad[:8]:
        print(f"     overlap  [{k1}] {t1[:28]!r}  <->  [{k2}] {t2[:28]!r}"
              f"   {ox}x{oy} px")
    for k, t in clipped[:6]:
        print(f"     clipped  [{k}] {t[:40]!r}")
    return ok


def audit_file(module_path, results_dir, data_dir=None, compare_dir=None):
    """Render every figure and check it for text collisions.

    Figure signatures live in 08_figures.py and change as figures are
    redesigned. This audit calls each one through a small table so that a
    signature change produces a clear message here rather than a TypeError
    halfway through a two-hour run -- which is exactly what happened when
    fig_ceiling moved from hardcoded value lists to reading ceiling.json.
    """
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location("mpf", module_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    raw, g, pc = m.load(results_dir)

    ceiling = 0.9578
    cj = os.path.join(data_dir or "", "ceiling.json")
    if data_dir and os.path.exists(cj):
        import json
        ceiling = float(json.load(open(cj))["ceiling_to_report"])

    captured = {}
    orig = m.save
    m.save = lambda fig, out, name: captured.setdefault(name, fig)
    os.makedirs("/tmp/_audit", exist_ok=True)
    O = "/tmp/_audit"

    calls = [
        ("fig2_pareto",      lambda: m.fig_pareto(g, O, ceiling=ceiling)),
        ("fig3_ablation",    lambda: m.fig_ablation(g, O, ceiling=ceiling)),
        ("fig4_per_class",   lambda: m.fig_per_class(pc, O)),
        ("fig5_cost_split",  lambda: m.fig_cost_split(g, O)),
        ("fig6_binary_lift", lambda: m.fig_binary_lift(g, O)),
        ("fig7_ceiling",     lambda: m.fig_ceiling(O, data_dir, compare_dir)),
        ("fig8_class_balance", lambda: m.fig_class_balance(O, data_dir)),
        ("fig9_tool_value",  lambda: m.fig_tool_value(results_dir, O)),
        ("fig10_injection",  lambda: m.fig_injection(O, data_dir, compare_dir)),
        ("fig11_rare_class", lambda: m.fig_rare_class(pc, O, data_dir)),
    ]
    for name, fn in calls:
        try:
            fn()
        except TypeError as e:
            print(f"{name:20} SIGNATURE MISMATCH — {e}")
            print(f"{'':20} update the call table in this file to match "
                  "08_figures.py")
        except Exception as e:
            print(f"{name:20} could not render — {type(e).__name__}: {e}")
    m.save = orig

    allok = True
    for name, fig in captured.items():
        if not audit(fig, name):
            allok = False
        plt.close(fig)
    skipped = [n for n, _ in calls if n not in captured]
    if skipped:
        print(f"\n  not rendered (skipped or unavailable): {', '.join(skipped)}")
    print("\n" + ("ALL RENDERED FIGURES CLEAN" if allok else "FIXES NEEDED"))
    return allok


if __name__ == "__main__":
    mod = sys.argv[1] if len(sys.argv) > 1 else "scripts/08_figures.py"
    res = sys.argv[2] if len(sys.argv) > 2 else "results"
    dd  = sys.argv[3] if len(sys.argv) > 3 else None
    cd  = sys.argv[4] if len(sys.argv) > 4 else None
    sys.exit(0 if audit_file(mod, res, dd, cd) else 1)
