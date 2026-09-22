"""The README's charts, from the saved results.

    python bench/plot_readme.py            # writes docs/charts/*-light.png and *-dark.png

Inputs (all in bench/results/, all written by the bench scripts, never by hand):
  jevbench-<tier>-<system>.json   bench/jevbench.py --out      per-item accuracy
  speed-<system>.json             bench/speed.py --label       latency and throughput

Systems drawn, in fixed colour order: rev on Qwen3.8-27B (blue), Jev (orange),
rev on Qwen3.5-2B on an M2 Pro (green). Three series is the most this palette
seats on one chart; add a fourth as a new chart, not a new colour.

Style: paper and ink, and the only colour on the page is data (the same rule
as Parley, whose tokens these are). IBM Plex: Serif for the title, Sans for
text, Mono for numbers. The fonts are fetched once into ~/.cache/rev/fonts
(OFL, from the google/fonts repo); without them the charts fall back to
DejaVu and say so.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "results"
OUT = HERE.parent / "docs" / "charts"

SYSTEMS = [  # key, label, results suffix, speed file
    ("27b", "rev, Qwen3.8-27B served by sglang (2x L40S)", "27b", "speed-remote-logprob.json"),
    ("jev", "Jev 1.13 (api.typesafe.ai)", "jev", "speed-jev.json"),
    ("2b", "rev, Qwen3.5-2B in-process (M2 Pro, 16 GB)", "2b", "speed-rev.json"),
]
# Parley's tokens: paper / ink, rules, and its first three speaker hues for the
# data. The dark hues are one step deeper than Parley's own so they sit in the
# validated lightness band on the dark paper.
THEMES = {
    "light": {"surface": "#f5f6f7", "text": "#16181d", "muted": "#6c7079", "grid": "#dfe2e6",
              "range": "#c9ced4", "series": ["#3b5bdb", "#e8590c", "#087f5b"]},
    "dark":  {"surface": "#131519", "text": "#e6e8eb", "muted": "#969ba3", "grid": "#2a2e35",
              "range": "#3b4048", "series": ["#5c7cfa", "#e8590c", "#0ca678"]},
}

FONT_DIR = Path(os.path.expanduser("~/.cache/rev/fonts"))
FONT_FILES = {  # file name -> where google/fonts keeps it (OFL)
    "IBMPlexSerif-SemiBold.ttf": "ofl/ibmplexserif/IBMPlexSerif-SemiBold.ttf",
    "IBMPlexMono-Regular.ttf": "ofl/ibmplexmono/IBMPlexMono-Regular.ttf",
    "IBMPlexSans-Variable.ttf": "ofl/ibmplexsans/IBMPlexSans%5Bwdth%2Cwght%5D.ttf",
}
FONTS = {"title": "IBM Plex Serif", "text": "IBM Plex Sans", "num": "IBM Plex Mono"}


def ensure_fonts() -> bool:
    """Register IBM Plex with matplotlib, fetching it on first use. False = fallback."""
    import matplotlib.font_manager as fm
    try:
        FONT_DIR.mkdir(parents=True, exist_ok=True)
        for name, path in FONT_FILES.items():
            f = FONT_DIR / name
            if not f.exists():
                urllib.request.urlretrieve(f"https://raw.githubusercontent.com/google/fonts/main/{path}", f)
        # Sans ships as a variable font; matplotlib wants one file per weight.
        if not (FONT_DIR / "IBMPlexSans-SemiBold.ttf").exists():
            from fontTools.ttLib import TTFont
            from fontTools.varLib.instancer import instantiateVariableFont
            for w, inst in ((400, "Regular"), (500, "Medium"), (600, "SemiBold")):
                font = TTFont(FONT_DIR / "IBMPlexSans-Variable.ttf")
                instantiateVariableFont(font, {"wght": w, "wdth": 100}).save(FONT_DIR / f"IBMPlexSans-{inst}.ttf")
        for f in FONT_DIR.glob("IBMPlex*-*.ttf"):
            if "Variable" not in f.name:
                fm.fontManager.addfont(str(f))
        return True
    except Exception as e:                      # noqa: BLE001
        print(f"[plot] IBM Plex unavailable ({e}); using DejaVu", file=sys.stderr)
        FONTS.update(title="DejaVu Serif", text="DejaVu Sans", num="DejaVu Sans Mono")
        return False
FAMILY_LABEL = {
    "adversarial": "adversarial", "ambiguous": "ambiguous", "judge_hard": "judge",
    "long_policy": "long policy", "multi_hop": "multi-hop", "probability": "probability",
    "routing_hard": "routing", "temporal_numeric": "dates & arithmetic",
    "tradeoff": "trade-off", "trap": "trap",
}


def load_json(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=10, length=0)
    ax.grid(axis="x", color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_family(FONTS["num"])


def figure(t, height):
    fig = plt.figure(figsize=(11, height), dpi=180, facecolor=t["surface"])
    plt.rcParams.update({"font.family": FONTS["text"], "text.color": t["text"],
                         "axes.labelcolor": t["muted"]})
    return fig


def title(ax, t, main, sub, y=1.16, dy=0.075):
    """Title in serif, subtitle in sans, both flush with the plot's left edge.
    `y` is axes-relative; tall plots pass a smaller one so it stays on the page."""
    ax.text(0, y, main, transform=ax.transAxes, ha="left", fontsize=15, color=t["text"],
            family=FONTS["title"], weight=600)
    ax.text(0, y - dy, sub, transform=ax.transAxes, ha="left", fontsize=9.5, color=t["muted"])


def num(ax, x, y, text, t, **kw):
    """A number in mono, in ink, never in the series colour."""
    kw.setdefault("fontsize", 9); kw.setdefault("color", t["muted"])
    ax.text(x, y, text, family=FONTS["num"], **kw)


def dot(ax, x, y, color, t, z=3):
    ax.plot([x], [y], "o", ms=9, color=color, markeredgecolor=t["surface"], markeredgewidth=2, zorder=z)


def legend(fig, t, labels, colors, x=0.5, y=0.02):
    handles = [plt.Line2D([], [], marker="o", ls="", ms=9, color=c, markeredgecolor=t["surface"],
                          markeredgewidth=2) for c in colors]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(x, y), ncol=len(labels),
               frameon=False, fontsize=10, labelcolor=t["muted"], handletextpad=0.4, columnspacing=1.8)


# --- 1. accuracy: tiers, then the hard families --------------------------------
def accuracy(theme):
    t = THEMES[theme]
    data = {k: {tier: load_json(f"jevbench-{tier}-{suf}.json") for tier in ("easy", "standard", "hard")}
            for k, _, suf, _ in SYSTEMS}
    rows = [("easy (48)", lambda d: d["easy"]["accuracy"]),
            ("standard (72)", lambda d: d["standard"]["accuracy"]),
            ("hard (111)", lambda d: d["hard"]["accuracy"])]
    fams = sorted(data["27b"]["hard"]["by_family"], key=lambda f: -data["27b"]["hard"]["by_family"][f])
    n = data["27b"]["hard"]["n_by_family"]
    frows = [(f"{FAMILY_LABEL.get(f, f)} ({n[f]})", (lambda f: lambda d: d["hard"]["by_family"][f])(f)) for f in fams]

    fig = figure(t, 8.4)
    ax = fig.add_axes([0.24, 0.09, 0.72, 0.78])
    style(ax, t)
    ys, labels = [], []
    y = 0
    for label, get in rows + [(None, None)] + frows:
        if label is None:
            y -= 0.6; continue
        vals = {k: get(data[k]) for k, _, _, _ in SYSTEMS if all(data[k].values())}
        lo, hi = min(vals.values()), max(vals.values())
        ax.plot([lo * 100, hi * 100], [y, y], color=t["range"], linewidth=3, solid_capstyle="round", zorder=1)
        for j, (k, _, _, _) in enumerate(SYSTEMS):
            if k not in vals:
                continue
            # Two systems at the same value would hide one another: split them
            # a little around the row so both stay visible.
            twins = [kk for kk in vals if kk != k and abs(vals[kk] - vals[k]) < 0.005]
            dy = 0.0
            if twins:
                order = [kk for kk, _, _, _ in SYSTEMS if kk in vals and (kk == k or kk in twins)]
                dy = (len(order) - 1) / 2 * 0.24 - order.index(k) * 0.24
            dot(ax, vals[k] * 100, y + dy, t["series"][j], t, z=4 if k == "27b" else 3)
        # direct labels: the served 27B above, Jev below, only where they differ from each other
        a, b = vals.get("27b"), vals.get("jev")
        if a is not None:
            num(ax, a * 100, y + 0.40, f"{a*100:.0f}", t, ha="center", va="bottom")
        if b is not None and (a is None or abs(a - b) > 0.005):
            num(ax, b * 100, y - 0.40, f"{b*100:.0f}", t, ha="center", va="top")
        ys.append(y); labels.append(label)
        y -= 1
    ax.set_xlim(15, 104); ax.set_ylim(y + 0.2, 0.9)
    ax.set_xticks([20, 40, 60, 80, 100]); ax.set_xticklabels([f"{v}%" for v in (20, 40, 60, 80, 100)])
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=10.5, color=t["text"])
    for lab in ax.get_yticklabels():
        lab.set_family(FONTS["text"])
    title(ax, t, "Where a frozen Qwen3.8-27B matches Jev, and where it does not",
          "Accuracy on JevBench's 231 public items, per tier and then per hard family. "
          "Same items for every system. 2026-09-22.", y=1.09, dy=0.04)
    ax.axhline(-2.8, color=t["grid"], linewidth=1)
    ax.text(16, -3.05, "hard tier by family", fontsize=9, color=t["muted"], va="top")
    legend(fig, t, [s[1] for s in SYSTEMS], t["series"])
    fig.savefig(OUT / f"accuracy-{theme}.png", facecolor=t["surface"])
    plt.close(fig)


# --- 2. latency: p50 with the p95 reach ----------------------------------------
def latency(theme):
    t = THEMES[theme]
    speed = {k: load_json(f)["results"] for k, _, _, f in SYSTEMS}
    kinds = [("short: clipboard paste", "short, ~230 tokens"),
             ("medium: JevBench standard", "medium, ~160 tokens"),
             ("long: JevBench hard", "long document, 1-4k tokens"),
             ("long: three questions per request", "long, three questions at once")]
    fig = figure(t, 4.6)
    ax = fig.add_axes([0.26, 0.17, 0.70, 0.62])
    style(ax, t)
    ys, labels = [], []
    for i, (key, label) in enumerate(kinds):
        for j, (k, _, _, _) in enumerate(SYSTEMS):
            y = -i * 1.0 - (j - 1) * 0.26
            r = speed[k][key]
            p50, p95 = r["p50_ms"], r["p95_ms"]
            ax.plot([p50, p95], [y, y], color=t["series"][j], linewidth=2, alpha=0.45, zorder=2,
                    solid_capstyle="round")
            dot(ax, p50, y, t["series"][j], t)
            num(ax, p95 * 1.08, y, f"{p50} / {p95} ms", t, va="center", fontsize=8.5)
        ys.append(-i); labels.append(label)
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(90, 9000)
    ax.set_xticks([100, 300, 1000, 3000]); ax.set_xticklabels(["100 ms", "300 ms", "1 s", "3 s"])
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=10.5, color=t["text"])
    for lab in ax.get_yticklabels():
        lab.set_family(FONTS["text"])
    ax.set_ylim(-len(kinds) + 0.45, 0.55)
    title(ax, t, "One question, end to end: median, with the line reaching p95",
          "Same client, same items. The 27B is two network hops away on a cluster; "
          "Jev via its API; the 2B in-process on an M2 Pro.")
    legend(fig, t, [s[1] for s in SYSTEMS], t["series"])
    fig.savefig(OUT / f"latency-{theme}.png", facecolor=t["surface"])
    plt.close(fig)


# --- 3. the gate: coverage against accuracy on hard ----------------------------
def gate(theme):
    t = THEMES[theme]
    fig = figure(t, 5.0)
    ax = fig.add_axes([0.10, 0.22, 0.86, 0.58])
    style(ax, t)
    ax.grid(axis="y", color=t["grid"], linewidth=1)
    for j, (k, label, suf, _) in enumerate(SYSTEMS):
        d = load_json(f"jevbench-hard-{suf}.json")
        if d is None:
            continue
        items = sorted(d["items"], key=lambda r: -r["confidence"])
        hit = np.array([r["choice"] == r["gold"] for r in items], dtype=float)
        cov = np.arange(1, len(items) + 1) / len(items)
        acc = np.cumsum(hit) / np.arange(1, len(items) + 1)
        start = int(0.1 * len(items))          # below ten items one flip moves the line 10 points
        ax.plot(cov[start:] * 100, acc[start:] * 100, color=t["series"][j], linewidth=2,
                solid_joinstyle="round", solid_capstyle="round", zorder=3 if k != "27b" else 4)
        # the point the README quotes: everything at confidence >= 0.9
        m = np.array([r["confidence"] >= 0.9 for r in items])
        if m.any():
            c90, a90 = m.mean(), hit[m].mean()
            dot(ax, c90 * 100, a90 * 100, t["series"][j], t, z=5)
            short = {"27b": "27B", "jev": "Jev", "2b": "2B"}[k]
            dx, dy, va = {"27b": (1.5, 1.2, "bottom"), "jev": (-1.5, -5.5, "top"), "2b": (1.5, 1.2, "bottom")}[k]
            ha = "right" if k == "jev" else "left"
            ax.text(c90 * 100 + dx, a90 * 100 + dy,
                    f"{short}: at 0.9 answers {c90*100:.0f}%, {a90*100:.0f}% of them right",
                    fontsize=8.5, color=t["muted"], va=va, ha=ha)
    ax.set_xlim(8, 100); ax.set_ylim(50, 101)
    ax.set_xticks([25, 50, 75, 100]); ax.set_xticklabels(["25%", "50%", "75%", "100%"])
    ax.set_yticks([50, 60, 70, 80, 90, 100]); ax.set_yticklabels(["50%", "60%", "70%", "80%", "90%", "100%"])
    ax.set_xlabel("share of hard items answered (most confident first)", fontsize=9.5)
    ax.set_ylabel("accuracy on what was answered", fontsize=9.5)
    title(ax, t, "What a confidence gate buys on the hard tier",
          "Items sorted by each system's own confidence. Act above a threshold, "
          "hand the rest to a person; the dot is the 0.9 gate.")
    legend(fig, t, [s[1] for s in SYSTEMS], t["series"])
    fig.savefig(OUT / f"gate-{theme}.png", facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    ensure_fonts()
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        accuracy(theme); latency(theme); gate(theme)
    print(f"-> {OUT}")
