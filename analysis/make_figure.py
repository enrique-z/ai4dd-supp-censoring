#!/usr/bin/env python3
"""Figure 1 -- the S4 decomposition, which the tables cannot show.

The manuscript reports S4 as four aggregate numbers and two correlations. Those
hide the two things a reviewer most needs to see: how wide the per-target spread
is behind each mean, and that the pre-registered P3 correlation genuinely
changes sign between the unmatched and the size-matched contrast rather than
merely weakening.

Panel A: per-target RMSE differences for the three contrasts, all 158 targets.
Naive minus exact-only is mostly negative -- keeping censored rows helps. The
size-matched mixture minus exact-only is mostly positive -- at equal n the same
rows hurt. Tobit minus naive is positive almost everywhere. The mean and its
95% bootstrap CI over targets are drawn from the ledger, so the summary marks
cannot drift from the text.

Panel B: the sign flip itself. Censored fraction against the per-target
difference, for the unmatched and the size-matched contrast, with a least
squares line each. The lines cross: the more censored a target, the more
deletion costs (unmatched, rho<0) and simultaneously the more the wrong labels
cost at fixed n (matched, rho>0). Two effects of opposite sign, which is the
paper's actual claim.

Per-target points are recomputed here from S4_downstream.targets in the raw
evidence; every aggregate mark is read from the ledger. The script asserts the
two agree before drawing, so a rerun that moves the evidence fails here rather
than producing a figure that disagrees with the prose.

Type-3 fonts are a silent desk-reject at many venues and matplotlib's PDF
backend emits them by default, so `pdf.fonttype` is forced to 42 (TrueType).

Usage: python3 analysis/make_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
import matplotlib.lines
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42     # never Type 3 -- see docstring
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent / "evidence"
RAW = EVIDENCE / "censoring_raw.json"
LEDGER = EVIDENCE / "censoring_ledger.jsonl"
OUT = HERE.parent / "fig_s4.pdf"
# Claims this figure draws. guard.py reads this so that a ledger row
# REPORTED IN A FIGURE counts as reported, instead of forcing every plotted
# number into a redundant table beside its own figure. The link is auditable:
# this file is written by the same script that reads the ledger and draws them.
CLAIMS_OUT = EVIDENCE / "figure_claims.json"

# (ledger stem, per-target field a, per-target field b, label)
# each contrast is a - b, per target
CONTRASTS = [
    ("naive_minus_exact",  "rmse_naive",           "rmse_exact_only",
     "naive $-$ exact-only"),
    ("matched_minus_exact", "rmse_naive_matched",  "rmse_exact_only",
     "size-matched $-$ exact-only"),
    ("tobit_minus_naive",  "rmse_censoring_aware", "rmse_naive",
     "Tobit $-$ naive"),
]

TOL = 5e-4          # ledger values are rounded to 4 dp
RHO_TOL = 2e-3      # ledger rho values are rounded to 3 dp


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho with average ranks for ties; no scipy dependency."""
    def rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average ranks within tie groups
        sv = v[order]
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


def main() -> int:
    L = {r["claim_id"]: r for r in
         (json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip())}
    targets = json.load(RAW.open())["S4_downstream"]["targets"]

    drawn: list[str] = []

    def led(cid: str) -> float:
        drawn.append(cid)
        return float(L[cid]["value"])

    n_targets = int(led("s4::n_targets"))
    if len(targets) != n_targets:
        raise SystemExit(
            f"target count drift: raw evidence has {len(targets)}, "
            f"ledger s4::n_targets says {n_targets}")

    cens = np.array([t["censored_frac"] for t in targets], dtype=float)

    diffs: dict[str, np.ndarray] = {}
    for stem, fa, fb, _label in CONTRASTS:
        d = np.array([t[fa] - t[fb] for t in targets], dtype=float)
        diffs[stem] = d
        want = led(f"s4::{stem}_mean")
        got = float(d.mean())
        if abs(got - want) > TOL:
            raise SystemExit(
                f"{stem}: per-target mean {got:.6f} disagrees with ledger "
                f"{want} (tol {TOL}). Evidence and ledger are out of sync.")

    # the pre-registered P3 correlations, reproduced from the same per-target rows
    for stem, cid in (("naive_minus_exact", "s4::p3_rho"),
                      ("matched_minus_exact", "s4::p3_rho_matched")):
        want = led(cid)
        got = spearman(cens, diffs[stem])
        if abs(got - want) > RHO_TOL:
            raise SystemExit(
                f"{cid}: recomputed Spearman {got:.6f} disagrees with ledger "
                f"{want} (tol {RHO_TOL}).")

    fig = plt.figure(figsize=(5.5, 2.05), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.03, hspace=0.03)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0])
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1], sharex=axB)

    # ---------------------------------------------------------------- panel A
    # Clipped to the bulk so the bootstrap CIs are legible; the number of
    # targets falling outside is stated in the figure caption (computed here
    # and printed below so the caption literal can be verified).
    XLIM = 0.5
    rng = np.random.default_rng(0)
    n_out = 0
    for i, (stem, _fa, _fb, _label) in enumerate(CONTRASTS):
        d = diffs[stem]
        keep = np.abs(d) <= XLIM
        n_out += int((~keep).sum())
        # points outside the window are dropped, not clipped: piling them on
        # the spine would read as a real cluster of targets at +/- XLIM.
        axA.scatter(d[keep],
                    i + rng.uniform(-0.15, 0.15, size=int(keep.sum())),
                    s=3.2, alpha=0.55, color="0.20", linewidths=0,
                    rasterized=True, zorder=2)
        m = float(L[f"s4::{stem}_mean"]["value"])
        lo = float(L[f"s4::{stem}_lo"]["value"])
        hi = float(L[f"s4::{stem}_hi"]["value"])
        drawn += [f"s4::{stem}_lo", f"s4::{stem}_hi"]
        axA.plot([lo, hi], [i - 0.34, i - 0.34], color="black", lw=1.4,
                 marker="|", ms=4.5, mew=1.4, zorder=4)
        axA.plot([m], [i - 0.34], marker="D", ms=3.4, color="black",
                 mfc="white", mew=1.1, zorder=5)

    axA.axvline(0.0, color="0.15", lw=0.8, ls="--", zorder=1)
    axA.set_yticks(range(len(CONTRASTS)))
    axA.set_yticklabels([c[3] for c in CONTRASTS], fontsize=9)
    axA.set_ylim(len(CONTRASTS) - 0.55, -0.62)
    axA.set_xlim(-XLIM, XLIM)
    axA.set_xlabel("per-target RMSE difference (log units)", fontsize=9)
    axA.tick_params(axis="x", labelsize=9)
    axA.tick_params(axis="y", length=0)
    axA.set_title(f"All {n_targets} targets, three contrasts",
                  fontsize=9, pad=4)
    for s in ("top", "right", "left"):
        axA.spines[s].set_visible(False)

    # ------------------------------------------------------------ panels B, C
    # Split rather than overlaid: the unmatched contrast spans a much wider
    # range, and on a shared y-axis it visually flattens the size-matched
    # slope into a line that looks null when it is not.
    for ax, (stem, rho_cid, colour, marker, label) in zip(
            (axB, axC),
            (("naive_minus_exact", "s4::p3_rho", "0.25", "o",
              "unmatched: naive $-$ exact-only"),
             ("matched_minus_exact", "s4::p3_rho_matched", "0.05", "^",
              "size-matched $-$ exact-only"))):
        d = diffs[stem]
        rho = float(L[rho_cid]["value"])
        ax.scatter(cens, d, s=5.0, alpha=0.60, color=colour, marker=marker,
                   linewidths=0, rasterized=True, zorder=2)
        b, a = np.polyfit(cens, d, 1)
        xs = np.linspace(cens.min(), cens.max(), 50)
        ax.plot(xs, a + b * xs, color=colour, lw=1.6, zorder=3)
        ax.axhline(0.0, color="0.15", lw=0.8, ls="--", zorder=1)
        # rho callout: unmatched (negative slope) → top-LEFT where cloud is
        # thin; size-matched (positive slope) → top-RIGHT which is empty.
        short = label.split(":")[0].strip() if ":" in label else label.split()[0]
        cx = 0.03 if b < 0 else 0.97
        cha = "left" if b < 0 else "right"
        ax.text(cx, 0.97, f"{short}\n$\\rho={rho:+.3f}$",
                transform=ax.transAxes, ha=cha, va="top", fontsize=9,
                color=colour, linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor=colour, linewidth=0.7, alpha=0.9))
        ax.tick_params(labelsize=9)
        if ax is axC:   # shared y-label on the lower panel only
            ax.set_ylabel("RMSE difference", fontsize=9, labelpad=2)
        ax.margins(y=0.16)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axB.set_title("The two effects have opposite sign",
                  fontsize=9, pad=4)
    axB.tick_params(axis="x", labelbottom=False)
    axC.set_xlabel("censored fraction of target", fontsize=9)

    # Resolve constrained layout before adding figure-level artists so that
    # axis positions and tight bounding boxes are final.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    _inv = fig.transFigure.inverted()

    posB = axB.get_position()
    posC = axC.get_position()

    # Panel letters placed at the top-left of each panel's FULL tight bounding
    # box (which includes titles, tick labels and axis labels).  get_tightbbox
    # returns display-pixel coordinates; we convert to figure [0-1] space.
    _letter_kw = dict(fontsize=11, fontweight="bold", fontfamily="sans-serif",
                      va="bottom", ha="left")
    for ax, letter in ((axA, "A"), (axB, "B")):
        tb = ax.get_tightbbox(renderer)
        fig_x, fig_y = _inv.transform((tb.x0, tb.y1))
        fig.text(fig_x, fig_y + 0.01, letter, **_letter_kw)

    # Light separator between the two right-column panels.
    sep_y = (posB.y0 + posC.y1) / 2
    fig.add_artist(
        matplotlib.lines.Line2D(
            [posB.x0, posB.x1], [sep_y, sep_y],
            transform=fig.transFigure, color="0.78", lw=0.8, clip_on=False))

    fig.savefig(OUT, bbox_inches="tight", dpi=400)
    plt.close(fig)

    CLAIMS_OUT.write_text(json.dumps(
        {"figure": OUT.name, "claim_ids": sorted(set(drawn))}, indent=1) + "\n")

    print(f"[figure] wrote {OUT}")
    print(f"[figure] declared {len(set(drawn))} claim ids -> {CLAIMS_OUT}")
    print(f"[figure] per-target means and both P3 correlations reproduce "
          f"the ledger within tolerance")
    print(f"[figure] n_out={n_out} of {3 * n_targets} points outside ±{XLIM} "
          f"(caption literal must match)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
