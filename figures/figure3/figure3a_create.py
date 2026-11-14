"""Figure 3a, "Deployed model over successive rounds": how far the deployed model drifts from
the starting model, round by round, under each acceptance rule. With --symlog it also draws
panel e (SI), the same view on the real-data trajectories.

    python figure3/figure3a_create.py bench/arms/e2_anthropic/drift --out fig3a.png
    python figure3/figure3a_create.py bench/arms/e2_anthropic_real/drift --out fig3e.png --symlog

Options
  DRIFT_DIR        one directory of trajectories (positional, required)
  --out PNG        the panel image (required; a .pdf name gives a vector file with editable text)
  --symlog         logarithmic axis beyond ±5 nats (real starts mix a +1,200-nat fix with ±1-nat wandering)
  --json PATH      also write the counted numbers as JSON

Input: trajectories (figure3_common.load_trajectories). A thin line is one trajectory: the Δ log
predictive density of the model deployed after each round against the starting model, on the
fresh draw for synthetic data and on the hidden fold for real data (figure3_common.round_delta);
round 0 is 0. A round whose score is unavailable is left out, never drawn as zero, and a
trajectory with no scored round is dropped; both are counted and printed. Thick line = median
over the rule's trajectories, band = interquartile range; the legend carries the final median.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator

try:
    import figure3_common as f3  # run as a file from the repo root
except ImportError:
    from figure3 import figure3_common as f3  # run as a module / imported by tests
common = f3.common

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------

def count(drift_dir) -> tuple[dict, dict]:
    """(numbers, {rule: [{round: Δ} per trajectory]}) in the house rule order."""
    by_rule, dropped, dropped_rounds = {}, 0, 0
    for rec in f3.load_trajectories(drift_dir):
        dropped_rounds += sum(f3.round_delta(r, rec) is None for r in rec["rounds"])
        pts = f3.series(rec)
        if len(pts) == 1:  # no scored round at all
            dropped += 1
            continue
        by_rule.setdefault(rec["rule"], []).append(pts)
    numbers = {"rules": {}, "n_dropped": dropped, "n_dropped_rounds": dropped_rounds}
    for rule in common.rule_order(by_rule):
        series = by_rule[rule]
        rounds = sorted({k for s in series for k in s})
        at = {k: [s[k] for s in series if k in s] for k in rounds}
        numbers["rules"][rule] = {"n": len(series), "rounds": rounds,
                                  "median": [float(np.median(at[k])) for k in rounds],
                                  "q1": [float(np.percentile(at[k], 25)) for k in rounds],
                                  "q3": [float(np.percentile(at[k], 75)) for k in rounds]}
    return numbers, {rule: by_rule[rule] for rule in numbers["rules"]}


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

def print_table(numbers: dict, drift_dir) -> None:
    rounds = sorted({k for d in numbers["rules"].values() for k in d["rounds"]})
    print(f"figure3a  Δ of the deployed model vs. the starting model (nats), median by round; "
          f"{Path(drift_dir).resolve().parent.name}")
    print(f"{'rule':12s} {'n':>3s}  " + " ".join(f"{'r' + str(k):>7s}" for k in rounds))
    for rule, d in numbers["rules"].items():
        by_round = dict(zip(d["rounds"], d["median"], strict=True))
        print(f"{rule:12s} {d['n']:3d}  " + " ".join(f"{by_round[k]:+7.2f}" if k in by_round else f"{'-':>7s}"
                                                    for k in rounds))
    print(f"left out: {numbers['n_dropped']} trajectories without any score, "
          f"{numbers['n_dropped_rounds']} rounds without a score")


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

def draw(numbers: dict, series_by_rule: dict, drift_dir, out_png, *, symlog: bool = False) -> None:
    rules = list(numbers["rules"])
    fig = Figure(figsize=(4.6, 3.3 + 0.2 * len(rules)))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for rule in rules:
        col, ls = common.rule_color(rule), "--" if rule == "control" else "-"
        for s in series_by_rule[rule]:
            ks = sorted(s)
            ax.plot(ks, [s[k] for k in ks], color=col, lw=0.6, alpha=0.3, ls=ls)
        d = numbers["rules"][rule]
        ax.fill_between(d["rounds"], d["q1"], d["q3"], color=col, alpha=0.12, lw=0)
        ax.plot(d["rounds"], d["median"], color=col, lw=2.2, ls=ls,
                label=f"{f3.legend_name(rule)} (n = {d['n']}), final {d['median'][-1]:+.2f}")
    ax.axhline(0, color="#c3c2b7", lw=0.7)
    ax.set_xlabel("refinement round")
    ax.set_ylabel("Δ log predictive density\nvs. the starting model (nats)")
    if symlog:
        ax.set_yscale("symlog", linthresh=5.0, linscale=1.0)
        ticks = [-10, 0, 10, 100, 1000]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:+d}" if t else "0" for t in ticks])
        ax.yaxis.set_minor_locator(NullLocator())
    ax.set_title("Deployed model over successive rounds", loc="left", fontsize=11, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0.08)
    legend_h = 0.2 * len(rules) + 0.1                       # inches, below the x label
    fig.legend(loc="lower center", bbox_to_anchor=(0.55, 0.3 / fig.get_figheight()), ncol=1, fontsize=8, frameon=False)
    fig.text(0.99, 0.01, f"thin = one problem, thick = median, band = IQR; {Path(drift_dir).resolve().parent.name}\n"
             "Δ on the fresh draw (synthetic) or the hidden fold (real data)",
             ha="right", va="bottom", fontsize=6.5, color="#666666", style="italic")
    fig.subplots_adjust(left=0.2, right=0.97, top=0.9, bottom=(legend_h + 0.85) / fig.get_figheight())
    fig.savefig(out_png, dpi=200)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drift_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--symlog", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    numbers, series_by_rule = count(args.drift_dir)
    if not numbers["rules"]:
        raise SystemExit(f"no scored trajectories in {args.drift_dir}")
    print_table(numbers, args.drift_dir)
    draw(numbers, series_by_rule, args.drift_dir, args.out, symlog=args.symlog)
    print(f"wrote {args.out}")
    if args.json:
        Path(args.json).write_text(json.dumps(numbers, indent=2))
    return numbers


if __name__ == "__main__":
    main()
