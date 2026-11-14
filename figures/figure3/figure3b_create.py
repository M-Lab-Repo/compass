"""Figure 3b, "Why the deployed model drifts": per acceptance rule, the fraction of trajectories
whose deployed model changed in each round (left) and the mean number of latent sample sites
in the deployed model (right); round 0 is the starting model.

    python figure3/figure3b_create.py bench/arms/e2_anthropic/drift --out fig3b.png

Options
  DRIFT_DIR        one directory of trajectories (positional, required)
  --out PNG        the panel image (required; a .pdf name gives a vector file with editable text)
  --json PATH      also write the counted numbers as JSON

Input: trajectories (figure3_common.load_trajectories). No held-out score is needed, so this
panel reads the raw records alone. "Changed" is the round's ``changed`` flag: the rule accepted
a candidate that differs from the current model. The site count of round 0 is
figure3_common.count_latent_sites of the starting program; later rounds read the
``n_sample_sites`` the run stored for the deployed model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

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

def _mean(xs) -> float:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else math.nan


def count(drift_dir) -> dict:
    """{rule: {rounds, frac_changed, sites, n, changed, n_rounds}} in the house rule order, round 0 included."""
    by_rule: dict[str, list[dict]] = {}
    for rec in f3.load_trajectories(drift_dir):
        by_rule.setdefault(rec["rule"], []).append(rec)
    out = {}
    for rule in common.rule_order(by_rule):
        recs = by_rule[rule]
        rounds = sorted({int(r["round"]) for rec in recs for r in rec["rounds"]})
        frac = [0.0]
        sites = [_mean(f3.count_latent_sites(rec.get("baseline0_code") or "") for rec in recs)]
        for g in rounds:
            at = [r for rec in recs for r in rec["rounds"] if int(r["round"]) == g]
            frac.append(sum(bool(r.get("changed")) for r in at) / len(at))
            sites.append(_mean(r.get("n_sample_sites") for r in at))
        out[rule] = {"rounds": [0, *rounds], "frac_changed": frac, "sites": sites, "n": len(recs),
                     "changed": sum(bool(r.get("changed")) for rec in recs for r in rec["rounds"]),
                     "n_rounds": sum(len(rec["rounds"]) for rec in recs)}
    return out


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

def print_table(numbers: dict, drift_dir) -> None:
    print(f"figure3b  rounds that changed the deployed model and its latent sites; {Path(drift_dir).resolve().parent.name}")
    for rule, d in numbers.items():
        print(f"{rule:12s} n = {d['n']}  changed {d['changed']}/{d['n_rounds']} rounds;  fraction by round "
              + " ".join(f"{x:.2f}" for x in d["frac_changed"]))
        print(f"{'':12s}        sites by round " + " ".join("nan" if math.isnan(s) else f"{s:.2f}" for s in d["sites"]))


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

def draw(numbers: dict, drift_dir, out_png) -> None:
    fig = Figure(figsize=(6.8, 3.4))
    FigureCanvasAgg(fig)
    ax_l, ax_r = fig.subplots(1, 2)
    for rule, d in numbers.items():
        col, ls = common.rule_color(rule), "--" if rule == "control" else "-"
        ax_l.plot(d["rounds"], d["frac_changed"], color=col, lw=1.6, ls=ls, marker="o", ms=3.2,
                  label=f3.legend_name(rule))
        ax_r.plot(d["rounds"], d["sites"], color=col, lw=1.6, ls=ls, marker="o", ms=3.2)
    rounds = sorted({k for d in numbers.values() for k in d["rounds"]})
    for ax in (ax_l, ax_r):
        ax.set_xticks(rounds)
        ax.set_xlabel("refinement round")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.margins(x=0.08)
    ax_l.set_ylim(-0.04, 1.08)
    ax_l.set_ylabel("fraction of trajectories whose\ndeployed model changed")
    ax_l.set_title("Does the deployed model change?", loc="left", fontsize=10, fontweight="bold")
    ax_r.set_ylim(bottom=0)
    ax_r.set_ylabel("latent quantities in the deployed\nmodel (mean over problems)")
    ax_r.set_title("How large does it grow?", loc="left", fontsize=10, fontweight="bold")
    referee = next((r for r in ("referee", "mcmc") if r in numbers), None)
    if referee is not None:  # the referee's line is flat: say so on the figure, not only in the legend
        d, col = numbers[referee], common.rule_color(referee)
        ax_l.text(rounds[-1], d["frac_changed"][-1] + 0.04, f"Referee: changed in {d['changed']} of {d['n_rounds']} rounds",
                  color=col, fontsize=6.5, ha="right", va="bottom")
        flat = all(abs(s - d["sites"][0]) < 1e-9 for s in d["sites"] if not math.isnan(s))
        note = (f"Referee: flat at {d['sites'][0]:.2f}, the starting models' mean" if flat
                else f"Referee: {d['sites'][0]:.2f} to {d['sites'][-1]:.2f}")
        ax_r.text(rounds[-1], d["sites"][-1] - 0.05 * ax_r.get_ylim()[1], note, color=col, fontsize=6.5, ha="right", va="top")
    fig.text(0.02, 0.975, "Why the deployed model drifts", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=len(numbers), fontsize=8, frameon=False)
    fig.text(0.99, 0.11, f"measured: {Path(drift_dir).resolve().parent.name}; round 0 = the starting model",
             ha="right", va="bottom", fontsize=6.5, color="#666666", style="italic")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.82, bottom=0.3, wspace=0.4)
    fig.savefig(out_png, dpi=200)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drift_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    numbers = count(args.drift_dir)
    if not numbers:
        raise SystemExit(f"no trajectories in {args.drift_dir}")
    print_table(numbers, args.drift_dir)
    draw(numbers, args.drift_dir, args.out)
    print(f"wrote {args.out}")
    if args.json:
        Path(args.json).write_text(json.dumps(numbers, indent=2))
    return numbers


if __name__ == "__main__":
    main()
