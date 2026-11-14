"""Figure 2d, "When nothing is wrong": how often each rule replaced a correctly specified model.

    python figure2/figure2d_create.py bench/arms/e5_anthropic --out fig2d.png
    python figure2/figure2d_create.py bench/arms/e5_anthropic bench/arms/e5_haiku ... --out fig2d_families.png

Options
  DIR ...          correct-baseline (E5) record directories: one gives panel d, several the families variant
  --out PNG        the panel image (required)
  --json PATH      also write the counted numbers as JSON

Input: correct-baseline (E5) record directories; one arm gives panel d, several give the
"across families" variant (one stem per arm within each rule). The unit is the run, not the
candidate: a reading rule replaces the model in a run when it accepts any candidate of that
run; the referee's decision is the run's own validation_passed. Runs whose pipeline crashed
are left out of every denominator; a run in which a rule returned no verdict at all leaves
that rule's denominator; a completed run with no runnable candidate counts as not replaced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

try:
    import figure2_common as common  # run as a file from the repo root
except ImportError:
    from figure2 import figure2_common as common  # run as a module / imported by tests

# --out foo.pdf (or .svg) gives a vector file; TrueType embedding keeps the text editable in Illustrator
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

K_ALPHA = 2 * common.BOUND_PCT  # the bound for a run that examines K = 2 candidates


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------

def count_arm(records: list[dict]) -> dict:
    """{"n_records", "n_crashed", "replaced": {rule: [runs, replaced]}} for one E5 arm."""
    if not records:
        raise ValueError("no records")
    if not all((r.get("flags") or {}).get("null_baseline") for r in records):
        raise ValueError("panel 2d needs correct-baseline (null_baseline) records")
    done = [r for r in records if (r.get("pipeline") or {}).get("ok", True)]
    if not done:
        raise ValueError("every pipeline crashed")
    rules = common.rule_order({k for r in done for c in r["pipeline"].get("candidates") or []
                               for k in (c.get("judges") or {})})
    replaced = {}
    for rule in rules:
        runs, n_replaced = 0, 0
        for r in done:
            cands = r["pipeline"].get("candidates") or []
            verdicts = [common.verdict(c, rule) for c in cands]
            if cands and all(v is None for v in verdicts):
                continue  # the rule never returned a verdict in this run
            runs += 1
            n_replaced += any(bool(v) for v in verdicts)
        replaced[rule] = [runs, n_replaced]
    replaced["referee"] = [len(done), sum(bool(r["pipeline"].get("validation_passed")) for r in done)]
    return {"n_records": len(done), "n_crashed": len(records) - len(done), "replaced": replaced}


def count(arm_dirs) -> dict:
    """{"rules": [...], "arms": {arm: count_arm(...)}, "k_alpha": 4.6}."""
    arms = {Path(d).name: count_arm(common.load_records(d)) for d in arm_dirs}
    rules = common.rule_order({r for a in arms.values() for r in a["replaced"] if r != "referee"})
    return {"rules": rules + ["referee"], "arms": arms, "k_alpha": K_ALPHA}


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

def print_table(numbers: dict) -> None:
    print("figure2d  runs that replaced a correct model (a rule with no verdict in a run drops that run)")
    arms = list(numbers["arms"])
    print(f"{'rule':40s} " + " ".join(f"{a:>16s}" for a in arms))
    for rule in numbers["rules"]:
        cells = []
        for a in arms:
            n, k = numbers["arms"][a]["replaced"].get(rule, [0, 0])
            cells.append(f"{k}/{n} {100 * k / n:3.0f}%" if n else "-")
        print(f"{rule:40s} " + " ".join(f"{c:>16s}" for c in cells))
    crashed = {a: numbers["arms"][a]["n_crashed"] for a in arms if numbers["arms"][a]["n_crashed"]}
    if crashed:
        print("crashed runs left out:", crashed)


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

def draw(numbers: dict, arm_dirs, out_png) -> None:
    names = {Path(d).name: common.arm_label(Path(d).name, common.load_records(d)) for d in arm_dirs}
    several = len(numbers["arms"]) > 1
    blocks = [(rule, [a for a in numbers["arms"] if numbers["arms"][a]["replaced"].get(rule, [0])[0]])
              for rule in numbers["rules"]]
    n_rows = sum(max(1, len(a)) for _, a in blocks) + len(blocks) - 1
    fig = Figure(figsize=(6.5, 0.3 * n_rows + 1.5))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    y, ticks, labels = 0, [], []
    for rule, arms in blocks:
        col = common.rule_color(rule)
        y0 = y
        for arm in arms or [None]:
            if arm is None:
                ax.text(K_ALPHA + 3, y, "no verdicts (0/0)", fontsize=7, va="center", color="#666666")
            else:
                n, k = numbers["arms"][arm]["replaced"][rule]
                pct = 100 * k / n
                ax.plot([0, pct], [y, y], color=col, lw=1.6, alpha=0.35, solid_capstyle="round")
                ax.scatter([pct], [y], s=30, color=col, zorder=3)
                text = f"{pct:.0f}% ({k}/{n})" + (f" {names[arm]}" if several else "")
                ax.text(pct + 1.5 if pct < 60 else pct - 1.5, y + 0.02, text, fontsize=7, va="bottom",
                        ha="left" if pct < 60 else "right", color="#333333")
            y += 1
        ticks.append((y0 + y - 1) / 2)
        labels.append(common.rule_name(rule))
        y += 1
    ax.axvline(K_ALPHA, color=common.RULE_COLOR["referee"], ls="--", lw=0.9)
    ax.text(K_ALPHA + 0.8, -0.6, f"K·α = {K_ALPHA:.1f}% (K = 2)", fontsize=7, color=common.RULE_COLOR["referee"])
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(y - 1, -1)
    ax.set_xlim(0, 105)
    n_rec = sorted({a["n_records"] for a in numbers["arms"].values()})
    ax.set_xlabel("runs that replaced a correctly specified model (%)"
                  + ("" if several else f"\nn = {n_rec[0]} records"))
    ax.set_title("When nothing is wrong", loc="left", fontsize=11, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.3, right=0.97, top=1 - 0.5 / fig.get_figheight(),
                        bottom=0.9 / fig.get_figheight())
    fig.savefig(out_png, dpi=200)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    numbers = count(args.arm_dirs)
    print_table(numbers)
    draw(numbers, args.arm_dirs, args.out)
    print(f"wrote {args.out}")
    if args.json:
        Path(args.json).write_text(json.dumps(numbers, indent=2))
    return numbers


if __name__ == "__main__":
    main()
