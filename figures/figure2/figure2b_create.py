"""Figure 2b, "Who accepts them": acceptance of confabulated or harmful revisions per rule.

    python figure2/figure2b_create.py bench/arms/e5_anthropic bench/arms/e1_anthropic ... --out fig2b.png

Options
  DIR ...          record directories, one stem each per rule (positional, required)
  --out PNG        the panel image (required)
  --json PATH      also write the counted numbers as JSON

Input: record directories. Rows are every pipeline candidate, template control and arbiter
case of every record (crashed runs simply contribute no rows); only rows whose label is
confabulated or harmful are kept. For each rule and arm the acceptance is
accepted / judged, where a row the rule gave no verdict on (None) is not judged. The filled
dot is the agent's own revisions (pipeline rows) with its n; template controls (square) and
arbiter cases (triangle) are drawn hollow on the same stem.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

try:
    import figure2_common as common  # run as a file from the repo root
except ImportError:
    from figure2 import figure2_common as common  # run as a module / imported by tests

# --out foo.pdf (or .svg) gives a vector file; TrueType embedding keeps the text editable in Illustrator
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

BAD = ("confabulated", "harmful")
SOURCES = ("pipeline", "control", "arbiter")


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------

def rows_of(record: dict) -> list[tuple[str, dict]]:
    """(source, row) for every judged row of a record."""
    rows = [("pipeline", c) for c in (record.get("pipeline") or {}).get("candidates") or []]
    rows += [("control", c) for c in record.get("control") or []]
    rows += [("arbiter", r) for r in ((record.get("arbiter") or {}).get("rules") or {}).get("rows") or []]
    return rows


def tally(rows, rule: str) -> list[int]:
    """[judged, accepted] of one rule over rows; a None verdict is not judged."""
    verdicts = [common.verdict(row, rule) for row in rows]
    judged = [v for v in verdicts if v is not None]
    return [len(judged), sum(bool(v) for v in judged)]


def count(arm_dirs) -> dict:
    """{"rules": [...], "arms": {arm: {rule: {"pipeline": [n, k], "control": [n, k], "arbiter": [n, k]}}}}."""
    labelled: dict[str, list[tuple[str, dict]]] = {}
    for d in arm_dirs:
        rows = []
        for rec in common.load_records(d):
            for source, row in rows_of(rec):
                label = common.label_of(row, rec)
                if label != "unavailable":
                    rows.append((source, row, label))
        labelled[Path(d).name] = rows
    # the rules are the judge keys found on any labelled row, plus the referee
    keys = {k for rows in labelled.values() for _, row, _ in rows for k in (row.get("judges") or {})}
    rules = common.rule_order(keys | {"mcmc"})
    arms = {}
    for arm, rows in labelled.items():
        bad = [(s, row) for s, row, label in rows if label in BAD]
        arms[arm] = {rule: {src: tally([row for s, row in bad if s == src], rule) for src in SOURCES}
                     for rule in rules}
    return {"rules": rules, "arms": arms}


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

def _cell(n: int, k: int) -> str:
    return f"{k}/{n} {100 * k / n:3.0f}%" if n else "-"


def print_table(numbers: dict) -> None:
    print("figure2b  accepted/judged among confabulated or harmful rows (None verdicts not judged)")
    print(f"{'rule':40s} {'arm':26s} {'pipeline':>14s} {'control':>14s} {'arbiter':>14s}")
    for rule in numbers["rules"]:
        for arm, cells in numbers["arms"].items():
            c = cells[rule]
            print(f"{rule:40s} {arm:26s} {_cell(*c['pipeline']):>14s} {_cell(*c['control']):>14s} "
                  f"{_cell(*c['arbiter']):>14s}")


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

def draw(numbers: dict, arm_dirs, out_png) -> None:
    names = {Path(d).name: common.arm_label(Path(d).name, common.load_records(d)) for d in arm_dirs}
    # one stem per arm that judged any row under the rule; arms never read by a model
    # (GPT-5, Gemini) have no independent:* keys, so those blocks are shorter
    blocks = [(rule, [arm for arm, cells in numbers["arms"].items()
                      if sum(n for n, _ in cells[rule].values()) > 0])
              for rule in numbers["rules"]]
    n_rows = sum(len(arms) for _, arms in blocks) + len(blocks) - 1
    fig = Figure(figsize=(7.0, 0.27 * n_rows + 1.6))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    y, ticks, tick_labels = 0, [], []
    for rule, arms in blocks:
        col = common.rule_color(rule)
        y0 = y
        for arm in arms:
            cells = numbers["arms"][arm][rule]
            n_p, k_p = cells["pipeline"]
            if n_p:
                pct = 100 * k_p / n_p
                ax.plot([0, pct], [y, y], color=col, lw=1.6, alpha=0.35, solid_capstyle="round")
                ax.scatter([pct], [y], s=28, color=col, zorder=3)
                ax.text(min(pct, 92) + 1.5 if pct < 60 else pct - 1.5, y + 0.02,
                        f"{pct:.0f}% (n = {n_p}) {names[arm]}", fontsize=6.5, va="bottom",
                        ha="left" if pct < 60 else "right", color="#333333")
            for src, marker in (("control", "s"), ("arbiter", "^")):
                n_s, k_s = cells[src]
                if n_s:
                    ax.scatter([100 * k_s / n_s], [y], marker=marker, s=18, facecolor="white",
                               edgecolor=col, lw=0.8, zorder=4)
            y += 1
        ticks.append((y0 + y - 1) / 2)
        tick_labels.append(common.rule_name(rule))
        y += 1  # a blank row between rule blocks
    ax.axvline(common.BOUND_PCT, color=common.RULE_COLOR["referee"], ls="--", lw=0.9)
    ax.text(common.BOUND_PCT + 0.8, -0.6, f"bound {common.BOUND_PCT}%", fontsize=7,
            color=common.RULE_COLOR["referee"])
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_ylim(y - 1, -1)
    ax.set_xlim(0, 105)
    ax.set_xlabel("acceptance of confabulated or harmful revisions (%)")
    ax.set_title("Who accepts them", loc="left", fontsize=11, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.5)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", ls="", color="#555555", ms=5, label="agent's revisions (n)"),
               Line2D([], [], marker="s", ls="", color="white", mec="#555555", ms=4, label="template controls"),
               Line2D([], [], marker="^", ls="", color="white", mec="#555555", ms=4, label="arbiter decoys")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.subplots_adjust(left=0.27, right=0.97, top=1 - 0.5 / fig.get_figheight(),
                        bottom=1.1 / fig.get_figheight())
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
