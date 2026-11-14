"""Figure 2a, "What the agent proposed": the agent's revisions per arm, by label.

    python figure2/figure2a_create.py bench/arms/e5_anthropic bench/arms/e1_anthropic ... --out fig2a.png

Options
  DIR ...          record directories, drawn top to bottom in the order given (positional, required)
  --out PNG        the panel image (required)
  --json PATH      also write the counted numbers as JSON

Input: record directories. Each bar counts the pipeline candidates of one arm (template controls
and arbiter decoys are other candidate populations and are not counted here) by the label of
figure2_common.label_of. Bars are grouped by experiment, under a title that says what the
experiment is (nothing is wrong / a fix exists / real datasets), and labelled by the proposer
model; within a group the arms keep the order given.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

try:
    import figure2_common as common  # run as a file from the repo root
except ImportError:
    from figure2 import figure2_common as common  # run as a module / imported by tests

# --out foo.pdf (or .svg) gives a vector file; TrueType embedding keeps the text editable in Illustrator
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------

def count(arm_dirs) -> dict[str, dict[str, int]]:
    """{arm name: {label: number of the agent's revisions}} in the order given, zeros kept."""
    out = {}
    for d in arm_dirs:
        records = common.load_records(d)
        tally = Counter(common.label_of(c, rec) for rec in records
                        for c in (rec.get("pipeline") or {}).get("candidates") or [])
        unknown = set(tally) - set(common.LABELS)
        if unknown:
            raise ValueError(f"{d}: unexpected labels {sorted(unknown)}")
        out[Path(d).name] = {label: int(tally.get(label, 0)) for label in common.LABELS}
    return out


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

def print_table(numbers: dict[str, dict[str, int]]) -> None:
    print("figure2a  the agent's revisions by reference label (pipeline candidates only)")
    print(f"{'arm':26s} {'genuine':>7s} {'confab':>6s} {'harmful':>7s} {'underp':>6s} "
          f"{'unavail':>7s} {'total':>5s}")
    for arm, c in numbers.items():
        print(f"{arm:26s} {c['genuine']:7d} {c['confabulated']:6d} {c['harmful']:7d} "
              f"{c['underpowered']:6d} {c['unavailable']:7d} {sum(c.values()):5d}")


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

def draw(numbers: dict[str, dict[str, int]], arm_dirs, out_png) -> None:
    """Bars grouped by experiment (figure2_common.CONDITION_ORDER), each group under a plain-words
    title and each bar labelled by its proposer model; within a group the arms keep the order given."""
    records = {Path(d).name: common.load_records(d) for d in arm_dirs}
    groups: dict[str, list[str]] = {}
    for arm in numbers:
        groups.setdefault(common.condition(arm, records[arm]) or "other", []).append(arm)
    rows = []  # ("title", experiment) or ("arm", arm), top to bottom
    for cond in (*common.CONDITION_ORDER, "other"):
        if cond in groups:
            rows.append(("title", cond))
            rows += [("arm", arm) for arm in groups[cond]]
    fig = Figure(figsize=(7.0, 0.42 * len(rows) + 1.9))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ys = dict(zip(rows, range(len(rows))[::-1], strict=True))
    for (kind, key), y in ys.items():
        if kind == "title":
            ax.axhline(y + 0.5, color="#dddddd", linewidth=0.6)
            ax.text(0, y - 0.05, common.CONDITION_TITLE.get(key, key), ha="left", va="center",
                    fontsize=9, fontweight="bold", color="#333333")
            continue
        left = 0
        real = common.condition(key, records[key]) == "real data"
        for label in common.LABELS:
            n = numbers[key][label]
            if n == 0:
                continue
            ax.barh(y, n, left=left, height=0.6, color=common.LABEL_COLOR[label],
                    edgecolor="#555555", linewidth=0.5,
                    hatch="//" if real else ("////" if label == "unavailable" else None))
            if n >= 6:  # a segment too narrow for its number is left unlabelled; see the table
                ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=8,
                        color="white" if label in ("genuine", "confabulated", "harmful") else "#333333")
            left += n
        ax.text(left + 2, y, f"n = {left}", va="center", fontsize=8, color="#333333")
    arms = [(key, y) for (kind, key), y in ys.items() if kind == "arm"]
    ax.set_yticks([y for _, y in arms])
    ax.set_yticklabels([common.arm_model(records[a]) or a for a, _ in arms], fontsize=9)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("revisions the agent proposed (count)")
    ax.set_title("What the agent proposed", loc="left", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(sum(c.values()) for c in numbers.values()) * 1.15)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.5)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=common.LABEL_COLOR[lab], edgecolor="#555555", label=lab)
               for lab in common.LABELS[:4]]
    handles.append(Patch(facecolor="white", edgecolor="#555555", hatch="////", label="label unavailable"))
    if "real data" in groups:
        handles.append(Patch(facecolor="#dddddd", edgecolor="#555555", hatch="//", label="real data (hatched bar)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              fontsize=8, frameon=False)
    fig.text(0.99, 0.01, "label from held-out data: a fresh draw for synthetic arms, the hidden fold for real data",
             ha="right", va="bottom", fontsize=7, color="#666666", style="italic")
    fig.subplots_adjust(left=0.2, right=0.97, top=0.92, bottom=0.2)
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
