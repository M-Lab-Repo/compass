"""Figure 4, "What a verdict costs": seconds and tokens per verdict for every acceptance rule, from
the cost block each record carries.

    python figure4/figure4_create.py bench/arms/e1_anthropic --out fig4.png
    python figure4/figure4_create.py bench/arms/e1_anthropic bench/arms/e1_anthropic_realdata bench/arms/e1_haiku bench/arms/e1_gpt bench/arms/e1_gemini_flash --out fig4.pdf --grouped

Options
  DIR ...          record directories (positional, required)
  --out PNG        the panel image (required; a .pdf name gives a vector file with editable text)
  --grouped        one figure for all directories: a row per proposer (model and data), the rules as
                   coloured bars side by side; without it, one block of bars per directory, stacked
  --json PATH      also write the counted numbers as JSON

Input: the ``cost`` block of each record. For a reading rule: the number of verdicts it gave in the
run (the agent's candidates, the template controls and the arbiter cases), the wall seconds of
those judge calls (for a language model, the API latency) and the tokens in and out. For the
referee: the MCMC seconds of the validator inside the pipeline over the candidates it judged, and
no tokens. Hidden-fold scoring, the arbiter fits and the control fits are MCMC that is not a
verdict of any rule; their seconds are printed, not drawn (for re-runs the hidden-fold seconds
are read from the scores/ side files). Records without a cost block, the oldest Sonnet records,
are skipped and counted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]  # the repo root, so "from figure2 import" works when run as a file
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from figure2 import figure2_common as common  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

EXCLUDED = ("hidden_scoring", "arbiter", "control")  # MCMC that is not a verdict of any rule


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------

def side_file_seconds(arm_dir) -> float:
    """MCMC seconds of the hidden-fold fits that figure2_score.py stored in scores/ side files."""
    total = 0.0
    for path in (Path(arm_dir) / "scores").glob("*.json"):
        for rows in json.loads(path.read_text()).values():
            for row in (rows.values() if isinstance(rows, dict) else []):
                for fold in (row.values() if isinstance(row, dict) else []):
                    if isinstance(fold, dict):
                        total += float(fold.get("seconds") or 0.0)
    return total


def count(arm_dirs) -> dict:
    """{arm: {rules: {rule: {verdicts, seconds, tokens}}, n_with_cost, n_without_cost, mcmc_seconds, n_candidates}}."""
    out = {}
    for d in arm_dirs:
        records = common.load_records(d)
        with_cost = [r for r in records if isinstance(r.get("cost"), dict)]
        raw: dict[str, dict] = {}
        for r in with_cost:
            for rule, b in (r["cost"].get("by_rule") or {}).items():
                if common.rule_family(rule) is None:
                    continue
                acc = raw.setdefault(rule, {"verdicts": 0, "wall_seconds": 0.0, "total_tokens": 0})
                acc["verdicts"] += int(b.get("verdicts") or 0)
                acc["wall_seconds"] += float(b.get("wall_seconds") or 0.0)
                acc["total_tokens"] += int(b.get("total_tokens") or 0)
        rules = {rule: {"verdicts": v["verdicts"], "seconds": v["wall_seconds"] / v["verdicts"],
                        "tokens": v["total_tokens"] / v["verdicts"]} for rule, v in raw.items() if v["verdicts"]}
        mcmc = {k: sum(float((r["cost"].get("mcmc_seconds") or {}).get(k) or 0.0) for r in with_cost)
                for k in ("pipeline_validation", *EXCLUDED)}
        mcmc["hidden_scoring"] += side_file_seconds(d)
        n_candidates = sum(len((r.get("pipeline") or {}).get("candidates") or []) for r in with_cost)
        if n_candidates:
            rules["referee"] = {"verdicts": n_candidates, "seconds": mcmc["pipeline_validation"] / n_candidates, "tokens": None}
        out[Path(d).name] = {"rules": {r: rules[r] for r in common.rule_order(rules)}, "n_with_cost": len(with_cost),
                             "n_without_cost": len(records) - len(with_cost), "mcmc_seconds": mcmc,
                             "n_candidates": n_candidates}
    return out


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

def print_table(numbers: dict) -> None:
    print("figure4  what a verdict costs: judge-call seconds and tokens per verdict; referee = validator MCMC seconds")
    for arm, c in numbers.items():
        m = c["mcmc_seconds"]
        print(f"{arm}: {c['n_with_cost']} records with a cost block, {c['n_without_cost']} without; "
              f"MCMC seconds not drawn: hidden-fold scoring {m['hidden_scoring']:,.0f}, arbiter {m['arbiter']:,.0f}, "
              f"control fits {m['control']:,.0f}")
        for rule, v in c["rules"].items():
            tokens = "-" if v["tokens"] is None else f"{v['tokens']:,.0f}"
            print(f"    {rule:36s} {v['verdicts']:5d} verdicts  {v['seconds']:6.1f} s  {tokens:>6s} tokens")


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

def draw(numbers: dict, arm_dirs, out_png) -> None:
    arms = list(numbers)
    labels = {Path(d).name: common.arm_label(Path(d).name, common.load_records(d)) for d in arm_dirs}
    n_rules = max(len(c["rules"]) for c in numbers.values())
    block = 0.36 * n_rules + 1.3                              # inches per arm: its bars, title and x labels
    fig = Figure(figsize=(6.4, block * len(arms) + 0.7))
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(arms), 2, squeeze=False, sharey="row")
    for (ax_s, ax_t), arm in zip(axes, arms, strict=True):
        c = numbers[arm]
        rules = list(c["rules"])
        ys = list(range(len(rules)))[::-1]
        for ax, key, xlabel in ((ax_s, "seconds", "seconds per verdict"), (ax_t, "tokens", "tokens per verdict (in + out)")):
            values = [c["rules"][r][key] or 0.0 for r in rules]
            top = max(values) or 1.0
            for y, rule, v in zip(ys, rules, values, strict=True):
                ax.barh(y, v, height=0.62, color=common.rule_color(rule), hatch="////" if rule == "control" else None,
                        edgecolor="white", linewidth=0.5)
                text = "—" if c["rules"][rule][key] is None else (f"{v:,.0f}" if key == "tokens" else f"{v:.1f}")
                ax.text(v + 0.02 * top, y, text, va="center", fontsize=7, color="#333333")
            ax.set_xlim(0, top * 1.3)
            ax.set_xlabel(xlabel, fontsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.grid(axis="x", color="#e5e5e5", linewidth=0.5)
            ax.set_axisbelow(True)
        ax_s.set_yticks(ys)
        ax_s.set_yticklabels([f"{common.rule_name(r).split(chr(10))[0]}\nn = {c['rules'][r]['verdicts']}" for r in rules], fontsize=7)
        ax_s.set_ylim(-0.6, len(rules) - 0.4)
        ax_s.set_title(f"What a verdict costs: {labels[arm]} ({c['n_with_cost']} records)", loc="left", fontsize=9.5, fontweight="bold")
    fig.text(0.99, 0.01, "judges: wall time of the model call; referee: MCMC of the validation inside the pipeline only\n"
             "(hidden-fold scoring, arbiter and control fits are MCMC without a verdict and are not drawn)",
             ha="right", va="bottom", fontsize=6.5, color="#666666", style="italic")
    fig.subplots_adjust(left=0.26, right=0.98, top=1 - 0.45 / fig.get_figheight(), bottom=0.75 / fig.get_figheight(),
                        hspace=1.3 / (0.36 * n_rules), wspace=0.15)
    fig.savefig(out_png, dpi=200)


CONDITION_SHORT = {"E5": "correct baselines", "E1": "synthetic problems", "real data": "real datasets"}


def row_label(name: str, records: list[dict]) -> str:
    """"Sonnet 4.6\\n(synthetic problems)": the proposer model and, in plain words, the data it worked on."""
    model = common.arm_model(records) or name
    cond = CONDITION_SHORT.get(common.condition(name, records) or "", "")
    return f"{model}\n({cond})" if cond else model


def draw_grouped(numbers: dict, arm_dirs, out_png) -> None:
    """All directories on one figure: a row per proposer, the rules as bars side by side, in the
    rule colours used everywhere; left seconds per verdict, right tokens per verdict."""
    arms = list(numbers)
    records = {Path(d).name: common.load_records(d) for d in arm_dirs}
    rules = common.rule_order({r for c in numbers.values() for r in c["rules"]})
    fig = Figure(figsize=(7.2, 0.9 * len(arms) + 1.9))
    FigureCanvasAgg(fig)
    ax_s, ax_t = fig.subplots(1, 2, sharey=True)
    height = 0.8 / len(rules)
    for ax, key, xlabel in ((ax_s, "seconds", "seconds per verdict"), (ax_t, "tokens", "tokens per verdict (in + out)")):
        top = max((c["rules"][r][key] or 0.0) for c in numbers.values() for r in c["rules"]) or 1.0
        for i, arm in enumerate(arms):
            y_row = len(arms) - 1 - i
            for j, rule in enumerate(rules):
                v = numbers[arm]["rules"].get(rule)
                if v is None:
                    continue
                y = y_row + (len(rules) - 1) / 2 * height - j * height
                value = v[key] or 0.0
                ax.barh(y, value, height=height * 0.9, color=common.rule_color(rule),
                        hatch="////" if rule == "control" else None, edgecolor="white", linewidth=0.4)
                text = "—" if v[key] is None else (f"{value:,.0f}" if key == "tokens" else f"{value:.1f}")
                ax.text(value + 0.015 * top, y, text, va="center", fontsize=6, color="#333333")
        ax.set_xlim(0, top * 1.25)
        ax.set_xlabel(xlabel, fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.5)
        ax.set_axisbelow(True)
    ax_s.set_yticks(range(len(arms))[::-1])
    ax_s.set_yticklabels([row_label(a, records[a]) for a in arms], fontsize=8)
    ax_s.set_ylim(-0.6, len(arms) - 0.4)
    ax_s.tick_params(axis="y", length=0)
    ax_s.set_title("What a verdict costs", loc="left", fontsize=11, fontweight="bold")
    handles = [Patch(facecolor=common.rule_color(r), hatch="////" if r == "control" else None, edgecolor="white",
                     label=common.rule_name(r).split("\n")[0]) for r in rules]
    fig.legend(handles=handles, loc="lower center", ncol=len(rules), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.45 / fig.get_figheight()))
    fig.text(0.99, 0.01, "judges: wall time of the model call; referee: MCMC of the validation inside the pipeline, no tokens\n"
             "(hidden-fold scoring, arbiter and control fits are MCMC without a verdict and are not drawn)",
             ha="right", va="bottom", fontsize=6.5, color="#666666", style="italic")
    fig.subplots_adjust(left=0.2, right=0.98, top=1 - 0.45 / fig.get_figheight(), bottom=1.4 / fig.get_figheight(), wspace=0.12)
    fig.savefig(out_png, dpi=200)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grouped", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    numbers = count(args.arm_dirs)
    empty = [arm for arm, c in numbers.items() if not c["rules"]]
    if empty:
        raise SystemExit(f"no record with a cost block in {empty}")
    print_table(numbers)
    (draw_grouped if args.grouped else draw)(numbers, args.arm_dirs, args.out)
    print(f"wrote {args.out}")
    if args.json:
        Path(args.json).write_text(json.dumps(numbers, indent=2))
    return numbers


if __name__ == "__main__":
    main()
