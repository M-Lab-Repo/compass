"""Figure 3c, "Outcome of the accepted revision": on each real dataset, what the first revision
each acceptance rule accepted turned out to be on the hidden fold, with the referee's
deployments at the right.

    python figure3/figure3c_create.py bench/arms/e1_anthropic_realdata_x2 --out fig3c.png

Options
  DIR ...          record directories of real-data runs, pooled (positional, required)
  --out PNG        the panel image (required; a .pdf name gives a vector file with editable text)
  --json PATH      also write the counted numbers as JSON

One row per dataset, one column per rule. Within one run the rule's outcome is the label
(figure2_common.label_of) of the first candidate, by iteration, that it accepted: genuine,
confabulated or harmful on the hidden fold; "underpowered" when the fold holds fewer than 10
points and the label would be confabulated; "accepted, unscored" when the candidate has no
label; "nothing accepted" when it accepted none; no cell when it returned no verdict. A
dataset's cell is the majority outcome over its runs (k/n printed when they disagree; ties go to
the worse outcome). The right margin gives the referee's deployments and the median hidden-fold
Δ of the deployed candidates. † marks datasets whose hidden fold holds fewer than 10 points.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

try:
    import figure3_common as f3  # run as a file from the repo root
except ImportError:
    from figure3 import figure3_common as f3  # run as a module / imported by tests
common = f3.common

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

NOTHING, UNSCORED = "nothing accepted", "unavailable"
RANK = {"harmful": 0, "confabulated": 1, "underpowered": 2, UNSCORED: 3, "genuine": 4, NOTHING: 5}  # ties: the worse


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------

def candidates(rec: dict) -> list[dict]:
    return (rec.get("pipeline") or {}).get("candidates") or []


def first_accepted(rec: dict, rule: str) -> str | None:
    """The label of the first candidate the rule accepted in this run; UNSCORED when it has no
    label, NOTHING when the rule accepted none, None when the rule returned no verdict at all."""
    cands = sorted(candidates(rec), key=lambda c: (c.get("iteration") is None, c.get("iteration") or 0))
    verdicts = [common.verdict(c, rule) for c in cands]
    if cands and all(v is None for v in verdicts):
        return None
    for c, v in zip(cands, verdicts, strict=True):
        if v:
            label = common.label_of(c, rec)
            return UNSCORED if label == "unavailable" else label
    return NOTHING


def majority(outcomes: list[str]) -> dict:
    counts = Counter(outcomes)
    label = min(counts, key=lambda o: (-counts[o], RANK[o]))
    return {"label": label, "k": counts[label], "n": len(outcomes)}


def count(arm_dirs) -> dict:
    """dataset -> rule -> {label, k, n}, plus the referee's deployments and fold sizes per dataset."""
    records = [r for d in arm_dirs for r in common.load_records(d) if r.get("type") not in common.SYNTHETIC_TYPES]
    done = [r for r in records if (r.get("pipeline") or {}).get("ok", True)]
    if not done:
        raise ValueError("no completed real-data record in " + ", ".join(str(d) for d in arm_dirs))
    rules = common.rule_order({k for r in done for c in candidates(r) for k in (c.get("judges") or {})} | {"mcmc"})
    names = ["referee" if r == "mcmc" else r for r in rules]
    out = {"rules": names, "datasets": f3.dataset_order(r["type"] for r in done), "rows": {}, "deployed": {},
           "n_hidden": {}, "n_records": len(done), "n_crashed": len(records) - len(done)}
    for ds in out["datasets"]:
        recs = sorted((r for r in done if r["type"] == ds),
                      key=lambda r: (int(r.get("seed") or 0), int(r.get("repeat") or 0),
                                     int((r.get("flags") or {}).get("split_seed") or 0)))
        out["rows"][ds] = {}
        for rule, name in zip(rules, names, strict=True):
            outcomes = [o for r in recs if (o := first_accepted(r, rule)) is not None]
            out["rows"][ds][name] = majority(outcomes) if outcomes else {"label": None, "k": 0, "n": 0}
        deployed = [c for r in recs for c in candidates(r) if c.get("deployed")]
        deltas = [d for d in (common.hidden_score(c).get("delta") for c in deployed) if isinstance(d, (int, float))]
        ses = [s for s in (common.hidden_score(c).get("se") for c in deployed) if isinstance(s, (int, float))]
        out["deployed"][ds] = {"deployed": len(deployed), "n": len(recs),
                               "delta_median": float(np.median(deltas)) if deltas else None,
                               "se_median": float(np.median(ses)) if ses else None}
        folds = [((r.get("pipeline") or {}).get("split_info") or {}).get("n_hidden") for r in recs]
        folds = [int(f) for f in folds if isinstance(f, (int, float))]
        out["n_hidden"][ds] = min(folds) if folds else None
    return out


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

def _delta_text(delta, se) -> str:
    fmt = "{:+.0f}" if abs(delta) >= 100 else "{:+.2f}"
    return "Δ " + fmt.format(delta) + ("" if se is None else " ± " + fmt.format(se).lstrip("+"))


def print_table(numbers: dict) -> None:
    print(f"figure3c  first accepted revision per real dataset and rule (majority over runs); "
          f"{numbers['n_records']} records, {numbers['n_crashed']} crashed skipped")
    for ds in numbers["datasets"]:
        d = numbers["deployed"][ds]
        extra = "" if d["delta_median"] is None else "  median hidden " + _delta_text(d["delta_median"], d["se_median"])
        print(f"{ds:12s} hidden fold {numbers['n_hidden'][ds]}  referee deployed {d['deployed']}/{d['n']}{extra}")
        for rule in numbers["rules"]:
            c = numbers["rows"][ds][rule]
            print(f"    {rule:36s} {str(c['label']):18s} {c['k']}/{c['n']}")


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

def _cell(ax, i, j, cell) -> None:
    pad, side, label = 0.08, 0.84, cell["label"]
    style = {None: dict(fc="white", ec="#c3c2b7", lw=0.7, ls=":"),
             NOTHING: dict(fc="white", ec="#c3c2b7", lw=0.7, hatch="////"),
             UNSCORED: dict(fc="white", ec="#52514e", lw=0.9)}
    ax.add_patch(Rectangle((j + pad, i + pad), side, side, **style.get(label, dict(fc=common.LABEL_COLOR.get(label, "white"), ec="none"))))
    note = {None: "no\nverdicts", UNSCORED: "accepted,\nunscored", "underpowered": "accepted,\nfold too small"}.get(label)
    if note:
        ax.text(j + 0.5, i + 0.5, note, ha="center", va="center", fontsize=5.5, color="#52514e")
    if cell["n"] > 1 and cell["k"] < cell["n"]:
        dark = label in ("genuine", "confabulated", "harmful")
        ax.text(j + 0.9, i + 0.9, f"{cell['k']}/{cell['n']}", ha="right", va="bottom", fontsize=5.6,
                color="white" if dark else "#52514e")


def _column(rule: str) -> str:
    if rule == "control":
        return "Control judge\n(no rationale)"
    return common.rule_name("mcmc" if rule == "referee" else rule).replace(" ", "\n", 1)


def draw(numbers: dict, arm_dirs, out_png) -> None:
    rules, datasets = numbers["rules"], numbers["datasets"]
    n_rules, n_ds = len(rules), len(datasets)
    small = [ds for ds in datasets if (numbers["n_hidden"].get(ds) or common.SMALL_FOLD) < common.SMALL_FOLD]
    fig = Figure(figsize=(0.62 * n_rules + 3.4, 0.5 * n_ds + 2.7))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for i, ds in enumerate(datasets):
        for j, rule in enumerate(rules):
            _cell(ax, i, j, numbers["rows"][ds][rule])
        d = numbers["deployed"][ds]
        text = f"deployed {d['deployed']}/{d['n']}"
        if d["delta_median"] is not None:
            text += "\n" + _delta_text(d["delta_median"], d["se_median"])
        ax.text(n_rules + 0.15, i + 0.5, text, ha="left", va="center", fontsize=6.5, color="#52514e", clip_on=False)
    ax.set_xlim(0, n_rules)
    ax.set_ylim(n_ds, 0)
    ax.set_xticks(np.arange(n_rules) + 0.5)
    ax.set_xticklabels([_column(r) for r in rules], fontsize=6.5)
    ax.set_yticks(np.arange(n_ds) + 0.5)
    ax.set_yticklabels([f"{f3.dataset_name(ds)}{'†' if ds in small else ''}\n({numbers['n_hidden'][ds]} hidden)"
                        for ds in datasets], fontsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Outcome of the accepted revision", loc="left", fontsize=11, fontweight="bold")
    ax.text(n_rules + 0.15, -0.1, "referee's final verdict\n(hidden-fold Δ, nats)", ha="left", va="bottom",
            fontsize=6.3, color="#898781", style="italic", clip_on=False)
    swatches = {"genuine": dict(fc=common.LABEL_COLOR["genuine"]), "confabulated": dict(fc=common.LABEL_COLOR["confabulated"]),
                "harmful": dict(fc=common.LABEL_COLOR["harmful"]), NOTHING: dict(fc="white", ec="#c3c2b7", hatch="////"),
                "underpowered": dict(fc=common.LABEL_COLOR["underpowered"]),
                "accepted, label unavailable": dict(fc="white", ec="#52514e", lw=0.9), "no verdicts": dict(fc="white", ec="#c3c2b7", ls=":")}
    seen = {c["label"] for by in numbers["rows"].values() for c in by.values()}
    names = ["genuine", "confabulated", "harmful", NOTHING]
    names += ["underpowered"] if "underpowered" in seen else []
    names += ["accepted, label unavailable"] if UNSCORED in seen else []
    names += ["no verdicts"] if None in seen else []
    fig.legend([Rectangle((0, 0), 1, 1, **swatches[n]) for n in names], names, loc="lower center", ncol=4, fontsize=7,
               frameon=False, bbox_to_anchor=(0.45, 0.42 / fig.get_figheight()), handlelength=1.0, handleheight=0.8)
    note = f"\n† hidden fold under {common.SMALL_FOLD} points ({', '.join(small)})" if small else ""
    fig.text(0.99, 0.01, f"measured: {', '.join(Path(d).name for d in arm_dirs)}, {numbers['n_records']} records; "
             f"hidden-fold label{note}", ha="right", va="bottom", fontsize=6.3, color="#666666", style="italic")
    fig.subplots_adjust(left=1.0 / fig.get_figwidth(), right=1 - 1.4 / fig.get_figwidth(), top=1 - 0.55 / fig.get_figheight(),
                        bottom=1.7 / fig.get_figheight())
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
