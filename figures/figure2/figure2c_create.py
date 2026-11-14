"""Figure 2c, "The theorem in one picture": the referee's acceptance rises with the data when a
revision helps and stays under its bound when it does not; a rule that only reads code has the
same acceptance in both worlds.

    python figure2/figure2c_create.py bench/arms/e1_anthropic --out fig2c.png [--reps 4000 --seed 0]

Options
  DIR              one record directory whose candidates give the two flat lines (positional, required)
  --out PNG        the panel image (required)
  --reps N         replicate datasets per sample size for the curves (default 4000)
  --seed N         random seed of the simulation (default 0)
  --json PATH      also write the numbers as JSON

Input: ONE record directory (the misspecified-baseline Sonnet arm in the paper). The two flat
lines are the execution check's and the self-evaluation's measured acceptance over every
pipeline candidate of that arm, whatever its label. The curves need no records: they are the
S gate (Delta >= 2 SE on leave-one-out log predictive densities) applied to the conjugate pair
m0: y ~ N(0, 1) against m1: y ~ N(mu, 1), mu ~ N(0, 1), in closed form, on `reps` replicate
datasets per sample size, under P+ (y ~ N(0.3, 1): the revision helps), a weaker P+ (0.15)
and P- (y ~ N(0, 1): it does not).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
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

NS = (10, 20, 50, 100, 200, 500, 1000)
MEAN_PLUS, MEAN_WEAK, MEAN_MINUS = 0.3, 0.15, 0.0
FLAT_RULES = ("execution", "self")  # the reading rules drawn as flat measured lines
BOUND = common.BOUND_PCT / 100.0    # Phi(-2)
_LOG_2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# The conjugate pair in closed form (identical to bench/figures/fig2c_theorem.py)
# ---------------------------------------------------------------------------

def loo_lpd_pair(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise leave-one-out log predictive densities (lpd_m0, lpd_m1) for y of shape
    (..., n). For m1 the posterior of mu without point i is N(S_-i / n, 1 / n) (prior
    precision 1 plus n - 1 observations), so its LOO predictive is N(S_-i / n, 1 + 1 / n)."""
    y = np.asarray(y, dtype=float)
    n = y.shape[-1]
    lpd0 = -0.5 * _LOG_2PI - 0.5 * y**2
    s_minus = y.sum(axis=-1, keepdims=True) - y
    mean = s_minus / n
    var = 1.0 + 1.0 / n
    lpd1 = -0.5 * np.log(2.0 * np.pi * var) - 0.5 * (y - mean) ** 2 / var
    return lpd0, lpd1


def s_gate_accepts(y: np.ndarray) -> np.ndarray:
    """The referee's S gate on each dataset (row) of y: Delta >= 2 SE with
    SE = sqrt(n) sd(d_i, ddof = 1), and SE > 0."""
    lpd0, lpd1 = loo_lpd_pair(y)
    d = lpd1 - lpd0
    delta = d.sum(axis=-1)
    se = d.std(axis=-1, ddof=1) * np.sqrt(d.shape[-1])
    return (se > 0) & (delta >= 2.0 * se)


def acceptance_curve(ns, mean: float, *, reps: int, rng: np.random.Generator) -> list[float]:
    """Acceptance frequency of the S gate over reps datasets y ~ N(mean, 1) per sample size."""
    return [float(s_gate_accepts(rng.normal(mean, 1.0, size=(reps, int(n)))).mean()) for n in ns]


def simulate(*, reps: int = 4000, seed: int = 0, ns=NS) -> dict:
    """The three curves, drawn from one generator in the order plus, weak, minus."""
    rng = np.random.default_rng(seed)
    ns = [int(n) for n in ns]
    return {"n": ns,
            "accept_plus": acceptance_curve(ns, MEAN_PLUS, reps=reps, rng=rng),
            "accept_plus_weak": acceptance_curve(ns, MEAN_WEAK, reps=reps, rng=rng),
            "accept_minus": acceptance_curve(ns, MEAN_MINUS, reps=reps, rng=rng),
            "reps": reps, "seed": seed, "bound": BOUND}


# ---------------------------------------------------------------------------
# The measured flat lines
# ---------------------------------------------------------------------------

def flat_rates(records: list[dict]) -> dict:
    """{rule: {"n", "accepted", "rate"}}: acceptance over every pipeline candidate of the
    records, any label; a candidate the rule gave no verdict on is not counted."""
    cands = [c for rec in records for c in (rec.get("pipeline") or {}).get("candidates") or []]
    out = {}
    for rule in FLAT_RULES:
        judged = [v for v in (common.verdict(c, rule) for c in cands) if v is not None]
        out[rule] = {"n": len(judged), "accepted": sum(bool(v) for v in judged),
                     "rate": (sum(bool(v) for v in judged) / len(judged)) if judged else None}
    return out


def count(arm_dir, *, reps: int = 4000, seed: int = 0) -> dict:
    numbers = simulate(reps=reps, seed=seed)
    numbers["flat"] = flat_rates(common.load_records(arm_dir))
    numbers["arm"] = Path(arm_dir).name
    return numbers


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

def print_table(numbers: dict) -> None:
    print(f"figure2c  S gate acceptance by sample size ({numbers['reps']} replicates per n, seed {numbers['seed']})")
    print(f"{'n':>18s} " + " ".join(f"{n:>7d}" for n in numbers["n"]))
    for key, label in (("accept_plus", f"P+ mean {MEAN_PLUS}"), ("accept_plus_weak", f"P+ mean {MEAN_WEAK}"),
                       ("accept_minus", f"P- mean {MEAN_MINUS}")):
        print(f"{label:>18s} " + " ".join(f"{v:7.4f}" for v in numbers[key]))
    for rule, f in numbers["flat"].items():
        rate = "-" if f["rate"] is None else f"{100 * f['rate']:.1f}%"
        print(f"flat: {rule} {f['accepted']}/{f['n']} = {rate}  (arm {numbers['arm']}, all candidate labels)")


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

def draw(numbers: dict, out_png) -> None:
    ns = numbers["n"]
    referee = common.RULE_COLOR["referee"]
    fig = Figure(figsize=(7.0, 3.6))
    FigureCanvasAgg(fig)
    ax_plus, ax_minus = fig.subplots(1, 2, sharey=True)
    for ax in (ax_plus, ax_minus):
        for rule, f in numbers["flat"].items():
            if f["rate"] is not None:
                ax.axhline(f["rate"], color=common.rule_color(rule), lw=1.4)
        ax.axhline(BOUND, color=referee, lw=0.9, ls="--")
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns], fontsize=7)
        ax.minorticks_off()
        ax.set_xlim(ns[0] / 1.35, ns[-1] * 1.35)
        ax.set_ylim(-0.04, 1.1)
        ax.set_xlabel("sample size n")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)
    ax_plus.plot(ns, numbers["accept_plus"], color=referee, lw=1.8, marker="o", ms=4)
    ax_plus.plot(ns, numbers["accept_plus_weak"], color=referee, lw=1.0, alpha=0.8, marker="o", ms=2.6)
    ax_minus.plot(ns, numbers["accept_minus"], color=referee, lw=1.8, marker="o", ms=4)
    ax_minus.text(ns[-1], BOUND + 0.02, f"bound Φ(-2) = {common.BOUND_PCT}%", color=referee,
                  fontsize=7, ha="right", va="bottom")
    ax_minus.text(np.sqrt(ns[0] * ns[-1]), 0.14,
                  f"referee accepts at most {100 * max(numbers['accept_minus']):.2f}% at any n",
                  fontsize=7, ha="center", color="#333333")
    for rule, f in numbers["flat"].items():
        if f["rate"] is not None:
            ax_plus.text(ns[0], f["rate"] + 0.025, f"{common.rule_name(rule)}: measured {100 * f['rate']:.0f}%",
                         fontsize=7, color="#333333", va="bottom")
    ax_plus.set_ylabel("acceptance probability")
    ax_plus.set_title("P$^{+}$: the revision helps", loc="left", fontsize=9)
    ax_minus.set_title("P$^{-}$: it does not", loc="left", fontsize=9)
    fig.suptitle("The theorem in one picture", x=0.02, ha="left", fontsize=11, fontweight="bold")
    handles = [Line2D([], [], color=referee, lw=1.8, marker="o", ms=4, label=f"Referee: S gate, y ~ N({MEAN_PLUS}, 1)"),
               Line2D([], [], color=referee, lw=1.0, alpha=0.8, marker="o", ms=2.6, label=f"Referee: weaker help, y ~ N({MEAN_WEAK}, 1)"),
               Line2D([], [], color=referee, lw=0.9, ls="--", label=f"referee's bound Φ(-2) = {common.BOUND_PCT}%")]
    for rule, f in numbers["flat"].items():
        if f["rate"] is not None:
            handles.append(Line2D([], [], color=common.rule_color(rule), lw=1.4,
                                  label=f"{common.rule_name(rule).split(chr(10))[0]}: measured, {numbers['arm']}"))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.85, bottom=0.33, wspace=0.08)
    fig.savefig(out_png, dpi=200)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    numbers = count(args.arm_dir, reps=args.reps, seed=args.seed)
    print_table(numbers)
    draw(numbers, args.out)
    print(f"wrote {args.out}")
    if args.json:
        Path(args.json).write_text(json.dumps(numbers, indent=2))
    return numbers


if __name__ == "__main__":
    main()
