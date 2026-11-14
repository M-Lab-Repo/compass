"""The two drift arms of Figure 3 as data: what each one runs, from which starting model, under
which acceptance rules, for how many rounds. Standard library only, so a --dry-run loads
nothing heavy. Panel c's arm is a census arm and lives in figure2/figure2_protocols.py
(MATRIX_ARMS names it).

A drift arm is one proposer family revising a starting model for ROUNDS rounds under each of
the RULES: one trajectory per (starting model, rule), the accepted revision of a round becoming
the model the next round starts from. The flag names are those of
``bench.drift.DriftConfig.flags()`` so re-run records stay comparable with the committed ones.
"""

from __future__ import annotations

try:
    import figure3_common as f3  # run as a file
except ImportError:
    from figure3 import figure3_common as f3  # run as a module / tests
from figure2 import (
    figure2_protocols as census,  # noqa: E402  (the repo root is on sys.path after figure3_common)
)

RULES = ("execution", "self", "control", "referee")  # one trajectory per rule
ROUNDS = 6                                           # proposer steps per trajectory
SEED = census.SEED                                   # split seed and MCMC seed of every trajectory
MATRIX_ARMS = ("real-data_sonnet-4.6",)  # panel c: a census arm, figure2's engine

# every DriftConfig flag with its default; a protocol overrides a few
FLAG_DEFAULTS = dict(arm="", rounds=ROUNDS, hidden_frac=0.25, split_seed=SEED, discovery=False,
                     null_baseline=False, live_scout=False, provider=None, model=None, smoke_on_data=False,
                     guard_replicate_sites=False, budget_scale=1.0, lit_threshold=6)
# start from the correct (generating) model; the problem statement is the original one, which
# names the feature of the data the correct model already handles
FROM_CORRECT = dict(null_baseline=True)
# start from the textbook model of a real dataset, live literature, double MCMC budget
REAL = dict(live_scout=True, smoke_on_data=True, guard_replicate_sites=True, budget_scale=2.0)


def _arm(kind: str, protocol: dict, legacy_dir: str, **items) -> dict:
    provider, model = census.FAMILIES["anthropic"]
    return {"kind": kind, "provider": provider, "model": model, "protocol": protocol, "seeds": (0,),
            "rules": RULES, "rounds": ROUNDS, "legacy_dir": legacy_dir, **items}


# One name per arm that reads without the paper: drift_<baseline>_<proposer>.
ARM_PROTOCOLS = {
    "drift_correct-baseline_sonnet-4.6": _arm("synthetic", FROM_CORRECT, "e2_anthropic", types=f3.common.SYNTHETIC_TYPES),
    "drift_real-data_sonnet-4.6": _arm("real", REAL, "e2_anthropic_real", datasets=census.REAL_SEVEN),
}
LEGACY_DIR = {arm: p["legacy_dir"] for arm, p in ARM_PROTOCOLS.items()}  # new name -> bench/arms/<dir>


def trajectory_stem(type_name: str, seed: int, rule: str) -> str:
    """``<type>_seed<s>_<rule>``, the file name bench.drift gave the committed trajectories."""
    return f"{type_name}_seed{seed}_{rule}"


def trajectories_of(arm: str) -> list[dict]:
    """Every trajectory of the arm in protocol order: {type, seed, rule, stem}."""
    p = ARM_PROTOCOLS[arm]
    names = p["types"] if p["kind"] == "synthetic" else p["datasets"]
    return [{"type": t, "seed": s, "rule": r, "stem": trajectory_stem(t, s, r)}
            for t in names for s in p["seeds"] for r in p["rules"]]


def record_flags(arm: str) -> dict:
    """The full flags dictionary a trajectory of this arm carries."""
    p = ARM_PROTOCOLS[arm]
    return {**FLAG_DEFAULTS, "arm": arm, "rounds": p["rounds"], "provider": p["provider"], "model": p["model"],
            **p["protocol"]}


def describe(arm: str) -> str:
    """One line a user can read: the starting models, rules, rounds and the committed records."""
    p = ARM_PROTOCOLS[arm]
    if p["kind"] == "synthetic":
        start = f"{len(p['types'])} synthetic problems, starting from the correct model"
    else:
        start = (f"{len(p['datasets'])} real datasets, starting from the textbook model, "
                 f"MCMC budget x{p['protocol'].get('budget_scale', 1.0):g}")
    return (f"{arm:36s} {start}; {len(p['rules'])} rules x {p['rounds']} rounds; "
            f"{len(trajectories_of(arm))} trajectories; committed records: bench/arms/{p['legacy_dir']}/drift")


def arms_help(arms) -> str:
    """The --help epilog of a per-panel script: its arms, one line each."""
    return "arms of this panel:\n" + "\n".join("  " + describe(a) for a in arms)
