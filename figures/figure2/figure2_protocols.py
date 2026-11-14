"""The census arms as data: what each one runs, with which model, and which independent
readers score it afterwards. Standard library only, so a --dry-run loads nothing heavy.

An arm is a protocol (correct baseline E5, misspecified baseline E1, or real data) applied by
one proposer family. Synthetic arms run the eight misspecification types at seeds 0, 1, 2 with
three repeats (72 records); real-data arms run textbook datasets at one or three hidden splits.
Figure 2 reads the first ten arms (CENSUS_ARMS); the eleventh, run by the same engine, is read
by Figure 4 (real data at the default budget). The
flag names are those of ``bench.run.RunConfig.flags()`` so the records stay comparable with the
committed ones.
"""

from __future__ import annotations

try:
    from figure2_common import SYNTHETIC_TYPES  # run as a file
except ImportError:
    from figure2.figure2_common import SYNTHETIC_TYPES  # run as a module / tests

SEED = 0                 # split seed and MCMC seed of every run
MAX_ITERATIONS = 2       # pipeline iteration budget
CONTROL_K = 3            # template controls per run
REAL_SEVEN = ("roaches", "ucbadmit", "baseball", "seeds", "radon", "kidiq", "golf")

# MCMC budgets (bench/run.py): the default, the hierarchical type's, and the per-arm scale
DEFAULT_BUDGET = dict(num_warmup=500, num_samples=1000, num_chains=2, target_accept=0.9)
HIER_BUDGET = dict(num_warmup=1000, num_samples=2000, num_chains=4, target_accept=0.95)


def scale_budget(budget: dict, scale: float) -> dict:
    out = dict(budget)
    if scale != 1.0:
        for key in ("num_warmup", "num_samples"):
            if key in out:
                out[key] = max(1, int(round(out[key] * scale)))
    return out


def budget_for(inst: dict, scale: float = 1.0) -> dict:
    """The instance's own budget (real datasets carry one), else the type default, scaled."""
    if inst.get("budget"):
        return scale_budget(inst["budget"], scale)
    return scale_budget(HIER_BUDGET if inst.get("type") == "missing_hierarchy" else DEFAULT_BUDGET, scale)


# every RunConfig flag with its default; a protocol overrides a few
FLAG_DEFAULTS = dict(arm="", provider=None, model=None, judge_providers=[], repeat=1, split_seed=SEED,
                     hidden_frac=0.25, discovery=False, null_baseline=False, verdict_only=False,
                     hint_level=None, max_iterations=MAX_ITERATIONS, control=False, live_scout=False,
                     smoke_on_data=False, guard_replicate_sites=False, budget_scale=1.0, lit_threshold=6)
SYNTHETIC_E5 = dict(discovery=True, null_baseline=True, verdict_only=True, control=True)
SYNTHETIC_E1 = dict(discovery=True, verdict_only=True, control=True)
REAL_X2 = dict(verdict_only=True, control=True, live_scout=True, smoke_on_data=True,
               guard_replicate_sites=True, budget_scale=2.0)
REAL_X1 = dict(verdict_only=True, control=True, live_scout=True, smoke_on_data=True)   # first pass: one split, default budget

FAMILIES = {"anthropic": ("anthropic", "claude-sonnet-4-6"),
            "haiku": ("anthropic", "claude-haiku-4-5-20251001"),
            "gpt": ("openai", "gpt-5"),
            "gemini_flash": ("google", "gemini-3.8-flash")}
SONNET_READERS = (("anthropic", "claude-haiku-4-5-20251001"), ("anthropic", "claude-opus-4-5-20251101"),
                  ("openai", "gpt-5"))
HAIKU_READERS = (("anthropic", "claude-sonnet-4-6"), ("anthropic", "claude-opus-4-5-20251101"),
                 ("openai", "gpt-5"))
GPT_READER = (("openai", "gpt-5"),)


def _synthetic(family: str, protocol: dict, readers, legacy_dir: str) -> dict:
    provider, model = FAMILIES[family]
    return {"kind": "synthetic", "provider": provider, "model": model, "protocol": protocol,
            "types": SYNTHETIC_TYPES, "seeds": (0, 1, 2), "repeats": 3, "readers": readers,
            "legacy_dir": legacy_dir}


def _real(family: str, readers, legacy_dir: str, *, datasets=REAL_SEVEN, split_seeds=(0, 1, 2),
          protocol=REAL_X2) -> dict:
    provider, model = FAMILIES[family]
    return {"kind": "real", "provider": provider, "model": model, "protocol": protocol,
            "datasets": datasets, "split_seeds": split_seeds, "repeats": 1, "readers": readers,
            "legacy_dir": legacy_dir}


# One name per arm that reads without the paper: <baseline>_<proposer>.
#   correct-baseline       the run starts from the correct model; nothing is wrong (the paper's E5)
#   misspecified-baseline  the run starts from a misspecified model; a fix exists (the paper's E1)
#   real-data              seven textbook datasets, three hidden splits, double MCMC budget
# The paper's committed records live under the older short names (legacy_dir, under bench/arms/).
ARM_PROTOCOLS = {
    "correct-baseline_sonnet-4.6": _synthetic("anthropic", SYNTHETIC_E5, SONNET_READERS, "e5_anthropic"),
    "misspecified-baseline_sonnet-4.6": _synthetic("anthropic", SYNTHETIC_E1, SONNET_READERS, "e1_anthropic"),
    "real-data_sonnet-4.6": _real("anthropic", GPT_READER, "e1_anthropic_realdata_x2"),
    "correct-baseline_haiku-4.5": _synthetic("haiku", SYNTHETIC_E5, HAIKU_READERS, "e5_haiku"),
    "misspecified-baseline_haiku-4.5": _synthetic("haiku", SYNTHETIC_E1, HAIKU_READERS, "e1_haiku"),
    "correct-baseline_gpt-5": _synthetic("gpt", SYNTHETIC_E5, (), "e5_gpt"),
    "misspecified-baseline_gpt-5": _synthetic("gpt", SYNTHETIC_E1, (), "e1_gpt"),
    "real-data_gpt-5": _real("gpt", (), "e1_gpt_realdata_x2"),
    "correct-baseline_gemini-3.8-flash": _synthetic("gemini_flash", SYNTHETIC_E5, (), "e5_gemini_flash"),
    "misspecified-baseline_gemini-3.8-flash": _synthetic("gemini_flash", SYNTHETIC_E1, (), "e1_gemini_flash"),
    # one more census arm, same engine, read by Figure 4 (not by Figure 2)
    "real-data-default-budget_sonnet-4.6": _real("anthropic", (), "e1_anthropic_realdata", split_seeds=(0,),
                                                 protocol=REAL_X1),
}
CENSUS_ARMS = tuple(ARM_PROTOCOLS)[:10]                                              # Figure 2, panels a and b
E5_ARMS = tuple(a for a, p in ARM_PROTOCOLS.items() if p["protocol"].get("null_baseline"))  # panel d
LEGACY_DIR = {arm: p["legacy_dir"] for arm, p in ARM_PROTOCOLS.items()}             # new name -> bench/arms/<dir>


def condition(name: str) -> str | None:
    """"E5", "E1" or "real data" from an arm name, new or legacy; None when unrecognisable."""
    if name.startswith("correct-baseline") or name.startswith("e5"):
        return "E5"
    if name.startswith("real-data") or "realdata" in name:
        return "real data"
    if name.startswith("misspecified-baseline") or name.startswith("e1"):
        return "E1"
    return None


def record_stem(type_name: str, seed: int, split_seed: int, repeat: int) -> str:
    """``<type>_seed<s>[_s<split>]_r<k>``, the file name bench.run gave the committed records."""
    stem = f"{type_name}_seed{seed}"
    if split_seed != SEED:
        stem += f"_s{split_seed}"
    return f"{stem}_r{repeat}"


def record_flags(arm: str, split_seed: int = SEED) -> dict:
    """The full flags dictionary a record of this arm carries."""
    p = ARM_PROTOCOLS[arm]
    return {**FLAG_DEFAULTS, "arm": arm, "provider": p["provider"], "model": p["model"],
            "repeat": p["repeats"], "split_seed": split_seed, **p["protocol"]}


def instances_of(arm: str) -> list[dict]:
    """Every run of the arm in protocol order: {type, seed, split_seed, repeat, stem}."""
    p = ARM_PROTOCOLS[arm]
    out = []
    if p["kind"] == "synthetic":
        for type_name in p["types"]:
            for seed in p["seeds"]:
                for r in range(1, p["repeats"] + 1):
                    out.append({"type": type_name, "seed": seed, "split_seed": SEED, "repeat": r,
                                "stem": record_stem(type_name, seed, SEED, r)})
    else:
        for dataset in p["datasets"]:
            for split in p["split_seeds"]:
                out.append({"type": dataset, "seed": 0, "split_seed": split, "repeat": 1,
                            "stem": record_stem(dataset, 0, split, 1)})
    return out


def readers_of(arm: str) -> tuple:
    """(provider, model) pairs of the independent readers that score this arm offline."""
    return tuple(ARM_PROTOCOLS[arm]["readers"])


def describe(arm: str) -> str:
    """One line a user can read: what the arm is, how many runs, its readers, its committed records."""
    p = ARM_PROTOCOLS[arm]
    if p["kind"] == "real":
        what = p["datasets"][0] if len(p["datasets"]) == 1 else f"{len(p['datasets'])} real datasets"
        cond = (f"{what} x {len(p['split_seeds'])} hidden split(s), "
                f"MCMC budget x{p['protocol'].get('budget_scale', 1.0):g}")
    elif p["protocol"].get("null_baseline"):
        cond = "starts from the correct model (the paper's E5)"
    else:
        cond = "starts from a misspecified model (the paper's E1)"
    readers = ", ".join(model for _, model in p["readers"]) or "none"
    return (f"{arm:40s} {cond}; {len(instances_of(arm))} runs; readers: {readers}; "
            f"committed records: bench/arms/{p['legacy_dir']}")


def arms_help(arms) -> str:
    """The --help epilog of a per-panel script: its arms, one line each."""
    return "arms of this panel:\n" + "\n".join("  " + describe(a) for a in arms)

