"""Synthetic benchmark records for the figure tests: the ``experiments.run`` record schema,
trimmed from a smoke record (no model code, reasoning or per-observation deltas), three
scored census records, one drift trajectory, and a writer that lays an arm directory out the
way the figure code expects (records, a summary file it must skip, a broken file it must
skip). No test reads the research repository's records; every arm is built under
``tmp_path`` from these."""

from __future__ import annotations

import copy
import json
from pathlib import Path


def judges(execution=True, self_=True, control=True):
    out = {}
    for name, v in (("execution", execution), ("self", self_), ("control", control)):
        out[name] = {"passed": v, "confidence": "high", "reasoning": "", "role": name,
                     "error": None if v is not None else "boom"}
    return out


def hidden(label, delta=-0.1, se=0.1):
    if label == "unavailable":
        return {"available": False, "reason": "boom", "label": "unavailable"}
    return {"available": True, "n_hidden": 38, "candidate_lppd": -83.0 + delta,
            "baseline_lppd": -83.0, "delta": delta, "se": se, "label": label}


def mcmc(passed: bool) -> dict:
    return {"passed": passed, "converged": True, "loo_delta": -0.25,
            "failure_reasons": [] if passed else ["no significant improvement"]}


def candidate(iteration: int) -> dict:
    return {"iteration": iteration, "proposal_names": ["Tighten intercept prior"],
            "rationale": "", "mcmc": mcmc(False), "judges": judges(),
            "hidden": hidden("confabulated"), "fresh": hidden("confabulated"),
            "deployed": False}


def control(name: str) -> dict:
    return {"name": name, "source": "template", "rationale": "", "mcmc": mcmc(False),
            "judges": judges(), "hidden": hidden("confabulated"),
            "fresh": hidden("confabulated")}


def template() -> dict:
    """The ``experiments.run`` record schema, trimmed from a smoke record."""
    flags = {"arm": "smoke", "provider": None, "model": None, "judge_providers": [],
             "repeat": 1, "split_seed": 0, "hidden_frac": 0.25, "discovery": True,
             "null_baseline": True, "verdict_only": True, "hint_level": None,
             "max_iterations": 2, "control": True, "live_scout": False}
    pipeline = {"ok": True, "status": "failed", "validation_passed": False, "n_proposals": 5,
                "proposal_names": ["Tighten intercept prior"], "iterations": 2,
                "null_baseline": True, "seconds": 554.6,
                "candidates": [candidate(0), candidate(1)],
                "split_info": {"n": 150, "n_visible": 112, "n_hidden": 38, "seed": 0,
                               "hidden_frac": 0.25}}
    return {"type": "overdispersion", "family": "counts", "seed": 0,
            "gold_class": "likelihood: Negative-Binomial", "decoy_label": "dispersion pinned huge",
            "budget": {"num_warmup": 500, "num_samples": 1000, "num_chains": 2,
                       "target_accept": 0.9},
            "max_iterations": 2, "arm": "smoke", "repeat": 1, "provider": None, "model": None,
            "flags": flags, "pipeline": pipeline, "arbiter": None,
            "control": [control("add_quadratic"), control("drop_covariate"),
                        control("link_swap")]}


def score(c: dict, label: str, judges_: dict, *, mcmc_: bool, deployed: bool = False,
          delta: float = -0.1, se: float = 0.1) -> None:
    # hidden fold and fresh draw agree, so hand-computed numbers hold under every label source
    c["hidden"] = hidden(label, delta, se)
    c["fresh"] = hidden(label, delta, se)
    c["judges"] = judges_
    c["mcmc"]["passed"] = mcmc_
    c["deployed"] = deployed


def records() -> tuple[dict, dict, dict]:
    """Three census records: (a) two confabulated candidates and three confabulated controls;
    (b) an unscored candidate, a harmful deployed one and one genuine control; (c) one genuine
    deployed candidate, no controls, an arbiter block."""
    t = template()
    assert t["flags"]["null_baseline"] and len(t["pipeline"]["candidates"]) == 2
    assert len(t["control"]) == 3

    a = copy.deepcopy(t)
    for c in a["pipeline"]["candidates"]:
        score(c, "confabulated", judges(), mcmc_=False)
    for i, c in enumerate(a["control"]):
        score(c, "confabulated", judges() if i == 0 else judges(self_=False, control=False),
              mcmc_=False)
    a["pipeline"]["validation_passed"] = False

    b = copy.deepcopy(a)
    b["type"], b["seed"] = "heavy_tails", 1
    c0, c1 = b["pipeline"]["candidates"]
    score(c0, "unavailable", judges(self_=False, control=False), mcmc_=False)
    score(c1, "harmful", judges(self_=None, control=False), mcmc_=True, deployed=True,
          delta=-9.0, se=2.0)
    b["control"] = b["control"][:1]
    score(b["control"][0], "genuine", judges(self_=False), mcmc_=True, delta=4.0, se=1.0)
    b["pipeline"]["validation_passed"] = True

    c = copy.deepcopy(a)
    c["type"], c["seed"] = "zero_inflation", 2
    c["pipeline"]["candidates"] = c["pipeline"]["candidates"][:1]
    score(c["pipeline"]["candidates"][0], "genuine", judges(), mcmc_=True, deployed=True,
          delta=6.0, se=1.5)
    c["pipeline"]["validation_passed"] = True
    c["control"] = None
    c["arbiter"] = {"n": 3, "rows": [], "rules": {"rows": [
        {"name": "gold (genuine)", "judges": judges(control=False), "mcmc_passed": True,
         "hidden_label": "genuine", "hidden_delta": 5.0, "hidden_se": 1.0},
        {"name": "decoy", "judges": judges(control=None), "mcmc_passed": False,
         "hidden_label": "harmful", "hidden_delta": -9.0, "hidden_se": 2.0},
        {"name": "no_change", "judges": judges(), "mcmc_passed": False,
         "hidden_label": None, "hidden_delta": None, "hidden_se": None},
    ]}}
    return a, b, c


def write_arm(d: Path, recs) -> Path:
    """An arm directory: one file per record, a summary file and a broken file to skip."""
    d.mkdir(parents=True)
    for rec in recs:
        (d / f"{rec['type']}_seed{rec['seed']}_r1.json").write_text(json.dumps(rec))
    (d / "arm_summary.json").write_text(json.dumps({"n_records": len(recs)}))
    (d / "broken.json").write_text("{not json")
    return d


def trajectory(rule: str, slope: float, rng, *, unscored_round: int | None = None) -> dict:
    """A drift trajectory of five rounds whose hidden-fold delta follows ``slope``."""
    rounds = [{"round": k, "accepted": True,
               "hidden": {"available": True, "n_hidden": 38, "label": "confabulated",
                          "delta": float(slope * k + rng.normal(0, 0.2)), "se": 0.3}}
              for k in range(1, 6)]
    if unscored_round is not None:
        rounds[unscored_round]["hidden"] = {"available": False, "reason": "boom",
                                            "label": "unavailable"}
    return {"type": "overdispersion", "rule": rule, "arm": "drift", "rounds": rounds}
