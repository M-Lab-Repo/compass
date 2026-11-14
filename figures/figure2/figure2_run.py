"""Run one Figure 2 arm: the agent revises a model on every instance of the arm, the live rules
judge each revision, and one raw record per run is written. Nothing derived is stored here;
figure2_score.py adds the held-out labels and the independent readers in side files.

    python figure2/figure2_run.py --arm e5_haiku --out-root runs/2026-11 [--only PREFIX] [--dry-run]

Options
  --arm ARM        one of the arms of figure2_protocols.ARM_PROTOCOLS (required): Figure 2's ten, plus the
                   two that Figures 3c and 4 read
  --out-root DIR   where the records go: DIR/<arm>/<type>_seed<s>_r<k>.json (required; bench/arms is refused)
  --only PREFIX    run only the instances whose file name starts with PREFIX: <type>_seed<s>_r<k> on
                   synthetic arms, <dataset>_seed0[_s<split>]_r1 on real-data arms (the type and
                   dataset names are listed in any figure2X_run.py and in figure2_protocols.py)
  --dry-run        print the runs that would happen and call nothing

One run, in order (run_one): build the instance (data, starting model, problem statement),
split off the 25% hidden fold the proposer never sees, call the pipeline on the visible fold,
judge every candidate it produced (execution check, the proposer judging itself, the same
model with the rationale withheld), score three template edits of the starting model the same
way, and on misspecified-baseline arms judge the planted gold fix, a decoy and an unchanged
copy. A run whose pipeline raises writes no record, so re-running the arm retries it; a run
whose record exists is skipped.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

try:
    import figure2_common as common  # run as a file from the repo root
    import figure2_protocols as protocols
except ImportError:
    from figure2 import figure2_common as common  # run as a module / imported by tests
    from figure2 import figure2_protocols as protocols

CODE_DIR = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("COMPASS_CACHE_DIR", CODE_DIR / ".test_cache"))  # MCMC trace cache; research: CODE_DIR / ".test_cache"
COMMITTED_ARMS = CODE_DIR / "bench" / "arms"
# llm.get_llm searches .env from the current working directory upward; an arm launched from
# elsewhere (its runs/ root, a scheduler) still gets the checkout's code/.env. override=False:
# an exported variable wins (the pattern of tutorials/demos/_common.py and figure2_score.py).
load_dotenv(CODE_DIR / ".env", override=False)


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------

def build_instance(type_name: str, seed: int, flags: dict) -> dict:
    """The instance exactly as a run sees it: data, baseline program, problem statement,
    visible/hidden split and MCMC budget. Shared with figure2_score.instance_for."""
    from bench import dgp, realdata
    from bench.holdout import split_visible_hidden

    inst = dgp.make_instance(type_name, seed) if type_name in dgp.TYPES else realdata.make_real_instance(type_name)
    if flags.get("discovery") and "gold_code" in inst:
        inst = dgp.neutralize(inst)       # the statement names no symptom (real data already neutral)
    baseline = inst["gold_code"] if flags.get("null_baseline") else inst["original_code"]
    budget = protocols.budget_for(inst, float(flags.get("budget_scale") or 1.0))
    hidden_frac = float(flags.get("hidden_frac") or 0.25)
    visible, hidden, split_info = split_visible_hidden(inst["data"], hidden_frac=hidden_frac,
                                                       seed=int(flags.get("split_seed", protocols.SEED)))
    return {"inst": inst, "baseline": baseline, "visible": visible, "hidden": hidden,
            "split_info": split_info, "budget": budget}


# ---------------------------------------------------------------------------
# Judging and validation helpers (same shapes as bench.run, so records stay comparable)
# ---------------------------------------------------------------------------

def _field(v, key, default=None):
    if v is None:
        return default
    return v.get(key, default) if isinstance(v, dict) else getattr(v, key, default)


def validation_summary(v) -> dict:
    """The referee's verdict on one program, as stored under ``mcmc``."""
    passed = bool(_field(v, "converged", False) and not _field(v, "failure_reasons", []))
    heldout = _field(v, "heldout")
    return {"passed": passed, "converged": bool(_field(v, "converged", False)),
            "rhat_max": _field(v, "rhat_max"), "ess_min": _field(v, "ess_min"),
            "divergences": _field(v, "divergences"), "waic_delta": _field(v, "waic_delta"),
            "loo_delta": _field(v, "loo_delta"), "ppc": _field(v, "posterior_predictive_check"),
            "confidence": _field(v, "confidence", ""),
            "failure_reasons": list(_field(v, "failure_reasons", []) or []),
            "hints": list(_field(v, "misspecification_hints", []) or []),
            "heldout_delta": heldout.get("lppd_delta") if isinstance(heldout, dict) else None}


def run_judges(judges: dict, code: str, baseline: str, *, proposals, dataset_summary: str,
               problem: str, rationale: str) -> dict:
    """Every rule's verdict on one program; a rule that raises is stored as an error."""
    out = {}
    for name, judge in judges.items():
        role = getattr(judge, "role", name)
        try:
            v = judge(code, baseline, proposals=proposals, dataset_summary=dataset_summary,
                      problem=problem, rationale=rationale)
            if not hasattr(v, "passed"):
                raise TypeError(f"judge returned {type(v).__name__}, not JudgeVerdict")
            out[name] = {"passed": bool(v.passed), "confidence": getattr(v, "confidence", None),
                         "reasoning": (getattr(v, "reasoning", "") or "")[:500], "role": role, "error": None}
        except Exception as e:  # a candidate program can break any rule
            out[name] = {"passed": None, "confidence": None, "reasoning": None, "role": role,
                         "error": f"{type(e).__name__}: {e}"[:500]}
    return out


def default_validate(refined_code: str, original_code: str, data: dict, budget: dict, proposals=None):
    """The referee: fit both programs on the visible fold and apply the gates."""
    from compass.agents.validator import run_full_validation
    return run_full_validation(refined_code, original_code, data, framework="numpyro", fast=True,
                               cache_dir=str(CACHE), seed=protocols.SEED, proposals=proposals, **budget)


class Ledger:
    """What one run cost, in the shape the committed records carry under ``cost`` (Figure 4 reads
    it): LLM usage per stage (pipeline, judges, control, arbiter), usage and wall seconds per
    acceptance rule, and MCMC seconds per stage."""

    STAGES = ("pipeline", "judges", "control", "arbiter")

    def __init__(self):
        self.stage, self.by_rule = {}, {}
        self.mcmc = {"pipeline_validation": 0.0, "arbiter": 0.0, "control": 0.0}
        self.t0 = time.perf_counter()

    @contextmanager
    def llm(self, stage: str):
        """Book every model call made inside the block under ``stage``."""
        from compass.llm import merge_usage, track_usage
        with track_usage(stage) as tracker:
            try:
                yield
            finally:
                self.stage[stage] = merge_usage(self.stage.get(stage), tracker.snapshot())

    @contextmanager
    def fits(self, stage: str):
        """Book the wall seconds of the MCMC inside the block under ``stage``."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.mcmc[stage] = round(self.mcmc[stage] + time.perf_counter() - t0, 3)

    def wrap(self, judges: dict) -> dict:
        """The same rules; every verdict books its model usage and wall seconds under its rule."""
        from compass.llm import merge_usage, track_usage

        def wrap(name, judge):
            @functools.wraps(judge)
            def tracked(*args, **kwargs):
                t0 = time.perf_counter()
                with track_usage(name) as tracker:
                    try:
                        return judge(*args, **kwargs)
                    finally:
                        prev = self.by_rule.get(name) or {}
                        self.by_rule[name] = {
                            **merge_usage(prev, tracker.snapshot()),
                            "verdicts": prev.get("verdicts", 0) + 1,
                            "wall_seconds": round(prev.get("wall_seconds", 0.0) + time.perf_counter() - t0, 3)}
            tracked.role = getattr(judge, "role", name)
            return tracked
        return {name: wrap(name, j) for name, j in judges.items()}

    def record(self) -> dict:
        from compass.llm import empty_usage
        return {**{s: self.stage.get(s) or empty_usage() for s in self.STAGES}, "by_rule": self.by_rule,
                "mcmc_seconds": dict(self.mcmc), "wall_seconds": round(time.perf_counter() - self.t0, 3)}


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def run_one(ctx: dict, arm: str, flags: dict, r: int, *, llm, run_refinement=None, validate=None,
            execution_check=None) -> dict:
    """The raw record of one run. ``run_refinement``, ``validate`` and ``execution_check``
    default to the pipeline's own functions; tests pass stubs."""
    import compass.judge as judge_mod
    from compass.agents.strategist import proposals_rationale
    from compass.graph import infer_data_shapes
    from compass.llm import track_usage

    if run_refinement is None:
        from compass.graph import run_refinement
    validate = validate or default_validate
    execution_check = execution_check or judge_mod.create_execution_check
    inst, baseline, visible, budget = ctx["inst"], ctx["baseline"], ctx["visible"], ctx["budget"]
    smoke = visible if flags.get("smoke_on_data") else None
    ledger = Ledger()
    judges = ledger.wrap({"execution": execution_check(infer_data_shapes(visible), data=smoke),
                          "self": judge_mod.create_self_judge(llm),
                          "control": judge_mod.create_control_judge(llm)})
    summary = common.dataset_summary(len(visible["y"]), inst["problem"])
    t0 = time.time()
    with track_usage("run") as usage:
        # 1. the agent revises the baseline on the visible fold; the referee inside the
        #    pipeline uses this arm's MCMC budget
        def validation_fn(refined, original, data, *, framework="numpyro", fast=False,
                          cache_dir=None, seed=0, proposals=None):
            with ledger.fits("pipeline_validation"):
                return validate(refined, original, data, budget, proposals)

        with ledger.llm("pipeline"):
            final = run_refinement(
                problem_description=inst["problem"], model_code=baseline, framework="numpyro",
                data=visible, max_iterations=int(flags.get("max_iterations") or protocols.MAX_ITERATIONS),
                thread_id=f"{arm}-{inst['type']}-{inst['seed']}-s{flags.get('split_seed', 0)}-r{r}",
                validator_kwargs={"validation_fn": validation_fn, "cache_dir": str(CACHE), "seed": protocols.SEED},
                scout_kwargs={"live": bool(flags.get("live_scout"))}, llm=llm,
                verdict_only=bool(flags.get("verdict_only")), smoke_on_data=bool(flags.get("smoke_on_data")),
                guard_replicate_sites=bool(flags.get("guard_replicate_sites")))

        # 2. every candidate the pipeline produced, judged by the live rules
        refined = final.get("refined_model_code") or ""
        proposals = final.get("proposals") or []
        passed = validation_summary(final.get("validation"))["passed"]
        history = list(final.get("candidate_history") or [])
        # an empty final bundle after verified rounds (the proposer re-served banned names, or the
        # gate dropped everything): the proposal fields describe the last verified candidate and
        # terminal_round says why the run ended (mirrors bench.run.run_pipeline)
        terminal_round = "empty_bundle" if (not proposals and history) else None
        if terminal_round:
            proposals = list(history[-1].get("proposals") or [])
            proposal_names = (list(history[-1].get("proposal_names") or [])
                              or [p.get("name", "?") for p in proposals])
        else:
            proposal_names = [p.get("name", "?") for p in proposals]
        if refined.strip() and not any((h.get("refined_model_code") or "") == refined for h in history):
            history.append({"iteration": final.get("iteration"), "refined_model_code": refined,
                            "proposal_names": proposal_names, "proposals": proposals,
                            "rationale": proposals_rationale(proposals), "passed": passed,
                            "validation": final.get("validation")})
        candidates = []
        with ledger.llm("judges"):
            for h in history:
                code, rationale = h.get("refined_model_code") or "", h.get("rationale") or ""
                candidates.append({
                    "iteration": h.get("iteration"), "code": code,
                    "proposal_names": list(h.get("proposal_names") or []), "rationale": rationale,
                    "mcmc": validation_summary(h.get("validation")),
                    "judges": run_judges(judges, code, baseline, proposals=h.get("proposals"),
                                         dataset_summary=summary, problem=inst["problem"], rationale=rationale),
                    "deployed": bool(passed and h.get("passed") and code == refined)})

        # 3. template controls: mechanical edits of the baseline, no rationale, same rules
        control = None
        if flags.get("control"):
            from bench import control_proposer
            control = []
            with ledger.llm("control"):
                for cand in control_proposer.propose(baseline, k=protocols.CONTROL_K,
                                                     seed=protocols.SEED + inst["seed"]):
                    try:
                        with ledger.fits("control"):
                            mcmc = validation_summary(validate(cand["code"], baseline, dict(visible), budget))
                        mcmc["error"] = None
                    except Exception as e:  # a template edit may not sample; record, do not abort
                        mcmc = validation_summary(None)
                        mcmc["error"] = f"{type(e).__name__}: {e}"[:500]
                    control.append({"name": cand["name"], "source": cand.get("source", "template"),
                                    "code": cand["code"], "rationale": "", "mcmc": mcmc,
                                    "judges": run_judges(judges, cand["code"], baseline, proposals=[{"name": cand["name"]}],
                                                         dataset_summary=summary, problem=inst["problem"], rationale="")})

        # 4. arbiter cases on misspecified synthetic baselines: the planted gold fix, a decoy
        #    and an unchanged copy, all against the original program
        arbiter = None
        if "gold_code" in inst and not flags.get("null_baseline"):
            cases = [{"name": "gold (genuine)", "refined_code": inst["gold_code"]},
                     {"name": f"decoy ({inst['decoy_label']})", "refined_code": inst["decoy_code"]},
                     {"name": "no_change", "refined_code": inst["original_code"]}]
            for c in cases:
                c.update(original_code=inst["original_code"], problem=inst["problem"], dataset_summary=summary)

            def fit_case(c):
                with ledger.fits("arbiter"):
                    return validate(c["refined_code"], c["original_code"], dict(visible), budget)

            with ledger.llm("arbiter"):
                rules = judge_mod.run_arbiter_comparison(cases, judges, validate=fit_case)
            arbiter = {"rules": rules}

    findings = final.get("literature_findings") or []
    return {
        "type": inst["type"], "family": inst.get("family"), "seed": inst["seed"],
        "gold_class": inst.get("gold_class"), "decoy_label": inst.get("decoy_label"),
        "budget": budget, "max_iterations": int(flags.get("max_iterations") or protocols.MAX_ITERATIONS),
        "arm": arm, "repeat": r, "provider": flags.get("provider"), "model": flags.get("model"),
        "run_date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "flags": flags,
        "pipeline": {"ok": True, "status": final.get("status"), "validation_passed": passed,
                     "refined_model_code": refined, "n_proposals": len(proposal_names),
                     "proposal_names": proposal_names, "terminal_round": terminal_round,
                     "iterations": final.get("iteration"), "validation": validation_summary(final.get("validation")),
                     "split_info": ctx["split_info"], "candidates": candidates,
                     "error_log": [str(e)[:400] for e in (final.get("error_log") or [])[-5:]],
                     "literature_sources": [str(_field(f, "paper_title") or _field(f, "arxiv_id") or "")[:120]
                                            for f in findings][:10],
                     "seconds": round(time.time() - t0, 1)},
        "control": control, "arbiter": arbiter, "usage": usage.snapshot(), "cost": ledger.record(),
    }


# ---------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------

def run_arm(arm: str, out_root, *, llm=None, only: str | None = None, dry_run: bool = False,
            **seams) -> list[tuple[str, str]]:
    """Every run of the arm, in protocol order; returns [(stem, "done"|"skip"|"ERROR ...")]."""
    out_root = Path(out_root).resolve()
    if COMMITTED_ARMS.resolve() in (out_root, *out_root.parents):
        raise ValueError(f"{out_root} is inside the committed bench/arms; re-runs go to their own root")
    proto = protocols.ARM_PROTOCOLS[arm]
    todo = [i for i in protocols.instances_of(arm) if not only or i["stem"].startswith(only)]
    out_dir = out_root / arm
    print(f"arm {arm}: {len(todo)} run(s), {proto['kind']}, proposer {proto['provider']}:{proto['model']}"
          f"{' (dry run)' if dry_run else ''}")
    if dry_run:
        for i in todo:
            print(f"  would run {out_dir / (i['stem'] + '.json')}")
        return [(i["stem"], "dry-run") for i in todo]
    out_dir.mkdir(parents=True, exist_ok=True)
    if llm is None:
        from compass.llm import get_llm
        llm = get_llm(provider=proto["provider"], model=proto["model"])
    statuses = []
    for n, i in enumerate(todo, 1):
        path = out_dir / f"{i['stem']}.json"
        t0 = time.time()
        if path.exists():
            status = "skip"
        else:
            try:
                flags = protocols.record_flags(arm, i["split_seed"])
                ctx = build_instance(i["type"], i["seed"], flags)
                record = run_one(ctx, arm, flags, i["repeat"], llm=llm, **seams)
                path.write_text(json.dumps(record, indent=2, default=str))
                status = "done"
            except Exception as exc:  # no record: the next pass retries this run
                status = f"ERROR {type(exc).__name__}: {exc}"[:200]
        print(f"[{n}/{len(todo)}] {path.name}: {status} ({time.time() - t0:.0f}s)", flush=True)
        statuses.append((i["stem"], status))
    return statuses


def main(argv=None) -> list[tuple[str, str]]:
    ap = argparse.ArgumentParser(description=__doc__, epilog=protocols.arms_help(protocols.ARM_PROTOCOLS),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=list(protocols.ARM_PROTOCOLS), metavar="ARM")
    ap.add_argument("--out-root", required=True, metavar="DIR")
    ap.add_argument("--only", metavar="PREFIX")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    return run_arm(args.arm, args.out_root, only=args.only, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
