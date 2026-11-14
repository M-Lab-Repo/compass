"""Run one drift arm: from one starting model, the agent proposes one revision per round, one
acceptance rule decides whether it replaces the model, and the accepted revision is the model
the next round starts from. One raw record per trajectory (one starting model, one rule),
rewritten after every round so a crash loses at most one round. Nothing derived is stored
here; figure3_score.py adds the held-out scores in side files.

    python figure3/figure3_run.py --arm drift_correct-baseline_sonnet-4.6 --out-root runs/2026-11 [--only PREFIX] [--dry-run]

Options
  --arm ARM        one of the drift arms of figure3_protocols.ARM_PROTOCOLS (required)
  --out-root DIR   records go to DIR/<arm>/drift/<type>_seed<s>_<rule>.json (required; bench/arms is refused)
  --only PREFIX    run only the trajectories whose file name starts with PREFIX (<type>_seed<s>_<rule>)
  --dry-run        print the trajectories that would run and call nothing

One round (run_round): the pipeline runs once, with a single iteration, on the current model.
For every rule but the referee a stub replaces the referee inside the pipeline (it always
passes), so the pipeline stops after one synthesis and the rule is applied to the candidate
here; for the referee the real validator runs inside the pipeline and its verdict is the rule.
A round with no candidate (after one retry) keeps the model. A trajectory file with fewer
rounds than the arm asks for is resumed from its last deployed model; one whose starting model
or hidden split differs from the protocol is refused.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

try:
    import figure3_common as f3  # run as a file from the repo root
    import figure3_protocols as protocols
except ImportError:
    from figure3 import figure3_common as f3  # run as a module / imported by tests
    from figure3 import figure3_protocols as protocols
from figure2 import (
    figure2_run as census,  # noqa: E402  build_instance, run_judges, validation_summary, the referee
)

CODE_DIR = Path(__file__).resolve().parents[1]
# llm.get_llm searches .env from the current working directory upward; an arm launched from
# elsewhere still gets the checkout's code/.env (override=False: an exported variable wins).
load_dotenv(CODE_DIR / ".env", override=False)

# ---------------------------------------------------------------------------
# One round
# ---------------------------------------------------------------------------

def build_judges(rule: str, visible: dict, flags: dict, *, llm, execution_check=None) -> dict:
    """The one reading rule of a trajectory as a judge, keyed by its name; {} for the referee,
    which decides inside the pipeline."""
    import compass.judge as judge_mod
    from compass.graph import infer_data_shapes
    if rule == "referee":
        return {}
    if rule == "execution":
        make = execution_check or judge_mod.create_execution_check
        return {rule: make(infer_data_shapes(visible), data=visible if flags.get("smoke_on_data") else None)}
    factory = judge_mod.create_self_judge if rule == "self" else judge_mod.create_control_judge
    return {rule: factory(llm)}


def candidate_of(final: dict) -> dict | None:
    """The candidate the pipeline produced this round: the last entry of its history that has
    code, else the final program; None when it produced nothing runnable."""
    history = [h for h in (final.get("candidate_history") or []) if (h.get("refined_model_code") or "").strip()]
    if history:
        return dict(history[-1])
    code = final.get("refined_model_code") or ""
    if not code.strip():
        return None
    from compass.agents.strategist import proposals_rationale
    proposals = final.get("proposals") or []
    return {"refined_model_code": code, "proposal_names": [p.get("name", "?") for p in proposals],
            "proposals": proposals, "rationale": proposals_rationale(proposals), "validation": final.get("validation")}


def run_round(ctx: dict, arm: str, flags: dict, rule: str, g: int, baseline: str, *, llm, judges: dict,
              run_refinement=None, validate=None) -> dict:
    """One proposer step on ``baseline`` and the rule's decision; returns the round's record."""
    from compass.llm import track_usage
    from compass.state import ValidationResult
    if run_refinement is None:
        from compass.graph import run_refinement
    validate = validate or census.default_validate
    inst, visible, budget = ctx["inst"], ctx["visible"], ctx["budget"]
    mcmc = {"seconds": 0.0}

    def referee(refined, original, data, *, framework="numpyro", fast=False, cache_dir=None, seed=0, proposals=None):
        t0 = time.perf_counter()
        try:
            return validate(refined, original, data, budget, proposals)
        finally:
            mcmc["seconds"] += time.perf_counter() - t0

    def stub(refined, original, data, **kw):  # the rule is applied below, not inside the pipeline
        return ValidationResult(converged=True, failure_reasons=[])

    t0 = time.time()
    with track_usage("proposer") as proposer:
        for attempt in (1, 2):  # one retry: a transient API failure must not end the trajectory
            final = run_refinement(
                problem_description=inst["problem"], model_code=baseline, framework="numpyro", data=visible,
                max_iterations=1,
                thread_id=f"{arm}-{inst['type']}-{inst['seed']}-{rule}-g{g}" + ("" if attempt == 1 else "-retry"),
                validator_kwargs={"validation_fn": referee if rule == "referee" else stub,
                                  "cache_dir": str(census.CACHE), "seed": protocols.SEED},
                scout_kwargs={"live": bool(flags.get("live_scout"))}, llm=llm, verdict_only=True,
                smoke_on_data=bool(flags.get("smoke_on_data")),
                guard_replicate_sites=bool(flags.get("guard_replicate_sites")))
            entry = candidate_of(final)
            if entry is not None:
                break
    accepted, candidate = False, None
    verdict = {"passed": None, "reasoning": "no candidate produced", "error": None}
    with track_usage("judge") as judge_usage:
        if entry is not None:
            code = entry["refined_model_code"]
            candidate = {"code": code, "proposal_names": list(entry.get("proposal_names") or []),
                         "rationale": entry.get("rationale") or ""}
            if rule == "referee":
                v = census.validation_summary(entry.get("validation"))
                verdict = {"passed": v["passed"], "error": None,
                           "reasoning": "verifier passed" if v["passed"] else "; ".join(map(str, v["failure_reasons"]))}
            else:
                verdict = census.run_judges(
                    judges, code, baseline, proposals=entry.get("proposals"),
                    dataset_summary=f3.common.dataset_summary(len(visible["y"]), inst["problem"]),
                    problem=inst["problem"], rationale=candidate["rationale"])[rule]
            accepted = bool(verdict["passed"])
    after = candidate["code"] if accepted else baseline
    return {"round": g, "accepted": accepted, "changed": accepted and after != baseline, "rule_verdict": verdict,
            "candidate": candidate, "baseline_code_after": after, "n_sample_sites": f3.count_latent_sites(after),
            "pipeline_status": final.get("status"),
            "pipeline_errors": [str(e)[:300] for e in (final.get("error_log") or [])[-3:]],
            "attempts": attempt, "seconds": round(time.time() - t0, 1),
            "cost": {"proposer": proposer.snapshot(), "judge": judge_usage.snapshot(),
                     "mcmc_seconds": {"referee": round(mcmc["seconds"], 3)}}}


# ---------------------------------------------------------------------------
# One trajectory, one arm
# ---------------------------------------------------------------------------

def run_trajectory(ctx: dict, arm: str, flags: dict, rule: str, path: Path, *, llm, run_refinement=None,
                   validate=None, execution_check=None) -> str:
    """Run or resume the trajectory stored at ``path``; returns "done", "resumed" or "skip"."""
    baseline0, rounds = ctx["baseline"], int(flags["rounds"])
    split_info = json.loads(json.dumps({k: v for k, v in ctx["split_info"].items() if k != "hidden_idx"}, default=str))
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("baseline0_code") != baseline0 or record.get("split_info") != split_info:
            raise ValueError(f"{path.name}: the stored trajectory starts from another model or hidden split; "
                             f"move it away to re-run it")
        if len(record["rounds"]) >= rounds:
            return "skip"
        status = "resumed"
    else:
        record = {"type": ctx["inst"]["type"], "seed": ctx["inst"]["seed"], "rule": rule, "arm": arm, "rounds": [],
                  "flags": flags, "split_info": split_info, "baseline0_code": baseline0,
                  "provider": flags.get("provider"), "model": flags.get("model"), "seconds": 0.0,
                  "run_date": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        status = "done"
    judges = build_judges(rule, ctx["visible"], flags, llm=llm, execution_check=execution_check)
    done = record["rounds"]
    baseline = done[-1]["baseline_code_after"] if done else baseline0
    path.parent.mkdir(parents=True, exist_ok=True)
    for g in range(len(done) + 1, rounds + 1):
        rnd = run_round(ctx, arm, flags, rule, g, baseline, llm=llm, judges=judges,
                        run_refinement=run_refinement, validate=validate)
        done.append(rnd)
        baseline = rnd["baseline_code_after"]
        record["seconds"] = round(record["seconds"] + rnd["seconds"], 1)
        path.write_text(json.dumps(record, indent=2, default=str))
    return status


def run_arm(arm: str, out_root, *, llm=None, only: str | None = None, dry_run: bool = False,
            **seams) -> list[tuple[str, str]]:
    """Every trajectory of the arm, in protocol order; returns [(stem, "done"|"resumed"|"skip"|"ERROR ...")]."""
    out_root = Path(out_root).resolve()
    if census.COMMITTED_ARMS.resolve() in (out_root, *out_root.parents):
        raise ValueError(f"{out_root} is inside the committed bench/arms; re-runs go to their own root")
    proto = protocols.ARM_PROTOCOLS[arm]
    todo = [t for t in protocols.trajectories_of(arm) if not only or t["stem"].startswith(only)]
    out_dir = out_root / arm / "drift"
    print(f"arm {arm}: {len(todo)} trajectories x {proto['rounds']} rounds, {proto['kind']}, "
          f"proposer {proto['provider']}:{proto['model']}{' (dry run)' if dry_run else ''}")
    if dry_run:
        for t in todo:
            print(f"  would run {out_dir / (t['stem'] + '.json')}")
        return [(t["stem"], "dry-run") for t in todo]
    if llm is None:
        from compass.llm import get_llm
        llm = get_llm(provider=proto["provider"], model=proto["model"])
    flags = protocols.record_flags(arm)
    statuses = []
    for n, t in enumerate(todo, 1):
        path = out_dir / f"{t['stem']}.json"
        t0 = time.time()
        try:
            ctx = census.build_instance(t["type"], t["seed"], flags)
            status = run_trajectory(ctx, arm, flags, t["rule"], path, llm=llm, **seams)
        except Exception as exc:  # the file keeps its finished rounds; the next pass resumes it
            status = f"ERROR {type(exc).__name__}: {exc}"[:200]
        print(f"[{n}/{len(todo)}] {path.name}: {status} ({time.time() - t0:.0f}s)", flush=True)
        statuses.append((t["stem"], status))
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
