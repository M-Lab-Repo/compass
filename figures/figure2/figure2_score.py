"""Score the raw records of one arm: the held-out labels every panel colours by, and the
verdicts of independent readers. Both go to side files next to the records, which are never
rewritten.

    python figure2/figure2_score.py runs/2026-11/e5_haiku [--readers P:M,...] [--force]

Options
  ARM_DIR          a directory of raw records written by figure2_run.py (positional, required)
  --readers LIST   comma-separated provider:model readers, e.g. anthropic:claude-opus-4-5-20251101,openai:gpt-5
                   (default: none; the per-panel figure2X_score.py files pass the arm's own readers)
  --force          recompute labels and verdicts that are already stored

Labels (scores/<run>.json): for every row of a record (the agent's candidates, the template
controls, the arbiter cases) the row's program and its baseline are fitted on the visible fold
and compared on the 25% hidden fold and, for synthetic data, on a fresh 150-point draw from the
true process: genuine if Delta > 2 SE, harmful if Delta < -2 SE, confabulated otherwise. MCMC,
no API key, cached fits reused.

Readers (readers/<run>.json): another model judges each stored row from the same prompt the
live judges saw (the symptom-free statement, N and that statement, the proposal names, the
proposer's justification, both programs), without running anything. Needs that provider's key.
Rows already scored or judged are skipped unless --force, so an interrupted pass resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import figure2_common as common  # run as a file from the repo root
    import figure2_run as engine
except ImportError:
    from figure2 import figure2_common as common  # run as a module / imported by tests
    from figure2 import figure2_run as engine

ARBITER_CODE = {"gold": "gold_code", "decoy": "decoy_code", "no_change": "original_code"}


# ---------------------------------------------------------------------------
# The instance behind a record
# ---------------------------------------------------------------------------

def instance_for(record: dict) -> dict:
    """Rebuild what the run saw and refuse to continue if it does not match the record."""
    ctx = engine.build_instance(record["type"], record["seed"], record.get("flags") or {})
    if record.get("budget") and record["budget"] != ctx["budget"]:
        raise ValueError(f"{record['type']} seed {record['seed']}: budget {ctx['budget']} differs from the "
                         f"recorded {record['budget']}")
    recorded = (record.get("pipeline") or {}).get("split_info") or {}
    if recorded.get("n_hidden") != ctx["split_info"]["n_hidden"]:
        raise ValueError(f"{record['type']} seed {record['seed']}: split mismatch (n_hidden "
                         f"{ctx['split_info']['n_hidden']} vs recorded {recorded.get('n_hidden')})")
    if recorded.get("hidden_idx") is not None and list(recorded["hidden_idx"]) != list(ctx["split_info"]["hidden_idx"]):
        raise ValueError(f"{record['type']} seed {record['seed']}: hidden_idx differs")
    if "gold_code" in ctx["inst"]:
        from bench.holdout import fresh_draw
        ctx["fresh"] = fresh_draw(record["type"], record["seed"])
    else:
        ctx["fresh"] = None
    return ctx


def rows_of(record: dict, ctx: dict) -> list[dict]:
    """Every row a label or a reader applies to: {where, key, code, baseline, proposals, rationale}."""
    inst, baseline = ctx["inst"], ctx["baseline"]
    rows = []
    for c in (record.get("pipeline") or {}).get("candidates") or []:
        rows.append({"where": "pipeline", "key": str(c.get("iteration")), "code": c["code"], "baseline": baseline,
                     "proposals": [{"name": n} for n in c.get("proposal_names") or []],
                     "rationale": c.get("rationale") or ""})
    for c in record.get("control") or []:
        rows.append({"where": "control", "key": c["name"], "code": c["code"], "baseline": baseline,
                     "proposals": [{"name": c["name"]}], "rationale": ""})
    for r in ((record.get("arbiter") or {}).get("rules") or {}).get("rows") or []:
        rows.append({"where": "arbiter", "key": r["name"], "code": inst[ARBITER_CODE[r["name"].split(" ", 1)[0]]],
                     "baseline": inst["original_code"], "proposals": None, "rationale": ""})
    return rows


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def default_scorer(code, baseline, visible, fold, *, budget, seed, cache_dir):
    from bench.holdout import score_pair_on_hidden
    return score_pair_on_hidden(code, baseline, visible, fold, budget=budget, seed=seed, cache_dir=cache_dir)


def _slim(score: dict) -> dict:
    return {k: v for k, v in score.items() if k != "per_obs_delta"}


def score_labels(record: dict, ctx: dict, stored: dict, *, force: bool, scorer) -> tuple[dict, int, int]:
    """Fill the scores file for one record; returns (scores, fits done, rows kept)."""
    scores = {"budget": ctx["budget"], "split_check": {"n_hidden": ctx["split_info"]["n_hidden"], "ok": True},
              "pipeline": dict(stored.get("pipeline") or {}), "control": dict(stored.get("control") or {}),
              "arbiter": dict(stored.get("arbiter") or {})}
    folds = {"hidden": ctx["hidden"]}
    if ctx["fresh"] is not None:
        folds["fresh"] = ctx["fresh"]
    done = kept = 0
    for row in rows_of(record, ctx):
        entry = scores[row["where"]].setdefault(row["key"], {})
        for fold_name, fold in folds.items():
            if fold_name in entry and not force:
                kept += 1
                continue
            t0 = time.time()
            score = scorer(row["code"], row["baseline"], ctx["visible"], fold,
                           budget=ctx["budget"], seed=engine.protocols.SEED, cache_dir=str(engine.CACHE))
            entry[fold_name] = {**_slim(score), "seconds": round(time.time() - t0, 3)}  # Figure 4 sums the seconds
            done += 1
    scores["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return scores, done, kept


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def default_judge_factory(provider: str, model: str):
    from compass.judge import create_independent_judge
    from compass.llm import get_llm
    return create_independent_judge(get_llm(provider=provider, model=model))


def key_available(provider: str) -> bool:
    """Is this provider's API key set (code/.env is read first)?"""
    from dotenv import load_dotenv

    from compass.llm import PROVIDERS
    load_dotenv(engine.CODE_DIR / ".env")
    spec = PROVIDERS.get(provider)
    if spec is None:
        return False
    return any(os.environ.get(k) for k in (spec["env_key"], *spec.get("env_aliases", ())))


def score_readers(record: dict, ctx: dict, stored: dict, readers: dict, *, force: bool) -> tuple[dict, int, int]:
    """Judge every row with every reader not yet stored; returns (readers file, verdicts, errors)."""
    inst, visible = ctx["inst"], ctx["visible"]
    neutralized = bool((record.get("flags") or {}).get("discovery") and "gold_code" in inst)
    out = {"problem_shown": inst["problem"], "models": sorted(set(stored.get("models") or []) | set(readers)),
           "pipeline": dict(stored.get("pipeline") or {}), "control": dict(stored.get("control") or {}),
           "arbiter": dict(stored.get("arbiter") or {})}
    summary = common.dataset_summary(len(visible["y"]), inst["problem"])
    n_verdicts = n_errors = 0
    for row in rows_of(record, ctx):
        entry = out[row["where"]].setdefault(row["key"], {})
        pending = {m: j for m, j in readers.items() if force or m not in entry}
        if not pending:
            continue
        verdicts = engine.run_judges(pending, row["code"], row["baseline"], proposals=row["proposals"],
                                     dataset_summary=summary, problem=inst["problem"], rationale=row["rationale"])
        for model, v in verdicts.items():
            entry[model] = {**v, "discovery_neutralized": neutralized}
            n_verdicts += 1
            n_errors += v["error"] is not None
    out["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return out, n_verdicts, n_errors


# ---------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------

def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


def score_arm(arm_dir, *, readers=(), force: bool = False, scorer=None, judge_factory=None) -> list[dict]:
    """Labels for every record of ``arm_dir`` (scores/), then the readers' verdicts (readers/).
    ``readers`` are (provider, model) pairs; one whose key is missing is skipped with a note."""
    arm_dir = Path(arm_dir)
    scorer = scorer or default_scorer
    judge_factory = judge_factory or default_judge_factory
    judges = {}
    for provider, model in readers:
        if judge_factory is default_judge_factory and not key_available(provider):
            print(f"reader {provider}:{model} skipped: no API key for {provider}")
            continue
        judges[model] = judge_factory(provider, model)
    report = []
    for path in sorted(arm_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except ValueError:
            continue
        if not (isinstance(record, dict) and "type" in record):
            continue
        t0 = time.time()
        ctx = instance_for(record)
        scores, done, kept = score_labels(record, ctx, _read(arm_dir / "scores" / path.name), force=force, scorer=scorer)
        _write(arm_dir / "scores" / path.name, scores)
        line = f"{path.stem}: labels {done} fitted, {kept} kept"
        entry = {"record": path.name, "fits": done, "kept": kept, "verdicts": 0, "errors": 0}
        if judges:
            rd, n_verdicts, n_errors = score_readers(record, ctx, _read(arm_dir / "readers" / path.name), judges, force=force)
            _write(arm_dir / "readers" / path.name, rd)
            line += f"; readers {len(judges)} model(s), {n_verdicts} new verdicts ({n_errors} errors)"
            entry.update(verdicts=n_verdicts, errors=n_errors)
        print(f"{line} ({time.time() - t0:.0f}s)", flush=True)
        report.append(entry)
    return report


def parse_readers(text: str) -> tuple:
    pairs = []
    for item in text.split(","):
        item = item.strip()
        if item:
            provider, _, model = item.partition(":")
            if not model:
                raise ValueError(f"reader {item!r} is not provider:model")
            pairs.append((provider.strip().lower(), model.strip()))
    return tuple(pairs)


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm_dir", metavar="ARM_DIR")
    ap.add_argument("--readers", default="", metavar="LIST")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    return score_arm(args.arm_dir, readers=parse_readers(args.readers), force=args.force)


if __name__ == "__main__":
    main()
