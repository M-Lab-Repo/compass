"""Score the trajectories of one drift arm: for every round, the model deployed after the round
against the starting model on the hidden fold ("hidden") and, for synthetic data, on a fresh
draw from the true process ("fresh"); and the round's candidate against the model it was
proposed for ("step"). Side files scores/<trajectory>.json; the records are never rewritten.

    python figure3/figure3_score.py runs/2026-11/drift_correct-baseline_sonnet-4.6/drift [--force]

Options
  DRIFT_DIR        a directory of trajectories written by figure3_run.py (positional, required)
  --force          recompute scores that are already stored (default: skip them, so a pass resumes)

MCMC only, no API key. Fits are cached by program, data and budget, so a model that stays
deployed for several rounds is fitted once. Labels as everywhere: genuine if Δ > 2 SE, harmful
if Δ < −2 SE, confabulated otherwise.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import figure3_common as f3  # run as a file from the repo root
except ImportError:
    from figure3 import figure3_common as f3  # run as a module / imported by tests
from figure2 import figure2_run as census  # noqa: E402
from figure2 import (
    figure2_score as census_score,  # noqa: E402  default_scorer, _slim, _read, _write
)


def instance_for(record: dict) -> dict:
    """Rebuild what the trajectory saw and refuse to continue if it does not match the record."""
    ctx = census.build_instance(record["type"], record["seed"], record.get("flags") or {})
    stem = f"{record['type']} seed {record['seed']} {record.get('rule')}"
    if ctx["baseline"] != record.get("baseline0_code"):
        raise ValueError(f"{stem}: the starting model differs from the protocol's")
    if ctx["split_info"]["n_hidden"] != (record.get("split_info") or {}).get("n_hidden"):
        raise ValueError(f"{stem}: hidden split mismatch")
    if "gold_code" in ctx["inst"]:
        from bench.holdout import fresh_draw
        ctx["fresh"] = fresh_draw(record["type"], record["seed"])
    else:
        ctx["fresh"] = None
    return ctx


def score_trajectory(record: dict, ctx: dict, stored: dict, *, force: bool, scorer) -> tuple[dict, int, int]:
    """Fill the scores file for one trajectory; returns (scores, fits done, scores kept)."""
    scores = {"budget": ctx["budget"], "split_check": {"n_hidden": ctx["split_info"]["n_hidden"], "ok": True},
              "rounds": dict(stored.get("rounds") or {})}
    baseline0, before = record["baseline0_code"], record["baseline0_code"]
    done = kept = 0
    for r in record["rounds"]:
        entry = scores["rounds"].setdefault(str(r["round"]), {})
        after = r["baseline_code_after"]
        wanted = {"hidden": (after, baseline0, ctx["hidden"])}
        if ctx["fresh"] is not None:
            wanted["fresh"] = (after, baseline0, ctx["fresh"])
        if r.get("candidate"):
            wanted["step"] = (r["candidate"]["code"], before, ctx["hidden"])
        for key, (code, base, fold) in wanted.items():
            if key in entry and not force:
                kept += 1
                continue
            t0 = time.time()
            score = scorer(code, base, ctx["visible"], fold, budget=ctx["budget"], seed=census.protocols.SEED,
                           cache_dir=str(census.CACHE))
            entry[key] = {**census_score._slim(score), "seconds": round(time.time() - t0, 3)}
            done += 1
        before = after
    scores["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return scores, done, kept


def score_arm(drift_dir, *, force: bool = False, scorer=None) -> list[dict]:
    """Scores for every trajectory of ``drift_dir``, written to its scores/ side files."""
    drift_dir = Path(drift_dir)
    scorer = scorer or census_score.default_scorer
    report = []
    for path in sorted(drift_dir.glob("*.json")):
        if not f3.TRAJECTORY_NAME.match(path.name):
            continue
        record = json.loads(path.read_text())
        if not (isinstance(record, dict) and isinstance(record.get("rounds"), list)):
            continue
        t0 = time.time()
        ctx = instance_for(record)
        side = drift_dir / "scores" / path.name
        scores, done, kept = score_trajectory(record, ctx, census_score._read(side), force=force, scorer=scorer)
        census_score._write(side, scores)
        print(f"{path.stem}: {done} fitted, {kept} kept ({time.time() - t0:.0f}s)", flush=True)
        report.append({"record": path.name, "fits": done, "kept": kept})
    return report


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drift_dir", metavar="DRIFT_DIR")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    return score_arm(args.drift_dir, force=args.force)


if __name__ == "__main__":
    main()
