"""Verifier-only adjudication: candidate vs original, no LLM involved.

Usage:
  python verify_candidate.py --data data.npz --original original.py \
      --candidate candidate.py [--warmup 3000 --samples 2000 --chains 2] \
      [--seed 0] [--cache .compass_cache] [--out verdict.json]

Prints the ValidationResult (verdict = accepted iff converged and no
failure_reasons; quote failure_reasons verbatim to the user).
"""
import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import dump_result, load_data, load_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=".compass_cache")
    ap.add_argument("--out", default="compass_verdict.json")
    args = ap.parse_args()

    from compass import run_full_validation

    res = run_full_validation(
        load_model(args.candidate), load_model(args.original), load_data(args.data),
        seed=args.seed, cache_dir=args.cache,
        num_warmup=args.warmup, num_samples=args.samples, num_chains=args.chains,
    )
    d = dataclasses.asdict(res) if dataclasses.is_dataclass(res) else dict(res.__dict__)
    d["accepted"] = bool(res.converged and not res.failure_reasons)
    dump_result(d, args.out)


if __name__ == "__main__":
    main()
