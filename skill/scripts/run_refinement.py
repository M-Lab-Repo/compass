"""Run the full COMPASS pipeline (agents + verifier) on a user's model.

Usage:
  python run_refinement.py --data data.npz --model model.py \
      --problem "symptoms, never the fix" [--summary "..."] \
      [--iterations 2] [--seed 0] [--cache .compass_cache] [--live-scout] \
      [--provider anthropic|openai|google] [--model-id ID] \
      [--smoke-on-data] [--guard-replicate-sites] \
      [--warmup N --samples N --chains N] [--out result.json]

Needs the API key of the chosen provider: ANTHROPIC_API_KEY by default,
OPENAI_API_KEY for --provider openai, GOOGLE_API_KEY (or GEMINI_API_KEY) for
--provider google. Prints the report and writes the final state as JSON.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import dump_result, load_data, load_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--summary", default="")
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=".compass_cache")
    ap.add_argument("--live-scout", action="store_true")
    ap.add_argument("--provider", default=None,
                    help="LLM provider for every agent: anthropic, openai or google "
                         "(default: BAYES_REFINE_PROVIDER, else anthropic)")
    ap.add_argument("--model-id", default=None, dest="model_id",
                    help="model identifier for the provider (default: the provider's "
                         "BAYES_REFINE_MODEL* variable, else its built-in default)")
    ap.add_argument("--smoke-on-data", action="store_true",
                    help="smoke-test candidates on the real data arrays instead of dummy "
                         "data derived from the recorded shapes")
    ap.add_argument("--guard-replicate-sites", action="store_true",
                    help="add the NumPyro site guard (no unobserved replicate or discrete "
                         "sample sites) to the synthesizer prompts")
    ap.add_argument("--synth-attempts", type=int, default=6,  # the package default is 3
                    help="synthesis retry budget per candidate (smoke-test fixes) (default 6; the package default is 3)")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--chains", type=int, default=None)
    ap.add_argument("--out", default="compass_result.json")
    args = ap.parse_args()

    from compass import run_refinement
    from compass.agents.validator import run_full_validation

    budget = {k: v for k, v in dict(num_warmup=args.warmup, num_samples=args.samples,
                                    num_chains=args.chains).items() if v is not None}

    def validation_fn(refined_code, original_code, data, **kwargs):
        # The validator node passes framework, fast, cache_dir, seed and proposals as
        # keywords; they are forwarded unchanged and the budget is added on top.
        return run_full_validation(refined_code, original_code, data, **kwargs, **budget)

    validator_kwargs = {"cache_dir": args.cache, "seed": args.seed}
    if budget:
        validator_kwargs["validation_fn"] = validation_fn

    graph_kwargs = {
        "validator_kwargs": validator_kwargs,
        "scout_kwargs": {"live": bool(args.live_scout)},
        "synthesizer_kwargs": {"max_attempts": args.synth_attempts},
        "smoke_on_data": bool(args.smoke_on_data),
        "guard_replicate_sites": bool(args.guard_replicate_sites),
    }
    if args.provider is not None or args.model_id is not None:
        from compass import get_llm

        graph_kwargs["llm"] = get_llm(provider=args.provider, model=args.model_id)

    final = run_refinement(
        problem_description=args.problem,
        dataset_summary=args.summary,
        model_code=load_model(args.model),
        data=load_data(args.data),
        max_iterations=args.iterations,
        **graph_kwargs,
    )
    print("\n===== REPORT =====\n")
    print(final.get("report") or "(no report produced)")
    keep = {k: final.get(k) for k in ("status", "iteration", "validation",
                                      "refined_model_code", "proposals",
                                      "feedback_history", "candidate_history", "report")}
    dump_result(keep, args.out)


if __name__ == "__main__":
    main()
