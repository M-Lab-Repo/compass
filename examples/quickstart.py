"""End-to-end COMPASS quickstart: refine a Poisson model of overdispersed
counts with the full pipeline (agents + verifier).

Requires the API key of the provider the agents run on: ANTHROPIC_API_KEY by
default, in the environment or in a .env file in the current working directory
or one of its parents; for another provider pass llm=get_llm(provider="openai")
or "google" (the `providers` extra). Runtime: several minutes, most of it MCMC;
the fit cache is written to .compass_cache/ in the working directory. The
verifier-only example that needs no key is verifier_only.py next to this file.
"""
import numpy as np

from compass import run_refinement

rng = np.random.default_rng(0)
data = {"y": rng.negative_binomial(2, 0.3, size=200), "n": 200}

MODEL = """
def model(n=None, y=None):
    lam = numpyro.sample("lam", dist.Exponential(0.1))
    with numpyro.plate("obs", n):
        numpyro.sample("y", dist.Poisson(lam), obs=y)
"""

final = run_refinement(
    problem_description=(
        "Count outcome whose sample variance is far above its mean, with "
        "occasional large values. The single-rate Poisson baseline may be "
        "too rigid for this dispersion."),
    model_code=MODEL,
    data=data,
    max_iterations=2,
    validator_kwargs={"cache_dir": ".compass_cache", "seed": 0},
)

print(final.get("report") or final.get("status"))
