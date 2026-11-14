"""Verifier-only COMPASS example: adjudicate a candidate model against a
baseline with no LLM and no API key.

Synthetic overdispersed counts (seed 0) are the data. The baseline is a
single-rate Poisson model, the candidate a NegativeBinomial2 model with a
dispersion parameter. ``run_full_validation`` fits both with NUTS and applies
the acceptance predicate (gates G, S, P, R); the script prints the verdict,
the failure reasons, the predictive margin elpd_diff +/- se and the wall time.

    python examples/verifier_only.py          # 300 warmup / 600 samples / 2 chains
    python examples/verifier_only.py --fast   # 150 / 300 / 2: a quicker check

The ``--fast`` budget is for trying the script out, not for a verdict: with
300 draws per chain the Poisson baseline's ESS falls below the gate's 400, so
the verifier refuses to certify the comparison ("original-model fit unreliable
(comparison baseline): ESS ...") although the candidate is clearly better. A
verdict to act on needs the default budget or more (the package default is
1000 / 2000 / 4). Measured on one CPU core: the default budget sampled in 11 s
(14 s in all, JAX import included) and printed "verdict: accepted (confidence
HIGH)" with elpd_diff -112.99 +/- 23.41. Set ``JAX_PLATFORMS=cpu`` to keep JAX
off an accelerator it cannot use.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

# The verifier executes model source strings in a namespace where `numpyro`,
# `dist` (numpyro.distributions) and `jnp` are already defined.
BASELINE = '''
def model(n=None, y=None):
    lam = numpyro.sample("lam", dist.Exponential(0.1))
    with numpyro.plate("obs", n):
        numpyro.sample("y", dist.Poisson(lam), obs=y)
'''

CANDIDATE = '''
def model(n=None, y=None):
    mu = numpyro.sample("mu", dist.Exponential(0.1))
    phi = numpyro.sample("phi", dist.HalfNormal(5.0))
    with numpyro.plate("obs", n):
        numpyro.sample("y", dist.NegativeBinomial2(mu, phi), obs=y)
'''

BUDGET = {"num_warmup": 300, "num_samples": 600, "num_chains": 2}
FAST_BUDGET = {"num_warmup": 150, "num_samples": 300, "num_chains": 2}


def make_data(seed: int = 0, n: int = 200) -> dict:
    """Overdispersed counts: NegativeBinomial(r=2, p=0.3), variance ~ 3.3x the mean."""
    rng = np.random.default_rng(seed)
    return {"y": rng.negative_binomial(2, 0.3, size=n), "n": n}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fast", action="store_true",
                    help="smaller MCMC budget (150 warmup / 300 samples / 2 chains) "
                         "for a quick check of the installation; not a verdict to act on "
                         "(the baseline's ESS falls below the gate at this budget)")
    args = ap.parse_args(argv)

    from compass import run_full_validation  # imports JAX and NumPyro: a few seconds

    budget = FAST_BUDGET if args.fast else BUDGET
    data = make_data()
    t0 = time.perf_counter()
    res = run_full_validation(CANDIDATE, BASELINE, data, seed=0, cache_dir=None, **budget)
    wall = time.perf_counter() - t0

    accepted = bool(res.converged and not res.failure_reasons)
    if res.repair:
        verdict = "accepted as repair"
    elif accepted:
        verdict = "accepted" + (f" (confidence {res.confidence})" if res.confidence else "")
    else:
        verdict = "refused"
    print(f"verdict: {verdict}")
    print("failure reasons:", "; ".join(res.failure_reasons) if res.failure_reasons else "none")
    if res.elpd_diff is not None and res.elpd_diff_se is not None:
        print(f"elpd_diff: {res.elpd_diff:+.2f} +/- {res.elpd_diff_se:.2f} "
              "(negative favours the candidate; the S gate needs |elpd_diff| >= 2 se)")
    else:
        print("elpd_diff: unavailable")
    print(f"S gate instrument: {res.s_gate or 'PSIS-LOO'}; "
          f"R-hat max {res.rhat_max:.3f}, ESS min {res.ess_min:.0f}, "
          f"divergences {res.divergences}, PPC {'passed' if res.posterior_predictive_check else 'failed'}")
    print(f"wall time: {wall:.1f} s (budget {budget['num_warmup']} warmup / "
          f"{budget['num_samples']} samples / {budget['num_chains']} chains, two fits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
