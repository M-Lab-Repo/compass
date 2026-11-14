"""The golden test case: a deliberately flawed model with a KNOWN refinement.

Spec reference: SPEC_bayesian_refinement_system.md §10. `GOLDEN_ORIGINAL`,
`GOLDEN_PROBLEM` and `GOLDEN_DATASET_SUMMARY` are verbatim from the spec.

`GOLDEN_REFINED_REFERENCE` is the hand-written ground-truth refinement the
validator can treat as "known good": it applies exactly the three refinements
the system MUST rediscover (spec §10):

1. Non-centered parameterization — ``alpha_j = mu_alpha + sigma_alpha * z_j``
   with ``z_j ~ Normal(0, 1)`` sampled inside the groups plate, and ``alpha``
   exposed via ``numpyro.deterministic`` so downstream diagnostics still see
   the school intercepts. (Papaspiliopoulos et al. 2007, arXiv 0708.3797.)
2. Tighter, scale-aware hyperpriors (Gelman et al. 2017, arXiv 1708.07487).
   GOLDEN_DATASET_SUMMARY says scores have mean=72, sd=15 (range 40-100):
   - ``mu_alpha ~ Normal(70, 20)``: the grand intercept must sit near the
     score mean; sd=20 still spans the entire 40-100 score range, so this is
     weakly informative, not dogmatic. The original ``Normal(0, 10)`` puts
     essentially zero prior mass near the actual data scale (~72).
   - ``sigma_alpha ~ HalfNormal(5)``: between-school sd is necessarily a
     fraction of the TOTAL score sd (15); HalfNormal(5) keeps ~95% of its
     mass below 10 while the original HalfNormal(10), combined with the
     centered parameterization and only 8 groups, invites funnel geometry.
3. Regularizing slope prior — ``beta ~ Normal(0, 10)`` instead of
   ``Normal(0, 100)``. study_hours has sd=2.1, so a slope of 10 already
   moves scores by ~21 points per sd of study hours; Normal(0, 10) remains
   generous while ruling out the absurd slopes Normal(0, 100) allows.

The residual ``sigma ~ HalfNormal(10)`` is kept: with total score sd 15 it is
already a reasonable weakly informative choice for observation noise and is
not one of the flagged flaws.

Both models share the EXACT function signature
``def model(X, group_idx, N_groups, y=None)`` — the synthesizer is required
to preserve it (spec §5.4) and tests enforce it via ast.
"""

GOLDEN_ORIGINAL = '''
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def model(X, group_idx, N_groups, y=None):
    mu_alpha = numpyro.sample("mu_alpha", dist.Normal(0, 10))       # vague
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(10))# vague
    with numpyro.plate("groups", N_groups):
        alpha = numpyro.sample("alpha", dist.Normal(mu_alpha, sigma_alpha))  # CENTERED (issue)
    beta = numpyro.sample("beta", dist.Normal(0, 100))              # absurdly vague
    sigma = numpyro.sample("sigma", dist.HalfNormal(10))
    mu = alpha[group_idx] + X * beta
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
'''

GOLDEN_PROBLEM = """Predicting student test scores across 8 schools.
Each school has 20-50 students. One continuous predictor (study hours).
Scores range 40-100. We suspect school-level effects but have few schools."""

GOLDEN_DATASET_SUMMARY = """N=280 students, 8 schools (groups of 20-50).
Outcome: continuous test score (mean=72, sd=15).
Predictor: study_hours (mean=5.2, sd=2.1). No missing data."""

GOLDEN_REFINED_REFERENCE = '''
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def model(X, group_idx, N_groups, y=None):
    # Scale-aware hyperpriors: scores have mean ~72, sd ~15 (range 40-100).
    mu_alpha = numpyro.sample("mu_alpha", dist.Normal(70, 20))
    # Between-school sd is a fraction of the total score sd (15).
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(5))
    # NON-CENTERED: sample standardized offsets, recover alpha deterministically.
    with numpyro.plate("groups", N_groups):
        z = numpyro.sample("z", dist.Normal(0, 1))
    alpha = numpyro.deterministic("alpha", mu_alpha + sigma_alpha * z)
    # Regularizing slope prior: study_hours sd is 2.1, so |beta| ~ 10 already
    # implies ~21-point score swings per sd of hours.
    beta = numpyro.sample("beta", dist.Normal(0, 10))
    sigma = numpyro.sample("sigma", dist.HalfNormal(10))
    mu = alpha[group_idx] + X * beta
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
'''
