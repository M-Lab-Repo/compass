"""Offline tests of the validator's acceptance predicate and its S-gate guards.

Every fast test here is MCMC-free: ``compass.agents.validator.run_mcmc`` is replaced
by a ``fake_run_mcmc`` (the pattern of the research suite's test_holdout.py) that hands back
canned ``InferenceData`` built with ``az.from_dict`` — posterior, sample_stats
(energy, diverging), log_likelihood and observed_data — so that
``compute_diagnostics`` and ``compare_models`` run FOR REAL on them. Only the
PPC (which needs the fitted model to regenerate replicates) and the
latent-structure audit are stubbed, and the model code is real NumPyro source
wherever the structural guard (``check_observed_sites``) runs.

Covers:

* each convergence gate's failure-reason string (divergences, R-hat, ESS,
  E-BFMI), computed by the real ``compute_diagnostics``;
* the no-significant-improvement refusal;
* defect V2 — a Pareto k-hat of inf gates and escalates like k > 0.7, NaN
  does neither;
* defect V1a — the masked / scaled observed-site guard (handlers.mask,
  ``.mask()``, ``obs_mask=``; an identically-True mask passes);
* defect V1b — the observed-data identity guard between the two fits;
* defect V1c — the degenerate pointwise-column guard;
* defect V3 — ``az.compare`` is pinned to ``ic="loo", scale="log"`` so the
  session-wide ``az.rcParams`` cannot change the margin;
* the TOST repair acceptance and one positive acceptance;
* the 2026-11-12 adversarial-review findings: F1 — observed values are
  compared at float32 precision when a side cast its response (a clipped
  response is still refused); F2 — the structural guard is COMPARATIVE (a
  data mask both programs share passes, a widened mask or an added scale is
  refused; shared constant pointwise columns are exempt from the degenerate
  rule and from the k-hat); F3 — the two-faced likelihood (response-dependent
  likelihood parameters) is refused against an honest baseline, noted when
  both programs condition on the response (AR(1)), skipped when the candidate
  cannot be un-conditioned; F4 — the wording of the degenerate-column,
  granularity and inf-k-hat refusals; F5 — the Stan-style observed_data
  identity fallback;
* slow tests: the real masked Poisson candidate, the shared-mask Poisson vs
  NegativeBinomial2 design (compared and certified) and the soft-clamp probe
  (refused), each through ``run_full_validation``.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest
from scipy.stats import norm, poisson

import compass.agents.validator as validator
import compass.resolution as resolution_mod
from compass.agents.validator import (
    ComparisonRefused,
    _khat_unreliable,
    check_observed_sites,
    compare_models,
    run_full_validation,
)

# ---------------------------------------------------------------------------
# Model zoo (real NumPyro source; the structural guard traces these)
# ---------------------------------------------------------------------------

POISSON = '''
def model(y=None):
    lam = numpyro.sample("lam", dist.Gamma(2.0, 0.2))
    with numpyro.plate("n", y.shape[0]):
        numpyro.sample("y", dist.Poisson(lam), obs=y)
'''

# the probe that was ACCEPTED before the fix: same likelihood, awkward points masked
POISSON_MASKED_HANDLER = POISSON.replace(
    '        numpyro.sample("y", dist.Poisson(lam), obs=y)',
    '        with numpyro.handlers.mask(mask=(y > 4)):\n'
    '            numpyro.sample("y", dist.Poisson(lam), obs=y)')
POISSON_MASKED_METHOD = POISSON.replace(
    'dist.Poisson(lam), obs=y', 'dist.Poisson(lam).mask(y > 4), obs=y')
POISSON_OBS_MASK_KW = POISSON.replace(
    'dist.Poisson(lam), obs=y', 'dist.Poisson(lam), obs=y, obs_mask=(y > 4)')
POISSON_MASK_ALL_TRUE = POISSON.replace(
    'dist.Poisson(lam), obs=y',
    'dist.Poisson(lam).mask(jnp.ones(y.shape[0], dtype=bool)), obs=y')
POISSON_SCALED = POISSON.replace(
    '        numpyro.sample("y", dist.Poisson(lam), obs=y)',
    '        with numpyro.handlers.scale(scale=0.5):\n'
    '            numpyro.sample("y", dist.Poisson(lam), obs=y)')
# a second observed site that is a plain latent being masked (prior, not data)
POISSON_MASKED_LATENT = '''
def model(y=None):
    with numpyro.handlers.mask(mask=False):
        lam = numpyro.sample("lam", dist.Gamma(2.0, 0.2))
    with numpyro.plate("n", y.shape[0]):
        numpyro.sample("y", dist.Poisson(lam), obs=y)
'''

# a different code string for the baseline so fake_run_mcmc can tell them apart
POISSON_BASELINE = POISSON + "\n# baseline\n"

# --- 2026-11-12 adversarial-review probes (F1–F5) ----------------------------
# plate sized from a kwarg: the un-conditioned call (y=None) runs, as the PPC
# requires of every candidate it passes
POISSON_N = '''
def model(y=None, N=400):
    lam = numpyro.sample("lam", dist.Gamma(2.0, 0.2))
    with numpyro.plate("n", N):
        numpyro.sample("y", dist.Poisson(lam), obs=y)
'''
NB2_N = '''
def model(y=None, N=400):
    lam = numpyro.sample("lam", dist.Gamma(2.0, 0.2))
    phi = numpyro.sample("phi", dist.Exponential(0.1))
    with numpyro.plate("n", N):
        numpyro.sample("y", dist.NegativeBinomial2(lam, phi), obs=y)
'''
# the F3 probe, ACCEPTED before the fix (elpd_diff -26.4 +/- 8.9, k-hat 0.12,
# PPC passing): an honest Poisson when sampled (y is None), a soft clamp on
# the observed values when scored
POISSON_SOFT_CLAMP = POISSON_N.replace(
    '    with numpyro.plate("n", N):\n        numpyro.sample("y", dist.Poisson(lam), obs=y)',
    '    rate = lam if y is None else jnp.where(y > 4, lam, 0.5 * y + 0.2 * lam)\n'
    '    with numpyro.plate("n", N):\n        numpyro.sample("y", dist.Poisson(rate), obs=y)')
assert "jnp.where(y > 4" in POISSON_SOFT_CLAMP
# the same probe sizing its plate from y: its un-conditioned trace cannot run
POISSON_SOFT_CLAMP_YSHAPE = POISSON_SOFT_CLAMP.replace(
    "def model(y=None, N=400):", "def model(y=None):").replace(
    'numpyro.plate("n", N)', 'numpyro.plate("n", y.shape[0])')
# the standard missing-data design: BOTH programs apply the data's own mask
MASKED_DATA_POISSON = '''
def model(y=None, mask=None, N=400):
    lam = numpyro.sample("lam", dist.Gamma(2.0, 0.2))
    m = mask if mask is not None else jnp.ones(N, dtype=bool)
    with numpyro.plate("n", N), numpyro.handlers.mask(mask=m):
        numpyro.sample("y", dist.Poisson(lam), obs=y)
'''
MASKED_DATA_NB2 = MASKED_DATA_POISSON.replace(
    '    lam = numpyro.sample("lam", dist.Gamma(2.0, 0.2))\n',
    '    lam = numpyro.sample("lam", dist.Gamma(2.0, 0.2))\n'
    '    phi = numpyro.sample("phi", dist.Exponential(0.1))\n').replace(
    "dist.Poisson(lam)", "dist.NegativeBinomial2(lam, phi)")
assert "NegativeBinomial2" in MASKED_DATA_NB2
# a candidate WIDENING the shared mask with one of its own
MASKED_DATA_NB2_EXTRA = MASKED_DATA_NB2.replace(
    "numpyro.handlers.mask(mask=m):", "numpyro.handlers.mask(mask=m & (y > 4)):")
# a candidate keeping the shared mask but scaling the likelihood
MASKED_DATA_NB2_SCALED = MASKED_DATA_NB2.replace(
    "numpyro.handlers.mask(mask=m):",
    "numpyro.handlers.mask(mask=m), numpyro.handlers.scale(scale=0.5):")
# a candidate that accepts the mask kwarg but scores EVERY point
UNMASKED_NB2_WITH_MASK_KWARG = MASKED_DATA_NB2.replace(", numpyro.handlers.mask(mask=m):", ":")
assert "handlers.mask" not in UNMASKED_NB2_WITH_MASK_KWARG
POISSON_N_SCALED = POISSON_N.replace(
    '    with numpyro.plate("n", N):',
    '    with numpyro.plate("n", N), numpyro.handlers.scale(scale=0.5):')
NB2_N_SCALED = NB2_N.replace(
    '    with numpyro.plate("n", N):',
    '    with numpyro.plate("n", N), numpyro.handlers.scale(scale=0.5):')
NB2_N_SCALED_OTHER = NB2_N_SCALED.replace("scale=0.5", "scale=0.25")
# AR(1): a legitimate conditional likelihood (the lagged response enters the
# location); with y=None the lag is a vector of zeros, so the un-conditioned
# call runs and the response dependence is visible on BOTH sides
AR1 = '''
def model(y=None, T=60):
    phi = numpyro.sample("phi", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    prev = jnp.zeros(T) if y is None else jnp.concatenate([jnp.zeros(1), y[:-1]])
    with numpyro.plate("t", T):
        numpyro.sample("y", dist.Normal(phi * prev, sigma), obs=y)
'''
AR1_INTERCEPT = AR1.replace(
    '    phi = numpyro.sample("phi", dist.Normal(0.0, 1.0))\n',
    '    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))\n'
    '    phi = numpyro.sample("phi", dist.Normal(0.0, 1.0))\n').replace(
    "dist.Normal(phi * prev, sigma)", "dist.Normal(mu + phi * prev, sigma)")
assert "mu + phi * prev" in AR1_INTERCEPT
IID_NORMAL_T = '''
def model(y=None, T=60):
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    with numpyro.plate("t", T):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
'''

CLEAN_DIAG = {"rhat_max": 1.001, "ess_min": 900.0, "divergences": 0, "ebfmi_min": 1.0}
BAD_DIAG = {"rhat_max": 1.31, "ess_min": 40.0, "divergences": 0, "ebfmi_min": 0.9}


# ---------------------------------------------------------------------------
# Canned fits
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def y400():
    return np.random.default_rng(0).poisson(10.0, size=400)


def _fit(y, lam, *, name="y", diverging=None, energy=None, seed=0, ll=None):
    """InferenceData with a Poisson pointwise log-likelihood under ``lam``
    draws of shape (chains, draws): exactly what az.from_numpyro would carry."""
    rng = np.random.default_rng(seed)
    lam = np.asarray(lam, dtype=float)
    chains, draws = lam.shape
    if ll is None:
        ll = poisson.logpmf(np.asarray(y)[None, None, :], lam[..., None])
    if diverging is None:
        diverging = np.zeros((chains, draws), dtype=bool)
    if energy is None:
        energy = rng.normal(0.0, 1.0, (chains, draws))
    return az.from_dict(
        posterior={"lam": lam},
        sample_stats={"energy": energy, "diverging": diverging},
        log_likelihood={name: ll},
        observed_data={name: np.asarray(y)},
    )


def _lam(center, sd=0.3, seed=1, chains=2, draws=600):
    return np.random.default_rng(seed).normal(center, sd, (chains, draws))


def _fit_normal(y, mu, *, sigma=1.0, name="y", seed=0):
    """InferenceData with a Gaussian pointwise log-likelihood under ``mu``
    draws of shape (chains, draws) — the continuous counterpart of ``_fit``
    (needed for the float32 round-trip probes: a Poisson log-pmf is undefined
    off the integers)."""
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=float)
    chains, draws = mu.shape
    ll = norm.logpdf(np.asarray(y, dtype=float)[None, None, :], mu[..., None], sigma)
    return az.from_dict(
        posterior={"mu": mu},
        sample_stats={"energy": rng.normal(0.0, 1.0, (chains, draws)),
                      "diverging": np.zeros((chains, draws), dtype=bool)},
        log_likelihood={name: ll},
        observed_data={name: np.asarray(y)},
    )


@pytest.fixture
def stubs(monkeypatch):
    """Stub the model-touching instruments; return the fit table ``fake_run_mcmc``
    reads (code string -> InferenceData). ``compute_diagnostics`` and
    ``compare_models`` stay REAL unless a test replaces them."""
    fits: dict = {}

    def fake_run_mcmc(code, data, *args, **kwargs):
        return None, fits[code]

    monkeypatch.setattr(validator, "run_mcmc", fake_run_mcmc)
    monkeypatch.setattr(validator, "run_ppc", lambda *a, **k: (True, []))
    monkeypatch.setattr(resolution_mod, "audit_resolution", lambda *a, **k: None)
    return fits


def _validate(fits, y, refined=POISSON, original=POISSON_BASELINE, **kw):
    kw.setdefault("auto_cv", False)  # escalation needs real refits; tested separately
    return run_full_validation(refined, original, {"y": y}, framework="numpyro",
                               fast=True, **kw)


# ---------------------------------------------------------------------------
# Convergence gates (G): each reason string, from the real compute_diagnostics
# ---------------------------------------------------------------------------

class TestConvergenceGates:
    def test_divergences(self, stubs, y400):
        div = np.zeros((2, 600), dtype=bool)
        div[0, :4] = True
        div[1, :3] = True
        stubs[POISSON] = _fit(y400, _lam(10.0), diverging=div)
        stubs[POISSON_BASELINE] = _fit(y400, _lam(13.0), seed=2)
        v = _validate(stubs, y400)
        assert v.converged is False and v.divergences == 7
        assert "7 divergent transitions" in v.failure_reasons

    def test_rhat(self, stubs, y400):
        lam = _lam(10.0)
        lam[1] += 2.0  # chains disagree -> R-hat >> 1.01
        stubs[POISSON] = _fit(y400, lam)
        stubs[POISSON_BASELINE] = _fit(y400, _lam(13.0), seed=2)
        v = _validate(stubs, y400)
        diag = validator.compute_diagnostics(stubs[POISSON])
        assert diag["rhat_max"] > validator.RHAT_THRESHOLD
        assert v.converged is False
        assert f"R-hat too high: {diag['rhat_max']:.3f}" in v.failure_reasons

    def test_ess(self, stubs, y400):
        rng = np.random.default_rng(3)
        lam = 10.0 + np.cumsum(rng.normal(0.0, 0.01, (2, 600)), axis=1)  # random walk
        stubs[POISSON] = _fit(y400, lam)
        stubs[POISSON_BASELINE] = _fit(y400, _lam(13.0), seed=2)
        v = _validate(stubs, y400)
        diag = validator.compute_diagnostics(stubs[POISSON])
        assert diag["ess_min"] < validator.ESS_THRESHOLD
        assert v.converged is False
        assert f"ESS too low: {diag['ess_min']:.0f}" in v.failure_reasons

    def test_ebfmi(self, stubs, y400):
        rng = np.random.default_rng(4)
        energy = np.cumsum(rng.normal(0.0, 0.1, (2, 600)), axis=1)  # sticky energy
        stubs[POISSON] = _fit(y400, _lam(10.0), energy=energy)
        stubs[POISSON_BASELINE] = _fit(y400, _lam(13.0), seed=2)
        v = _validate(stubs, y400)
        diag = validator.compute_diagnostics(stubs[POISSON])
        assert diag["ebfmi_min"] < validator.EBFMI_THRESHOLD
        assert v.converged is False
        assert f"E-BFMI low: {diag['ebfmi_min']:.2f}" in v.failure_reasons

    def test_clean_fit_passes_every_gate(self, stubs, y400):
        stubs[POISSON] = _fit(y400, _lam(10.0))
        stubs[POISSON_BASELINE] = _fit(y400, _lam(13.0), seed=2)
        v = _validate(stubs, y400)
        assert v.converged is True
        assert not any(r.startswith(("R-hat", "ESS", "E-BFMI")) or "divergent" in r
                       for r in v.failure_reasons)


# ---------------------------------------------------------------------------
# Significance gate (S)
# ---------------------------------------------------------------------------

class TestSignificanceGate:
    def test_no_significant_improvement_refusal(self, stubs, y400):
        # the same posterior law on both sides: delta-eLPD is Monte-Carlo noise
        stubs[POISSON] = _fit(y400, _lam(10.0, seed=11))
        stubs[POISSON_BASELINE] = _fit(y400, _lam(10.0, seed=12), seed=2)
        v = _validate(stubs, y400)
        assert v.converged is True
        reason = next(r for r in v.failure_reasons
                      if r.startswith("no significant improvement: delta-eLPD"))
        assert f"{v.elpd_diff:.2f}" in reason and f"(se {v.elpd_diff_se:.2f})" in reason

    def test_positive_acceptance(self, stubs, y400):
        # refined near the truth (10), baseline off by 30%: a real, large win
        stubs[POISSON] = _fit(y400, _lam(10.0))
        stubs[POISSON_BASELINE] = _fit(y400, _lam(13.0), seed=2)
        v = _validate(stubs, y400)
        assert v.converged is True
        assert v.failure_reasons == []
        assert v.elpd_diff < 0 and v.elpd_diff_se > 0
        assert abs(v.elpd_diff) >= 2 * v.elpd_diff_se
        assert v.loo_delta < 0 and v.waic_delta < 0
        assert v.khat_max is not None and np.isfinite(v.khat_max)
        assert v.posterior_predictive_check is True
        assert v.repair is None and v.s_gate is None
        assert v.confidence in ("HIGH", "MEDIUM")


# ---------------------------------------------------------------------------
# Defect V2 — k-hat = inf gates and escalates; NaN does neither
# ---------------------------------------------------------------------------

def _comparison(diff=-20.0, se=3.0, khat=0.3, **extra):
    return {"waic": 10.0, "loo": 11.0, "waic_delta": 2 * diff, "loo_delta": 2 * diff,
            "elpd_diff": diff, "elpd_diff_se": se,
            "significant_improvement": bool(diff < 0 and abs(diff) >= 2 * se),
            "khat_max": khat, "khat_refined": khat, "khat_original": 0.3, **extra}


class TestInfiniteKhat:
    @pytest.mark.parametrize("k,expected", [
        (None, False), (float("nan"), False), (0.3, False), (0.7, False),
        (0.71, True), (float("inf"), True), (-float("inf"), True),
    ])
    def test_predicate(self, k, expected):
        assert _khat_unreliable(k) is expected

    def test_inf_gate_fires(self, stubs, y400):
        # arviz returns k = inf for degenerate importance weights; the real
        # compare_models now refuses such fits earlier (V1c), so the gate's
        # own inf branch is driven through a canned comparison here.
        stubs[POISSON] = _fit(y400, _lam(10.0))
        stubs[POISSON_BASELINE] = _fit(y400, _lam(13.0), seed=2)
        stubs_cmp = _comparison(khat=float("inf"))
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(validator, "compare_models", lambda r, o: dict(stubs_cmp))
            v = _validate(stubs, y400)
        assert v.converged is True  # comparison-reliability failure only
        # F4 wording: arviz returns inf whenever at most 4 weights exceed the
        # tail cutoff — degenerate (constant) columns, but tied weights too
        assert ("PSIS unreliable: refined-model Pareto-k-hat is inf "
                "(degenerate or tied importance weights)") in v.failure_reasons
        assert v.khat_max == float("inf")

    def test_finite_wording_unchanged_and_nan_silent(self, stubs, y400):
        stubs[POISSON] = _fit(y400, _lam(10.0))
        stubs[POISSON_BASELINE] = _fit(y400, _lam(13.0), seed=2)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(validator, "compare_models",
                       lambda r, o: _comparison(khat=0.95))
            high = _validate(stubs, y400)
            mp.setattr(validator, "compare_models",
                       lambda r, o: _comparison(khat=float("nan")))
            nan = _validate(stubs, y400)
        assert "PSIS unreliable: refined-model Pareto-k-hat 0.95 > 0.7" in high.failure_reasons
        assert not any("PSIS" in r for r in nan.failure_reasons)

    def test_inf_triggers_the_escalation_ladder_nan_does_not(self, stubs, y400):
        from compass import crossval

        stubs[POISSON] = _fit(y400, _lam(10.0))
        stubs[POISSON_BASELINE] = _fit(y400, _lam(13.0), seed=2)
        calls: list = []
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(crossval, "infer_cv_spec",
                       lambda *a, **k: calls.append("infer_cv_spec") or None)
            mp.setattr(crossval, "reloo_refined",
                       lambda *a, **k: calls.append("reloo_refined") or None)
            mp.setattr(validator, "compare_models",
                       lambda r, o: _comparison(khat=float("inf")))
            v = _validate(stubs, y400, auto_cv=True)
            assert calls == ["infer_cv_spec", "reloo_refined"]  # ladder attempted
            assert any("Pareto-k-hat is inf" in r for r in v.failure_reasons)
            calls.clear()
            mp.setattr(validator, "compare_models",
                       lambda r, o: _comparison(khat=float("nan")))
            _validate(stubs, y400, auto_cv=True)
            assert calls == []  # NaN: no information, no escalation


# ---------------------------------------------------------------------------
# Defect V1a — masked / scaled observed sites, decided by tracing
# ---------------------------------------------------------------------------

class TestMaskedSiteGuard:
    @pytest.mark.parametrize("code,site", [
        (POISSON_MASKED_HANDLER, "y"), (POISSON_MASKED_METHOD, "y"),
        (POISSON_OBS_MASK_KW, "y_observed"),
    ])
    def test_masked_observed_site_is_named_with_its_count(self, y400, code, site):
        reasons = check_observed_sites(code, {"y": y400})
        assert len(reasons) == 1
        excluded = int((y400 <= 4).sum())
        assert excluded > 0
        assert reasons[0].startswith(f"observed site {site!r} is masked")
        assert f"{excluded} of 400 observations are excluded" in reasons[0]

    def test_scaled_observed_site(self, y400):
        reasons = check_observed_sites(POISSON_SCALED, {"y": y400})
        assert len(reasons) == 1
        assert reasons[0].startswith("observed site 'y' carries a log-likelihood scale (0.5")

    @pytest.mark.parametrize("code", [POISSON, POISSON_MASK_ALL_TRUE, POISSON_MASKED_LATENT])
    def test_clean_programs_pass(self, y400, code):
        # plain; a mask that is identically True; a masked PRIOR (latent site)
        assert check_observed_sites(code, {"y": y400}) == []

    def test_mask_decided_by_trace_not_text(self):
        # the mask expression is all-True on THIS data: harmless, so it passes
        y = np.full(50, 9)
        assert check_observed_sites(POISSON_MASKED_HANDLER, {"y": y}) == []

    def test_refusal_precedes_the_comparison(self, stubs, y400):
        stubs[POISSON_MASKED_HANDLER] = _fit(y400, _lam(10.0))
        stubs[POISSON_BASELINE] = _fit(y400, _lam(13.0), seed=2)

        def must_not_run(*a, **k):
            raise AssertionError("compare_models ran on a structurally refused candidate")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(validator, "compare_models", must_not_run)
            v = _validate(stubs, y400, refined=POISSON_MASKED_HANDLER)
        assert v.converged is True  # a comparison verdict, not a convergence one
        reason = next(r for r in v.failure_reasons
                      if r.startswith("model comparison refused: "))
        assert "observed site 'y' is masked" in reason
        assert v.elpd_diff is None and v.elpd_diff_se is None and v.khat_max is None
        assert v.repair is None and v.confidence == "LOW"

    def test_untraceable_program_is_refused(self, stubs, y400):
        bad = "def model(y=None):\n    raise RuntimeError('boom')\n"
        stubs[bad] = _fit(y400, _lam(10.0))
        stubs[POISSON_BASELINE] = _fit(y400, _lam(13.0), seed=2)
        v = _validate(stubs, y400, refined=bad)
        reason = next(r for r in v.failure_reasons
                      if r.startswith("model comparison refused: "))
        assert "could not be traced" in reason and "boom" in reason

    def test_stan_skips_the_numpyro_trace(self, stubs, y400):
        # Stan source cannot be traced as NumPyro; the guard is numpyro-only
        stan = "data { int N; }"
        stubs[stan] = _fit(y400, _lam(10.0))
        stubs[POISSON_BASELINE] = _fit(y400, _lam(13.0), seed=2)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(validator, "run_ppc", lambda *a, **k: (None, []))
            v = run_full_validation(stan, POISSON_BASELINE, {"y": y400},
                                    framework="stan", fast=True, auto_cv=False)
        assert not any(r.startswith("model comparison refused") for r in v.failure_reasons)


# ---------------------------------------------------------------------------
# Defect V1b — the two fits must have scored the same observations
# ---------------------------------------------------------------------------

class TestObservedDataIdentity:
    PREFIX = "model comparison refused: observed data differ between candidate and baseline"

    def test_single_renamed_site_with_identical_values_is_compared(self, y400):
        # a refinement may rename its one observed site: the observations decide
        comp = compare_models(_fit(y400, _lam(10.0), name="obs"), _fit(y400, _lam(13.0)))
        assert comp["significant_improvement"] is True

    def test_renamed_site_with_different_values_is_refused(self, y400):
        clipped = np.minimum(y400, 12)
        with pytest.raises(ComparisonRefused, match="observed values differ") as e:
            compare_models(_fit(clipped, _lam(10.0), name="obs"), _fit(y400, _lam(13.0)))
        assert "'obs' (baseline 'y')" in str(e.value)

    def test_different_variable_sets_are_refused(self, y400):
        two = _fit(y400, _lam(10.0))
        two.log_likelihood["y2"] = two.log_likelihood["y"]
        with pytest.raises(ComparisonRefused, match="log-likelihood variables") as e:
            compare_models(two, _fit(y400, _lam(13.0)))
        assert str(e.value).startswith(self.PREFIX)

    def test_different_shapes(self, y400):
        with pytest.raises(ComparisonRefused, match=r"\(300,\) vs \(400,\) observations"):
            compare_models(_fit(y400[:300], _lam(10.0)), _fit(y400, _lam(13.0)))

    def test_different_values(self, y400):
        clipped = np.minimum(y400, 12)
        n_diff = int((clipped != y400).sum())
        with pytest.raises(ComparisonRefused,
                           match=f"observed values differ at {n_diff} of 400 positions"):
            compare_models(_fit(clipped, _lam(10.0)), _fit(y400, _lam(13.0)))

    def test_dtype_only_difference_is_allowed(self, y400):
        comp = compare_models(_fit(y400.astype(np.int32), _lam(10.0)),
                              _fit(y400.astype(np.float64), _lam(13.0), seed=2))
        assert comp["significant_improvement"] is True

    def test_mismatch_lands_in_failure_reasons(self, stubs, y400):
        stubs[POISSON] = _fit(np.minimum(y400, 12), _lam(10.0))
        stubs[POISSON_BASELINE] = _fit(y400, _lam(13.0), seed=2)
        v = _validate(stubs, y400)
        assert v.converged is True
        assert any(r.startswith(self.PREFIX) for r in v.failure_reasons)
        assert v.elpd_diff is None


# ---------------------------------------------------------------------------
# Defect V1c — degenerate pointwise columns of the refined fit
# ---------------------------------------------------------------------------

class TestDegenerateColumns:
    def test_constant_columns_are_counted_and_refused(self, y400):
        lam = _lam(10.0)
        ll = poisson.logpmf(y400[None, None, :], lam[..., None])
        ll[..., :9] = 0.0  # what MaskedDistribution.log_prob returns on masked points
        with pytest.raises(ComparisonRefused, match="9 of 400 pointwise log-likelihood") as e:
            compare_models(_fit(y400, lam, ll=ll), _fit(y400, _lam(13.0), seed=2))
        # F4 wording: a saturated column (probabilities clipped to 1) is the
        # third way to a constant log-likelihood, next to masking and clamping
        assert "constant across all posterior draws (masked, clamped or saturated)" in str(e.value)
        assert "at positions where the baseline's are not" in str(e.value)

    def test_non_finite_column_is_degenerate(self, y400):
        lam = _lam(10.0)
        ll = poisson.logpmf(y400[None, None, :], lam[..., None])
        ll[0, 5, 17] = -np.inf  # one draw outside the support
        with pytest.raises(ComparisonRefused, match="1 of 400 pointwise"):
            compare_models(_fit(y400, lam, ll=ll), _fit(y400, _lam(13.0), seed=2))

    def test_original_side_is_not_gated_here(self, y400):
        # a degenerate ORIGINAL inflates the baseline (biases AGAINST acceptance,
        # conservative) — not refused by this guard
        lam = _lam(13.0)
        ll = poisson.logpmf(y400[None, None, :], lam[..., None])
        ll[..., :9] = 0.0
        comp = compare_models(_fit(y400, _lam(10.0)), _fit(y400, lam, ll=ll, seed=2))
        assert set(comp) >= {"elpd_diff", "significant_improvement"}

    def test_degenerate_refined_fit_lands_in_failure_reasons(self, stubs, y400):
        lam = _lam(10.0)
        ll = poisson.logpmf(y400[None, None, :], lam[..., None])
        ll[..., :9] = 0.0
        stubs[POISSON] = _fit(y400, lam, ll=ll)
        stubs[POISSON_BASELINE] = _fit(y400, _lam(13.0), seed=2)
        v = _validate(stubs, y400)
        assert any(r.startswith("model comparison refused: 9 of 400 pointwise")
                   for r in v.failure_reasons)
        assert v.converged is True and v.elpd_diff is None


# ---------------------------------------------------------------------------
# Defect V3 — az.compare scale pinned against the session rcParams
# ---------------------------------------------------------------------------

def test_compare_models_ignores_session_ic_scale(y400):
    ref, orig = _fit(y400, _lam(10.0)), _fit(y400, _lam(13.0), seed=2)
    baseline = compare_models(ref, orig)
    assert az.rcParams["stats.ic_scale"] == "log"  # precondition of the test
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(az.rcParams, "stats.ic_scale", "deviance")
        mp.setitem(az.rcParams, "stats.information_criterion", "waic")
        under_deviance = compare_models(ref, orig)
    assert az.rcParams["stats.ic_scale"] == "log"
    for key in ("elpd_diff", "elpd_diff_se", "loo_delta", "waic_delta", "khat_max"):
        assert under_deviance[key] == pytest.approx(baseline[key]), key
    assert under_deviance["significant_improvement"] is baseline["significant_improvement"]


# ---------------------------------------------------------------------------
# The repair-equivalence clause (TOST) through the same stubs
# ---------------------------------------------------------------------------

class TestRepairClause:
    def test_tost_repair_acceptance(self, stubs, y400):
        refined_fit = _fit(y400, _lam(10.0))
        original_fit = _fit(y400, _lam(10.0, seed=12), seed=2)
        stubs[POISSON] = refined_fit
        stubs[POISSON_BASELINE] = original_fit

        def split_diag(idata, mcmc=None):
            # baseline fails G, and still fails at the doubled-budget retry
            return dict(BAD_DIAG if idata is original_fit else CLEAN_DIAG)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(validator, "compute_diagnostics", split_diag)
            mp.setattr(validator, "compare_models",
                       lambda r, o: _comparison(diff=0.5, se=1.0))
            v = _validate(stubs, y400)
        assert v.repair is not None and v.repair.startswith("accepted as computational repair")
        assert "R-hat 1.310" in v.repair and "still failing at doubled budget" in v.repair
        assert v.converged is True and v.failure_reasons == []
        assert v.elpd_diff == pytest.approx(0.5) and v.elpd_diff_se == pytest.approx(1.0)

    def test_tost_repair_refused_when_khat_is_inf(self, stubs, y400):
        refined_fit = _fit(y400, _lam(10.0))
        original_fit = _fit(y400, _lam(10.0, seed=12), seed=2)
        stubs[POISSON] = refined_fit
        stubs[POISSON_BASELINE] = original_fit
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(validator, "compute_diagnostics",
                       lambda idata, mcmc=None: dict(
                           BAD_DIAG if idata is original_fit else CLEAN_DIAG))
            mp.setattr(validator, "compare_models",
                       lambda r, o: _comparison(diff=0.5, se=1.0, khat=float("inf")))
            v = _validate(stubs, y400)
        assert v.repair is None
        assert any("Pareto-k-hat is inf" in r for r in v.failure_reasons)
        assert any(r.startswith("original-model fit unreliable") for r in v.failure_reasons)


# ---------------------------------------------------------------------------
# The real thing (slow): the masked Poisson candidate end to end
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_masked_candidate_is_refused_end_to_end(trace_cache_dir):
    """The 2026-11-11 probe: y ~ Poisson(10), n = 400, candidate = the same
    model under handlers.mask(mask=(y > 4)). Before the fix it was ACCEPTED as
    a significant improvement (elpd_diff -49 +/- 14.5, PPC passing); now the
    structural guard refuses the comparison before any elpd is computed."""
    y = np.random.default_rng(0).poisson(10.0, size=400)
    excluded = int((y <= 4).sum())
    assert excluded > 0, "probe data must contain masked points"
    v = run_full_validation(POISSON_MASKED_HANDLER, POISSON, {"y": y},
                            framework="numpyro", fast=True, num_warmup=300,
                            num_samples=600, num_chains=2, seed=0,
                            cache_dir=trace_cache_dir)
    reason = next((r for r in v.failure_reasons
                   if r.startswith("model comparison refused: observed site 'y' is masked")),
                  None)
    assert reason is not None, v.failure_reasons
    assert f"{excluded} of 400 observations are excluded" in reason
    assert v.elpd_diff is None and v.khat_max is None and v.repair is None
    assert v.converged, f"the masked fit itself should converge: {v.failure_reasons}"


# ---------------------------------------------------------------------------
# F1 — observed values compared at the coarser precision
# ---------------------------------------------------------------------------

class TestObservedValueTolerance:
    """A candidate that passes its float64 response through ``jnp.asarray``
    (x64 off) observes a float32 copy; ``az.from_numpyro`` records that copy as
    the fit's observed_data, and exact equality refused 36 released
    candidates with 'observed values differ at N of N positions'."""

    def test_float32_cast_on_one_side_is_compared(self):
        rng = np.random.default_rng(5)
        y = rng.normal(5.0, 1.0, 300)                       # float64, not float32-representable
        assert not np.array_equal(y.astype(np.float32), y)  # the old check's trigger
        comp = compare_models(_fit_normal(y.astype(np.float32), _lam(5.0, sd=0.05)),
                              _fit_normal(y, _lam(7.0, sd=0.05), seed=2))
        assert comp["significant_improvement"] is True      # compared, and the truth wins

    def test_clipped_float32_response_is_still_refused(self):
        rng = np.random.default_rng(5)
        y = rng.normal(5.0, 1.0, 300)
        clipped = np.minimum(y, 5.5).astype(np.float32)
        n_diff = int((y > 5.5).sum())
        assert n_diff > 0
        with pytest.raises(ComparisonRefused,
                           match=f"observed values differ at {n_diff} of 300 positions"):
            compare_models(_fit_normal(clipped, _lam(5.0, sd=0.05)),
                           _fit_normal(y, _lam(7.0, sd=0.05), seed=2))

    def test_float64_pairs_keep_exact_equality(self):
        rng = np.random.default_rng(6)
        y = rng.normal(5.0, 1.0, 100)
        with pytest.raises(ComparisonRefused, match="observed values differ at 100 of 100"):
            compare_models(_fit_normal(y + 1e-9, _lam(5.0, sd=0.05)),
                           _fit_normal(y, _lam(7.0, sd=0.05), seed=2))

    def test_same_observed_values_dtype_rules(self):
        same = validator._same_observed_values
        rng = np.random.default_rng(7)
        y = rng.normal(size=16)
        assert same(y.astype(np.float32), y) == (True, 0)            # cast on one side
        assert same(y.astype(np.float32), y.astype(np.float32)) == (True, 0)
        assert same(y + 1e-9, y)[0] is False                         # both float64: exact
        assert same(np.arange(4, dtype=np.int64), np.arange(4, dtype=np.int32)) == (True, 0)
        assert same(np.array([True, False]), np.array([1, 0])) == (True, 0)
        assert same(np.arange(4), np.arange(4) + 1) == (False, 4)     # integers: exact, counted
        assert same(np.arange(4.0), np.arange(4)) == (True, 0)        # int vs float: tolerant
        assert same(np.zeros(3), np.zeros(4)) == (False, -1)          # shapes
        obj = np.array([{"a": 1}, {"a": 2}], dtype=object)
        assert same(obj, obj.copy()) == (True, 0)
        assert same(obj, np.array([{"a": 1}, {"a": 3}], dtype=object)) == (False, -1)
        z = y.copy()
        z[0] = np.nan
        assert same(z.astype(np.float32), z) == (True, 0)            # NaN == NaN on both paths
        assert same(z, z.copy()) == (True, 0)


# ---------------------------------------------------------------------------
# F4 / F5 — identity-check wording for granularity, Stan-style observed_data
# ---------------------------------------------------------------------------

class TestIdentityCheckWording:
    def test_pointwise_granularity_mismatch_is_named(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=(14, 4))
        mu = _lam(0.0, sd=0.1)
        cell_ll = norm.logpdf(y[None, None], mu[..., None, None], 1.0)   # (2, 600, 14, 4)
        row_ll = cell_ll.sum(axis=-1)                                     # to_event(1): (2, 600, 14)
        refined = az.from_dict(posterior={"mu": mu}, log_likelihood={"y": row_ll},
                               observed_data={"y": y})
        original = az.from_dict(posterior={"mu": mu}, log_likelihood={"y": cell_ll},
                                observed_data={"y": y})
        with pytest.raises(ComparisonRefused, match=(
                r"pointwise granularity differs \(shapes \(14,\) vs \(14, 4\)\); "
                r"score both models at the same level")) as e:
            compare_models(refined, original)
        assert "observed data differ" not in str(e.value)

    def test_same_ndim_size_mismatch_keeps_the_observation_count_wording(self, y400):
        with pytest.raises(ComparisonRefused, match=r"\(300,\) vs \(400,\) observations"):
            compare_models(_fit(y400[:300], _lam(10.0)), _fit(y400, _lam(13.0)))

    def test_stan_style_log_lik_variable_compares_the_observed_arrays(self, y400):
        # Stan fits carry the pointwise variable as ``log_lik`` and the data
        # block's arrays as observed_data (F5): identity is checked on the
        # arrays both fits share, not degraded to shapes only.
        def fit(y, lam):
            ll = poisson.logpmf(np.asarray(y)[None, None, :], lam[..., None])
            return az.from_dict(posterior={"lam": lam}, log_likelihood={"log_lik": ll},
                                observed_data={"y": np.asarray(y), "x": np.arange(400.0)})
        comp = compare_models(fit(y400, _lam(10.0)), fit(y400, _lam(13.0, seed=2)))
        assert comp["significant_improvement"] is True
        with pytest.raises(ComparisonRefused, match="variable 'y': observed values differ"):
            compare_models(fit(np.minimum(y400, 12), _lam(10.0)), fit(y400, _lam(13.0, seed=2)))


# ---------------------------------------------------------------------------
# F2 — degenerate columns are judged against the baseline's at the same positions
# ---------------------------------------------------------------------------

class TestSharedDegenerateColumns:
    def _pair(self, y400, ref_const, orig_const, n=9):
        lam_r, lam_o = _lam(10.0), _lam(13.0, seed=2)
        ll_r = poisson.logpmf(y400[None, None, :], lam_r[..., None])
        ll_o = poisson.logpmf(y400[None, None, :], lam_o[..., None])
        ll_r[..., :n] = ref_const
        ll_o[..., :n] = orig_const
        return _fit(y400, lam_r, ll=ll_r), _fit(y400, lam_o, ll=ll_o, seed=2)

    def test_shared_constant_columns_are_exempt_and_skip_the_khat(self, y400):
        # the same missing cells, masked in both programs: log_prob is exactly
        # 0 on both sides, arviz's k-hat is inf there by construction
        ref, orig = self._pair(y400, 0.0, 0.0)
        shared = validator.shared_degenerate_columns(ref, orig)
        assert shared is not None and shared.shape == (400,)
        assert shared[:9].all() and shared.sum() == 9
        comp = compare_models(ref, orig)                    # compared, not refused
        assert comp["significant_improvement"] is True
        assert np.isfinite(comp["khat_refined"]) and np.isfinite(comp["khat_original"])
        assert np.isfinite(comp["khat_max"])

    def test_shared_positions_with_different_constants_are_refused(self, y400):
        # a hand-picked constant (a Delta with its own log density) is not a
        # shared missing cell: the refined side gains 5 nats per point for free
        ref, orig = self._pair(y400, 5.0, 0.0)
        with pytest.raises(ComparisonRefused, match="9 of 400 pointwise log-likelihood"):
            compare_models(ref, orig)
        assert validator.shared_degenerate_columns(ref, orig) is None

    def test_non_finite_columns_stay_degenerate_even_when_shared(self, y400):
        lam_r, lam_o = _lam(10.0), _lam(13.0, seed=2)
        ll_r = poisson.logpmf(y400[None, None, :], lam_r[..., None])
        ll_o = poisson.logpmf(y400[None, None, :], lam_o[..., None])
        ll_r[0, 5, 17] = -np.inf
        ll_o[0, 5, 17] = -np.inf
        with pytest.raises(ComparisonRefused, match="1 of 400 pointwise"):
            compare_models(_fit(y400, lam_r, ll=ll_r), _fit(y400, lam_o, ll=ll_o, seed=2))

    def test_refined_only_degenerate_columns_are_still_counted(self, y400):
        ref, orig = self._pair(y400, 0.0, 0.0, n=9)
        ll_r = np.asarray(ref.log_likelihood["y"].values).copy()
        ll_r[..., 9:12] = 0.0                                # three cells only the candidate drops
        ref2 = _fit(y400, np.asarray(ref.posterior["lam"].values), ll=ll_r)
        with pytest.raises(ComparisonRefused, match="3 of 400 pointwise"):
            compare_models(ref2, orig)

    def test_shared_mask_fits_pass_through_run_full_validation(self, stubs, y400):
        mask = np.random.default_rng(1).uniform(size=400) < 0.8
        lam_r, lam_o = _lam(10.0), _lam(13.0, seed=2)
        ll_r = np.where(mask, poisson.logpmf(y400[None, None, :], lam_r[..., None]), 0.0)
        ll_o = np.where(mask, poisson.logpmf(y400[None, None, :], lam_o[..., None]), 0.0)
        stubs[MASKED_DATA_NB2] = _fit(y400, lam_r, ll=ll_r)
        stubs[MASKED_DATA_POISSON] = _fit(y400, lam_o, ll=ll_o, seed=2)
        v = run_full_validation(MASKED_DATA_NB2, MASKED_DATA_POISSON, {"y": y400, "mask": mask},
                                framework="numpyro", fast=True, auto_cv=False)
        assert v.converged is True
        assert v.failure_reasons == [], v.failure_reasons
        assert v.elpd_diff < 0 and v.khat_max is not None and np.isfinite(v.khat_max)


# ---------------------------------------------------------------------------
# F2 — the structural guard is comparative: masks and scales the baseline shares
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def masked_data(y400):
    mask = np.random.default_rng(1).uniform(size=400) < 0.8
    return {"y": y400, "mask": mask}


class TestComparativeMaskGuard:
    def test_shared_data_mask_passes(self, masked_data):
        assert int((~masked_data["mask"]).sum()) > 0
        assert check_observed_sites(MASKED_DATA_NB2, masked_data,
                                    original_code=MASKED_DATA_POISSON) == []

    def test_single_program_form_still_refuses_the_mask(self, masked_data):
        reasons = check_observed_sites(MASKED_DATA_POISSON, masked_data)
        excluded = int((~masked_data["mask"]).sum())
        assert len(reasons) == 1
        assert reasons[0].startswith("observed site 'y' is masked")
        assert f"{excluded} of 400 observations are excluded from its log-likelihood, so" in reasons[0]

    def test_candidate_widening_the_shared_mask_is_refused(self, masked_data):
        reasons = check_observed_sites(MASKED_DATA_NB2_EXTRA, masked_data,
                                       original_code=MASKED_DATA_POISSON)
        extra = int((masked_data["mask"] & (masked_data["y"] <= 4)).sum())
        assert extra > 0
        assert len(reasons) == 1
        assert reasons[0].startswith("observed site 'y' is masked")
        assert (f"{extra} of 400 observations are excluded from its log-likelihood but "
                "scored by the baseline") in reasons[0]

    def test_candidate_adding_a_mask_to_an_unmasked_baseline_is_refused(self, masked_data):
        reasons = check_observed_sites(MASKED_DATA_NB2, masked_data,
                                       original_code=UNMASKED_NB2_WITH_MASK_KWARG)
        excluded = int((~masked_data["mask"]).sum())
        assert len(reasons) == 1
        # an unmasked baseline scores every point: the single-program wording applies
        assert reasons[0].startswith("observed site 'y' is masked")
        assert f"{excluded} of 400 observations are excluded from its log-likelihood, so" in reasons[0]

    def test_same_mask_plus_scale_is_refused(self, masked_data):
        reasons = check_observed_sites(MASKED_DATA_NB2_SCALED, masked_data,
                                       original_code=MASKED_DATA_POISSON)
        assert len(reasons) == 1
        assert reasons[0].startswith("observed site 'y' carries a log-likelihood scale (0.5")

    def test_equal_scales_pass_and_different_scales_refuse(self, y400):
        assert check_observed_sites(NB2_N_SCALED, {"y": y400}, original_code=POISSON_N_SCALED) == []
        reasons = check_observed_sites(NB2_N_SCALED_OTHER, {"y": y400},
                                       original_code=POISSON_N_SCALED)
        assert len(reasons) == 1
        assert reasons[0].startswith("observed site 'y' carries a log-likelihood scale (0.25")
        assert "that differs from the baseline's (0.5)" in reasons[0]
        # scaled candidate against an UNSCALED baseline: the single-program wording
        reasons = check_observed_sites(NB2_N_SCALED, {"y": y400}, original_code=POISSON_N)
        assert len(reasons) == 1 and "a reweighted observation set" in reasons[0]

    def test_unmasked_candidate_against_a_masked_baseline_passes(self, masked_data):
        # the candidate scores MORE points than the baseline (conservative)
        assert check_observed_sites(UNMASKED_NB2_WITH_MASK_KWARG, masked_data,
                                    original_code=MASKED_DATA_POISSON) == []

    def test_structure_record_is_descriptive(self, masked_data):
        out = validator.compare_observed_structure(MASKED_DATA_NB2, MASKED_DATA_POISSON, masked_data)
        assert out["reasons"] == [] and out["notes"] == []
        assert out["sites"]["y"]["paired_with"] == "y"
        assert out["sites"]["y"]["excluded"] == int((~masked_data["mask"]).sum())
        assert out["sites"]["y"]["response_dependent"] is False  # probed (mask is a kwarg), honest


# ---------------------------------------------------------------------------
# F3 — the two-faced likelihood (response-dependent likelihood parameters)
# ---------------------------------------------------------------------------

class TestTwoFacedLikelihood:
    PREFIX = "observed site 'y': the likelihood parameters depend on the observed response"

    def test_soft_clamp_is_refused_against_an_honest_baseline(self, y400):
        reasons = check_observed_sites(POISSON_SOFT_CLAMP, {"y": y400}, original_code=POISSON_N)
        assert len(reasons) == 1
        assert reasons[0].startswith(self.PREFIX)
        assert "'rate'" in reasons[0] and "while the baseline's do not" in reasons[0]
        n_low = int((y400 <= 4).sum())
        assert n_low > 0 and f"{n_low} of 400 positions" in reasons[0]
        assert "un-conditioned trace" in reasons[0]

    def test_baseline_sized_from_y_is_probed_by_permutation(self, y400):
        # the baseline's un-conditioned call cannot run (plate from y.shape); it
        # is probed with a permuted response instead, found honest, and the
        # candidate is still refused
        reasons = check_observed_sites(POISSON_SOFT_CLAMP, {"y": y400}, original_code=POISSON)
        assert len(reasons) == 1 and reasons[0].startswith(self.PREFIX)

    def test_honest_candidates_pass(self, y400):
        assert check_observed_sites(NB2_N, {"y": y400}, original_code=POISSON_N) == []
        assert check_observed_sites(POISSON_N, {"y": y400}, original_code=POISSON_N) == []

    def test_ar1_on_both_sides_is_a_note_not_a_refusal(self):
        rng = np.random.default_rng(3)
        y = np.zeros(60)
        for t in range(1, 60):
            y[t] = 0.6 * y[t - 1] + rng.normal()
        notes: list[str] = []
        assert check_observed_sites(AR1_INTERCEPT, {"y": y}, original_code=AR1, notes=notes) == []
        assert len(notes) == 1
        assert notes[0].startswith("observed site 'y': the likelihood parameters depend on the "
                                   "observed response in both the candidate and the baseline")
        assert "'loc'" in notes[0]

    def test_ar1_candidate_against_an_iid_baseline_is_refused(self):
        # the documented trade-off of the comparative rule: a candidate that
        # starts conditioning the likelihood on the response is refused unless
        # the baseline did so too
        rng = np.random.default_rng(4)
        y = rng.normal(size=60)
        reasons = check_observed_sites(AR1_INTERCEPT, {"y": y}, original_code=IID_NORMAL_T)
        assert len(reasons) == 1 and reasons[0].startswith(self.PREFIX)

    def test_unconditionable_candidate_is_not_refused_on_this_check(self, y400):
        # its un-conditioned trace cannot run (plate from y.shape); the PPC
        # reports that separately, the structural guard stays silent
        assert check_observed_sites(POISSON_SOFT_CLAMP_YSHAPE, {"y": y400},
                                    original_code=POISSON) == []

    def test_single_program_form_records_a_note_only(self, y400):
        notes: list[str] = []
        assert check_observed_sites(POISSON_SOFT_CLAMP, {"y": y400}, notes=notes) == []
        assert len(notes) == 1 and "no baseline program was given" in notes[0]

    def test_refusal_reaches_the_failure_reasons(self, stubs, y400):
        stubs[POISSON_SOFT_CLAMP] = _fit(y400, _lam(10.0))
        stubs[POISSON_N] = _fit(y400, _lam(13.0), seed=2)
        v = _validate(stubs, y400, refined=POISSON_SOFT_CLAMP, original=POISSON_N)
        reason = next(r for r in v.failure_reasons if r.startswith("model comparison refused: "))
        assert reason.startswith("model comparison refused: " + self.PREFIX)
        assert v.converged is True and v.elpd_diff is None and v.repair is None

    def test_note_reaches_the_hints(self, stubs):
        rng = np.random.default_rng(3)
        y = np.zeros(60)
        for t in range(1, 60):
            y[t] = 0.6 * y[t - 1] + rng.normal()
        stubs[AR1_INTERCEPT] = _fit_normal(y, _lam(0.0, sd=0.05))
        stubs[AR1] = _fit_normal(y, _lam(0.0, sd=0.05, seed=12), seed=2)
        v = run_full_validation(AR1_INTERCEPT, AR1, {"y": y}, framework="numpyro",
                                fast=True, auto_cv=False)
        assert not any(r.startswith("model comparison refused") for r in v.failure_reasons)
        assert any("in both the candidate and the baseline" in h for h in v.misspecification_hints)


# ---------------------------------------------------------------------------
# The real thing (slow): the shared-mask design is compared, the soft clamp refused
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_shared_mask_design_is_compared_end_to_end(trace_cache_dir):
    """F2: the standard missing-data design — BOTH programs apply the data's
    own mask — was refused twice before the fix (masked site, then constant-0
    columns). Poisson vs NegativeBinomial2 on overdispersed counts with ~20%
    missing cells must be compared, and the true overdispersion certified. The
    missing cells keep their simulated values so the PPC, which compares the
    full arrays, is not confounded."""
    rng = np.random.default_rng(2)
    n = 120
    lam = np.exp(2.0 + 0.7 * rng.normal(size=n))
    y = rng.poisson(lam)
    mask = rng.uniform(size=n) < 0.8
    assert 0 < int((~mask).sum()) < n
    data = {"y": y, "mask": mask, "N": n}
    v = run_full_validation(MASKED_DATA_NB2, MASKED_DATA_POISSON, data, framework="numpyro",
                            fast=True, num_warmup=300, num_samples=800, num_chains=2,
                            seed=0, cache_dir=trace_cache_dir)
    assert not any(r.startswith("model comparison refused") for r in v.failure_reasons), \
        v.failure_reasons
    assert v.elpd_diff is not None and v.elpd_diff < 0 and v.elpd_diff_se > 0
    assert v.loo_delta is not None and v.loo_delta < 0
    assert v.khat_max is not None and np.isfinite(v.khat_max)   # the shared cells' inf is skipped
    assert v.converged, v.failure_reasons
    assert v.failure_reasons == [], v.failure_reasons


@pytest.mark.slow
def test_soft_clamp_candidate_is_refused_end_to_end(trace_cache_dir):
    """F3: ``rate = lam if y is None else jnp.where(y > 4, lam, 0.5 * y + 0.2 * lam)``
    — an honest Poisson when sampled (so the PPC passes), a soft clamp when
    scored — was ACCEPTED (elpd_diff -26.4 +/- 8.9, k-hat 0.12). The structural
    guard now refuses it before any elpd is computed."""
    y = np.random.default_rng(0).poisson(10.0, size=400)
    v = run_full_validation(POISSON_SOFT_CLAMP, POISSON_N, {"y": y}, framework="numpyro",
                            fast=True, num_warmup=300, num_samples=600, num_chains=2,
                            seed=0, cache_dir=trace_cache_dir)
    reason = next((r for r in v.failure_reasons if r.startswith(
        "model comparison refused: observed site 'y': the likelihood parameters depend on "
        "the observed response")), None)
    assert reason is not None, v.failure_reasons
    assert "while the baseline's do not" in reason
    assert v.elpd_diff is None and v.khat_max is None and v.repair is None
    assert v.converged, f"the probe itself should converge: {v.failure_reasons}"
