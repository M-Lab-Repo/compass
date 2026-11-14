"""compass.crossval — the latent-variable significance gate.

Covers, in rough order of statistical importance:

1. Marginalization CORRECTNESS on a conjugate normal-normal case with an
   analytic answer. This is the regression test for the conditional-prior
   property: held-out units must be scored under p(t | theta_s) with theta_s
   from the TRAIN posterior — sampling t from the MARGINAL prior instead
   (prior mean 0, data at ~50) would be tens of nats off per unit.
2. The compare_models contract: sign conventions, the 2*SE rule, the SE>0
   degeneracy floor (identical models must never certify).
3. `infer_cv_spec`: derives the unit plate + local latents from the program's
   plate structure, refuses ambiguous/non-latent structures.
4. Reliability guards: MC-bias gap and fold-fit R-hat demotion logic.
5. The full ladder (slow): PSIS k-hat blocks an overdispersion refinement;
   auto-escalation re-adjudicates by K-fold CV and certifies it; auto_cv=False
   preserves the conservative block.
6. The 2026-11-12 findings: F1 — spec inference resolves a renamed observed
   site fed by a float32-cast response (``infer_cv_spec`` and ``infer_obs_spec``
   match values at the coarser precision); F2 — the reloo rung neither flags
   nor refits positions where both fits share the same constant column.

MCMC goes through `trace_cache_dir`. The research tree keeps that cache on disk across
sessions, so its two marginalization tests (1.) are unmarked; the package cache is a
fresh per-session directory, so every test that samples is marked slow here.
"""

from __future__ import annotations

import numpy as np
import pytest

from compass.crossval import (
    CVSpec,
    _fold_reliability,
    _mc_reliability,
    _slice_units,
    _summarize,
    crossval_compare,
    held_out_unit_lppd,
    infer_cv_spec,
    infer_obs_spec,
)

# ---------------------------------------------------------------------------
# Model zoo (module-level strings, _exec_model convention: define `model`)
# ---------------------------------------------------------------------------

# Conjugate-checkable centered hierarchy: t_u ~ N(mu, TAU0), y_um ~ N(t_u, S0),
# with TAU0/S0 fixed constants so the held-out marginal given mu is analytic:
# y_u | mu ~ MVN(mu*1, S0^2*I + TAU0^2*J).
TAU0, S0 = 4.0, 2.0
CENTERED_HIER = f'''
def model(y=None, J=14, M=4):
    mu = numpyro.sample("mu", dist.Normal(0.0, 100.0))
    with numpyro.plate("unit", J):
        t = numpyro.sample("t", dist.Normal(mu, {TAU0}))
    with numpyro.plate("unit_row", J, dim=-2), numpyro.plate("item", M, dim=-1):
        numpyro.sample("y", dist.Normal(t[:, None], {S0}), obs=y)
'''

POOLED_NORMAL = '''
def model(y=None, J=14, M=4):
    mu = numpyro.sample("mu", dist.Normal(0.0, 100.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(20.0))
    with numpyro.plate("unit_row", J, dim=-2), numpyro.plate("item", M, dim=-1):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
'''

# Per-observation latent (the PSIS k-hat bomb): lognormal-Poisson overdispersion.
POISSON_PLAIN = '''
def model(y=None, N=60):
    eta = numpyro.sample("eta", dist.Normal(0.0, 5.0))
    with numpyro.plate("obs", N):
        numpyro.sample("y", dist.Poisson(jnp.exp(eta)), obs=y)
'''

POISSON_ODP = '''
def model(y=None, N=60):
    eta = numpyro.sample("eta", dist.Normal(0.0, 5.0))
    tau = numpyro.sample("tau", dist.HalfNormal(2.0))
    with numpyro.plate("obs", N):
        t = numpyro.sample("t", dist.Normal(0.0, 1.0))
        numpyro.sample("y", dist.Poisson(jnp.exp(eta + tau * t)), obs=y)
'''

# Mask-aware hierarchy whose observation SITE ("obs") is not named after the
# data kwarg feeding it ("y") — the entry-mask scheme must index numpyro's
# log_likelihood by the site name, not the kwarg (defect V7).
MASK_AWARE_RENAMED_SITE = '''
def model(y=None, mask=None, J=10, M=4):
    mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
    with numpyro.plate("unit", J):
        t = numpyro.sample("t", dist.Normal(mu, 1.0))
    m = mask if mask is not None else jnp.ones((J, M), dtype=bool)
    with numpyro.plate("unit_row", J, dim=-2), numpyro.plate("item", M, dim=-1):
        with numpyro.handlers.mask(mask=m):
            numpyro.sample("obs", dist.Normal(t[:, None], 1.0), obs=y)
'''

# The renamed, float32-cast observed site (F1): ``jnp.asarray`` of a float64
# response is a float32 copy, which exact value matching never equated with
# the kwarg, so spec inference returned None (and the ladder stayed blocked).
CENTERED_HIER_CAST_RENAMED = CENTERED_HIER.replace(
    f'numpyro.sample("y", dist.Normal(t[:, None], {S0}), obs=y)',
    f'numpyro.sample("obs", dist.Normal(t[:, None], {S0}), '
    'obs=None if y is None else jnp.asarray(y))')
assert 'numpyro.sample("obs"' in CENTERED_HIER_CAST_RENAMED
# the baseline names its site the same way (the dry-run traces BOTH programs
# and looks the site up by name; a pair with different site names stays
# un-escalated, as before)
POOLED_NORMAL_RENAMED = POOLED_NORMAL.replace('numpyro.sample("y"', 'numpyro.sample("obs"')
# site "counts" fed by kwarg "y" (the plate is already called "obs")
POISSON_PLAIN_RENAMED = POISSON_PLAIN.replace('numpyro.sample("y"', 'numpyro.sample("counts"')

# Two latent plates both tied to the obs (row AND column effects): the unit
# choice is ambiguous and inference must refuse.
CROSSED_EFFECTS = '''
def model(y=None, I=9, K=8):
    with numpyro.plate("row", I):
        a = numpyro.sample("a", dist.Normal(0.0, 1.0))
    with numpyro.plate("col", K):
        b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    with numpyro.plate("row_o", I, dim=-2), numpyro.plate("col_o", K, dim=-1):
        numpyro.sample("y", dist.Normal(a[:, None] + b[None, :], 1.0), obs=y)
'''


@pytest.fixture(scope="module")
def hier_data():
    rng = np.random.default_rng(7)
    J, M = 14, 4
    t = rng.normal(50.0, TAU0, size=J)
    y = rng.normal(t[:, None], S0, size=(J, M))
    return {"y": y, "J": J, "M": M}


@pytest.fixture(scope="module")
def odp_data():
    rng = np.random.default_rng(11)
    N = 60
    lam = np.exp(2.0 + 0.8 * rng.normal(size=N))
    y = rng.poisson(lam).astype(np.int32)
    return {"y": y, "N": N}


# ---------------------------------------------------------------------------
# 1. Marginalization correctness (conjugate reference)
# ---------------------------------------------------------------------------

def _analytic_unit_lppd(y_u: np.ndarray, mu: float) -> float:
    """log N(y_u; mu*1, S0^2 I + TAU0^2 11^T) via the matrix inversion lemma."""
    m = y_u.size
    r = y_u - mu
    # Sigma = s2*I + t2*J; Sigma^{-1} = I/s2 - (t2/(s2*(s2+m*t2))) J
    s2, t2 = S0 ** 2, TAU0 ** 2
    quad = (r @ r) / s2 - (t2 / (s2 * (s2 + m * t2))) * r.sum() ** 2
    logdet = (m - 1) * np.log(s2) + np.log(s2 + m * t2)
    return float(-0.5 * (m * np.log(2 * np.pi) + logdet + quad))


@pytest.mark.slow
def test_held_out_lppd_matches_conjugate_truth(hier_data, trace_cache_dir):
    """Conditional-prior marginalization reproduces the analytic marginal.

    The train posterior for mu sits near 50 while the PRIOR for mu is
    Normal(0, 100): an implementation that marginalized t under the marginal
    prior (prior-predictive draws) instead of p(t | mu_s) would miss the
    analytic value by tens of nats per unit. Tolerance 1.5 nats/unit mean.
    """
    from compass.agents.validator import run_mcmc

    J = hier_data["J"]
    train_idx, test_idx = np.arange(4, J), np.arange(0, 4)
    spec = CVSpec(per_unit_keys=["y"], local_latents=["t"], obs_site="y",
                  unit_size_keys=["J"])
    train = _slice_units(hier_data, train_idx, spec)
    test = _slice_units(hier_data, test_idx, spec)

    _, idata = run_mcmc(CENTERED_HIER, train, "numpyro", 300, 600, 2, seed=0,
                        cache_dir=trace_cache_dir, target_accept=0.9)
    mu_hat = float(np.asarray(idata.posterior["mu"].values).mean())
    assert abs(mu_hat - 50.0) < 3.0, "train posterior did not locate the data scale"

    lppd, mc_gap = held_out_unit_lppd(
        CENTERED_HIER, idata, test, spec, n_units=len(test_idx),
        s_use=200, r_prior=400, seed=0)

    analytic = np.array([_analytic_unit_lppd(test["y"][u], mu_hat)
                         for u in range(len(test_idx))])
    err = np.abs(lppd - analytic).mean()
    assert err < 1.5, f"mean |lppd - analytic| = {err:.2f} nats (marginalization broken?)"
    assert np.all(mc_gap >= -0.5) and mc_gap.mean() < 0.5


@pytest.mark.slow
def test_held_out_lppd_model_without_local_latent(hier_data, trace_cache_dir):
    """A spec listing latents the model lacks must degrade to the exact
    plug-in score (R forced to 1, zero MC gap) — this is the original-model
    side of an add-random-effects refinement."""
    from compass.agents.validator import run_mcmc

    J = hier_data["J"]
    spec = CVSpec(per_unit_keys=["y"], local_latents=["t"], obs_site="y",
                  unit_size_keys=["J"])
    train = _slice_units(hier_data, np.arange(4, J), spec)
    test = _slice_units(hier_data, np.arange(0, 4), spec)
    _, idata = run_mcmc(POOLED_NORMAL, train, "numpyro", 300, 600, 2, seed=0,
                        cache_dir=trace_cache_dir, target_accept=0.9)
    lppd, mc_gap = held_out_unit_lppd(
        POOLED_NORMAL, idata, test, spec, n_units=4, s_use=200, r_prior=64, seed=0)
    assert lppd.shape == (4,) and np.all(np.isfinite(lppd))
    assert np.allclose(mc_gap, 0.0), "no local latent -> no marginalization MC gap"


# ---------------------------------------------------------------------------
# 2. Contract: signs, 2*SE rule, degeneracy floor
# ---------------------------------------------------------------------------

def test_summarize_sign_conventions():
    ref = np.array([-1.0, -1.2, -0.8, -1.1, -0.9, -1.0])
    orig = ref - 2.0                     # refined better by 2 nats/unit
    res = _summarize(ref, orig)
    assert res["lppd_delta"] == pytest.approx(12.0)
    assert res["elpd_diff"] == pytest.approx(-12.0)   # negative favors refined
    assert res["loo_delta"] == pytest.approx(-24.0)   # deviance scale
    assert res["significant_improvement"] is True

    worse = _summarize(orig, ref)
    assert worse["elpd_diff"] == pytest.approx(12.0)
    assert worse["significant_improvement"] is False


def test_summarize_degenerate_se_never_certifies():
    same = np.array([-1.0, -2.0, -3.0])
    res = _summarize(same, same.copy())
    assert res["lppd_delta"] == pytest.approx(0.0)
    assert res["significant_improvement"] is False    # SE=0 floor

    # constant positive shift: SE is exactly 0 -> still must refuse
    res2 = _summarize(same + 1.0, same)
    assert res2["elpd_diff_se"] == pytest.approx(0.0)
    assert res2["significant_improvement"] is False


def test_entry_mask_score_indexes_log_likelihood_by_site_name(monkeypatch):
    """Defect V7: ``log_likelihood`` is keyed by the observation SITE ("obs"),
    the data dict by the kwarg ("y"); indexing the former by the latter raised
    KeyError for every model whose site is not named after its kwarg. Offline:
    the fold fits are canned, only the real numpyro scoring runs."""
    import arviz as az

    from compass import crossval

    rng = np.random.default_rng(5)
    J, M = 10, 4
    y = rng.normal(0.0, 1.0, (J, M)) + rng.normal(0.0, 1.0, (J, 1))
    data = {"y": y, "J": J, "M": M}
    idata = az.from_dict(posterior={"mu": rng.normal(0.0, 0.1, (1, 40)),
                                    "t": rng.normal(0.0, 1.0, (1, 40, J))})
    monkeypatch.setattr(crossval, "run_mcmc", lambda *a, **k: (None, idata))
    monkeypatch.setattr(crossval, "compute_diagnostics", lambda *a, **k: {
        "rhat_max": 1.0, "ess_min": 1000.0, "divergences": 0, "ebfmi_min": 1.0})

    res = crossval.entry_mask_score(MASK_AWARE_RENAMED_SITE, MASK_AWARE_RENAMED_SITE, data,
                                    obs_key="y", obs_site="obs", seed=0, budget=(1, 1, 1))
    assert res["scheme"].startswith("entry_mask") and res["n_held_out_entries"] > 0
    assert res["n_units"] > 0 and np.isfinite(res["lppd_delta"])
    assert res["significant_improvement"] is False      # identical fits: SE floor

    # the adapter forwards BOTH names from the CVSpec
    spec = CVSpec(per_unit_keys=["y"], local_latents=["t"], obs_site="obs",
                  obs_kwarg="y", unit_size_keys=["J"])
    out = crossval_compare(MASK_AWARE_RENAMED_SITE, MASK_AWARE_RENAMED_SITE, data, spec,
                           scheme="entry_mask", budget=(1, 1, 1), seed=0)
    assert out["s_gate"].startswith("entry_mask") and out["khat_refined"] is None
    assert out["elpd_diff"] == pytest.approx(0.0)

    # the kwarg alone (old call shape) still works when site == kwarg
    same_name = MASK_AWARE_RENAMED_SITE.replace('numpyro.sample("obs"', 'numpyro.sample("y"')
    res2 = crossval.entry_mask_score(same_name, same_name, data, obs_key="y", seed=0,
                                     budget=(1, 1, 1))
    assert res2["n_units"] == res["n_units"]


def test_slice_units_respects_per_key_axes():
    data = {"y": np.arange(12).reshape(3, 4), "x": np.arange(8).reshape(2, 4),
            "J": 4, "note": "shared"}
    spec = CVSpec(per_unit_keys=["y", "x"], local_latents=[], obs_site="y",
                  unit_size_keys=["J"], unit_axis=1, per_unit_axes={"y": 1, "x": 1})
    out = _slice_units(data, np.array([0, 2]), spec)
    assert out["y"].shape == (3, 2) and out["x"].shape == (2, 2)
    assert np.array_equal(out["y"][:, 1], data["y"][:, 2])
    assert out["J"] == 2 and out["note"] == "shared"


# ---------------------------------------------------------------------------
# 3. Plate-structure inference
# ---------------------------------------------------------------------------

def test_infer_cv_spec_finds_unit_plate(odp_data):
    got = infer_cv_spec(POISSON_ODP, odp_data, POISSON_PLAIN)
    assert got is not None
    spec = got["spec"]
    assert got["plate"] == "obs" and got["n_units"] == 60
    assert spec.local_latents == ["t"] and spec.obs_site == "y"
    assert spec.per_unit_keys == ["y"] and spec.unit_size_keys == ["N"]


def test_infer_cv_spec_shared_size_plates(hier_data):
    """Latent plate 'unit' and obs plate 'unit_row' share only their SIZE —
    the size fallback must still find the unit structure."""
    got = infer_cv_spec(CENTERED_HIER, hier_data, POOLED_NORMAL)
    assert got is not None
    assert got["n_units"] == hier_data["J"]
    assert got["spec"].local_latents == ["t"]
    assert got["spec"].per_unit_axes["y"] == 0


def test_infer_obs_spec_glm_yes_latent_no(odp_data):
    """The reloo rung applies exactly where the CV rung does not: a plain GLM
    (no latent on the obs plate) slices per observation; a per-obs latent
    model must be refused here (it belongs to the CV rung)."""
    got = infer_obs_spec(POISSON_PLAIN, odp_data)
    assert got is not None
    assert got["n_units"] == 60 and got["spec"].local_latents == []
    assert got["spec"].per_unit_keys == ["y"] and got["spec"].unit_size_keys == ["N"]

    assert infer_obs_spec(POISSON_ODP, odp_data) is None

    # 2-D obs: pointwise order cannot be identified with row order -> refuse
    rng = np.random.default_rng(3)
    two_d = {"y": rng.normal(size=(14, 4)), "J": 14, "M": 4}
    assert infer_obs_spec(POOLED_NORMAL, two_d) is None


def test_infer_cv_spec_resolves_a_float32_cast_renamed_site(hier_data):
    """F1: the site "obs" is fed by ``jnp.asarray(y)``; the kwarg is found at
    float32 precision, so the escalation ladder can derive its folds."""
    assert hier_data["y"].dtype == np.float64
    got = infer_cv_spec(CENTERED_HIER_CAST_RENAMED, hier_data, POOLED_NORMAL_RENAMED)
    assert got is not None, "spec inference lost the cast response"
    assert got["spec"].obs_site == "obs" and got["spec"].obs_kwarg == "y"
    assert got["spec"].local_latents == ["t"] and got["n_units"] == hier_data["J"]


def test_infer_obs_spec_resolves_a_renamed_site_by_value(odp_data):
    """The reloo rung's slicing spec used to require the site to be a data key;
    a renamed site is now matched by value (at the coarser precision, F1)."""
    got = infer_obs_spec(POISSON_PLAIN_RENAMED, odp_data)
    assert got is not None
    assert got["spec"].obs_site == "counts" and got["spec"].obs_kwarg == "y"
    assert got["n_units"] == 60


def test_reloo_skips_shared_degenerate_positions(odp_data, monkeypatch):
    """F2: cells both programs mask have a constant pointwise log-likelihood
    (k-hat = inf by construction) in BOTH fits; they carry no information and
    must be neither flagged nor refit. Without the exemption the three shared
    cells would be 'flagged' and sent to the exact-refit path."""
    import arviz as az
    from scipy.stats import poisson

    from compass import crossval

    y = odp_data["y"]
    rng = np.random.default_rng(0)

    def fit(center, seed):
        eta = rng.normal(np.log(center), 0.02, (2, 300))
        ll = poisson.logpmf(y[None, None, :], np.exp(eta)[..., None])
        ll[..., :3] = 0.0                               # the shared missing cells
        return az.from_dict(posterior={"eta": eta}, log_likelihood={"y": ll},
                            observed_data={"y": y})

    ref, orig = fit(float(y.mean()), 1), fit(float(y.mean()) * 1.3, 2)
    calls: list = []
    monkeypatch.setattr(crossval, "infer_obs_spec", lambda *a, **k: calls.append(1) or None)
    out = crossval.reloo_refined(POISSON_PLAIN, POISSON_PLAIN, odp_data, ref, orig)
    assert calls == [], "shared degenerate cells were flagged for an exact refit"
    # nothing else exceeds the threshold, so the rung does not apply at all
    k = np.asarray(az.loo(ref, pointwise=True).pareto_k, dtype=float)
    assert np.isinf(k[:3]).all()
    if (k[3:] <= 0.7).all():
        assert out is None


def test_infer_cv_spec_refuses_non_latent_and_ambiguous(hier_data):
    assert infer_cv_spec(POOLED_NORMAL, hier_data) is None  # nothing local
    rng = np.random.default_rng(0)
    crossed = {"y": rng.normal(size=(9, 8)), "I": 9, "K": 8}
    assert infer_cv_spec(CROSSED_EFFECTS, crossed) is None  # row AND col plates
    tiny = {"y": rng.normal(size=(4, 3)), "J": 4, "M": 3}
    assert infer_cv_spec(CENTERED_HIER.replace("J=14", "J=4").replace("M=4", "M=3"),
                         tiny) is None                       # too few units


# ---------------------------------------------------------------------------
# 4. Reliability guards (pure logic)
# ---------------------------------------------------------------------------

def test_mc_reliability_thresholds():
    res = {"lppd_delta_se": 4.0}
    _mc_reliability(res, 0.9, 0.8)       # 1.7 < 2.0 = se/2
    assert res["mc_unreliable"] is False
    res = {"lppd_delta_se": 4.0}
    _mc_reliability(res, 1.5, 0.8)       # 2.3 > 2.0
    assert res["mc_unreliable"] is True
    res = {"lppd_delta_se": float("nan")}
    _mc_reliability(res, 0.1, 0.0)       # no SE to compare against -> any gap fails
    assert res["mc_unreliable"] is True


def test_fold_reliability_rhat_gate():
    res = {}
    _fold_reliability(res, [{"rhat_max": 1.004, "divergences": 0},
                            {"rhat_max": 1.008, "divergences": 2}])
    assert res["cv_fit_unreliable"] is False and res["fold_divergences_total"] == 2
    res = {}
    _fold_reliability(res, [{"rhat_max": 1.02, "divergences": 0}])
    assert res["cv_fit_unreliable"] is True
    res = {}
    _fold_reliability(res, [{"rhat_max": float("nan"), "divergences": 0}])
    assert res["cv_fit_unreliable"] is True


# ---------------------------------------------------------------------------
# 5. End-to-end: CV certifies the true structure; null never certifies; the
#    validator ladder escalates exactly where PSIS is blocked. (slow: MCMC)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_kfold_certifies_hierarchy_and_null_refuses(hier_data, trace_cache_dir):
    spec = CVSpec(per_unit_keys=["y"], local_latents=["t"], obs_site="y",
                  unit_size_keys=["J"])
    win = crossval_compare(CENTERED_HIER, POOLED_NORMAL, hier_data, spec,
                           scheme="kfold", K=3, budget=(300, 600, 2),
                           s_use=200, r_prior=60, seed=0,
                           cache_dir=trace_cache_dir)
    assert win["elpd_diff"] < 0, "true hierarchy must win out of sample"
    assert win["reliability_note"] is None
    assert win["significant_improvement"] is True
    assert win["khat_refined"] is None and win["s_gate"].startswith("kfold")

    null = crossval_compare(CENTERED_HIER, CENTERED_HIER, hier_data, spec,
                            scheme="kfold", K=3, budget=(300, 600, 2),
                            s_use=200, r_prior=60, seed=0,
                            cache_dir=trace_cache_dir)
    assert null["significant_improvement"] is False   # identical fits -> SE floor


@pytest.mark.slow
def test_validator_ladder_escalates_odp(odp_data, trace_cache_dir):
    """The flagship completeness hole: adding a per-observation random effect.

    Without auto_cv the refined-side k-hat gate must block (PSIS can only
    over-estimate the refined model's gain); with auto_cv the validator derives
    the fold structure from the plates, re-adjudicates by real K-fold CV, and
    certifies — converged, no failure reasons, s_gate documents the escalation.
    """
    from compass.agents.validator import run_full_validation

    blocked = run_full_validation(
        POISSON_ODP, POISSON_PLAIN, odp_data, fast=True,
        num_warmup=300, num_samples=800, num_chains=2,
        cache_dir=trace_cache_dir, seed=0, auto_cv=False)
    assert any("Pareto-k-hat" in r for r in blocked.failure_reasons), \
        f"expected a k-hat block, got {blocked.failure_reasons}"

    res = run_full_validation(
        POISSON_ODP, POISSON_PLAIN, odp_data, fast=True,
        num_warmup=300, num_samples=800, num_chains=2,
        cache_dir=trace_cache_dir, seed=0)
    assert res.s_gate is not None and "auto-escalated" in res.s_gate
    assert res.converged, f"refined fit did not converge: {res.failure_reasons}"
    assert not res.failure_reasons, f"ladder did not certify: {res.failure_reasons}"
    assert res.loo_delta is not None and res.loo_delta < 0
