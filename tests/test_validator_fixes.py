"""Fast regression tests for the validator findings sweep + the PSIS gate.

Every test here is MCMC-free: real MCMC paths are stubbed via monkeypatch and
the PPC/lppd paths run on tiny synthetic ``az.from_dict`` posteriors. Covers:

* x64 leak — ``run_ppc`` / ``_heldout_lppd`` restore ``jax_enable_x64``;
* trace-cache contract — ``trace_cache_path`` computes the path ``run_mcmc``
  writes (x64 an explicit param, legacy prefixed keys still found);
* object-dtype data — deterministic cache keys (no pointer hashing);
* Stan PPC — ambiguous observed-array fallback returns None (descriptive),
  replicate names dedup by VALUE across groups;
* run_full_validation — PPC exceptions surface their cause, PSIS k-hat > 0.7
  gate (never flips ``converged``), ``khat_max`` populated;
* prediction scoring honors direction; confidence treats PPC None as neutral;
* localization keeps per-site failures instead of collapsing by stat;
* F5 (2026-11-12) — ``_run_stan`` hands the numeric array entries of the data
  to ``az.from_cmdstanpy`` as ``observed_data`` when the converter accepts it;
* F1 (2026-11-12) — ``_resolve_obs_kwarg`` / the PPC's un-conditioning match a
  float32-cast observed site against its float64 data kwarg.
"""

from __future__ import annotations

import numpy as np
import pytest

import compass.agents.validator as validator
from compass.agents.validator import (
    _cache_key,
    _heldout_lppd,
    _match_observed,
    confidence_from_margin,
    localize_misspecification,
    run_full_validation,
    run_mcmc,
    run_ppc,
    score_proposal_predictions,
    trace_cache_path,
)

# Module-level numpyro.enable_x64() — the exact shape of LLM-generated code
# that used to leak process-global float64 out of the PPC / held-out paths.
X64_PLATE_MODEL = '''
numpyro.enable_x64()
def model(y=None):
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    with numpyro.plate("data", 10):
        numpyro.sample("y", dist.Normal(mu, 1.0), obs=y)
'''

X64_SCALAR_MODEL = '''
numpyro.enable_x64()
def model(y=None):
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    numpyro.sample("y", dist.Normal(mu, 1.0), obs=y)
'''

TINY_MODEL = "def model(y=None):\n    pass\n"


def _mu_idata(n_draws: int = 50):
    import arviz as az

    rng = np.random.default_rng(0)
    return az.from_dict(posterior={"mu": rng.normal(0.0, 1.0, (2, n_draws))})


# ---------------------------------------------------------------------------
# Finding 1 — x64 snapshot/restore in the PPC and held-out lppd paths
# ---------------------------------------------------------------------------

class TestX64Restore:
    @pytest.fixture(autouse=True)
    def _restore_flag_after_test(self):
        import jax

        before = bool(jax.config.jax_enable_x64)
        yield
        if bool(jax.config.jax_enable_x64) != before:
            jax.config.update("jax_enable_x64", before)

    def test_exec_model_alone_leaks_the_flag(self):
        # Precondition guard: if numpyro ever stops flipping the global flag,
        # the restore tests below would be vacuous — fail loudly instead.
        import jax

        assert bool(jax.config.jax_enable_x64) is False
        validator._exec_model(X64_PLATE_MODEL)
        assert bool(jax.config.jax_enable_x64) is True
        jax.config.update("jax_enable_x64", False)

    def test_run_ppc_restores_x64(self):
        import jax

        before = bool(jax.config.jax_enable_x64)
        run_ppc(_mu_idata(), X64_PLATE_MODEL, {"y": np.zeros(10)})
        assert bool(jax.config.jax_enable_x64) is before

    def test_heldout_lppd_restores_x64(self):
        import jax

        before = bool(jax.config.jax_enable_x64)
        val = _heldout_lppd(X64_SCALAR_MODEL, _mu_idata(), {"y": np.zeros(6)})
        assert isinstance(val, float) and np.isfinite(val)
        assert bool(jax.config.jax_enable_x64) is before


# ---------------------------------------------------------------------------
# Findings 3 + 5 — trace-cache contract and object-dtype keys
# ---------------------------------------------------------------------------

class TestTraceCacheContract:
    BUDGET = dict(num_warmup=10, num_samples=10, num_chains=1)

    def test_run_mcmc_writes_the_path_trace_cache_path_reports(self, tmp_path,
                                                               monkeypatch):
        # No sampling: the runner is stubbed; only the cache plumbing is real.
        monkeypatch.setattr(validator, "_run_numpyro",
                            lambda *a, **k: ("sampler", "fake-idata"))
        data = {"y": np.arange(4.0)}
        run_mcmc(TINY_MODEL, data, cache_dir=tmp_path, **self.BUDGET)

        expected = trace_cache_path(tmp_path, TINY_MODEL, data, **self.BUDGET)
        assert expected.exists()  # the PUBLIC path is the one actually written

        sampler, idata = run_mcmc(TINY_MODEL, data, cache_dir=tmp_path,
                                  **self.BUDGET)
        assert sampler is None and idata == "fake-idata"  # cache hit

    def test_x64_is_an_explicit_key_dimension(self, tmp_path):
        data = {"y": np.arange(4.0)}
        p32 = trace_cache_path(tmp_path, TINY_MODEL, data, x64=False)
        p64 = trace_cache_path(tmp_path, TINY_MODEL, data, x64=True)
        import jax

        default = trace_cache_path(tmp_path, TINY_MODEL, data)
        assert p32 != p64
        assert default == (p64 if jax.config.jax_enable_x64 else p32)

    def test_legacy_prefixed_key_format_is_preserved(self, tmp_path):
        # Old code hashed f"x64={flag};{code}" AS the model code; the new
        # explicit param must produce byte-identical keys so existing cache
        # pickles are still found.
        import hashlib

        import arviz as az
        import jax
        import numpyro

        data = {"y": np.arange(4.0)}
        h = hashlib.sha256()
        h.update(f"numpyro={numpyro.__version__};jax={jax.__version__};"
                 f"arviz={az.__version__}".encode())
        h.update(f"x64=False;{TINY_MODEL}".encode())
        h.update(b"numpyro")
        h.update(str((10, 10, 1, 0.8)).encode())
        h.update(str(0).encode())
        for key in sorted(data):
            arr = np.asarray(data[key])
            h.update(key.encode())
            h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
        legacy = tmp_path / f"trace_{h.hexdigest()}.pkl"
        assert trace_cache_path(tmp_path, TINY_MODEL, data, x64=False,
                                **self.BUDGET) == legacy

    def test_object_dtype_data_keys_are_value_based(self):
        budget = (10, 10, 1, 0.8)
        # two structurally-equal but DISTINCT objects -> same key (the old
        # pointer-based tobytes() hashing made these differ)
        k1 = _cache_key(TINY_MODEL, "numpyro", budget, 0, {"meta": {"a": 1}}, x64=False)
        k2 = _cache_key(TINY_MODEL, "numpyro", budget, 0, {"meta": {"a": 1}}, x64=False)
        k3 = _cache_key(TINY_MODEL, "numpyro", budget, 0, {"meta": {"a": 2}}, x64=False)
        assert k1 == k2
        assert k1 != k3


# ---------------------------------------------------------------------------
# Findings 2 + 6 — Stan PPC observed-array matching and replicate dedup
# ---------------------------------------------------------------------------

class TestStanPPCMatching:
    def test_ambiguous_same_size_fallback_returns_none(self):
        # Two same-size observed arrays (covariate + response): a first-match
        # guess could score replicates against the covariate -> must be None.
        arrays = {"x": np.zeros(50), "counts": np.ones(50)}
        assert _match_observed("z_rep", (50,), arrays, None) is None

    def test_unambiguous_same_size_fallback_still_matches(self):
        arrays = {"counts": np.ones(50), "n": np.asarray(50.0)}
        hit = _match_observed("z_rep", (50,), arrays, None)
        assert hit is not None and hit.size == 50

    def test_ambiguous_match_makes_ppc_unavailable_not_failed(self):
        import arviz as az

        rng = np.random.default_rng(1)
        idata = az.from_dict(posterior={"z_rep": rng.normal(0, 1, (2, 100, 50))})
        data = {"x": rng.normal(0, 1, 50), "counts": rng.normal(0, 1, 50)}
        # None (descriptive/unavailable), NOT a wrong True/False verdict.
        assert run_ppc(idata, "", data, framework="stan") is None

    def test_replicate_names_dedup_by_value_across_groups(self):
        import arviz as az

        rng = np.random.default_rng(2)
        rep = rng.normal(0.0, 1.0, (2, 200, 80))
        # y_rep lands in BOTH groups (as with cmdstan CSV parsing); the names
        # are equal strings but distinct objects -> must still dedup.
        idata = az.from_dict(posterior={"y_rep": rep},
                             posterior_predictive={"y_rep": rep})
        y = rng.normal(0.0, 1.0, 80)
        verdict, details = run_ppc(idata, "", {"y": y}, framework="stan",
                                   return_details=True)
        assert verdict is True
        assert len(details) == 4  # one per statistic, NOT doubled


# ---------------------------------------------------------------------------
# Findings 4 + gate — run_full_validation with stubbed MCMC internals
# ---------------------------------------------------------------------------

CLEAN_DIAG = {"rhat_max": 1.001, "ess_min": 900.0, "divergences": 0,
              "ebfmi_min": 1.0}


def _stub_validation_internals(monkeypatch, *, khat=0.3, khat_original=None,
                               ppc=(True, [])):
    """Stub every MCMC-touching dependency of run_full_validation.

    ``khat`` stubs the REFINED model's Pareto-k (the gating one);
    ``khat_original`` the original's (recorded, never gating). The structural
    S-gate guard (``check_observed_sites``, defect V1a) traces the refined
    program and refuses one it cannot trace, so it is stubbed clean here too
    (the model code in these tests is the placeholder string "c").
    """
    monkeypatch.setattr(validator, "run_mcmc", lambda *a, **k: (None, object()))
    monkeypatch.setattr(validator, "compute_diagnostics",
                        lambda idata, mcmc=None: dict(CLEAN_DIAG))
    monkeypatch.setattr(validator, "check_observed_sites", lambda *a, **k: [])
    ko = khat if khat_original is None else khat_original
    finite = [k for k in (khat, ko) if not np.isnan(k)]
    comparison = {"waic": 10.0, "loo": 11.0, "waic_delta": -5.0,
                  "loo_delta": -5.0, "elpd_diff": -20.0, "elpd_diff_se": 3.0,
                  "significant_improvement": True,
                  "khat_max": max(finite) if finite else float("nan"),
                  "khat_refined": khat, "khat_original": ko}
    monkeypatch.setattr(validator, "compare_models",
                        lambda r, o: dict(comparison))
    if isinstance(ppc, Exception):
        def _raise(*a, **k):
            raise ppc
        monkeypatch.setattr(validator, "run_ppc", _raise)
    else:
        monkeypatch.setattr(validator, "run_ppc", lambda *a, **k: ppc)


class TestPsisReliabilityGate:
    def test_high_refined_khat_records_failure_but_not_convergence(self, monkeypatch):
        _stub_validation_internals(monkeypatch, khat=0.95)
        v = run_full_validation("c", "c", {"y": np.zeros(4)})
        assert v.converged is True  # comparison-reliability failure only
        assert ("PSIS unreliable: refined-model Pareto-k-hat 0.95 > 0.7"
                in v.failure_reasons)
        assert v.khat_max == 0.95

    def test_high_ORIGINAL_khat_never_gates(self, monkeypatch):
        # A badly misspecified original (e.g. Poisson on overdispersed counts,
        # roaches k-hat 4.29) inflates the BASELINE elpd — conservative, so it
        # must not block the refinement. Recorded via khat_max only.
        _stub_validation_internals(monkeypatch, khat=0.3, khat_original=4.29)
        v = run_full_validation("c", "c", {"y": np.zeros(4)})
        assert v.failure_reasons == []
        assert v.khat_max == 4.29

    def test_low_khat_passes_clean(self, monkeypatch):
        _stub_validation_internals(monkeypatch, khat=0.3)
        v = run_full_validation("c", "c", {"y": np.zeros(4)})
        assert v.failure_reasons == []
        assert v.khat_max == 0.3

    def test_nan_khat_does_not_gate(self, monkeypatch):
        _stub_validation_internals(monkeypatch, khat=float("nan"))
        v = run_full_validation("c", "c", {"y": np.zeros(4)})
        assert not any("PSIS" in r for r in v.failure_reasons)


class TestPpcErrorSurfaced:
    def test_ppc_exception_cause_lands_in_failure_reason(self, monkeypatch):
        exc = ValueError("cannot un-condition observed site(s) ['obs']")
        _stub_validation_internals(monkeypatch, ppc=exc)
        v = run_full_validation("c", "c", {"y": np.zeros(4)})
        assert v.converged is True
        assert v.posterior_predictive_check is False
        reason = next(r for r in v.failure_reasons
                      if r.startswith("posterior predictive check failed"))
        assert "cannot un-condition" in reason  # cause surfaced, not discarded

    def test_genuine_ppc_false_keeps_the_plain_reason(self, monkeypatch):
        _stub_validation_internals(monkeypatch, ppc=(False, []))
        v = run_full_validation("c", "c", {"y": np.zeros(4)})
        assert "posterior predictive check failed" in v.failure_reasons


# ---------------------------------------------------------------------------
# Finding 7 — prediction scoring honors direction on the delta metrics
# ---------------------------------------------------------------------------

class TestDirectionAwareScoring:
    def _score(self, direction, loo_delta):
        p = {"name": "p", "prediction": {"metric": "loo_delta",
                                         "direction": direction}}
        return score_proposal_predictions(
            [p], diagnostics=dict(CLEAN_DIAG),
            comparison={"loo_delta": loo_delta, "waic_delta": loo_delta},
            ppc_passed=True)[0]

    def test_improve_still_means_negative_delta(self):
        assert self._score("improve", -3.0)["passed"] is True
        assert self._score("improve", 3.0)["passed"] is False

    def test_increase_means_positive_delta(self):
        assert self._score("increase", 3.0)["passed"] is True
        assert self._score("increase", -3.0)["passed"] is False

    def test_decrease_means_the_canonical_good_direction(self):
        assert self._score("decrease", -3.0)["passed"] is True


# ---------------------------------------------------------------------------
# Finding 8 — PPC None is neutral in the confidence rule
# ---------------------------------------------------------------------------

class TestConfidencePpcNone:
    KW = dict(converged=True, significant_improvement=True, elpd_diff=-160.0,
              elpd_diff_se=9.0, ess_min=1200.0, divergences=0)

    def test_none_caps_at_medium_not_low(self):
        assert confidence_from_margin(ppc_passed=None, **self.KW) == "MEDIUM"

    def test_true_and_false_semantics_unchanged(self):
        assert confidence_from_margin(ppc_passed=True, **self.KW) == "HIGH"
        assert confidence_from_margin(ppc_passed=False, **self.KW) == "LOW"


# ---------------------------------------------------------------------------
# Finding 9 — localization keeps per-site failures
# ---------------------------------------------------------------------------

class TestPerSiteLocalization:
    DIAG = {"rhat_max": 1.0, "divergences": 0}

    def test_later_site_does_not_overwrite_earlier_failure(self):
        details = [
            {"site": "a", "stat": "sd", "passed": False, "side": "high"},
            {"site": "b", "stat": "sd", "passed": False, "side": "low"},
        ]
        hints = localize_misspecification(self.DIAG, details)
        # site a's over-dispersion hint survives site b's sd-low detail
        assert any("OVER-DISPERSION" in h and "'a'" in h for h in hints)

    def test_hints_name_their_site_and_dedup_exact_repeats(self):
        d = {"site": "a", "stat": "zeros", "passed": False, "side": "high"}
        hints = localize_misspecification(self.DIAG, [d, dict(d)])
        assert len(hints) == 1 and "'a'" in hints[0] and "ZERO-INFLATION" in hints[0]


# ---------------------------------------------------------------------------
# Theory-driven hardenings: baseline-fit gate + SE degeneracy floor
# ---------------------------------------------------------------------------

BAD_DIAG = {"rhat_max": 1.25, "ess_min": 40.0, "divergences": 7, "ebfmi_min": 1.0}


class TestBaselineFitGate:
    def _stub_split_diags(self, monkeypatch, refined_diag, original_diag):
        refined_marker, original_marker = object(), object()
        calls = iter([refined_marker, original_marker])
        monkeypatch.setattr(validator, "run_mcmc",
                            lambda *a, **k: (None, next(calls)))
        monkeypatch.setattr(
            validator, "compute_diagnostics",
            lambda idata, mcmc=None: dict(
                refined_diag if idata is refined_marker else original_diag))
        comparison = {"waic": 10.0, "loo": 11.0, "waic_delta": -5.0,
                      "loo_delta": -5.0, "elpd_diff": -20.0, "elpd_diff_se": 3.0,
                      "significant_improvement": True, "khat_max": 0.3,
                      "khat_refined": 0.3, "khat_original": 0.3}
        monkeypatch.setattr(validator, "compare_models",
                            lambda r, o: dict(comparison))
        monkeypatch.setattr(validator, "run_ppc", lambda *a, **k: (True, []))
        monkeypatch.setattr(validator, "check_observed_sites", lambda *a, **k: [])

    def test_bad_original_fit_blocks_without_flipping_converged(self, monkeypatch):
        # A non-mixing BASELINE under-estimates elpd(M0) and would manufacture
        # a false accept — comparison-reliability failure, refined still True.
        self._stub_split_diags(monkeypatch, CLEAN_DIAG, BAD_DIAG)
        v = run_full_validation("c", "c", {"y": np.zeros(4)})
        assert v.converged is True
        reason = next(r for r in v.failure_reasons
                      if r.startswith("original-model fit unreliable"))
        assert "divergences" in reason and "R-hat" in reason and "ESS" in reason

    def test_clean_original_fit_adds_no_reason(self, monkeypatch):
        self._stub_split_diags(monkeypatch, CLEAN_DIAG, CLEAN_DIAG)
        v = run_full_validation("c", "c", {"y": np.zeros(4)})
        assert not any(r.startswith("original-model") for r in v.failure_reasons)


class TestSeDegeneracyFloor:
    def test_zero_se_never_certifies_improvement(self, monkeypatch):
        import pandas as pd

        class _Elpd(dict):
            pareto_k = np.array([0.1])

        def fake_waic(idata, scale=None):
            return _Elpd(elpd_waic=-10.0 if idata == "refined" else -5.0)

        def fake_loo(idata, scale=None, pointwise=False):
            return _Elpd(elpd_loo=-10.0 if idata == "refined" else -5.0)

        # refined ranked best, magnitude 5.0, but dse == 0 -> must NOT certify
        cmp_df = pd.DataFrame(
            {"elpd_diff": [0.0, 5.0], "dse": [0.0, 0.0]},
            index=["refined", "original"])
        seen: dict = {}

        def fake_compare(d, **kw):
            seen.update(kw)
            return cmp_df

        monkeypatch.setattr(validator.az, "waic", fake_waic)
        monkeypatch.setattr(validator.az, "loo", fake_loo)
        monkeypatch.setattr(validator.az, "compare", fake_compare)
        # the string "idata" stand-ins carry no log_likelihood group, so the
        # V1 comparability guard has nothing to check and passes through
        comp = validator.compare_models("refined", "original")
        assert comp["elpd_diff"] < 0 and comp["elpd_diff_se"] == 0.0
        assert comp["significant_improvement"] is False  # the floor
        # V3: the scale is pinned explicitly, never read from az.rcParams
        assert seen == {"ic": "loo", "scale": "log"}


# ---------------------------------------------------------------------------
# Defect V8 — the Stan seed is passed through unchanged (0 is valid, <0 is not)
# ---------------------------------------------------------------------------

class TestStanSeedPassthrough:
    """``seed if seed > 0 else 1`` mapped seeds 0 and 1 onto the same cmdstan
    run under two different trace-cache keys. cmdstan accepts seed 0."""

    def _fake_cmdstanpy(self, monkeypatch, seen: dict):
        import sys
        import types

        class _Model:
            def __init__(self, stan_file):
                seen["stan_file"] = stan_file

            def sample(self, **kw):
                seen.update(kw)
                return "fit"

        fake = types.ModuleType("cmdstanpy")
        fake.CmdStanModel = _Model
        monkeypatch.setitem(sys.modules, "cmdstanpy", fake)
        monkeypatch.setattr(validator.az, "from_cmdstanpy", lambda posterior: "idata")

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_seed_reaches_cmdstan_unchanged(self, monkeypatch, tmp_path, seed):
        seen: dict = {}
        self._fake_cmdstanpy(monkeypatch, seen)
        fit, idata = validator._run_stan("data { }", {}, 10, 10, 1, seed, 0.8,
                                         cache_dir=tmp_path)
        assert (fit, idata) == ("fit", "idata")
        assert seen["seed"] == seed  # 0 stays 0; no silent remap to 1

    def test_negative_seed_is_refused_before_compilation(self, monkeypatch, tmp_path):
        seen: dict = {}
        self._fake_cmdstanpy(monkeypatch, seen)
        with pytest.raises(ValueError, match="non-negative"):
            validator._run_stan("data { }", {}, 10, 10, 1, -1, 0.8, cache_dir=tmp_path)
        assert seen == {}  # no model was built, nothing sampled


# ---------------------------------------------------------------------------
# F5 — the Stan fit carries observed_data (identity check beyond shapes)
# ---------------------------------------------------------------------------

class TestStanObservedData:
    def _fake_cmdstanpy(self, monkeypatch):
        import sys
        import types

        class _Model:
            def __init__(self, stan_file):
                pass

            def sample(self, **kw):
                return "fit"

        fake = types.ModuleType("cmdstanpy")
        fake.CmdStanModel = _Model
        monkeypatch.setitem(sys.modules, "cmdstanpy", fake)

    def test_numeric_array_entries_are_forwarded(self, monkeypatch, tmp_path):
        self._fake_cmdstanpy(monkeypatch)
        seen: dict = {}

        def fake_from_cmdstanpy(posterior, observed_data=None):
            seen["posterior"] = posterior
            seen["observed_data"] = observed_data
            return "idata"

        monkeypatch.setattr(validator.az, "from_cmdstanpy", fake_from_cmdstanpy)
        data = {"N": 4, "y": np.arange(4), "x": [0.5, 1.5, 2.5, 3.5],
                "label": "counts", "flag": True, "names": np.array(["a", "b"])}
        fit, idata = validator._run_stan("data { }", data, 10, 10, 1, 0, 0.8, cache_dir=tmp_path)
        assert (fit, idata) == ("fit", "idata")
        assert set(seen["observed_data"]) == {"y", "x"}   # arrays only: no scalars, no strings
        np.testing.assert_array_equal(seen["observed_data"]["y"], np.arange(4))
        np.testing.assert_array_equal(seen["observed_data"]["x"], [0.5, 1.5, 2.5, 3.5])

    def test_converter_without_the_argument_is_called_plainly(self, monkeypatch, tmp_path):
        # an arviz whose from_cmdstanpy takes no observed_data must not be
        # handed one (TypeError) — the capability is read off the signature
        self._fake_cmdstanpy(monkeypatch)
        monkeypatch.setattr(validator.az, "from_cmdstanpy", lambda posterior: "idata")
        _, idata = validator._run_stan("data { }", {"y": np.arange(4)}, 10, 10, 1, 0, 0.8,
                                       cache_dir=tmp_path)
        assert idata == "idata"

    def test_no_arrays_means_no_observed_data(self, monkeypatch, tmp_path):
        self._fake_cmdstanpy(monkeypatch)
        seen: dict = {}

        def fake_from_cmdstanpy(posterior, **kw):
            seen.update(kw)
            return "idata"

        monkeypatch.setattr(validator.az, "from_cmdstanpy", fake_from_cmdstanpy)
        validator._run_stan("data { }", {"N": 3}, 10, 10, 1, 0, 0.8, cache_dir=tmp_path)
        assert seen == {}


# ---------------------------------------------------------------------------
# F1 — a float32-cast, renamed observed site still resolves to its data kwarg
# ---------------------------------------------------------------------------

# the shape of 36 released candidates: the response goes through jnp.asarray
# (x64 off -> float32) and the site is not named after the kwarg
CAST_RENAMED_SITE = '''
def model(x=None, y=None):
    y = None if y is None else jnp.asarray(y)
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    with numpyro.plate("n", x.shape[0]):
        numpyro.sample("obs", dist.Normal(mu + 0.0 * x, 1.0), obs=y)
'''


class TestFloat32CastResolution:
    def _data(self):
        rng = np.random.default_rng(0)
        return {"x": rng.normal(size=12), "y": rng.normal(size=12)}   # float64

    def test_resolver_finds_the_kwarg_for_a_float32_cast_site(self):
        import jax.numpy as jnp

        data = self._data()
        site_value = np.asarray(jnp.asarray(data["y"]))              # what the trace records
        assert site_value.dtype == np.float32
        assert not np.array_equal(site_value, data["y"])             # exact matching failed here
        assert validator._resolve_obs_kwarg("obs", site_value, data) == "y"
        # a transformed response is still unresolvable
        assert validator._resolve_obs_kwarg("obs", np.minimum(site_value, 0.0), data) is None

    def test_ppc_unconditions_a_float32_cast_renamed_site(self):
        # before F1: ValueError("cannot un-condition observed site(s) ['obs'] ...")
        verdict, details = run_ppc(_mu_idata(), CAST_RENAMED_SITE, self._data(),
                                   return_details=True)
        assert verdict in (True, False)
        assert [d["site"] for d in details] == ["obs"] * 4
