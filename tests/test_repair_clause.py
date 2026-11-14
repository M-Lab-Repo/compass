"""v4 Component 3 — the repair-equivalence clause.

Completeness hole it closes: a refinement whose value is inferential (an
identification constraint, a reparameterization) leaves delta-eLPD ~ 0, so
the (G)∧(S)∧(P) predicate could never accept it — even when the BASELINE
fails the convergence gates and the refinement is the fix. The clause
accepts, labeled as repair, iff baseline failed G ∧ refined passes ∧ the
comparison is reliable ∧ TOST-equivalent (|Δ| ≤ 2·SE and |Δ|+2·SE ≤ 4 nats).

The fast tests script every precondition through monkeypatched instruments
(the clause is pure logic over their outputs); the slow test runs the real
L11 additive-confound pair end to end. I2 guard: a converged baseline can
never trigger the clause.
"""

from __future__ import annotations

import numpy as np
import pytest

import compass.agents.validator as validator_mod
import compass.resolution as resolution_mod
from compass.agents.validator import run_full_validation

GOOD_DIAG = {"rhat_max": 1.004, "ess_min": 900.0, "divergences": 0, "ebfmi_min": 0.9}
BAD_DIAG = {"rhat_max": 1.31, "ess_min": 40.0, "divergences": 0, "ebfmi_min": 0.9}


def _comparison(diff=0.5, se=1.0, khat=0.3, **extra):
    sig = bool(diff < 0 and abs(diff) >= 2 * se)
    return {"waic": 0.0, "loo": 0.0, "waic_delta": 2 * diff, "loo_delta": 2 * diff,
            "elpd_diff": diff, "elpd_diff_se": se,
            "significant_improvement": sig, "khat_max": khat,
            "khat_refined": khat, "khat_original": khat, **extra}


@pytest.fixture()
def stubbed(monkeypatch):
    """Monkeypatch every instrument; returns a dict the test can tweak."""
    knobs = {"orig_diag": dict(BAD_DIAG), "refined_diag": dict(GOOD_DIAG),
             "comparison": _comparison(), "ppc": (True, [])}

    monkeypatch.setattr(validator_mod, "run_mcmc",
                        lambda *a, **k: (None, object()))
    diags = {"n": 0}

    def fake_diagnostics(idata, mcmc=None):
        # per validation call: 1 = refined, 2 = original, 3 = the repair
        # clause's baseline retry at doubled budget (defaults to still-bad)
        diags["n"] += 1
        k = (diags["n"] - 1) % 3
        if k == 0:
            return dict(knobs["refined_diag"])
        if k == 1:
            return dict(knobs["orig_diag"])
        return dict(knobs.get("orig_retry_diag", knobs["orig_diag"]))

    monkeypatch.setattr(validator_mod, "compute_diagnostics", fake_diagnostics)
    monkeypatch.setattr(validator_mod, "compare_models",
                        lambda *a, **k: dict(knobs["comparison"]))
    monkeypatch.setattr(validator_mod, "run_ppc",
                        lambda *a, **k: knobs["ppc"])
    monkeypatch.setattr(resolution_mod, "audit_resolution",
                        lambda *a, **k: None)
    # the structural S-gate guard (V1a) traces the refined program; the
    # placeholder "def model(): pass" takes no data, so stub it clean
    monkeypatch.setattr(validator_mod, "check_observed_sites",
                        lambda *a, **k: [])
    knobs["_reset"] = lambda: diags.update(n=0)
    return knobs


def _run(stubbed=None, **kwargs):
    if stubbed is not None:
        stubbed["_reset"]()
    return run_full_validation("def model(): pass", "def model(): pass",
                               {"y": np.ones(4)}, framework="numpyro",
                               fast=True, **kwargs)


def test_repair_fires_on_all_preconditions(stubbed):
    res = _run(stubbed)
    assert res.repair is not None and "computational repair" in res.repair
    assert "R-hat 1.310" in res.repair          # names the baseline evidence
    assert res.converged and res.failure_reasons == []
    assert res.elpd_diff == pytest.approx(0.5)
    assert res.elpd_diff_se == pytest.approx(1.0)


def test_repair_never_fires_when_baseline_converged(stubbed):
    """The I2 guard: every published baseline converges by construction."""
    stubbed["orig_diag"] = dict(GOOD_DIAG)
    res = _run(stubbed)
    assert res.repair is None
    assert any("no significant improvement" in r for r in res.failure_reasons)


def test_repair_requires_tost_margin(stubbed):
    stubbed["comparison"] = _comparison(diff=3.9, se=0.05)   # |diff| > 2*SE
    res = _run(stubbed)
    assert res.repair is None

    stubbed["comparison"] = _comparison(diff=0.1, se=3.0)    # CI escapes ±4
    res = _run(stubbed)
    assert res.repair is None
    assert any("original-model fit unreliable" in r for r in res.failure_reasons)


def test_repair_requires_reliable_comparison(stubbed):
    stubbed["comparison"] = _comparison(khat=1.2)            # khat gate reason
    res = _run(stubbed)
    assert res.repair is None
    assert any("Pareto-k-hat" in r for r in res.failure_reasons)

    stubbed["comparison"] = _comparison(se=0.0)              # degenerate SE
    res = _run(stubbed)
    assert res.repair is None

    stubbed["comparison"] = _comparison(
        reliability_note="CV fold fit unreliable")           # demoted comparison
    res = _run(stubbed)
    assert res.repair is None


def test_repair_refused_when_khat_is_inf(stubbed):
    """Defect V2: k-hat = inf (degenerate importance weights) is the WORST
    PSIS verdict; the clause must refuse it like k > 0.7, and the inf gate
    reason must be recorded (the old finite-only test let inf through)."""
    stubbed["comparison"] = _comparison(khat=float("inf"))
    res = _run(stubbed)
    assert res.repair is None
    assert any("Pareto-k-hat is inf" in r for r in res.failure_reasons)

    stubbed["comparison"] = _comparison(khat=float("nan"))   # not estimable
    res = _run(stubbed)
    assert res.repair is None                                # no positive evidence
    assert not any("Pareto-k-hat" in r for r in res.failure_reasons)  # NaN never gates


def test_repair_never_fires_on_a_refused_comparison(stubbed, monkeypatch):
    """A masked/scaled observed site refuses the comparison outright (V1);
    the clause cannot read equivalence off a comparison that did not run."""
    stubbed["comparison"] = _comparison()                    # TOST-equivalent
    monkeypatch.setattr(validator_mod, "check_observed_sites",
                        lambda *a, **k: ["observed site 'y' is masked (test)"])
    res = _run(stubbed)
    assert res.repair is None
    assert res.failure_reasons[-1].startswith("model comparison refused: ")
    assert any("original-model fit unreliable" in r for r in res.failure_reasons)


def test_repair_requires_clean_refined_gates(stubbed):
    stubbed["refined_diag"] = dict(BAD_DIAG)                 # refined fails G
    res = _run(stubbed)
    assert res.repair is None and not res.converged

    stubbed["refined_diag"] = dict(GOOD_DIAG)
    stubbed["ppc"] = (False, [])                             # PPC failure blocks
    res = _run(stubbed)
    assert res.repair is None
    assert any("posterior predictive" in r for r in res.failure_reasons)


def test_repair_blocked_when_baseline_passes_on_retry(stubbed):
    """The abuse guard measured on the latent grid: a marginally under-mixed
    baseline that clears G at doubled budget is NOT broken, so a cosmetic,
    predictively-equivalent rival must not be crowned a repair."""
    stubbed["orig_retry_diag"] = dict(GOOD_DIAG)
    res = _run(stubbed)
    assert res.repair is None
    assert any("original-model fit unreliable" in r for r in res.failure_reasons)


def test_repair_kill_switch(stubbed):
    res = _run(stubbed, repair_clause=False)
    assert res.repair is None
    assert any("original-model fit unreliable" in r for r in res.failure_reasons)


def test_repair_renders_and_routes(stubbed):
    from compass.graph import _render_validation, validation_passed
    from compass.state import WorkflowState

    res = _run(stubbed)
    s = WorkflowState()
    s.validation = res
    assert validation_passed(s)                  # routes to success
    assert "REPAIR CLAUSE:" in _render_validation(s)


# --- the real thing: L11 additive confound, end to end (slow) ----------------

BASELINE_UNIDENTIFIED = '''
def model(y=None, J=12, M=4):
    mu = numpyro.sample("mu", dist.Normal(0.0, 50.0))
    m = numpyro.sample("m", dist.Normal(0.0, 50.0))
    with numpyro.plate("unit", J):
        t = numpyro.sample("t", dist.Normal(m, 1.0))
    with numpyro.plate("unit_row", J, dim=-2), numpyro.plate("item", M, dim=-1):
        numpyro.sample("y", dist.Normal(mu + t[:, None], 1.0), obs=y)
'''

GOLD_IDENTIFIED = '''
def model(y=None, J=12, M=4):
    mu = numpyro.sample("mu", dist.Normal(0.0, 50.0))
    with numpyro.plate("unit", J):
        t = numpyro.sample("t", dist.Normal(0.0, 1.0))
    with numpyro.plate("unit_row", J, dim=-2), numpyro.plate("item", M, dim=-1):
        numpyro.sample("y", dist.Normal(mu + t[:, None], 1.0), obs=y)
'''


@pytest.mark.slow
def test_identification_repair_accepted_end_to_end(trace_cache_dir):
    """Same data law, one identified: chains disagree on (mu, m) in the
    baseline while predictions match; the clause converts the refusal into a
    labeled repair. Without the clause: refused (the old completeness hole)."""
    rng = np.random.default_rng(31)
    J, M = 12, 4
    t = rng.normal(0.0, 1.0, size=J)
    y = rng.normal(1.5 + t[:, None], 1.0, size=(J, M))
    data = {"y": y, "J": J, "M": M}

    kwargs = dict(framework="numpyro", fast=True, num_warmup=300,
                  num_samples=600, num_chains=2,
                  cache_dir=trace_cache_dir, seed=0)

    blocked = run_full_validation(GOLD_IDENTIFIED, BASELINE_UNIDENTIFIED, data,
                                  repair_clause=False, **kwargs)
    assert blocked.converged, f"gold fit unhealthy: {blocked.failure_reasons}"
    assert blocked.failure_reasons, "expected the pre-clause refusal"
    assert any("original-model fit unreliable" in r
               for r in blocked.failure_reasons)

    repaired = run_full_validation(GOLD_IDENTIFIED, BASELINE_UNIDENTIFIED, data,
                                   **kwargs)
    assert repaired.repair is not None, (
        f"clause did not fire: {repaired.failure_reasons}")
    assert repaired.converged and repaired.failure_reasons == []
    assert abs(repaired.elpd_diff) <= 2 * repaired.elpd_diff_se
