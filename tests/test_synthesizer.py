"""Code Synthesizer tests (spec §5.4, §11 "Synthesizer tests").

Structural tests use FixtureLLM + an injected ast-only validator so the
generate->validate->fix loop runs in milliseconds. Exactly ONE test pays for
a real subprocess MCMC smoke run (JAX compile ~20-40s).
"""

from __future__ import annotations

import ast
import os
import re
import sys

import numpy as np
import pytest

from compass.agents.synthesizer import (
    GOLDEN_DEFAULT_DATA_SHAPES,
    GUARD_REPLICATE_SITES_HINT,
    GUARD_REPLICATE_SITES_RULE,
    compare_models_waic_loo,
    create_synthesizer_agent,
    generate_dummy_data,
    guard_hint,
    validate_numpyro_code,
    validate_stan_code,
)
from compass.fixtures.golden_model import GOLDEN_ORIGINAL, GOLDEN_REFINED_REFERENCE
from compass.schemas import SynthesizedCode
from compass.state import ModelSpec, WorkflowState

GOLDEN_SHAPES = {"X": [280], "y": [280], "group_idx": [280], "N_groups": 8}

PROPOSAL_NAMES = [
    "non_centered_parameterization",
    "weakly_informative_hyperpriors",
    "regularize_slope_prior",
]

# Missing colon on the def line -> SyntaxError at ast.parse time.
BROKEN_CODE = '''
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def model(X, group_idx, N_groups, y=None)
    mu_alpha = numpyro.sample("mu_alpha", dist.Normal(70, 20))
'''

# Parses fine but fails at runtime: alpha has shape (2,) while X has the
# shared leading dim (32 rows), so `alpha + X * beta` cannot broadcast.
# (NOTE: indexing a length-2 vector with group_idx does NOT raise in JAX —
# out-of-bounds gather indices are silently clamped — so we use a direct
# shape mismatch to get a deterministic runtime failure.)
RUNTIME_BROKEN_CODE = '''
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def model(X, group_idx, N_groups, y=None):
    with numpyro.plate("groups", 2):
        alpha = numpyro.sample("alpha", dist.Normal(0.0, 1.0))
    beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    mu = alpha + X * beta
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
'''


def _broken_syntax_msg() -> str:
    """The interpreter's own message for BROKEN_CODE (version-robust)."""
    try:
        ast.parse(BROKEN_CODE)
    except SyntaxError as e:
        return e.msg
    raise AssertionError("BROKEN_CODE unexpectedly parsed")


def fast_validator(code: str, data_shapes: dict) -> dict:
    """ast-only validator injected so agent-loop tests skip the slow smoke run."""
    try:
        ast.parse(code)
        return {"compiled": True, "stage": "ast", "stdout": "", "stderr": ""}
    except SyntaxError as e:
        return {"compiled": False, "stage": "ast", "stdout": "", "stderr": f"SyntaxError: {e}"}


def make_state(proposals) -> WorkflowState:
    """Build the synthesizer input state. `proposals` is a list of proposal
    dicts — tests pass the `proposals_input` conftest cascade fixture
    (real_proposals.json if promoted, else the frozen fallback)."""
    return WorkflowState(
        original_model=ModelSpec(
            framework="numpyro",
            source_code=GOLDEN_ORIGINAL,
            data_shapes=dict(GOLDEN_SHAPES),
        ),
        proposals=list(proposals),
        status="proposed",
    )


# ---------------------------------------------------------------------------
# Happy path: broken first draft -> FIX_PROMPT with exact error -> success
# ---------------------------------------------------------------------------
def test_happy_path_with_fix_loop(make_fixture_llm, proposals_input):
    fake = make_fixture_llm(
        SynthesizedCode(code=BROKEN_CODE, applied_proposals=PROPOSAL_NAMES),
        SynthesizedCode(code=GOLDEN_REFINED_REFERENCE, applied_proposals=PROPOSAL_NAMES),
    )
    state = make_state(proposals_input["proposals"])
    run = create_synthesizer_agent(fake, validator=fast_validator)
    update = run(state)

    assert update["refined_model_code"] == GOLDEN_REFINED_REFERENCE
    assert update["status"] == "synthesized"
    assert state.refined_model_code == GOLDEN_REFINED_REFERENCE
    assert state.status == "synthesized"

    # one synth call + one fix call, fix prompt carried the EXACT error text
    assert len(fake.calls) == 2
    assert _broken_syntax_msg() in fake.calls[1]
    assert BROKEN_CODE.strip() in fake.calls[1]  # failed code fed back too

    # synth prompt carried the original code and rendered proposals
    assert "Normal(0, 100)" in fake.calls[0]
    assert "non_centered_parameterization" in fake.calls[0]
    assert "regularize_slope_prior" in fake.calls[0]

    # result is valid Python
    ast.parse(update["refined_model_code"])


# ---------------------------------------------------------------------------
# Refinement assertions on the result code
# ---------------------------------------------------------------------------
def test_refinements_applied_and_signature_preserved(make_fixture_llm, proposals_input):
    fake = make_fixture_llm(
        SynthesizedCode(code=GOLDEN_REFINED_REFERENCE, applied_proposals=PROPOSAL_NAMES)
    )
    update = create_synthesizer_agent(fake, validator=fast_validator)(
        make_state(proposals_input["proposals"])
    )
    code = update["refined_model_code"]

    # has NumPyro imports (spec §11)
    assert "import numpyro" in code
    assert "numpyro.distributions" in code

    # vague slope prior is gone
    assert "Normal(0, 100)" not in code
    assert "Normal(0,100)" not in code

    # non-centered parameterization is present: a standard-normal site or the
    # z-offsets + deterministic alpha recovery
    noncentered = re.search(r"Normal\(\s*0(\.0)?\s*,\s*1(\.0)?\s*\)", code) or (
        '"z"' in code and "numpyro.deterministic" in code and "alpha" in code
    )
    assert noncentered

    # signature def model(X, group_idx, N_groups, y=None) preserved, via ast
    tree = ast.parse(code)
    fns = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "model"
    ]
    assert len(fns) == 1
    fn = fns[0]
    assert [a.arg for a in fn.args.args] == ["X", "group_idx", "N_groups", "y"]
    assert len(fn.args.defaults) == 1
    assert isinstance(fn.args.defaults[0], ast.Constant)
    assert fn.args.defaults[0].value is None


# ---------------------------------------------------------------------------
# Real validation: ONE subprocess smoke success + cheap failure paths
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_validate_numpyro_smoke_success_on_golden_refined():
    # The single slow test: real subprocess NUTS run (JAX compile ~20-40s).
    res = validate_numpyro_code(GOLDEN_REFINED_REFERENCE, GOLDEN_SHAPES)
    assert res["compiled"] is True
    assert res["stage"] == "smoke"
    # Pass is signalled by the unforgeable sentinel as the LAST stdout line
    # (candidate-printed "SUCCESS" strings must not spoof a pass).
    from compass.agents.synthesizer import _SMOKE_SENTINEL
    assert res["stdout"].splitlines()[-1].strip() == _SMOKE_SENTINEL


# A bare body written against the validator's pre-seeded namespace (numpyro,
# dist, jnp, jax, np) — the form of every benchmark and real-data program.
# The smoke harness shares that namespace, so this must PASS.
IMPORT_FREE_CODE = '''
def model(y=None):
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    with numpyro.plate("data", 16):
        numpyro.sample("y", dist.Normal(mu, 1.0), obs=y)
'''

# Any name outside that namespace is still a genuine missing import.
UNDEFINED_NAME_CODE = '''
def model(y=None):
    mu = numpyro.sample("mu", tfd.Normal(0.0, 1.0))
    with numpyro.plate("data", 16):
        numpyro.sample("y", dist.Normal(mu, 1.0), obs=y)
'''


@pytest.mark.slow
def test_validate_numpyro_bare_body_passes_smoke():
    res = validate_numpyro_code(IMPORT_FREE_CODE, {"y": [16]})
    assert res["compiled"] is True
    assert res["stage"] == "smoke"


def test_validate_numpyro_undefined_name_fails_smoke():
    res = validate_numpyro_code(UNDEFINED_NAME_CODE, {"y": [16]})
    assert res["compiled"] is False
    assert res["stage"] == "smoke"
    combined = res["stdout"] + res["stderr"]
    # The harness reports str(e) — the NameError MESSAGE, not the class name.
    assert "name 'tfd' is not defined" in combined


# Golden fallback requires an EXACT signature match: a strict subset like
# model(X, y=None) would receive ALL four golden kwargs from the harness and
# die with a TypeError that is not the code's fault — so with no data_shapes
# it must degrade to the honest ast-only verdict instead.
SUBSET_SIGNATURE_CODE = '''
import numpyro
import numpyro.distributions as dist

def model(X, y=None):
    beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    numpyro.sample("y", dist.Normal(X * beta, sigma), obs=y)
'''


def test_signature_subset_of_golden_does_not_match():
    from compass.agents.synthesizer import _signature_matches_golden

    assert _signature_matches_golden(SUBSET_SIGNATURE_CODE) is False
    assert _signature_matches_golden(GOLDEN_REFINED_REFERENCE) is True


def test_validate_numpyro_subset_signature_skips_smoke_without_shapes():
    # No data_shapes + non-golden (subset) signature: ast-only skip, never a
    # smoke run with guaranteed-wrong kwargs (TypeError churn in the fix loop).
    res = validate_numpyro_code(SUBSET_SIGNATURE_CODE, {})
    assert res["compiled"] is True
    assert res["stage"] == "ast"
    assert "smoke test skipped" in res["stdout"]


def test_validate_numpyro_syntax_error_caught_at_ast_stage():
    res = validate_numpyro_code("def model(:", GOLDEN_SHAPES)
    assert res["compiled"] is False
    assert res["stage"] == "ast"
    assert "syntaxerror" in res["stderr"].lower()


def test_validate_numpyro_runtime_failure_at_smoke_stage():
    res = validate_numpyro_code(RUNTIME_BROKEN_CODE, GOLDEN_SHAPES)
    assert res["compiled"] is False
    assert res["stage"] == "smoke"
    combined = (res["stdout"] + res["stderr"]).lower()
    assert "shape" in combined or "broadcast" in combined


# ---------------------------------------------------------------------------
# Dummy data generation
# ---------------------------------------------------------------------------
def test_generate_dummy_data_golden_shapes():
    d = generate_dummy_data(GOLDEN_SHAPES)
    assert d["X"].dtype == np.float32 and d["X"].shape == (32,)
    assert d["y"].dtype == np.float32 and d["y"].shape == (32,)
    assert d["group_idx"].dtype == np.int32 and d["group_idx"].shape == (32,)
    assert d["group_idx"].min() >= 0
    assert d["group_idx"].max() < 8
    assert d["N_groups"] == 8 and isinstance(d["N_groups"], int)


def test_generate_dummy_data_shrinks_scalar_row_count_with_arrays():
    # A scalar equal to a shrunk leading dim (Stan-style row count N: 280
    # beside X/y: [280]) must shrink with the arrays, or a model that plates
    # over N hits a harness-induced shape error.
    d = generate_dummy_data({"N": 280, "X": [280], "y": [280]})
    assert d["X"].shape == (32,) and d["y"].shape == (32,)
    assert d["N"] == 32 and isinstance(d["N"], int)
    # small count scalars not encoding a shrunk dim pass through unchanged
    d2 = generate_dummy_data(GOLDEN_SHAPES)
    assert d2["N_groups"] == 8


def test_generate_dummy_data_defensive_without_count_scalar():
    # No N_* scalar anywhere: indices must still be valid (small positive range)
    d = generate_dummy_data({"item_idx": [100], "x": [100]}, rows=16, seed=1)
    assert d["item_idx"].dtype == np.int32 and d["item_idx"].shape == (16,)
    assert d["item_idx"].min() >= 0
    assert d["x"].dtype == np.float32 and d["x"].shape == (16,)
    # module default shapes mirror the golden dataset
    assert GOLDEN_DEFAULT_DATA_SHAPES["N_groups"] == 8


# ---------------------------------------------------------------------------
# Single proposal
# ---------------------------------------------------------------------------
def test_single_proposal_works(make_fixture_llm, proposals_input):
    single = [proposals_input["proposals"][0]]  # non-centered only
    fake = make_fixture_llm(
        SynthesizedCode(code=GOLDEN_REFINED_REFERENCE, applied_proposals=[single[0]["name"]])
    )
    state = make_state(single)
    update = create_synthesizer_agent(fake, validator=fast_validator)(state)

    assert update["status"] == "synthesized"
    assert update["refined_model_code"] == GOLDEN_REFINED_REFERENCE
    assert "non_centered_parameterization" in fake.calls[0]
    assert "weakly_informative_hyperpriors" not in fake.calls[0]


# ---------------------------------------------------------------------------
# Exhaustion: always-broken output -> empty code + error_log, status untouched
# ---------------------------------------------------------------------------
def test_exhaustion_after_max_attempts(make_fixture_llm, proposals_input):
    fake = make_fixture_llm(
        SynthesizedCode(code=BROKEN_CODE, applied_proposals=PROPOSAL_NAMES)
    )  # single queued output repeats forever
    state = make_state(proposals_input["proposals"])
    run = create_synthesizer_agent(fake, max_attempts=2, validator=fast_validator)
    update = run(state)

    assert update["refined_model_code"] == ""
    assert state.refined_model_code == ""
    assert "status" not in update
    assert state.status != "synthesized"

    # error_log entries are plain strings pipeline-wide (report renders `- {e}`)
    entry = update["error_log"][-1]
    assert isinstance(entry, str)
    assert entry.startswith("synthesizer: 2 attempts exhausted")
    assert _broken_syntax_msg() in entry
    assert state.error_log[-1] is entry

    # max_attempts=2 -> 1 synth call + 1 fix call
    assert len(fake.calls) == 2


# ---------------------------------------------------------------------------
# compare_models_waic_loo: behavioral check on synthetic log-likelihoods
# (no MCMC — log_likelihood arrays are constructed directly)
# ---------------------------------------------------------------------------
def test_compare_models_waic_loo_prefers_better_fit():
    import arviz as az

    rng = np.random.default_rng(0)
    # 2 chains x 200 draws x 50 obs; "bad" is uniformly ~5 nats worse per obs,
    # so the refined-vs-original elpd gap is ~250 with tiny uncertainty.
    ll_good = rng.normal(-1.0, 0.05, (2, 200, 50))
    posterior = {"mu": rng.normal(size=(2, 200))}
    good = az.from_dict(posterior=posterior, log_likelihood={"y": ll_good})
    bad = az.from_dict(posterior=posterior, log_likelihood={"y": ll_good - 5.0})

    res = compare_models_waic_loo(good, bad)  # refined = the better model
    assert set(res) == {
        "elpd_waic_refined", "elpd_waic_original", "elpd_waic_delta",
        "elpd_loo_refined", "elpd_loo_original", "elpd_loo_delta",
        "preferred",
    }
    assert res["preferred"] == "refined"
    # Sign convention (docstring): POSITIVE elpd delta = refined better.
    assert res["elpd_waic_delta"] == pytest.approx(250.0, abs=5.0)
    assert res["elpd_loo_delta"] == pytest.approx(250.0, abs=5.0)
    assert res["elpd_waic_refined"] > res["elpd_waic_original"]
    assert res["elpd_loo_refined"] > res["elpd_loo_original"]

    flipped = compare_models_waic_loo(bad, good)  # refined = the WORSE model
    assert flipped["preferred"] == "original"
    assert flipped["elpd_waic_delta"] < 0
    assert flipped["elpd_loo_delta"] < 0


# ---------------------------------------------------------------------------
# Stan framework dispatch: a stan ModelSpec must route to validate_stan_code
# (recorded via monkeypatch — no CmdStan compile is ever paid)
# ---------------------------------------------------------------------------
STAN_CANDIDATE = """
data { int<lower=0> N; vector[N] y; }
parameters { real mu; real<lower=0> sigma; }
model { mu ~ normal(0, 10); sigma ~ normal(0, 5); y ~ normal(mu, sigma); }
"""


def test_stan_framework_dispatches_to_stan_validator(make_fixture_llm,
                                                     proposals_input,
                                                     monkeypatch):
    recorded = []

    def recording_stan_validator(code):
        recorded.append(code)
        return {"compiled": True, "stage": "smoke", "stdout": "", "stderr": ""}

    monkeypatch.setattr("compass.agents.synthesizer.validate_stan_code",
                        recording_stan_validator)

    fake = make_fixture_llm(
        SynthesizedCode(code=STAN_CANDIDATE, applied_proposals=PROPOSAL_NAMES)
    )
    state = WorkflowState(
        original_model=ModelSpec(framework="stan",
                                 source_code="// original stan model"),
        proposals=list(proposals_input["proposals"]),
        status="proposed",
    )
    update = create_synthesizer_agent(fake, validator=None)(state)

    # The recorder was called with the candidate: framework dispatch chose
    # the Stan branch, not the NumPyro smoke test.
    assert recorded == [STAN_CANDIDATE]
    assert update["status"] == "synthesized"
    assert update["refined_model_code"] == STAN_CANDIDATE


# ---------------------------------------------------------------------------
# B3: validate_stan_code actually exercised (slow: cold CmdStan compile ~40s)
# ---------------------------------------------------------------------------

_VALID_STAN = """
data { int<lower=0> N; vector[N] y; }
parameters { real mu; real<lower=0> sigma; }
model { y ~ normal(mu, sigma); }
"""

# stanc parse error: the model statement is left unterminated.
_INVALID_STAN = """
data { int<lower=0> N; vector[N] y; }
parameters { real mu; real<lower=0> sigma; }
model { y ~ normal(mu, sigma
"""


def _require_cmdstan() -> None:
    """Skip unless cmdstanpy (the ``stan`` extra) AND a CmdStan toolchain are installed."""
    cmdstanpy = pytest.importorskip("cmdstanpy", reason="needs the stan extra (cmdstanpy)")
    try:
        cmdstanpy.cmdstan_path()
    except Exception as exc:  # no toolchain: cmdstanpy raises ValueError
        pytest.skip(f"needs a CmdStan toolchain: {exc}")


@pytest.mark.slow
def test_validate_stan_code_compiles_valid_program():
    _require_cmdstan()
    verdict = validate_stan_code(_VALID_STAN)
    assert verdict["compiled"] is True
    assert verdict["stage"] == "smoke"


@pytest.mark.slow
def test_validate_stan_code_rejects_invalid_program():
    _require_cmdstan()
    verdict = validate_stan_code(_INVALID_STAN)
    assert verdict["compiled"] is False
    assert verdict["stderr"]


# ---------------------------------------------------------------------------
# Smoke test on REAL data (data=...) instead of shape-derived dummies
# ---------------------------------------------------------------------------
REAL_LIKE_DATA = {
    "distance": np.array([2.0, 3.0, 4.0], dtype=np.float32),
    "tries": np.array([1443, 694, 455], dtype=np.int32),
    "dept": np.array([0, 1, 5], dtype=np.int32),
    "n_dept": 6,
    "y": np.array([1346.0, 577.0, 337.0], dtype=np.float32),
}


def _probe_key() -> tuple:
    """The cache key of the smoke-interpreter probe for this process's interpreter."""
    from compass.agents import synthesizer as syn

    return (sys.executable, syn._SMOKE_INTERPRETER_FLAGS)


def _capture_smoke_subprocess(monkeypatch):
    """Stub subprocess.run: record the script and the npz it loads, report a pass."""
    import subprocess

    from compass.agents import synthesizer as syn

    captured: dict = {}
    # the once-per-process interpreter probe is booked as passed, so the stub only ever
    # sees the smoke run itself (the probe has its own tests further down)
    monkeypatch.setitem(syn._SMOKE_INTERPRETER_VERDICT, _probe_key(), None)

    def fake_run(argv, **kw):
        script = open(argv[-1]).read()  # [interpreter, *flags, script_path]
        captured["script"] = script
        captured["argv"] = list(argv)
        captured["env"] = kw.get("env")
        npz_path = re.search(r"np\.load\('([^']+)'\)", script).group(1)
        with np.load(npz_path) as z:
            captured["npz"] = {k: z[k] for k in z.files}
        return subprocess.CompletedProcess(argv, 0, stdout=syn._SMOKE_SENTINEL + "\n", stderr="")

    monkeypatch.setattr(syn.subprocess, "run", fake_run)
    return captured


def test_validate_numpyro_with_data_writes_the_real_arrays(monkeypatch):
    from compass.agents import synthesizer as syn

    captured = _capture_smoke_subprocess(monkeypatch)
    monkeypatch.setattr(syn, "generate_dummy_data",
                        lambda *a, **k: pytest.fail("dummy data generated despite data="))
    res = validate_numpyro_code(IMPORT_FREE_CODE, {"wrong": [999]}, data=REAL_LIKE_DATA)
    assert res["compiled"] is True and res["stage"] == "smoke"
    npz = captured["npz"]
    assert set(npz) == set(REAL_LIKE_DATA)
    assert npz["tries"].dtype == np.int32 and npz["tries"].tolist() == [1443, 694, 455]
    assert npz["dept"].dtype == np.int32 and npz["dept"].tolist() == [0, 1, 5]
    assert npz["distance"].dtype == np.float32
    assert npz["n_dept"].ndim == 0 and int(npz["n_dept"]) == 6  # scalar -> 0-d, .item() in script
    assert repr(IMPORT_FREE_CODE) in captured["script"]


def test_validate_numpyro_without_data_still_uses_dummy_data(monkeypatch):
    from compass.agents import synthesizer as syn

    captured = _capture_smoke_subprocess(monkeypatch)
    seen: list = []
    real_gen = syn.generate_dummy_data

    def spy(shapes, *a, **k):
        seen.append(shapes)
        return real_gen(shapes, *a, **k)

    monkeypatch.setattr(syn, "generate_dummy_data", spy)
    res = validate_numpyro_code(IMPORT_FREE_CODE, {"y": [16]})
    assert res["compiled"] is True and seen == [{"y": [16]}]
    assert captured["npz"]["y"].shape == (16,)


def test_create_synthesizer_agent_data_routes_default_check_through_partial(
        monkeypatch, make_fixture_llm, proposals_input):
    from compass.agents import synthesizer as syn

    calls: list = []

    def recorder(code, data_shapes, *a, **kw):
        calls.append((code, data_shapes, kw))
        return {"compiled": True, "stage": "smoke", "stdout": "", "stderr": ""}

    monkeypatch.setattr(syn, "validate_numpyro_code", recorder)
    fake = make_fixture_llm(
        SynthesizedCode(code=GOLDEN_REFINED_REFERENCE, applied_proposals=PROPOSAL_NAMES))
    state = make_state(proposals_input["proposals"])
    update = create_synthesizer_agent(fake, data=REAL_LIKE_DATA)(state)
    assert update["status"] == "synthesized"
    # call site unchanged: (code, data_shapes) positional, data bound by the partial
    assert calls == [(GOLDEN_REFINED_REFERENCE, GOLDEN_SHAPES, {"data": REAL_LIKE_DATA})]
    assert calls[0][2]["data"] is REAL_LIKE_DATA


def test_create_synthesizer_agent_without_data_calls_the_bare_validator(
        monkeypatch, make_fixture_llm, proposals_input):
    from compass.agents import synthesizer as syn

    calls: list = []

    def recorder(code, data_shapes, *a, **kw):
        calls.append(kw)
        return {"compiled": True, "stage": "smoke", "stdout": "", "stderr": ""}

    monkeypatch.setattr(syn, "validate_numpyro_code", recorder)
    fake = make_fixture_llm(
        SynthesizedCode(code=GOLDEN_REFINED_REFERENCE, applied_proposals=PROPOSAL_NAMES))
    create_synthesizer_agent(fake)(make_state(proposals_input["proposals"]))
    assert calls == [{}]  # default path: no data kwarg at all


def test_injected_validator_ignores_data(monkeypatch, make_fixture_llm, proposals_input):
    from compass.agents import synthesizer as syn

    monkeypatch.setattr(syn, "validate_numpyro_code",
                        lambda *a, **k: pytest.fail("default validator must not run"))
    seen: list = []

    def injected(code, data_shapes):  # strict two-arg contract
        seen.append((code, data_shapes))
        return fast_validator(code, data_shapes)

    fake = make_fixture_llm(
        SynthesizedCode(code=GOLDEN_REFINED_REFERENCE, applied_proposals=PROPOSAL_NAMES))
    update = create_synthesizer_agent(fake, validator=injected, data=REAL_LIKE_DATA)(
        make_state(proposals_input["proposals"]))
    assert update["status"] == "synthesized"
    assert seen == [(GOLDEN_REFINED_REFERENCE, GOLDEN_SHAPES)]


# Not ported: the research test ``test_validate_numpyro_on_real_visible_fold_passes_where_
# dummy_data_fails`` (the smoke test on a real visible fold of the golf and ucbadmit data)
# needs the benchmark's real-data loaders, which live in figures/bench
# (``bench.realdata`` / ``bench.holdout``) and are tested there.


# ---------------------------------------------------------------------------
# P4: the smoke-test interpreter gets a scrubbed environment and runs isolated.
# The child exec's LLM-written code at module level, so the caller's provider
# keys must never reach it.
# ---------------------------------------------------------------------------
def test_smoke_test_env_is_an_allowlist_that_drops_credentials_and_project_knobs():
    from compass.agents.synthesizer import smoke_test_env

    kept = {
        "PATH": "/usr/bin", "HOME": "/home/u", "LANG": "C.UTF-8", "LC_ALL": "C",
        "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        "TMPDIR": "/tmp/x", "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": "0",
        "KMP_DUPLICATE_LIB_OK": "TRUE", "KMP_AFFINITY": "disabled",  # Intel OpenMP knobs
    }
    dropped = {
        "ANTHROPIC_API_KEY": "sk-1", "OPENAI_API_KEY": "sk-2", "FAKE_API_KEY": "x",
        "GITHUB_TOKEN": "t", "HF_TOKEN": "t",
        "BAYES_REFINE_PROVIDER": "openai", "BAYES_REFINE_NO_CACHE": "1",
        "AWS_SECRET_ACCESS_KEY": "s", "SSH_AUTH_SOCK": "/s", "USER": "u",
        "JAX_TOKEN": "nope",  # a kept prefix never rescues a credential-shaped name
        "PYTHONPATH": "/code",  # inert under -I (P1): no longer passed through
    }
    assert smoke_test_env({**kept, **dropped}) == kept
    assert smoke_test_env({}) == {}
    # default source is the process environment, and the result is a new dict
    env = smoke_test_env()
    assert env is not os.environ and env.get("PATH") == os.environ.get("PATH")


def test_smoke_subprocess_runs_isolated_with_the_scrubbed_env(monkeypatch):
    """Fast pin through the stubbed subprocess: argv carries -I and the script last,
    env is smoke_test_env() of the caller's environment (sentinels gone)."""
    monkeypatch.setenv("FAKE_API_KEY", "sentinel-123")
    monkeypatch.setenv("MY_SERVICE_TOKEN", "sentinel-456")
    monkeypatch.setenv("BAYES_REFINE_PROVIDER", "openai")
    captured = _capture_smoke_subprocess(monkeypatch)

    res = validate_numpyro_code(IMPORT_FREE_CODE, {"y": [16]})

    assert res["compiled"] is True
    argv = captured["argv"]
    assert argv[0] == sys.executable and argv[1] == "-I" and argv[-1].endswith("smoke_test.py")
    env = captured["env"]
    assert env is not None
    for name in ("FAKE_API_KEY", "MY_SERVICE_TOKEN", "BAYES_REFINE_PROVIDER"):
        assert name not in env
    assert not any(k.endswith(("_API_KEY", "_TOKEN")) or k.startswith("BAYES_REFINE_")
                   for k in env)
    assert env["PATH"] == os.environ["PATH"]
    assert env.get("HOME") == os.environ.get("HOME")


def test_candidate_code_cannot_read_the_callers_api_keys(monkeypatch):
    """The real smoke subprocess: a candidate printing os.environ.get('FAKE_API_KEY') at
    module level must not see the caller's sentinel, while the kept runtime knob
    (JAX_PLATFORMS) does arrive and the interpreter is isolated. The candidate exits
    before NUTS, so this costs an interpreter start plus the numpyro import only."""
    monkeypatch.setenv("FAKE_API_KEY", "sentinel-should-not-leak")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    candidate = (
        "import os, sys\n"
        "print('FAKE_API_KEY=' + repr(os.environ.get('FAKE_API_KEY')))\n"
        "print('JAX_PLATFORMS=' + repr(os.environ.get('JAX_PLATFORMS')))\n"
        "print('isolated=' + repr(sys.flags.isolated))\n"
        "raise SystemExit(0)\n"
    )

    res = validate_numpyro_code(candidate, {"y": [16]})

    assert res["compiled"] is False  # no sentinel line: the candidate bailed out before NUTS
    assert res["stage"] == "smoke"
    assert "FAKE_API_KEY=None" in res["stdout"]
    assert "sentinel-should-not-leak" not in res["stdout"] + res["stderr"]
    assert "JAX_PLATFORMS='cpu'" in res["stdout"]
    assert "isolated=1" in res["stdout"]


# ---------------------------------------------------------------------------
# P1 (2026-11-12): the smoke interpreter is probed once per process. -I ignores the
# user site-packages and PYTHONPATH, so a layout with numpy / jax / numpyro there
# must raise ONE clear EnvironmentError, never a smoke verdict per candidate.
# ---------------------------------------------------------------------------
_PROBE_ARGV_TAIL = ["-c", "import numpy, jax, numpyro"]
_PREAMBLE_TRACE = (
    "Traceback (most recent call last):\n"
    '  File "/tmp/x/smoke_test.py", line 2, in <module>\n'
    "    import jax\n"
    "ModuleNotFoundError: No module named 'jax'\n"
)
_CANDIDATE_TRACE = (
    "Traceback (most recent call last):\n"
    '  File "/tmp/x/smoke_test.py", line 14, in <module>\n'
    "    exec(compile(...), _ns)\n"
    '  File "<candidate>", line 2, in <module>\n'
    "ModuleNotFoundError: No module named 'numpyro.contrib.nothing'\n"
)


def test_smoke_interpreter_probe_failure_raises_before_any_candidate_runs(monkeypatch):
    """A user-site / PYTHONPATH-only installation: the probe fails, validate_numpyro_code
    raises a clear EnvironmentError (cause + remedy), no harness is ever written or run,
    and the verdict is cached so the second call costs no interpreter start."""
    import subprocess

    from compass.agents import synthesizer as syn

    monkeypatch.setattr(syn, "_SMOKE_INTERPRETER_VERDICT", {})
    monkeypatch.setenv("FAKE_API_KEY", "sentinel-123")
    monkeypatch.setenv("PYTHONPATH", "/somewhere/user/site")
    runs: list[dict] = []

    def fake_run(argv, **kw):
        runs.append({"argv": list(argv), "env": kw.get("env")})
        assert argv[-2:] == _PROBE_ARGV_TAIL, "only the probe may reach the interpreter"
        return subprocess.CompletedProcess(
            argv, 1, stdout="",
            stderr="Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\n"
                   "ModuleNotFoundError: No module named 'numpyro'\n")

    monkeypatch.setattr(syn.subprocess, "run", fake_run)
    with pytest.raises(EnvironmentError) as info:
        validate_numpyro_code(IMPORT_FREE_CODE, {"y": [16]})
    msg = str(info.value)
    assert "No module named 'numpyro'" in msg
    assert "-I" in msg and "user site-packages" in msg and "pip install --user" in msg
    assert "interpreter's own environment" in msg and "virtual environment" in msg
    assert f"{sys.executable} -m pip install numpy jax numpyro" in msg
    assert "No candidate was smoke-tested" in msg
    # exactly one interpreter start — the probe, run as the harness is run
    assert len(runs) == 1
    probe = runs[0]
    assert probe["argv"][0] == sys.executable and probe["argv"][1] == "-I"
    assert probe["argv"][-2:] == _PROBE_ARGV_TAIL
    assert "FAKE_API_KEY" not in probe["env"] and "PYTHONPATH" not in probe["env"]
    assert probe["env"]["PATH"] == os.environ["PATH"]
    # cached: the second call raises again without another interpreter start
    with pytest.raises(EnvironmentError):
        validate_numpyro_code(IMPORT_FREE_CODE, {"y": [16]}, data={"y": np.ones(4, np.float32)})
    assert len(runs) == 1


def test_smoke_interpreter_probe_passes_once_and_is_cached(monkeypatch):
    """A working interpreter is probed once per process; later smoke runs skip it."""
    import subprocess

    from compass.agents import synthesizer as syn

    monkeypatch.setattr(syn, "_SMOKE_INTERPRETER_VERDICT", {})
    runs: list[list[str]] = []

    def fake_run(argv, **kw):
        runs.append(list(argv))
        if argv[-2:] == _PROBE_ARGV_TAIL:
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout=syn._SMOKE_SENTINEL + "\n", stderr="")

    monkeypatch.setattr(syn.subprocess, "run", fake_run)
    for _ in range(3):
        assert validate_numpyro_code(IMPORT_FREE_CODE, {"y": [16]})["compiled"] is True
    assert len(runs) == 4
    assert runs[0][-2:] == _PROBE_ARGV_TAIL and runs[0][1] == "-I"
    assert all(r[-1].endswith("smoke_test.py") for r in runs[1:])
    assert syn._SMOKE_INTERPRETER_VERDICT == {_probe_key(): None}


def test_smoke_interpreter_probe_timeout_is_an_environment_error(monkeypatch):
    import subprocess

    from compass.agents import synthesizer as syn

    monkeypatch.setattr(syn, "_SMOKE_INTERPRETER_VERDICT", {})

    def fake_run(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw.get("timeout"))

    monkeypatch.setattr(syn.subprocess, "run", fake_run)
    with pytest.raises(EnvironmentError, match="did not finish within"):
        validate_numpyro_code(IMPORT_FREE_CODE, {"y": [16]})


def test_preamble_module_not_found_is_an_environment_error_not_a_smoke_verdict(monkeypatch):
    """The harness imports numpy / jax / numpyro before it exec's the candidate: a
    ModuleNotFoundError for one of them with no <candidate> frame is the environment
    failing (raise the same EnvironmentError), while a candidate's own missing module —
    a <candidate> frame, or any other package — stays an ordinary smoke failure."""
    import subprocess

    from compass.agents import synthesizer as syn

    monkeypatch.setitem(syn._SMOKE_INTERPRETER_VERDICT, _probe_key(), None)
    stderr = {"text": _PREAMBLE_TRACE}

    def fake_run(argv, **kw):
        assert argv[-1].endswith("smoke_test.py")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr=stderr["text"])

    monkeypatch.setattr(syn.subprocess, "run", fake_run)
    with pytest.raises(EnvironmentError) as info:
        validate_numpyro_code(IMPORT_FREE_CODE, {"y": [16]})
    msg = str(info.value)
    assert "No module named 'jax'" in msg and "before the candidate ran" in msg
    assert "user site-packages" in msg and "pip install --user" in msg
    # the candidate's own import of a missing package: the fix loop's business
    stderr["text"] = _CANDIDATE_TRACE
    res = validate_numpyro_code(IMPORT_FREE_CODE, {"y": [16]})
    assert res["compiled"] is False and res["stage"] == "smoke"
    assert "numpyro.contrib.nothing" in res["stderr"]
    # ... and so is any package outside the runtime, even without a <candidate> frame
    stderr["text"] = "ModuleNotFoundError: No module named 'tensorflow_probability'\n"
    assert validate_numpyro_code(IMPORT_FREE_CODE, {"y": [16]})["compiled"] is False


def test_preamble_import_failure_helper():
    from compass.agents.synthesizer import _preamble_import_failure

    assert _preamble_import_failure(_PREAMBLE_TRACE) == "ModuleNotFoundError: No module named 'jax'"
    assert _preamble_import_failure(
        "ModuleNotFoundError: No module named 'numpyro.distributions'") == \
        "ModuleNotFoundError: No module named 'numpyro.distributions'"
    assert _preamble_import_failure("ModuleNotFoundError: No module named 'numpy'\n") == \
        "ModuleNotFoundError: No module named 'numpy'"
    assert _preamble_import_failure(_CANDIDATE_TRACE) is None
    assert _preamble_import_failure("ModuleNotFoundError: No module named 'numpyrox'") is None
    assert _preamble_import_failure("ValueError: shapes\n") is None
    assert _preamble_import_failure("") is None


def test_synthesizer_agent_lets_the_environment_error_through_without_a_fix_loop(
        make_fixture_llm, proposals_input, monkeypatch):
    """The default validator's EnvironmentError is not a smoke verdict: it propagates
    out of the node after ONE LLM call — no fix attempts, no give-up entry — so the
    driver sees the environment problem instead of max_attempts burnt per proposal."""
    from compass.agents import synthesizer as syn

    def broken(code, data_shapes, timeout=120, *, data=None):
        raise syn._smoke_environment_error("No module named 'jax'")

    monkeypatch.setattr(syn, "validate_numpyro_code", broken)
    fake = make_fixture_llm(
        SynthesizedCode(code=GOLDEN_REFINED_REFERENCE, applied_proposals=PROPOSAL_NAMES))
    state = make_state(proposals_input["proposals"])
    run = create_synthesizer_agent(fake, max_attempts=3)
    with pytest.raises(EnvironmentError, match="No module named 'jax'"):
        run(state)
    assert len(fake.calls) == 1
    assert state.error_log == [] and state.refined_model_code == ""


# ---------------------------------------------------------------------------
# Structured-output retries: an empty tool call is retried with a nudge, then
# treated like an exhausted synthesizer (no crash; the pipeline routes on "")
# ---------------------------------------------------------------------------
def test_empty_structured_reply_is_retried_with_a_nudge(make_fixture_llm, proposals_input):
    good = SynthesizedCode(code=GOLDEN_REFINED_REFERENCE, applied_proposals=PROPOSAL_NAMES)
    fake = make_fixture_llm({}, good)          # first reply: empty tool call -> ValidationError
    state = make_state(proposals_input["proposals"])
    run = create_synthesizer_agent(fake, max_attempts=2, validator=fast_validator)
    update = run(state)
    assert update["status"] == "synthesized" and update["refined_model_code"] == GOLDEN_REFINED_REFERENCE
    assert len(fake.calls) == 2
    assert "did not fit the required output schema" not in fake.calls[0]
    assert "did not fit the required output schema" in fake.calls[1]
    assert state.error_log == []


def test_persistently_empty_structured_reply_gives_up_cleanly(make_fixture_llm, proposals_input):
    fake = make_fixture_llm({})                 # repeats forever
    state = make_state(proposals_input["proposals"])
    run = create_synthesizer_agent(fake, max_attempts=3, validator=fast_validator)
    update = run(state)
    assert update["refined_model_code"] == "" and "status" not in update
    assert len(fake.calls) == 3                 # 1 + STRUCTURED_RETRIES, no fix calls
    entry = update["error_log"][-1]
    assert entry.startswith("synthesizer: structured output for SynthesizedCode invalid after 3 attempts")
    assert "ValidationError" in entry


def test_empty_reply_during_the_fix_loop_gives_up_with_the_validation_error(
        make_fixture_llm, proposals_input):
    broken = SynthesizedCode(code=BROKEN_CODE, applied_proposals=PROPOSAL_NAMES)
    fake = make_fixture_llm(broken, {}, {}, {})  # synth ok, every fix reply empty
    state = make_state(proposals_input["proposals"])
    run = create_synthesizer_agent(fake, max_attempts=2, validator=fast_validator)
    update = run(state)
    assert update["refined_model_code"] == ""
    entry = update["error_log"][-1]
    assert "structured output for SynthesizedCode invalid after 3 attempts" in entry
    assert "last validation error" in entry and _broken_syntax_msg() in entry
    assert len(fake.calls) == 4


# ---------------------------------------------------------------------------
# --guard-replicate-sites: an opt-in prompt rule; prompts untouched when off
# ---------------------------------------------------------------------------
def test_guard_replicate_sites_off_leaves_both_prompts_as_before(make_fixture_llm,
                                                                 proposals_input):
    """Off (the default), the guard's template variables render to the empty string:
    no rule, no hint, no placeholder, and the seams around them read exactly as the
    prompts did before the knob existed."""
    fake = make_fixture_llm(
        SynthesizedCode(code=BROKEN_CODE, applied_proposals=PROPOSAL_NAMES),
        SynthesizedCode(code=GOLDEN_REFINED_REFERENCE, applied_proposals=PROPOSAL_NAMES),
    )
    run = create_synthesizer_agent(fake, validator=fast_validator)
    assert run(make_state(proposals_input["proposals"]))["status"] == "synthesized"
    synth, fix = fake.calls
    for call in (synth, fix):
        assert "site guard" not in call.lower() and "{guard" not in call
        assert GUARD_REPLICATE_SITES_RULE.strip() not in call
        assert GUARD_REPLICATE_SITES_HINT not in call
    assert "for posterior predictive checks.\nHuman:" in synth
    assert "(full source, with imports).\nHuman:" in fix
    assert "```\n\nReturn the complete fixed program." in fix


def test_guard_replicate_sites_adds_the_rule_and_a_targeted_fix_hint(make_fixture_llm,
                                                                     proposals_input):
    """On, both prompts carry the rule; the fix prompt carries the one-line hint only when
    the smoke-test error mentions enumeration, a discrete site or a replicate site name."""
    fake = make_fixture_llm(
        SynthesizedCode(code=GOLDEN_REFINED_REFERENCE, applied_proposals=PROPOSAL_NAMES))
    verdicts = iter([
        {"compiled": False, "stage": "smoke", "stdout": "",
         "stderr": "NotImplementedError: site y_rep has a discrete distribution; enumerate it"},
        {"compiled": False, "stage": "smoke", "stdout": "",
         "stderr": "ValueError: Incompatible shapes for broadcasting"},
        {"compiled": True, "stage": "smoke", "stdout": "", "stderr": ""},
    ])
    run = create_synthesizer_agent(fake, max_attempts=3, guard_replicate_sites=True,
                                   validator=lambda code, shapes: next(verdicts))
    update = run(make_state(proposals_input["proposals"]))
    assert update["status"] == "synthesized"
    synth, fix_discrete, fix_shapes = fake.calls
    rule = GUARD_REPLICATE_SITES_RULE.strip()
    assert rule in synth and rule in fix_discrete and rule in fix_shapes
    assert "{guard" not in synth + fix_discrete + fix_shapes
    assert GUARD_REPLICATE_SITES_HINT in fix_discrete
    assert GUARD_REPLICATE_SITES_HINT not in fix_shapes
    # the hint sits between the error block and the closing instruction
    assert fix_discrete.index("enumerate it") < fix_discrete.index(GUARD_REPLICATE_SITES_HINT)
    assert fix_discrete.index(GUARD_REPLICATE_SITES_HINT) < fix_discrete.index(
        "Return the complete fixed program.")


def test_guard_hint_triggers_on_enumeration_discrete_sites_or_replicate_names():
    for error in ("ValueError: Enumerate support requires ...",
                  "discrete latent variables are not supported by NUTS",
                  "site y_rep: ...", "obs_pred has shape (32,)", "mu_hat is unobserved",
                  "y_tilde", "z_ppc", "n_sim"):
        assert guard_hint(error) == f"\n{GUARD_REPLICATE_SITES_HINT}\n", error
    for error in ("TypeError: unsupported operand", "SyntaxError: invalid syntax",
                  "repeat the fit", "prediction failed", ""):
        assert guard_hint(error) == "", error

