"""Smoke tests for the distilled COMPASS package: imports, graph construction
with a fixture LLM, and one real (tiny-budget) verifier round-trip."""
import re
from pathlib import Path

import numpy as np
import pytest

# The public surface shipped in 0.1.0; every later release must keep it.
API_0_1_0 = {
    "__version__",
    "build_refinement_graph",
    "build_report",
    "get_llm",
    "infer_data_shapes",
    "ModelSpec",
    "PaperFinding",
    "run_full_validation",
    "run_refinement",
    "ValidationResult",
    "WorkflowState",
}


def test_public_api_surface():
    import compass

    assert API_0_1_0 <= set(compass.__all__)
    for name in compass.__all__:
        assert getattr(compass, name, None) is not None, name
    assert compass.__version__
    # the package version is declared once, in pyproject.toml (no tomllib on 3.10)
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.MULTILINE)
    assert match is not None, "pyproject.toml has no version line"
    assert compass.__version__ == match.group(1)


def test_graph_builds_with_fixture_llm():
    from compass import build_refinement_graph
    from compass.llm import FixtureLLM

    app = build_refinement_graph(llm=FixtureLLM(outputs=["unused"]),
                                 data={"y": np.zeros(4)})
    assert app is not None


def test_infer_data_shapes_and_summary():
    from compass import infer_data_shapes
    from compass.graph import summarize_dataset

    data = {"y": np.arange(6.0), "n": 6}
    shapes = infer_data_shapes(data)
    assert shapes["y"] == [6]
    assert "y" in summarize_dataset(data)


@pytest.mark.slow
def test_verifier_round_trip(tmp_path):
    """Identical candidate vs original on tiny conjugate data: the verifier
    must run end to end and refuse (no significant improvement is possible),
    never accept."""
    from compass import run_full_validation

    rng = np.random.default_rng(0)
    data = {"y": rng.normal(1.0, 1.0, size=40)}
    model = """
def model(y=None):
    mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5.0))
    with numpyro.plate("obs", 40):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
"""
    res = run_full_validation(model, model, data, seed=0,
                              cache_dir=str(tmp_path),
                              num_warmup=500, num_samples=500, num_chains=2)
    assert isinstance(res.failure_reasons, list)
    accepted = bool(res.converged and not res.failure_reasons)
    assert not accepted
    assert res.loo == res.loo  # finite (not NaN) on the happy path


def test_every_submodule_imports():
    """Every module of the package imports with no key in the environment
    (the autouse fixture removes them); a module that needs an optional
    dependency at import time would surface here."""
    import importlib
    import pkgutil

    import compass

    names = [m.name for m in pkgutil.walk_packages(compass.__path__, prefix="compass.")]
    assert {"compass.agents.validator", "compass.crossval", "compass.graph",
            "compass.judge", "compass.llm", "compass.resolution"} <= set(names)
    for name in names:
        module = importlib.import_module(name)
        assert module.__name__ == name
