"""Shared fixtures for the compass package tests.

Every test here is offline. All but the ``@pytest.mark.slow`` ones (real MCMC, the
subprocess smoke run) are fast. The autouse ``isolated_env`` fixture keeps the suite
offline: ``compass.llm.get_llm`` calls ``load_dotenv(find_dotenv(usecwd=True))``, which
would pick up a live key from a ``.env`` in the current working directory or one of its
parents (the research checkout holds one), and the shell may export a key too. Both are
neutralised per test, so the suite behaves identically with and without keys.

Live tests are an explicit opt-in. A test marked ``llm`` (``tests/test_strategist.py``) skips
itself unless the environment variable ``COMPASS_LIVE_TESTS`` holds a non-empty value; when it
does, ``isolated_env`` leaves the provider keys and the ``.env`` lookup alone for that test,
so the key is read the way ``get_llm`` reads it for a user (``ANTHROPIC_API_KEY`` exported, or
in a ``.env`` file in the current working directory or one of its parents). Example:

    COMPASS_LIVE_TESTS=1 pytest tests/test_strategist.py -m llm

The variable has no effect on the unmarked tests, which stay offline.

The figure tests (``tests/figures/``) need the ``figures`` extra (matplotlib, Pillow, pypdf); when
it is not installed the directory is not collected, so ``pytest tests`` passes on a plain
``pip install -e '.[dev]'`` and runs everything after ``pip install -e '.[dev,figures]'``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

LIVE_TESTS_VAR = "COMPASS_LIVE_TESTS"

# the directory whose tests need an optional extra, skipped at collection when it is absent
collect_ignore: list[str] = []
if any(importlib.util.find_spec(m) is None for m in ("matplotlib", "PIL", "pypdf")):
    collect_ignore.append("figures")

PROVIDER_ENV = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "BAYES_REFINE_PROVIDER",
    "BAYES_REFINE_MODEL",
    "BAYES_REFINE_MODEL_OPENAI",
    "BAYES_REFINE_MODEL_GOOGLE",
)


def pytest_configure(config):
    # ``slow`` is registered in pytest.ini. The research tree's live-LLM marker travels with
    # the ported strategist test; registering it keeps the run warning-free.
    config.addinivalue_line(
        "markers",
        f"llm: needs a live provider key; skipped unless {LIVE_TESTS_VAR} is set (see conftest)")


@pytest.fixture(autouse=True)
def isolated_env(request, monkeypatch):
    """No dotenv lookup and no provider key or BAYES_REFINE_* setting reaches a test.

    The one exception is a test marked ``llm`` while ``COMPASS_LIVE_TESTS`` is set: that is
    the live-test opt-in, and the test needs the real key and the real ``.env`` lookup.
    """
    if request.node.get_closest_marker("llm") and os.environ.get(LIVE_TESTS_VAR):
        return
    import compass.llm as llm_mod

    monkeypatch.setattr(llm_mod, "load_dotenv", lambda *a, **k: None)
    for var in PROVIDER_ENV:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="session")
def golden_data() -> dict:
    """The synthetic 8-schools-style dataset (dict of numpy arrays)."""
    from compass.fixtures.golden_data import generate_golden_dataset

    return generate_golden_dataset()


@pytest.fixture(scope="session")
def golden_model_code() -> str:
    """Source of the deliberately flawed hierarchical regression."""
    from compass.fixtures.golden_model import GOLDEN_ORIGINAL

    return GOLDEN_ORIGINAL


@pytest.fixture(scope="session")
def golden_problem() -> str:
    from compass.fixtures.golden_model import GOLDEN_PROBLEM

    return GOLDEN_PROBLEM


@pytest.fixture(scope="session")
def golden_dataset_summary() -> str:
    from compass.fixtures.golden_model import GOLDEN_DATASET_SUMMARY

    return GOLDEN_DATASET_SUMMARY


# --- fixture cascade inputs ---------------------------------------------------
# The agent tests consume these instead of the FROZEN_* constants, as the research suite
# does: ``load_real_or_frozen`` prefers a promoted ``fixtures/real_<name>.json`` over the
# frozen object. No promoted file ships with the package, so these are the frozen outputs
# as plain dicts/lists. FROZEN_* stays the right choice for FixtureLLM canned OUTPUTS and
# exact-equality expectations.


@pytest.fixture(scope="session")
def analysis_input() -> dict:
    from compass.fixtures.frozen_outputs import load_real_or_frozen

    return load_real_or_frozen("analysis")


@pytest.fixture(scope="session")
def literature_input() -> list:
    from compass.fixtures.frozen_outputs import load_real_or_frozen

    return load_real_or_frozen("literature")


@pytest.fixture(scope="session")
def proposals_input() -> dict:
    from compass.fixtures.frozen_outputs import load_real_or_frozen

    return load_real_or_frozen("proposals")


@pytest.fixture
def make_fixture_llm():
    """Factory: make_fixture_llm(out1, out2, ...) -> FixtureLLM with that queue."""
    from compass.llm import FixtureLLM

    def _make(*outputs):
        return FixtureLLM(outputs=list(outputs))

    return _make


@pytest.fixture(scope="session")
def trace_cache_dir(tmp_path_factory) -> Path:
    """Disk cache for MCMC traces, shared by the slow tests of one session so a fit sampled
    by one test is reused by the next (``run_mcmc`` keys it by code, data, budget and seed).

    The research tree keeps a persistent ``.test_cache`` next to its code; the package
    suite uses a fresh per-session directory, which is why every test that samples here
    carries ``@pytest.mark.slow``.
    """
    return tmp_path_factory.mktemp("trace_cache")
