"""The runnable examples and the skill's runner scripts.

Fast: the key-free verifier example imports as a module and both of its model
strings compile and trace on its data; both skill scripts answer ``--help``
with exit code 0 (they are run as subprocesses, the way a user runs them).
Slow: the example runs end to end with ``--fast`` and prints a verdict line.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "verifier_only.py"
SKILL_SCRIPTS = ROOT / "skill" / "scripts"


def _load_module(path: Path):
    """Import a script by path (examples/ is not a package)."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace_model(src: str, data: dict) -> dict:
    """Execute a model string the way the verifier does and trace it on `data`."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro import handlers

    namespace = {"numpyro": numpyro, "dist": dist, "jnp": jnp}
    exec(compile(src, "<model>", "exec"), namespace)
    model = namespace["model"]
    assert callable(model)
    return handlers.trace(handlers.seed(model, 0)).get_trace(**data)


def test_verifier_example_imports_and_its_models_compile():
    mod = _load_module(EXAMPLE)
    data = mod.make_data()
    assert data["n"] == len(data["y"]) == 200
    assert data["y"].var() > 2 * data["y"].mean(), "the counts must be overdispersed"
    for src in (mod.BASELINE, mod.CANDIDATE):
        compile(src, "<model>", "exec")
        trace = _trace_model(src, data)
        assert trace["y"]["is_observed"]
        assert trace["y"]["value"].shape == (200,)
    # the candidate adds exactly one latent (the dispersion parameter)
    latents = {name for name, site in _trace_model(mod.CANDIDATE, data).items()
               if site["type"] == "sample" and not site["is_observed"]}
    assert latents == {"mu", "phi"}
    assert mod.FAST_BUDGET["num_samples"] < mod.BUDGET["num_samples"]


def test_verifier_example_help_exits_zero():
    proc = subprocess.run([sys.executable, str(EXAMPLE), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "--fast" in proc.stdout


@pytest.mark.parametrize("script", ["run_refinement.py", "verify_candidate.py"])
def test_skill_script_help_exits_zero(script):
    proc = subprocess.run([sys.executable, str(SKILL_SCRIPTS / script), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("usage:")
    assert "--data" in proc.stdout


@pytest.mark.slow
def test_verifier_example_runs_with_fast(tmp_path):
    """Two real NUTS fits at the --fast budget; the script must print a verdict line."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    proc = subprocess.run([sys.executable, str(EXAMPLE), "--fast"],
                          capture_output=True, text=True, timeout=1800,
                          cwd=tmp_path, env=env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    lines = proc.stdout.splitlines()
    verdict = [line for line in lines if line.startswith("verdict: ")]
    assert len(verdict) == 1, proc.stdout
    assert verdict[0].split(": ", 1)[1].split(" ")[0] in {"accepted", "refused"}
    assert any(line.startswith("failure reasons: ") for line in lines)
    assert any(line.startswith("elpd_diff: ") for line in lines)
    assert any(line.startswith("wall time: ") for line in lines)
