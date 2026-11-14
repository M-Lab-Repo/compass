"""Code Synthesizer agent — turns ranked proposals + the original model into
ONE complete, runnable refined model (spec §5.4).

Pipeline: render proposals -> SYNTH_PROMPT -> structured `SynthesizedCode` ->
validate (ast parse, then a subprocess MCMC smoke test for NumPyro, or a
cmdstanpy compile for Stan) -> on failure, feed the EXACT error text back via
FIX_PROMPT -> repeat up to `max_attempts`.

Model code is always handled as a SOURCE STRING: the smoke test exec's the
string in the same pre-seeded namespace the validator uses (numpyro, dist,
jnp, jax, np — the runtime contract for every program in the system) and
obtains `model` from it — the code is never imported as a module. Any other
undefined name still fails the smoke stage and recycles through the fix loop.
SECURITY: the subprocess is a correctness device, NOT a sandbox — the smoke
test executes LLM-WRITTEN code ON THE HOST, at module level, with the
filesystem and network privileges of the host process. What the child does
NOT get is the caller's environment: the interpreter runs in isolated mode
(``-I``) with the allowlisted, credential-scrubbed environment built by
``smoke_test_env`` (no ``*_API_KEY`` / ``*_TOKEN`` / ``BAYES_REFINE_*``
variables; see the constants above it). The validator's own IN-PROCESS
``exec`` of the same programs (``agents.validator._exec_model``, used for the
real MCMC comparison and the resolution audit) is unchanged and runs inside
the host process itself. Callers that need containment must provide it
externally (container, seccomp, etc.).
Because ``-I`` also ignores the user site-packages and PYTHONPATH, the smoke
interpreter is probed once per process (``_probe_smoke_interpreter``): when it
cannot import numpy / jax / numpyro, ``validate_numpyro_code`` raises
``EnvironmentError`` before any candidate runs instead of failing every
candidate at the smoke stage (and the fix loop with it).

`compare_models_waic_loo` is a lightweight convenience for WAIC/LOO model
comparison (spec §5.4 "add a model comparison function"). The Validator agent
owns the FULL comparison (convergence gates, deltas, PPC, failure routing);
this helper only summarizes az.waic/az.loo/az.compare for two InferenceData
objects.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping

import numpy as np
from langchain_core.prompts import ChatPromptTemplate

from compass.llm import STRUCTURED_RETRIES, StructuredOutputError, invoke_structured  # noqa: F401
from compass.schemas import SynthesizedCode
from compass.state import WorkflowState

# Sensible default when state.original_model.data_shapes is missing/empty:
# the golden 8-schools regression (spec §10).
GOLDEN_DEFAULT_DATA_SHAPES = {
    "X": [280],
    "y": [280],
    "group_idx": [280],
    "N_groups": 8,
}

# How much error text we feed back into the fix prompt (keep it bounded; the
# actual error is almost always at the tail of stderr/stdout).
_MAX_ERROR_CHARS = 6000

# Smoke-pass sentinel. Checked as the LAST stdout line (not a substring
# anywhere): candidate code printing "SUCCESS"-style strings at module level
# must not be able to spoof a pass. The harness prints it only after mcmc.run
# returns, so a later FAILURE line can never trail a genuine sentinel.
_SMOKE_SENTINEL = "SMOKE-OK-7f3d9c1e"

# --guard-replicate-sites: an opt-in NumPyro prompt rule. On two real datasets the
# synthesizer kept adding a discrete posterior-predictive replicate site
# (numpyro.sample("y_rep", ...)) that NUTS cannot sample, exhausted its repair
# attempts and produced no runnable candidate. The rule reaches SYNTH_PROMPT and
# FIX_PROMPT through a template variable that is the empty string when the guard
# is off, so the default prompt text is byte-identical to the unguarded one.
GUARD_REPLICATE_SITES_RULE = """

NumPyro site guard (the smoke test and the verifier sample with NUTS, which cannot
handle discrete sites):
- Every numpyro.sample site must be either observed (obs=...) or a continuous
  latent parameter.
- Do NOT add posterior-predictive replicate sites such as y_rep, y_pred or any
  *_rep site: the verifier generates replicates itself from the observed site.
- Discrete latent variables must be marginalized analytically (e.g. a mixture
  likelihood) or avoided; never sample them as sites."""
GUARD_REPLICATE_SITES_HINT = (
    "Hint (site guard): the error points at a discrete or replicate sample site. Remove "
    "every unobserved replicate site (y_rep and the like) and marginalize or drop any "
    "discrete latent so only observed sites and continuous latents remain.")
# Replicate-site name suffixes (mirrors agents.validator._PPC_VAR_SUFFIXES).
_REPLICATE_SITE_RE = re.compile(r"\b\w+_(?:rep|hat|pred|tilde|ppc|sim)\b")
_GUARD_ERROR_RE = re.compile(r"enumerat|discrete", re.IGNORECASE)


def guard_hint(error: str) -> str:
    """The one-line targeted fix hint of ``--guard-replicate-sites``: non-empty (and
    newline-padded for the prompt) when the smoke-test error mentions enumeration,
    a discrete site or a replicate site name; the empty string otherwise."""
    if _GUARD_ERROR_RE.search(error) or _REPLICATE_SITE_RE.search(error):
        return f"\n{GUARD_REPLICATE_SITES_HINT}\n"
    return ""


def _signature_matches_golden(code: str) -> bool:
    """True if the candidate's model signature uses EXACTLY the golden-default
    data names (X / y / group_idx / N_groups) — the only case where falling
    back to GOLDEN_DEFAULT_DATA_SHAPES for the smoke test makes sense. A
    strict-subset signature (e.g. ``def model(X, y=None)``) must NOT match:
    the harness calls ``mcmc.run(key, **dummy_data)`` with ALL golden kwargs,
    so a subset signature would fail with a guaranteed TypeError that is not
    the code's fault."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "model":
            args = {a.arg for a in node.args.args + node.args.kwonlyargs}
            return args == set(GOLDEN_DEFAULT_DATA_SHAPES)
    return False


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYNTH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert probabilistic programming (PPL) engineer, fluent in
NumPyro and Stan. Your task: apply ALL of the ranked refinement proposals to
the original Bayesian model and produce ONE complete, runnable model program.

Hard requirements:
1. Output ONE complete program — full source, not a diff or fragment. Include
   every import the code needs.
2. PRESERVE the original model's function signature and data inputs EXACTLY:
   same argument names, same order, same defaults. Never add, remove, or
   rename data arguments.
3. Apply the proposals in their given implementation order (priority 1
   first). Implement every change row exactly as proposed (exact
   distributions and hyperparameters). Do not invent extra refinements.
4. The code must compile and run as-is in an MCMC smoke test.

NumPyro-specific guidance:
- Use numpyro.distributions (imported as dist) for all distributions.
- Use numpyro.plate for vectorized sampling over groups/observations.
- Use numpyro.deterministic for derived quantities so they appear in the
  posterior (e.g., group intercepts recovered from a reparameterization).
- For non-centered parameterization, reparameterize MANUALLY: sample
  standardized offsets z ~ Normal(0, 1) inside the plate, then recover the
  group-level parameter as a deterministic site, e.g.
  alpha = numpyro.deterministic("alpha", mu_alpha + sigma_alpha * z).

Stan-specific guidance:
- Use proper block structure: data, parameters, transformed parameters,
  model, and generated quantities blocks.
- Use target += for custom log-density contributions.
- Include a generated quantities block producing posterior predictive draws
  NAMED `y_rep` (the validator's Stan PPC looks for a `y_rep` / `*_rep`
  generated quantity), plus log_lik if feasible, for posterior predictive checks.{guard_rules}""",
        ),
        (
            "human",
            """## Framework
{framework}

## Original model code
```
{original_code}
```

## Ranked refinement proposals (apply ALL, in implementation order)
{proposals}

## Data shapes (a smoke test will call the model with dummy data of exactly these shapes)
{data_shapes}

Produce the complete refined model program now.""",
        ),
    ]
)

FIX_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert probabilistic programming debugger. You are given
model code and the EXACT error output produced when compiling or running it.

Fix ONLY what the error indicates. Preserve everything else verbatim: the
function signature and data inputs, the chosen priors and hyperparameters,
the model structure, site names, and comments. Do not refactor, rename, or
"improve" anything beyond the minimal fix. Return the complete corrected
program (full source, with imports).{guard_rules}""",
        ),
        (
            "human",
            """## Framework
{framework}

## Code that failed
```
{code}
```

## Exact error output
```
{error}
```
{guard_hint}
Return the complete fixed program.""",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Proposal rendering (human-readable change tables for the prompt)
# ---------------------------------------------------------------------------
def render_proposals(proposals: list) -> str:
    """Render proposal dicts (Proposal.model_dump() shape) as prompt text."""
    if not proposals:
        return "(no proposals provided — reproduce the original model unchanged)"
    blocks = []
    for p in proposals:
        if hasattr(p, "model_dump"):  # tolerate Proposal objects
            p = p.model_dump()
        lines = [
            f"### Proposal (priority {p.get('priority', '?')}): {p.get('name', '?')}",
            f"- change_type: {p.get('change_type', '')}",
            f"- description: {p.get('description', '')}",
            f"- expected_improvement: {p.get('expected_improvement', '')}",
            f"- risk: {p.get('risk_assessment', '')}",
            f"- supporting papers: {'; '.join(p.get('supporting_papers') or [])}",
        ]
        changes = p.get("changes") or []
        if changes:
            lines.append("- changes:")
            lines.append("  | parameter | current | proposed | reason |")
            lines.append("  |---|---|---|---|")
            for c in changes:
                lines.append(
                    "  | {parameter} | {current} | {proposed} | {reason} |".format(
                        parameter=c.get("parameter", ""),
                        current=c.get("current", ""),
                        proposed=c.get("proposed", ""),
                        reason=c.get("reason", ""),
                    )
                )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Dummy data generation for the smoke test
# ---------------------------------------------------------------------------
def _is_index_key(key: str) -> bool:
    k = key.lower()
    return k.endswith("_idx") or k == "group"


def _find_index_count(idx_key: str, scalars: dict) -> int:
    """Find the scalar that bounds an index array (e.g. group_idx -> N_groups).

    Defensive: if no matching scalar exists, fall back to a small positive
    count so generated indices are always valid for SOME plate.
    """
    base = idx_key.lower()
    if base.endswith("_idx"):
        base = base[:-4]
    base = base.strip("_").rstrip("s")
    if base:
        for name, val in scalars.items():
            if base in name.lower() and val > 0:
                return int(val)
    for name, val in scalars.items():
        if name.lower().startswith(("n_", "num_")) and val > 0:
            return int(val)
    return 4


def generate_dummy_data(data_shapes: dict, rows: int = 32, seed: int = 0) -> dict:
    """Build a dict of numpy arrays/scalars matching `data_shapes`, shrunk.

    - int scalars pass through unchanged (e.g. N_groups: 8 stays 8) UNLESS
      the scalar equals an array leading dim that gets shrunk (e.g. N: 280
      alongside X: [280]) — then it is rewritten to the shrunk value so
      models that plate over the passed row count still see consistent shapes;
    - keys ending in "_idx" (or named "group") become int32 index arrays whose
      values are valid for the corresponding count scalar (e.g. N_groups);
    - everything else becomes standard-normal float32;
    - every array's LEADING dim is shrunk to min(dim, rows), so arrays that
      share a leading dim (e.g. X/y/group_idx all [280]) shrink consistently
      and models with a shared N still broadcast.
    """
    rng = np.random.default_rng(seed)
    scalars: dict = {}
    shapes: dict = {}
    for key, raw in (data_shapes or {}).items():
        if isinstance(raw, (list, tuple)):
            try:
                shapes[key] = [max(1, int(d)) for d in raw]
            except (TypeError, ValueError):
                shapes[key] = [rows]
        else:
            try:
                scalars[key] = int(raw)
            except (TypeError, ValueError):
                shapes[key] = [rows]  # unparseable spec -> defensive 1-D float

    # Leading dims that actually get shrunk (dim > rows): a scalar with the
    # same value encodes that dim (a Stan-style row count) and must shrink
    # with it; small count scalars (value <= rows, e.g. N_groups) stay put.
    # scalars is rewritten in place so index arrays below stay in range.
    shrunk_dims = {shape[0] for shape in shapes.values() if shape and shape[0] > rows}
    out: dict = {}
    for key in scalars:
        if scalars[key] in shrunk_dims:
            scalars[key] = rows
        out[key] = scalars[key]

    for key, shape in shapes.items():
        shrunk = tuple([min(shape[0], rows)] + list(shape[1:])) if shape else ()
        if _is_index_key(key):
            count = _find_index_count(key, scalars)
            out[key] = rng.integers(0, count, size=shrunk).astype(np.int32)
        elif key == "y":
            # The observed site (system-wide convention: obs_site defaults to
            # "y") must get SUPPORT-RESPECTING dummy values: standard-normal
            # floats are negative/non-integer, which is outside the support
            # of count likelihoods (NegativeBinomial2's log_prob is NaN
            # there), so candidates swapping the likelihood family failed
            # NUTS initialization in the smoke test through no fault of
            # their own. Small POSITIVE integers lie in the support of every
            # observation family used here (Normal, Student-t, Poisson,
            # NegBin/NB2, Binomial(n>=20), Gamma, LogNormal, zero-inflated
            # wrappers).
            out[key] = (1 + rng.poisson(2.0, size=shrunk)).astype(np.float32)
        else:
            out[key] = rng.standard_normal(shrunk).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Smoke-test interpreter environment (P4, 2026-11-11; P1, 2026-11-12)
# ---------------------------------------------------------------------------
# Defect fixed here: the smoke subprocess inherited the caller's FULL
# environment — provider API keys included — while exec'ing LLM-written code at
# module level. `smoke_test_env` is an ALLOWLIST: only the runtime essentials
# below pass through, and even among those anything credential-shaped
# (…_API_KEY, …_TOKEN) or a project knob (BAYES_REFINE_*) is dropped. The
# interpreter also runs in isolated mode (-I: every PYTHON* variable and the
# user site-packages are ignored, the script directory is not put on
# sys.path); the generated harness imports only numpy / jax / numpyro from the
# interpreter's OWN site-packages. PYTHONPATH is therefore NOT in the
# allowlist: -I ignores it, so passing it through was inert (it used to be
# kept "for a harness variant without -I" — such a variant must add it back
# together with the flag change).
# The flip side of -I is probed once, not discovered candidate by candidate: an
# installation whose numpy / jax / numpyro live in the user site
# (``pip install --user``) or are reachable only through PYTHONPATH makes
# EVERY candidate fail the smoke stage with ModuleNotFoundError, and the fix
# loop would burn max_attempts LLM calls per proposal on an environment
# problem. ``_probe_smoke_interpreter`` runs the import once per process and
# ``validate_numpyro_code`` raises ``EnvironmentError`` instead of a verdict.
_SMOKE_ENV_KEEP = frozenset({
    "PATH", "HOME", "LANG", "LANGUAGE", "TMPDIR", "TMP", "TEMP",
    "MPLBACKEND", "JAX_PLATFORMS", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
    # an interpreter does not even start on Windows without these
    "SYSTEMROOT", "SYSTEMDRIVE", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
    "COMSPEC", "PATHEXT",
})
# locale, and the numerical runtime's own knobs (accelerator, thread counts;
# KMP_ is Intel OpenMP — KMP_DUPLICATE_LIB_OK, KMP_AFFINITY, ...)
_SMOKE_ENV_KEEP_PREFIXES = ("LC_", "XLA_", "JAX_", "CUDA_", "OMP_", "KMP_", "MKL_",
                            "OPENBLAS_", "TF_", "NUMPYRO_")
_SMOKE_ENV_DROP_SUFFIXES = ("_API_KEY", "_TOKEN")
_SMOKE_ENV_DROP_PREFIXES = ("BAYES_REFINE_",)
_SMOKE_INTERPRETER_FLAGS = ("-I",)
# What the harness preamble imports; the probe imports exactly these.
_SMOKE_RUNTIME_MODULES = ("numpy", "jax", "numpyro")
_SMOKE_PROBE_TIMEOUT = 300  # seconds; a cold jax import can take a while
# The probe's verdict per (interpreter, flags): None when the runtime imports,
# else the error text. Filled once per process (see _probe_smoke_interpreter).
_SMOKE_INTERPRETER_VERDICT: dict[tuple[str, tuple[str, ...]], str | None] = {}
# A runtime module missing from the HARNESS PREAMBLE (see _preamble_import_failure).
_PREAMBLE_MISSING_MODULE_RE = re.compile(
    r"ModuleNotFoundError: No module named '(?:numpy|jax|numpyro)(?:\.[\w.]*)?'")


def smoke_test_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The scrubbed environment handed to the smoke-test interpreter.

    From ``environ`` (default ``os.environ``) keep a variable only if its
    name is in ``_SMOKE_ENV_KEEP`` or starts with one of
    ``_SMOKE_ENV_KEEP_PREFIXES`` — and never if it ends in ``_API_KEY`` /
    ``_TOKEN`` or starts with ``BAYES_REFINE_``, whatever else matches.
    Pure: a new dict, nothing read but ``environ``.
    """
    source = os.environ if environ is None else environ
    env: dict[str, str] = {}
    for key, value in source.items():
        name = str(key).upper()
        if name.endswith(_SMOKE_ENV_DROP_SUFFIXES) or name.startswith(_SMOKE_ENV_DROP_PREFIXES):
            continue
        if name in _SMOKE_ENV_KEEP or name.startswith(_SMOKE_ENV_KEEP_PREFIXES):
            env[str(key)] = str(value)
    return env


def _smoke_environment_error(detail: str) -> OSError:
    """The one error raised when the smoke-test interpreter cannot import the runtime:
    an ``EnvironmentError`` (``OSError`` is its Python 3 name; ruff UP024).

    Names the cause — the interpreter runs in isolated mode, so packages that live
    in the user site-packages (``pip install --user``) or are reachable only through
    PYTHONPATH are invisible to it — and the remedy (install into the interpreter's
    own environment, or run from a venv that has the packages)."""
    flags = " ".join(_SMOKE_INTERPRETER_FLAGS)
    modules = ", ".join(_SMOKE_RUNTIME_MODULES)
    return OSError(
        f"the smoke-test interpreter cannot import the NumPyro runtime ({modules}): "
        f"{detail.strip()}\n"
        f"The smoke test runs `{sys.executable} {flags}`, and isolated mode ignores "
        "PYTHONPATH, every PYTHON* variable and the user site-packages (pip install "
        "--user), so the packages must be installed in the interpreter's own "
        f"environment: install them into this interpreter (`{sys.executable} -m pip "
        f"install {' '.join(_SMOKE_RUNTIME_MODULES)}`) or run from a virtual environment "
        "that has them. No candidate was smoke-tested.")


def _probe_smoke_interpreter(interpreter: str | None = None,
                             flags: tuple[str, ...] = _SMOKE_INTERPRETER_FLAGS) -> None:
    """Check ONCE per process that the smoke-test interpreter, run exactly as the
    harness runs it (``interpreter``, ``flags``, ``smoke_test_env()``), can import
    numpy, jax and numpyro.

    Raises the ``EnvironmentError`` of ``_smoke_environment_error`` when it cannot;
    returns None when it can. The verdict is cached either way in
    ``_SMOKE_INTERPRETER_VERDICT``, so a broken layout costs one interpreter start
    per process — never one per candidate, and never an LLM fix attempt."""
    interpreter = interpreter or sys.executable
    key = (interpreter, tuple(flags))
    if key not in _SMOKE_INTERPRETER_VERDICT:
        probe = "import " + ", ".join(_SMOKE_RUNTIME_MODULES)
        try:
            result = subprocess.run(
                [interpreter, *flags, "-c", probe], capture_output=True, text=True,
                timeout=_SMOKE_PROBE_TIMEOUT, env=smoke_test_env())
        except subprocess.TimeoutExpired:
            verdict: str | None = f"`{probe}` did not finish within {_SMOKE_PROBE_TIMEOUT}s"
        else:
            if result.returncode == 0:
                verdict = None
            else:
                tail = (result.stderr or result.stdout or "").strip().splitlines()
                verdict = (f"`{probe}` exited with status {result.returncode}"
                           + (f": {tail[-1].strip()}" if tail else ""))
        _SMOKE_INTERPRETER_VERDICT[key] = verdict
    verdict = _SMOKE_INTERPRETER_VERDICT[key]
    if verdict is not None:
        raise _smoke_environment_error(verdict)


def _preamble_import_failure(stderr: str) -> str | None:
    """The runtime import that failed in the HARNESS PREAMBLE of a smoke run, or None.

    The harness imports numpy / jax / numpyro before it exec's the candidate (which
    is compiled with the filename ``<candidate>``), so a ``ModuleNotFoundError`` for
    one of them whose traceback has no ``<candidate>`` frame is the interpreter's
    environment failing, not the candidate: ``validate_numpyro_code`` then raises the
    probe's ``EnvironmentError`` instead of returning a smoke verdict. A candidate's
    own missing module — any other name, or a ``<candidate>`` frame — stays an
    ordinary smoke failure for the fix loop."""
    if "<candidate>" in stderr:
        return None
    m = _PREAMBLE_MISSING_MODULE_RE.search(stderr)
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# Validation: ast parse + subprocess MCMC smoke test (NumPyro)
# ---------------------------------------------------------------------------
def _build_smoke_script(code: str, npz_path: str) -> str:
    """Assemble the smoke-test script (spec §5.4). Dummy data is loaded from
    a .npz written with np.savez — never repr'd into the source.

    The candidate is exec'd in the same pre-seeded namespace the validator's
    ``_exec_model`` uses (numpyro, dist, jnp, jax, np): that namespace is the
    runtime contract for every program in this system — the benchmark grid and
    the real-data baselines are bare ``def model`` bodies written against it —
    so an isolated namespace would reject programs the verifier accepts."""
    return (
        "import numpy as np\n"
        "import jax\n"
        "import jax.numpy as jnp\n"
        "import numpyro\n"
        "import numpyro.distributions as dist\n"
        "from numpyro.infer import MCMC, NUTS\n"
        f"_npz = np.load({npz_path!r})\n"
        "dummy_data = {k: (_npz[k].item() if _npz[k].ndim == 0 else _npz[k])\n"
        "              for k in _npz.files}\n"
        "\n"
        "_ns = {'numpyro': numpyro, 'dist': dist, 'jnp': jnp, 'jax': jax,\n"
        "       'np': np, 'numpy': np}\n"
        f"exec(compile({code!r}, '<candidate>', 'exec'), _ns)\n"
        "model = _ns['model']\n"
        "\n"
        "kernel = NUTS(model)\n"
        "mcmc = MCMC(kernel, num_warmup=10, num_samples=10, num_chains=1,\n"
        "            progress_bar=False)\n"
        "try:\n"
        "    mcmc.run(jax.random.PRNGKey(0), **dummy_data)\n"
        f"    print({_SMOKE_SENTINEL!r})\n"
        "except Exception as e:\n"
        "    print(f'FAILURE: {type(e).__name__}: {e}')\n"
    )


def validate_numpyro_code(code: str, data_shapes: dict, timeout: int = 120, *,
                          data: dict | None = None) -> dict:
    """Two-stage check: ast.parse (catches syntax errors cheaply, spec §13
    pitfall), then a subprocess NUTS smoke test (10 warmup / 10 samples /
    1 chain) on dummy data shaped like `data_shapes`.

    ``data`` (keyword-only): when given, the smoke test runs on THESE arrays
    (dtype preserved, scalars as 0-d) instead of dummy data — shape-derived
    dummies are semantically wrong for real datasets (an integer ``tries`` or
    ``dept`` column comes out as negative float32), so any candidate using
    them as a Binomial total_count or an index dies through no fault of its
    own. ``data_shapes`` is then ignored. ``None`` keeps the dummy-data path.

    Returns {"compiled": bool, "stage": "ast"|"smoke", "stdout": str, "stderr": str}.

    Raises ``EnvironmentError`` (never a verdict) when the smoke interpreter itself
    cannot import numpy / jax / numpyro — found by the once-per-process probe before
    any candidate runs, or by a ``ModuleNotFoundError`` in the harness preamble.
    """
    try:
        ast.parse(code)
    except SyntaxError as e:
        return {"compiled": False, "stage": "ast", "stdout": "", "stderr": f"SyntaxError: {e}"}

    if data is not None:
        payload = {k: np.asarray(v) for k, v in data.items()}
    elif not data_shapes:
        if _signature_matches_golden(code):
            data_shapes = GOLDEN_DEFAULT_DATA_SHAPES
        else:
            # No shapes and a non-golden signature: dummy data would be
            # guaranteed-wrong kwargs (TypeError churn through the fix loop),
            # so degrade honestly to the ast-only verdict.
            return {
                "compiled": True,
                "stage": "ast",
                "stdout": "smoke test skipped: no data_shapes provided and the "
                          "model signature does not match the golden default",
                "stderr": "",
            }

    if data is None:
        payload = generate_dummy_data(data_shapes)
    # P1: the interpreter is probed once per process; a layout it cannot import the
    # runtime from raises here, before any candidate runs, instead of failing every
    # candidate at the smoke stage.
    _probe_smoke_interpreter()
    with tempfile.TemporaryDirectory() as td:
        npz_path = os.path.join(td, "dummy_data.npz")
        np.savez(npz_path, **payload)
        script_path = os.path.join(td, "smoke_test.py")
        with open(script_path, "w") as f:
            f.write(_build_smoke_script(code, npz_path))
        try:
            # P4: isolated interpreter + scrubbed environment (see smoke_test_env);
            # LLM-written code still runs on the host — this is not a sandbox.
            result = subprocess.run(
                [sys.executable, *_SMOKE_INTERPRETER_FLAGS, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=td,
                env=smoke_test_env(),
            )
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            return {
                "compiled": False,
                "stage": "smoke",
                "stdout": stdout,
                "stderr": f"smoke test timed out after {timeout}s",
            }
    stdout_lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    passed = bool(stdout_lines) and stdout_lines[-1].strip() == _SMOKE_SENTINEL
    if not passed:
        missing = _preamble_import_failure(result.stderr or "")
        if missing:  # the harness died before the candidate ran: environment, not code
            raise _smoke_environment_error(
                f"the smoke harness failed before the candidate ran: {missing}")
    return {
        "compiled": passed,
        "stage": "smoke",
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def validate_stan_code(code: str) -> dict:
    """Compile-only check for Stan programs via cmdstanpy.CmdStanModel.

    SLOW: a cold CmdStan compile takes ~30-90s (stanc + C++ toolchain), which
    is why this is used only when framework == "stan" and is never exercised
    in the default test suite. Returns the same dict shape as
    validate_numpyro_code, with stage "smoke" (compilation IS the smoke test
    for Stan — there is no cheap MCMC dry run without compiling first).
    """
    try:
        from cmdstanpy import CmdStanModel
    except Exception as e:  # cmdstanpy/CmdStan not installed
        return {
            "compiled": False,
            "stage": "smoke",
            "stdout": "",
            "stderr": f"cmdstanpy unavailable: {e}",
        }
    with tempfile.TemporaryDirectory() as td:
        stan_file = os.path.join(td, "model.stan")
        with open(stan_file, "w") as f:
            f.write(code)
        try:
            CmdStanModel(stan_file=stan_file)  # compiles; raises on failure
            return {"compiled": True, "stage": "smoke", "stdout": "", "stderr": ""}
        except Exception as e:
            return {"compiled": False, "stage": "smoke", "stdout": "", "stderr": str(e)}


def _format_error(verdict: dict) -> str:
    """Exact stderr/stdout from a failed validation, bounded for the prompt."""
    parts = [f"validation stage: {verdict.get('stage', '?')}"]
    if verdict.get("stderr"):
        parts.append("--- stderr ---\n" + str(verdict["stderr"]))
    if verdict.get("stdout"):
        parts.append("--- stdout ---\n" + str(verdict["stdout"]))
    text = "\n".join(parts)
    return text[-_MAX_ERROR_CHARS:]


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
def create_synthesizer_agent(
    llm,
    max_attempts: int = 3,
    validator: Callable[[str, dict], dict] | None = None,
    data: dict | None = None,
    guard_replicate_sites: bool = False,
):
    """Build the synthesizer node: generate -> validate -> fix loop (§5.4).

    `validator(code, data_shapes) -> verdict dict` may be injected for tests
    (e.g. an ast-only checker); by default the framework-appropriate real
    validator is used (NumPyro subprocess smoke test / Stan compile).

    ``data``: when given (and no validator is injected), the default NumPyro
    check smoke-tests candidates on these real arrays rather than on dummy
    data derived from ``data_shapes`` — the call-site contract
    ``check(code, data_shapes)`` is unchanged (``functools.partial``).

    ``guard_replicate_sites`` (default False, ``--guard-replicate-sites``): adds
    ``GUARD_REPLICATE_SITES_RULE`` to both prompts (no unobserved replicate or
    discrete sample sites) and, when a smoke-test error mentions enumeration, a
    discrete site or a replicate site name, the one-line ``guard_hint`` to the
    fix prompt. Off, both template variables render as the empty string.
    """
    synth_chain = SYNTH_PROMPT | llm.with_structured_output(SynthesizedCode)
    fix_chain = FIX_PROMPT | llm.with_structured_output(SynthesizedCode)
    guard_rules = GUARD_REPLICATE_SITES_RULE if guard_replicate_sites else ""

    def run(state: WorkflowState) -> dict:
        spec = state.original_model
        framework = ((spec.framework if spec else "") or "numpyro").lower()
        # No golden fallback here: validate_numpyro_code applies it only when
        # the candidate's signature actually matches the golden names, and
        # degrades to ast-only otherwise (no misleading shapes in the prompt).
        data_shapes = dict((spec.data_shapes if spec else None) or {})
        original_code = (spec.source_code if spec else "") or ""
        proposals = list(state.proposals or [])  # dicts, already priority-sorted

        if validator is not None:
            check = validator
        elif framework == "stan":
            check = lambda code, shapes: validate_stan_code(code)  # noqa: E731
        elif data is not None:
            # resolved per run (not at factory time) so the module attribute
            # stays the single patch point, exactly like the bare default below
            check = functools.partial(validate_numpyro_code, data=data)
        else:
            check = validate_numpyro_code

        def give_up(reason: str) -> dict:
            # Exhausted: leave status untouched — the orchestrator routes on the
            # empty refined_model_code (spec §5.6 route_after_synthesis).
            # error_log entries are plain strings pipeline-wide (the report node
            # renders them as `- {e}` bullets), so format rather than append a dict.
            state.refined_model_code = ""
            state.error_log.append(f"synthesizer: {reason}")
            return {"refined_model_code": "", "error_log": state.error_log}

        try:
            result = invoke_structured(
                synth_chain,
                {
                    "framework": framework,
                    "original_code": original_code,
                    "proposals": render_proposals(proposals),
                    "data_shapes": json.dumps(data_shapes),
                    "guard_rules": guard_rules,
                },
                nudge_field="proposals",
                what="SynthesizedCode",
                schema_hint="in particular `code` and `applied_proposals`",
            )
        except StructuredOutputError as e:
            return give_up(str(e))

        last_error = ""
        for attempt in range(1, max_attempts + 1):
            code = result.code
            verdict = check(code, data_shapes)
            if verdict.get("compiled"):
                state.refined_model_code = code
                state.status = "synthesized"
                return {"refined_model_code": code, "status": "synthesized"}
            last_error = _format_error(verdict)
            if attempt < max_attempts:
                try:
                    result = invoke_structured(
                        fix_chain,
                        {"framework": framework, "code": code, "error": last_error,
                         "guard_rules": guard_rules,
                         "guard_hint": guard_hint(last_error) if guard_replicate_sites else ""},
                        nudge_field="error",
                        what="SynthesizedCode",
                    )
                except StructuredOutputError as e:
                    return give_up(f"{e}; last validation error: {last_error}")

        return give_up(f"{max_attempts} attempts exhausted; last error: {last_error}")

    return run


# ---------------------------------------------------------------------------
# Model comparison convenience (full comparison lives in the Validator)
# ---------------------------------------------------------------------------
def compare_models_waic_loo(idata_refined, idata_original) -> dict:
    """Small WAIC/LOO comparison of two arviz.InferenceData (need log_likelihood).

    Returns elpd values, deltas (refined - original; POSITIVE elpd delta =
    refined predicts better) and az.compare's preferred model. The Validator
    agent owns the authoritative comparison and pass/fail routing.
    """
    import arviz as az  # lazy: arviz import is slow and only needed here

    waic_r, waic_o = az.waic(idata_refined), az.waic(idata_original)
    loo_r, loo_o = az.loo(idata_refined), az.loo(idata_original)
    comp = az.compare({"refined": idata_refined, "original": idata_original}, ic="loo")
    return {
        "elpd_waic_refined": float(waic_r.elpd_waic),
        "elpd_waic_original": float(waic_o.elpd_waic),
        "elpd_waic_delta": float(waic_r.elpd_waic - waic_o.elpd_waic),
        "elpd_loo_refined": float(loo_r.elpd_loo),
        "elpd_loo_original": float(loo_o.elpd_loo),
        "elpd_loo_delta": float(loo_r.elpd_loo - loo_o.elpd_loo),
        "preferred": str(comp.index[0]),
    }
