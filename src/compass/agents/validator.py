"""Validator agent — the no-LLM deterministic gatekeeper (spec §5.5).

Runs the refined and original models with real MCMC, computes ArviZ
convergence diagnostics, compares predictive performance (WAIC / LOO), and
runs a posterior predictive check. Emits a `state.ValidationResult` whose
`failure_reasons` are specific enough for the orchestrator to route on
("47 divergent transitions", not just "failed").

Conventions used throughout:

* Model code is ALWAYS a source string. It is exec'd into a fresh namespace
  pre-seeded with ``numpyro``, ``numpyro.distributions as dist``,
  ``jax.numpy as jnp`` and ``jax``; the namespace must define ``model``.
* Comparison sign convention (spec §5.5/§13): deltas are computed on the
  *deviance* scale (lower = better), ``delta = refined - original``, so a
  NEGATIVE delta means the refined model is better. ``elpd_diff`` (from
  ``az.compare`` on its default log scale) is likewise signed so that
  negative = refined better.
* Trace cache: when ``cache_dir`` is given, the converted ``InferenceData``
  is pickled under a key of sha256(x64 flag + model_code + framework +
  budget + seed + data bytes). A cache hit skips sampling entirely and
  returns ``(None, idata)``.

Thresholds (spec §5.5): R-hat < 1.01, ESS > 400, 0 divergences,
E-BFMI > 0.3.

Comparability guards on the S gate (defect V1, 2026-11-11; hardened after the
adversarial review of 2026-11-12, findings F1–F5): the refined program is
traced on the data NEXT TO the original program, so every structural rule is
comparative (``compare_observed_structure``). The comparison is refused when a
refined observed site is masked beyond the baseline's own mask
(``handlers.mask`` / ``.mask()`` / ``obs_mask=``), scaled differently from the
baseline, or governed by likelihood parameters that depend on the observed
response while the baseline's do not (the two-faced likelihood, F3);
``compare_models`` refuses fits whose observation sets differ — compared at
float32 precision when either side cast its response through ``jnp.asarray``
(F1) — or whose refined pointwise log-likelihood has degenerate columns the
baseline does not share at the same positions (F2, F4). Such a refusal is the
failure reason ``model comparison refused: ...`` — never an elpd number.

The structural guards (mask / scale / response dependence) are NumPyro-only:
they work by tracing the program under numpyro's effect handlers, which a Stan
program has no equivalent of. On the Stan path the S gate rests on the
fit-level checks alone — observation-set identity through ``observed_data``
(passed to ``az.from_cmdstanpy`` since F5) and the degenerate-column rule.
"""

from __future__ import annotations

import hashlib
import pickle
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive

from compass.state import ValidationResult, WorkflowState

# ---------------------------------------------------------------------------
# Thresholds and budgets (spec §5.5 / §11)
# ---------------------------------------------------------------------------

RHAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400.0
EBFMI_THRESHOLD = 0.3
KHAT_THRESHOLD = 0.7  # PSIS-LOO Pareto k-hat above which importance sampling is unreliable

FULL_BUDGET = (1000, 2000, 4)  # (num_warmup, num_samples, num_chains)
FAST_BUDGET = (200, 500, 2)


class ComparisonRefused(ValueError):
    """The S-gate comparison was REFUSED on structural grounds (defect V1).

    Raised by ``compare_models`` (observation-set identity / degenerate
    pointwise columns) and by ``run_full_validation`` (masked or scaled
    observed sites found by ``check_observed_sites``) when the two fits cannot
    be scored on the same observations — in which case *no* elpd difference is
    evidence of anything. Distinct from an ordinary failure (``model
    comparison unavailable: ...``) so that the message, which always starts
    with ``model comparison refused: ``, can be routed and reported as a
    verdict rather than as a tooling error.
    """

    def __init__(self, detail: str):
        super().__init__(f"model comparison refused: {detail}")


def _khat_unreliable(k) -> bool:
    """PSIS reliability predicate on one Pareto k-hat (defect V2).

    True when ``k > KHAT_THRESHOLD`` OR ``k`` is non-finite but not NaN. The
    two non-finite values mean opposite things and must not be lumped
    together:

    * ``inf`` is what arviz returns for DEGENERATE importance weights — e.g. a
      pointwise log-likelihood column that is constant across draws, exactly
      what a masked observation produces. It is the worst possible
      reliability verdict, yet the former test ``np.isfinite(k) and k > 0.7``
      was False for it and silently disabled both the gate and the
      auto-escalation. ``inf`` must gate (and escalate) like ``k > 0.7``.
    * ``NaN`` means k-hat could NOT be estimated (no information, e.g. too few
      draws for the tail fit). The documented behaviour — pinned by
      tests/test_validator_fixes.py::test_nan_khat_does_not_gate — is that NaN
      does not gate; it stays that way.
    """
    if k is None:
        return False
    k = float(k)
    if np.isnan(k):
        return False
    return (not np.isfinite(k)) or k > KHAT_THRESHOLD


# ---------------------------------------------------------------------------
# Model-code execution (source string -> callable), spec convention #4
# ---------------------------------------------------------------------------

def _exec_model(model_code: str) -> Callable:
    """Exec NumPyro model source in a fresh namespace and return `model`."""
    namespace: dict[str, Any] = {
        "numpyro": numpyro,
        "dist": dist,
        "jnp": jnp,
        "jax": jax,
        "np": np,
        "numpy": np,
    }
    exec(compile(model_code, "<model_code>", "exec"), namespace)
    model = namespace.get("model")
    if not callable(model):
        raise ValueError("model code must define a callable named 'model'")
    return model


@contextmanager
def _restored_x64():
    """Snapshot/restore the process-global ``jax_enable_x64`` flag.

    exec'ing candidate source runs its module-level statements in THIS
    process, and LLM-generated code loves ``numpyro.enable_x64()`` — without
    the restore that line would silently flip every subsequent model in the
    session (including tournament rivals) to float64. Every ``_exec_model``
    caller (MCMC, PPC, held-out lppd) must run under this guard.
    """
    before = bool(jax.config.jax_enable_x64)
    try:
        yield before
    finally:
        if bool(jax.config.jax_enable_x64) != before:
            jax.config.update("jax_enable_x64", before)


# ---------------------------------------------------------------------------
# Trace cache
# ---------------------------------------------------------------------------

def _cache_key(model_code: str, framework: str, budget: tuple, seed: int,
               data: dict, x64: bool) -> str:
    h = hashlib.sha256()
    # Library versions: a numpyro/jax/arviz bump must invalidate cached traces
    # so convergence tests re-verify against the live stack, not stale pickles.
    h.update(f"numpyro={numpyro.__version__};jax={jax.__version__};"
             f"arviz={az.__version__}".encode())
    # The x64 flag is folded in ahead of the source (float32 vs float64 runs
    # of the same model are different numerics). The "x64=<flag>;<code>" byte
    # layout is deliberate: it matches caches written by older code that
    # prefixed the flag onto model_code itself, so existing pickles still hit.
    h.update(f"x64={bool(x64)};{model_code}".encode())
    h.update(framework.encode())
    h.update(str(budget).encode())
    h.update(str(seed).encode())
    for key in sorted(data):
        arr = np.asarray(data[key])
        h.update(key.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        if arr.dtype == object:
            # Object arrays' tobytes() serializes raw POINTERS — keys would be
            # nondeterministic across processes (permanent misses) and two
            # objects reusing a freed address could collide into a wrong cache
            # hit for the sole arbiter. Hash a canonical pickle instead.
            h.update(pickle.dumps(data[key], protocol=4))
        else:
            h.update(arr.tobytes())
    return h.hexdigest()


def trace_cache_path(cache_dir, model_code: str, data: dict,
                     framework: str = "numpyro", num_warmup: int = 1000,
                     num_samples: int = 2000, num_chains: int = 4,
                     seed: int = 0, target_accept: float = 0.8,
                     x64: bool | None = None) -> Path:
    """Deterministic pickle path for one (model, data, budget, seed, x64) run.

    ``x64`` defaults to the CURRENT ``jax_enable_x64`` flag, which is exactly
    what ``run_mcmc`` snapshots before sampling — so external callers passing
    the raw model code compute the same path ``run_mcmc`` writes.
    """
    if x64 is None:
        x64 = bool(jax.config.jax_enable_x64)
    budget = (num_warmup, num_samples, num_chains, target_accept)
    key = _cache_key(model_code, framework, budget, seed, data, x64)
    return Path(cache_dir) / f"trace_{key}.pkl"


# ---------------------------------------------------------------------------
# MCMC runners
# ---------------------------------------------------------------------------

def _run_numpyro(model_code: str, data: dict, num_warmup: int,
                 num_samples: int, num_chains: int, seed: int,
                 target_accept: float):
    model = _exec_model(model_code)
    kernel = NUTS(model, target_accept_prob=target_accept)
    with warnings.catch_warnings():
        # "not enough devices to run parallel chains" — expected on CPU.
        warnings.simplefilter("ignore", UserWarning)
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=False,
        )
        mcmc.run(
            jax.random.PRNGKey(seed),
            **data,
            extra_fields=("energy", "diverging"),  # energy enables E-BFMI
        )
    idata = az.from_numpyro(mcmc)
    return mcmc, idata


def _run_stan(model_code: str, data: dict, num_warmup: int, num_samples: int,
              num_chains: int, seed: int, target_accept: float,
              cache_dir=None):
    """Sample a Stan program with cmdstanpy.

    ``seed`` is passed to cmdstan UNCHANGED and must be non-negative (defect
    V8): the former ``seed if seed > 0 else 1`` mapped seeds 0 and 1 onto the
    same cmdstan run while the trace cache keyed them separately, so two
    "independent" seeds returned one fit. cmdstan accepts seed 0; a negative
    seed is a caller error and raises ``ValueError`` (before any compilation).

    The numeric array entries of ``data`` (the program's data block) are handed
    to ``az.from_cmdstanpy`` as ``observed_data`` when it accepts that argument
    (F5): without them the fit carried no observed values and the S gate's
    identity check (``_check_comparable_fits``) degraded to shapes only. The
    structural guards (mask / scale / response dependence) remain NumPyro-only
    — see the module docstring.
    """
    import inspect
    import logging
    import tempfile

    if seed < 0:
        raise ValueError(f"Stan seed must be a non-negative integer, got {seed}")

    try:
        import cmdstanpy
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "The Stan backend requires cmdstanpy: "
            "pip install 'compass-refine[stan]' (plus a CmdStan toolchain)"
        ) from exc

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    # Stable filename when a cache_dir exists -> cmdstanpy reuses the
    # compiled executable across calls instead of recompiling.
    if cache_dir is not None:
        stan_dir = Path(cache_dir)
        stan_dir.mkdir(parents=True, exist_ok=True)
    else:
        stan_dir = Path(tempfile.mkdtemp(prefix="stan_model_"))
    code_hash = hashlib.sha256(model_code.encode()).hexdigest()[:16]
    stan_file = stan_dir / f"stan_model_{code_hash}.stan"
    if not stan_file.exists() or stan_file.read_text() != model_code:
        stan_file.write_text(model_code)

    model = cmdstanpy.CmdStanModel(stan_file=str(stan_file))
    fit = model.sample(
        data=data or None,
        chains=num_chains,
        iter_warmup=num_warmup,
        iter_sampling=num_samples,
        seed=seed,  # passed through unchanged (validated non-negative above)
        adapt_delta=target_accept,
        show_progress=False,
        show_console=False,
    )
    observed = {
        k: np.asarray(v) for k, v in (data or {}).items()
        if isinstance(v, np.ndarray | list | tuple)
        and np.asarray(v).ndim >= 1 and np.asarray(v).dtype.kind in "biuf"
    }
    try:
        accepts_observed = "observed_data" in inspect.signature(az.from_cmdstanpy).parameters
    except (TypeError, ValueError):
        accepts_observed = False
    converter_kwargs: dict[str, Any] = {}
    if observed and accepts_observed:
        converter_kwargs["observed_data"] = observed
    idata = az.from_cmdstanpy(posterior=fit, **converter_kwargs)
    return fit, idata


def run_mcmc(model_code: str, data: dict, framework: str = "numpyro",
             num_warmup: int = 1000, num_samples: int = 2000,
             num_chains: int = 4, seed: int = 0, target_accept: float = 0.8,
             cache_dir=None):
    """Sample `model_code` on `data`; return (mcmc_or_fit, idata).

    On a trace-cache hit the first element is None: sampling (and, for Stan,
    compilation) is skipped entirely.

    Runs under `_restored_x64` (see its docstring for why exec'ing candidate
    source can flip process-global JAX config). The pre-run x64 flag is also
    folded into the cache key via `trace_cache_path`'s default.
    """
    data = data or {}
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = trace_cache_path(
            cache_dir, model_code, data, framework,
            num_warmup, num_samples, num_chains, seed, target_accept,
        )
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return None, pickle.load(f)

    with _restored_x64():
        if framework == "numpyro":
            sampler, idata = _run_numpyro(
                model_code, data, num_warmup, num_samples, num_chains, seed,
                target_accept,
            )
        elif framework == "stan":
            sampler, idata = _run_stan(
                model_code, data, num_warmup, num_samples, num_chains, seed,
                target_accept, cache_dir=cache_dir,
            )
        else:
            raise ValueError(f"unknown framework: {framework!r}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(idata, f)
    return sampler, idata


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_diagnostics(idata, mcmc=None) -> dict:
    """Convergence diagnostics over ALL posterior variables and dimensions.

    Returns {rhat_max, ess_min, divergences, ebfmi_min}. With a single chain
    R-hat is undefined and reported as NaN (arviz 0.23 returns NaN values);
    E-BFMI is NaN when no energy was recorded.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rhat_ds = az.rhat(idata)
        ess_ds = az.ess(idata)
        rhat_values = [np.asarray(rhat_ds[v].values, dtype=float)
                       for v in rhat_ds.data_vars]
        ess_values = [np.asarray(ess_ds[v].values, dtype=float)
                      for v in ess_ds.data_vars]
        rhat_max = float(np.nanmax([np.nanmax(v) for v in rhat_values])) \
            if rhat_values else float("nan")
        ess_min = float(np.nanmin([np.nanmin(v) for v in ess_values])) \
            if ess_values else float("nan")

        divergences = 0
        sample_stats = getattr(idata, "sample_stats", None)
        if sample_stats is not None and "diverging" in sample_stats:
            divergences = int(np.asarray(sample_stats["diverging"]).sum())
        elif mcmc is not None and hasattr(mcmc, "get_extra_fields"):
            extra = mcmc.get_extra_fields()
            if "diverging" in extra:
                divergences = int(np.asarray(extra["diverging"]).sum())

        try:
            ebfmi_min = float(np.nanmin(np.asarray(az.bfmi(idata))))
        except (TypeError, ValueError):
            ebfmi_min = float("nan")  # no energy in sample_stats

    return {
        "rhat_max": rhat_max,
        "ess_min": ess_min,
        "divergences": divergences,
        "ebfmi_min": ebfmi_min,
    }


# ---------------------------------------------------------------------------
# Model comparison (WAIC / LOO), spec sign convention: negative = better
# ---------------------------------------------------------------------------

def _observed_arrays(idata) -> dict[str, np.ndarray]:
    """``{name: values}`` of an InferenceData's observed_data group ({} if absent)."""
    group = getattr(idata, "observed_data", None)
    if group is None:
        return {}
    return {str(name): np.asarray(group[name].values) for name in group.data_vars}


def _pointwise_shapes(idata) -> dict[str, tuple] | None:
    """``{name: obs_shape}`` (the dims after chain/draw) of the log_likelihood
    group, or None when the fit carries no pointwise log-likelihood at all."""
    group = getattr(idata, "log_likelihood", None)
    if group is None:
        return None
    return {str(name): tuple(np.shape(group[name].values)[2:]) for name in group.data_vars}


# Observed values are compared at the COARSER of the two precisions (F1): a
# candidate that passes its float64 response through ``jnp.asarray`` (x64 off)
# observes a float32 copy, which ``np.array_equal`` calls different at every
# non-representable position — 36 released candidates do exactly
# ``y = jnp.asarray(y)``. Four float32 ulp sit well above the round-trip error
# and far below any edit of the data that could move an elpd.
_FLOAT32_RTOL = 4.0 * float(np.finfo(np.float32).eps)


def _coarse_precision_pair(a: np.ndarray, b: np.ndarray) -> bool:
    """True when ``a`` and ``b`` are both numeric and must be compared with the
    float32 tolerance: either side is a float of <= 32 bits, or one side is
    float and the other integer (a cast happened somewhere). Integer, bool and
    object pairs — and float64 pairs — keep exact equality (F1)."""
    numeric = {"b", "i", "u", "f"}
    ka, kb = a.dtype.kind, b.dtype.kind
    if ka not in numeric or kb not in numeric:
        return False
    low_float = any(x.dtype.kind == "f" and x.dtype.itemsize <= 4 for x in (a, b))
    mixed = (ka == "f") != (kb == "f")
    return low_float or mixed


def _same_observed_values(a, b) -> tuple[bool, int]:
    """``(same, n_diff)`` for two observed arrays (F1).

    Shapes must match. Numeric pairs where either side is float32 (or
    narrower) or where only one side is float are compared with
    ``np.isclose(rtol=_FLOAT32_RTOL, atol=0, equal_nan=True)`` after promotion
    to float64; everything else (int / bool / object / float64 on both sides)
    must be exactly equal. ``n_diff`` counts the differing positions (-1 when
    the arrays cannot be compared elementwise: a shape mismatch, an object
    dtype that refuses ``==``). Used wherever an observed site value is matched
    against a data array: the fit identity check, ``_freed_obs_kwargs``,
    ``_resolve_obs_kwarg`` and crossval's spec inference.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False, -1
    try:
        if _coarse_precision_pair(a, b):
            close = np.isclose(a.astype(np.float64), b.astype(np.float64),
                               rtol=_FLOAT32_RTOL, atol=0.0, equal_nan=True)
            n_diff = int(np.sum(~close))
            return n_diff == 0, n_diff
        if a.dtype.kind == "f" and b.dtype.kind == "f":
            close = np.isclose(a, b, rtol=0.0, atol=0.0, equal_nan=True)  # exact, NaN == NaN
            n_diff = int(np.sum(~close))
            return n_diff == 0, n_diff
        same = bool(np.array_equal(a, b))
        if same or a.dtype == object or b.dtype == object:
            return same, 0 if same else -1
        with np.errstate(invalid="ignore"):
            return False, int(np.sum(~(a == b)))
    except Exception:
        return False, -1


def _pair_pointwise_variables(ref_shapes, orig_shapes):
    """Pair the two fits' log_likelihood variables: by name when the sets
    agree, by position when each fit has exactly one (a refinement may rename
    its single observed site, "y" -> "obs": the observations are the evidence,
    not the name). None when either fit has no log_likelihood group;
    ``ComparisonRefused`` when the variable sets cannot be paired."""
    if ref_shapes is None or orig_shapes is None:
        return None
    if set(ref_shapes) == set(orig_shapes):
        return [(name, name) for name in sorted(ref_shapes)]
    if len(ref_shapes) == 1 and len(orig_shapes) == 1:
        return [(next(iter(ref_shapes)), next(iter(orig_shapes)))]
    raise ComparisonRefused(
        "observed data differ between candidate and baseline (log-likelihood "
        f"variables {sorted(ref_shapes)} vs {sorted(orig_shapes)})")


def _degenerate_columns(idata) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per log_likelihood variable, ``(constant, nonfinite, value)`` over its
    points in C order (V1c): ``constant`` marks the columns with ``np.ptp == 0``
    over ALL posterior draws, ``nonfinite`` the columns carrying a non-finite
    value, ``value`` the constant a constant column takes (NaN elsewhere).

    A constant column means the observation does not depend on the parameters
    at all — the signature of a masked point (numpyro's
    ``MaskedDistribution.log_prob`` is exactly 0 there), a clamped one, or a
    saturated one (probabilities clipped to 1 give log-prob exactly 0; F4) —
    and PSIS has nothing to reweight: arviz reports k-hat = inf for such
    columns.
    """
    group = getattr(idata, "log_likelihood", None)
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if group is None:
        return out
    for name in group.data_vars:
        arr = np.asarray(group[name].values, dtype=float)
        if arr.ndim < 2:
            continue
        flat = arr.reshape(arr.shape[0] * arr.shape[1], -1)  # (draws, points)
        nonfinite = ~np.isfinite(flat).all(axis=0)
        with np.errstate(invalid="ignore"):
            constant = (np.ptp(flat, axis=0) == 0) & ~nonfinite
        value = np.where(constant, flat[0], np.nan)
        out[str(name)] = (constant, nonfinite, value)
    return out


def _degenerate_pointwise_columns(refined_idata, original_idata=None,
                                  pairs=None) -> tuple[int, int, dict[str, np.ndarray]]:
    """``(n_degenerate, n_points, shared)`` over the refined fit's
    log_likelihood variables (V1c; comparative since F2).

    A refined column counts as degenerate when it is constant across all
    posterior draws or non-finite — UNLESS the original fit's paired column
    (``pairs`` maps refined to original variable names; same positions, the
    shapes were checked by the caller) is constant with the SAME value: a
    missing cell that both programs mask contributes the same constant to
    both elpds and carries no information either way. ``shared`` holds those
    exempt positions per refined variable (flat C order — the order of arviz's
    pointwise outputs) so the PSIS k-hat, inf there by construction, can skip
    them. Refined-only degenerate columns stay degenerate, and so does a
    constant whose VALUE differs from the baseline's (a ``Delta`` with a
    hand-picked log density is not a shared missing cell) and any non-finite
    column (PSIS cannot score it on either side).
    """
    ref = _degenerate_columns(refined_idata)
    orig = _degenerate_columns(original_idata) if original_idata is not None else {}
    pairs = dict(pairs or {})
    bad = total = 0
    shared: dict[str, np.ndarray] = {}
    for name, (constant, nonfinite, value) in ref.items():
        total += int(constant.size)
        exempt = np.zeros_like(constant)
        o_name = pairs.get(name, name)
        if o_name in orig:
            o_constant, _o_nonfinite, o_value = orig[o_name]
            if o_constant.shape == constant.shape:
                with np.errstate(invalid="ignore"):
                    exempt = constant & o_constant & np.isclose(value, o_value,
                                                                rtol=1e-9, atol=1e-12)
        bad += int(((constant | nonfinite) & ~exempt).sum())
        if exempt.any():
            shared[name] = exempt
    return bad, total, shared


def shared_degenerate_columns(refined_idata, original_idata) -> np.ndarray | None:
    """Flat boolean mask (arviz pointwise order) of the positions where BOTH
    fits' single log_likelihood variable is constant with the same value —
    the shared missing cells of a masked design (F2) — or None when there is
    no such position, the fits cannot be paired, or they carry several
    log_likelihood variables. Used by ``compare_models`` (through
    ``_check_comparable_fits``) and by crossval's reloo rung so that a k-hat
    of inf at a shared missing cell is neither gated on nor refit."""
    try:
        pairs = _pair_pointwise_variables(_pointwise_shapes(refined_idata),
                                          _pointwise_shapes(original_idata))
    except ComparisonRefused:
        return None
    if pairs is None or len(pairs) != 1:
        return None
    _bad, _total, shared = _degenerate_pointwise_columns(refined_idata, original_idata,
                                                         dict(pairs))
    mask = shared.get(pairs[0][0])
    return mask if mask is not None and mask.any() else None


def _check_comparable_fits(refined_idata, original_idata) -> dict[str, np.ndarray]:
    """Refuse (``ComparisonRefused``) unless both fits scored the SAME
    observations and the refined fit's pointwise columns are informative where
    the baseline's are (V1b, V1c; F1, F2, F4, F5). Returns the shared-degenerate
    positions per refined log_likelihood variable (``_degenerate_pointwise_columns``)
    for the caller's k-hat computation.

    Each ``InferenceData`` carries its own ``log_likelihood`` / ``observed_data``
    groups, and ``az.compare`` happily differences two elpds computed over
    different observation sets — a candidate that masks part of the data (or
    subsets / clamps it inside the program) then "wins" by the excluded
    points' contribution. The identity check requires pairable log_likelihood
    variables (``_pair_pointwise_variables``), the same per-point shapes — a
    different NUMBER of dims is a different pointwise granularity (a
    ``to_event`` row-level likelihood vs a per-entry plate) and is named as
    such (F4) — and the same observed values, compared with
    ``_same_observed_values`` (F1: float32 tolerance when either side cast its
    response, exact equality otherwise). When the paired variables are not
    themselves recorded as observed data (Stan: ``log_lik`` next to ``y``), every
    observed array both fits carry is compared instead (F5). Fits without a
    log_likelihood group skip the check: ``az.loo`` fails on them anyway and
    the caller records the comparison as unavailable.
    """
    ref_shapes = _pointwise_shapes(refined_idata)
    orig_shapes = _pointwise_shapes(original_idata)
    pairs = _pair_pointwise_variables(ref_shapes, orig_shapes)
    if pairs is None:
        return {}

    def _label(r: str, o: str) -> str:
        return repr(r) if r == o else f"{r!r} (baseline {o!r})"

    for r_name, o_name in pairs:
        r_shape, o_shape = ref_shapes[r_name], orig_shapes[o_name]
        if r_shape == o_shape:
            continue
        if len(r_shape) != len(o_shape):
            raise ComparisonRefused(
                f"pointwise granularity differs (shapes {r_shape} vs {o_shape}); score "
                f"both models at the same level (variable {_label(r_name, o_name)}: a "
                "to_event row-level likelihood and a per-entry plate cannot be "
                "differenced pointwise)")
        raise ComparisonRefused(
            "observed data differ between candidate and baseline (variable "
            f"{_label(r_name, o_name)}: {r_shape} vs {o_shape} observations)")

    ref_obs, orig_obs = _observed_arrays(refined_idata), _observed_arrays(original_idata)

    def _require_same(name: str, a: np.ndarray, b: np.ndarray) -> None:
        if a.shape != b.shape:
            raise ComparisonRefused(
                "observed data differ between candidate and baseline (variable "
                f"{name}: observed shapes {a.shape} vs {b.shape})")
        same, n_diff = _same_observed_values(a, b)
        if not same:
            where = f"at {n_diff} of {a.size} positions" if n_diff >= 0 else "in value"
            raise ComparisonRefused(
                "observed data differ between candidate and baseline (variable "
                f"{name}: observed values differ {where})")

    compared_values = False
    for r_name, o_name in pairs:
        in_ref, in_orig = r_name in ref_obs, o_name in orig_obs
        if in_ref != in_orig:
            raise ComparisonRefused(
                "observed data differ between candidate and baseline (variable "
                f"{_label(r_name, o_name)} recorded as observed in only one fit)")
        if in_ref:
            _require_same(_label(r_name, o_name), ref_obs[r_name], orig_obs[o_name])
            compared_values = True
    if not compared_values and ref_obs and orig_obs:
        # Stan (F5): the pointwise variable is ``log_lik`` while the data block's
        # arrays are the observed_data; identity is then checked on every array
        # both fits recorded (both programs were handed the same dict).
        for key in sorted(set(ref_obs) & set(orig_obs)):
            _require_same(repr(key), ref_obs[key], orig_obs[key])

    bad, total, shared = _degenerate_pointwise_columns(refined_idata, original_idata,
                                                       dict(pairs))
    if bad:
        raise ComparisonRefused(
            f"{bad} of {total} pointwise log-likelihood values of the refined model "
            "are degenerate — constant across all posterior draws (masked, clamped or "
            "saturated) or non-finite — at positions where the baseline's are not: a "
            "candidate cannot be scored by PSIS-LOO on observations it does not model")
    return shared


def compare_models(refined_idata, original_idata) -> dict:
    """WAIC/LOO comparison of refined vs original.

    Before anything is computed the two fits must pass ``_check_comparable_fits``
    (defect V1): same observation set in both log_likelihood groups, identical
    observed values (at float32 precision when a side cast its response, F1),
    and no degenerate pointwise column on the refined side that the baseline
    does not share at the same position (F2); otherwise ``ComparisonRefused``
    is raised (never a dict a caller could mistake for evidence). Shared
    degenerate positions — the missing cells both programs mask — give arviz a
    Pareto k-hat of inf by construction (the importance weights there are all
    equal); they carry no information in either model and are skipped when the
    per-model k-hat maxima are taken.

    Deltas use scale="deviance" (lower = better) and are
    ``refined - original``: NEGATIVE delta = refined is better.
    ``elpd_diff`` / ``elpd_diff_se`` come from ``az.compare`` called with an
    EXPLICIT ``ic="loo", scale="log"`` (defect V3: the session-wide
    ``az.rcParams`` must not be able to change the margin); ``elpd_diff`` is
    signed so negative = refined better.
    ``significant_improvement`` (spec §13): delta < 0 AND
    |elpd_diff| >= 2 * elpd_diff_se.
    PSIS reliability (Pareto-k) is reported PER MODEL and the gate in
    ``run_full_validation`` blocks only on the REFINED model's k-hat: a
    k-hat > 0.7 means PSIS tends to OVER-estimate that model's elpd, so a
    high k-hat on the refined candidate inflates its apparent gain (a
    false-acceptance risk — must block), while a high k-hat on the original
    inflates the BASELINE and so biases the comparison *against* acceptance
    (conservative — recorded, never blocking; a badly misspecified original
    is precisely the textbook source of high k-hat, e.g. a Poisson fit to
    overdispersed counts). ``khat_max`` (max over both) is kept for report
    visibility; ``khat_refined`` / ``khat_original`` carry the split.
    """
    shared = _check_comparable_fits(refined_idata, original_idata)
    # Shared-degenerate positions (F2): skipped in the k-hat maxima below. az.loo
    # scores a single log_likelihood variable (it raises on several), so the one
    # entry of ``shared`` is the one that can match the pareto_k vector.
    exempt = next(iter(shared.values())) if len(shared) == 1 else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        waic_refined = az.waic(refined_idata, scale="deviance")
        waic_original = az.waic(original_idata, scale="deviance")
        loo_refined = az.loo(refined_idata, scale="deviance", pointwise=True)
        loo_original = az.loo(original_idata, scale="deviance", pointwise=True)
        # ic/scale pinned explicitly (V3): arviz reads both from rcParams when
        # omitted, so a session that set stats.ic_scale="deviance" would have
        # doubled elpd_diff AND dse here (the 2*SE test is scale-invariant but
        # the recorded margin, the TOST clause's ±4-nat CI and the confidence
        # z-score are not).
        cmp_df = az.compare(
            {"refined": refined_idata, "original": original_idata},
            ic="loo", scale="log",
        )

        def _khat(lo):
            pk = getattr(lo, "pareto_k", None)
            if pk is None:
                return float("nan")
            arr = np.asarray(pk, dtype=float).ravel()
            if exempt is not None and exempt.shape == arr.shape:
                arr = arr[~exempt]  # shared missing cells: inf by construction, no information
            return float(np.nanmax(arr)) if arr.size else float("nan")

        khat_refined = _khat(loo_refined)
        khat_original = _khat(loo_original)
        khat_max = (float(np.nanmax([khat_refined, khat_original]))
                    if not (np.isnan(khat_refined) and np.isnan(khat_original))
                    else float("nan"))

    waic = float(waic_refined["elpd_waic"])
    loo = float(loo_refined["elpd_loo"])
    waic_delta = float(waic_refined["elpd_waic"] - waic_original["elpd_waic"])
    loo_delta = float(loo_refined["elpd_loo"] - loo_original["elpd_loo"])

    # az.compare puts elpd_diff=0 on the best-ranked row; the magnitude and
    # its SE (dse) live on the other row.
    worse_rows = cmp_df[cmp_df["elpd_diff"] > 0]
    if len(worse_rows) == 0:
        magnitude, elpd_diff_se = 0.0, 0.0
    else:
        magnitude = float(worse_rows["elpd_diff"].iloc[0])
        elpd_diff_se = float(worse_rows["dse"].iloc[0])
    best = cmp_df.index[0]
    elpd_diff = -magnitude if best == "refined" else magnitude

    # Degeneracy floor: SE == 0 (pointwise differences numerically identical)
    # would make |elpd_diff| >= 2*SE vacuously true — such a comparison carries
    # no evidence and must not certify an improvement.
    significant_improvement = bool(
        loo_delta < 0
        and elpd_diff < 0
        and elpd_diff_se > 0
        and abs(elpd_diff) >= 2 * elpd_diff_se
    )
    return {
        "waic": waic,
        "loo": loo,
        "waic_delta": waic_delta,
        "loo_delta": loo_delta,
        "elpd_diff": elpd_diff,
        "elpd_diff_se": elpd_diff_se,
        "significant_improvement": significant_improvement,
        "khat_max": khat_max,
        "khat_refined": khat_refined,
        "khat_original": khat_original,
    }


# ---------------------------------------------------------------------------
# Posterior predictive check
# ---------------------------------------------------------------------------

def _trace_under(model, kwargs: dict, seed: int) -> dict:
    """Trace the model's sites under the given kwargs (seeded execution)."""
    seeded = handlers.seed(model, jax.random.PRNGKey(seed))
    with handlers.trace() as tr:
        seeded(**kwargs)
    return tr


# Posterior-predictive test statistics. Following Calibration-Not-Compilation
# [13] we track {mean, sd, max, #zeros}: mean catches location/link misfit, sd
# catches over/under-dispersion, max catches heavy-tail failures the moments
# miss, and #zeros catches zero-inflation. #zeros is degenerate for continuous
# data (observed and replicated both ~0) and passes there harmlessly.
_PPC_STATS = ("mean", "sd", "max", "zeros")
# max and #zeros block only on the HIGH tail — the DANGEROUS "the model cannot
# reproduce the data" direction (a maximum it can't reach = heavy-tail
# truncation; more zeros than it can produce = zero-inflation). A model that
# OVER-covers (observed statistic below the band — e.g. a Negative-Binomial
# whose heavy tail predicts a larger max than observed) is reported (side="low")
# but not failed: over-coverage is not a misfit the arbiter should reject.
# mean and sd stay two-sided (a location or dispersion miss matters either way).
_PPC_ONE_SIDED_HIGH = frozenset({"max", "zeros"})


def _ppc_statistic(arr2d, which: str):
    """Per-row (per-draw) value of a test statistic over ``(n_rows, n_points)``."""
    if which == "mean":
        return arr2d.mean(axis=1)
    if which == "sd":
        return arr2d.std(axis=1)
    if which == "max":
        return arr2d.max(axis=1)
    if which == "zeros":  # count of (near-)zero entries — a zero-inflation probe
        return np.isclose(arr2d, 0.0).sum(axis=1).astype(float)
    raise ValueError(f"unknown PPC statistic {which!r}")


def _ppc_stat_details(replicated, observed, label: str = "y"):
    """Central-98% posterior-predictive check over {mean, sd, max, #zeros}.

    Returns ``(passed, details)``. ``passed`` is True iff EVERY statistic's
    observed value lies inside the [1, 99] percentile band of its per-draw
    replicated distribution; ``details`` is one dict per statistic —
    ``{site, stat, observed, lo, hi, passed, side}`` — where ``side`` is
    ``"in"`` on a pass and ``"high"``/``"low"`` naming which tail an observed
    value fell into on a miss (used by ``localize_misspecification``).
    """
    rep = np.asarray(replicated, dtype=float)
    rep = rep.reshape(rep.shape[0], -1)
    obs = np.asarray(observed, dtype=float).ravel()
    details: list[dict] = []
    passed = True
    for stat in _PPC_STATS:
        rep_s = _ppc_statistic(rep, stat)
        obs_s = float(_ppc_statistic(obs[None, :], stat)[0])
        lo, hi = (float(x) for x in np.percentile(rep_s, [1.0, 99.0]))
        in_band = lo <= obs_s <= hi
        side = "in" if in_band else ("high" if obs_s > hi else "low")
        # one-sided-high statistics fail ONLY when the data exceed the model's
        # reach (side == "high"); harmless over-coverage (side == "low") passes.
        ok = (obs_s <= hi) if stat in _PPC_ONE_SIDED_HIGH else in_band
        details.append({"site": label, "stat": stat, "observed": obs_s,
                        "lo": lo, "hi": hi, "passed": bool(ok), "side": side})
        passed = passed and bool(ok)
    return passed, details


def _ppc_stat_check(replicated, observed) -> bool:
    """Boolean central-98% check (all of {mean, sd, max, #zeros} in band)."""
    return _ppc_stat_details(replicated, observed)[0]


def run_ppc(idata, model_code: str, data: dict, framework: str = "numpyro",
            seed: int = 1, return_details: bool = False):
    """Replicate the observed data from the posterior; sanity-check fit.

    For every observed quantity, the observed {mean, sd, max, #zeros} must fall
    within the central 98% interval of the replicated statistics. Dispatches by
    framework: numpyro un-conditions the observed sites and regenerates
    replicates with ``Predictive``; Stan reads the posterior-predictive
    generated quantities (e.g. ``y_rep``) straight from the trace.

    Returns the boolean/None verdict. With ``return_details=True`` returns
    ``(verdict, details)`` where ``details`` is the per-statistic breakdown from
    ``_ppc_stat_details`` (used for misspecification localization).
    """
    if framework == "numpyro":
        verdict, details = _run_ppc_numpyro(idata, model_code, data, seed=seed)
    elif framework == "stan":
        verdict, details = _run_ppc_stan(idata, data)
    else:
        raise NotImplementedError(f"run_ppc does not support framework {framework!r}")
    return (verdict, details) if return_details else verdict


def _run_ppc_numpyro(idata, model_code: str, data: dict, seed: int = 1):
    """NumPyro PPC: un-condition every observed site and regenerate replicates
    with ``Predictive``, then apply the central-98% mean+sd test.

    Runs under ``_restored_x64``: this path exec's the model source too, and
    in ``run_full_validation`` it runs AFTER both MCMC calls — an unguarded
    ``numpyro.enable_x64()`` here would leak into every later model."""
    with _restored_x64():
        return _ppc_numpyro_guarded(idata, model_code, data, seed)


def _traced_site_split(model, data: dict, seed: int):
    """One seeded trace -> (trace, observed site names, latent site names)."""
    seeded = handlers.seed(model, jax.random.PRNGKey(seed))
    with handlers.trace() as tr:
        seeded(**data)
    observed_sites = [name for name, site in tr.items()
                      if site["type"] == "sample" and site.get("is_observed")]
    latent_sites = {name for name, site in tr.items()
                    if site["type"] == "sample" and not site.get("is_observed")}
    return tr, observed_sites, latent_sites


# ---------------------------------------------------------------------------
# S-gate structural guard (defect V1a): the refined program must score EVERY
# observation it is handed, unweighted. Decided by TRACING the program on the
# data, never by reading its text.
# ---------------------------------------------------------------------------

def _distribution_chain(fn, limit: int = 32):
    """Yield ``fn`` and every distribution it wraps through ``base_dist``
    (Masked -> Expanded -> Independent -> ... chains), bounded."""
    seen = 0
    while fn is not None and seen < limit:
        yield fn
        seen += 1
        fn = getattr(fn, "base_dist", None)


def _effective_mask(site) -> np.ndarray | None:
    """AND of every ``MaskedDistribution`` mask along the site's distribution
    chain, broadcast to the site's value shape when possible; None when the
    site is unmasked or every mask is identically True (numpyro does not even
    wrap the distribution for a scalar ``True``)."""
    value = np.asarray(site["value"])
    mask = None
    for d in _distribution_chain(site["fn"]):
        if not isinstance(d, dist.MaskedDistribution):
            continue
        m = np.asarray(getattr(d, "_mask", True))
        if m.dtype != bool:
            m = m.astype(bool)
        if mask is None:
            mask = m
        else:
            try:
                mask = mask & m
            except ValueError:
                mask = np.broadcast_to(mask, value.shape) & np.broadcast_to(m, value.shape)
    if mask is None or bool(np.all(mask)):
        return None
    try:
        return np.array(np.broadcast_to(mask, value.shape))
    except ValueError:
        return np.array(mask)


def _site_scale(site) -> np.ndarray | None:
    """The site's ``handlers.scale`` factor as a float array; None when the
    site is unscaled or scaled by exactly 1 everywhere."""
    scale = site.get("scale")
    if scale is None:
        return None
    s = np.asarray(scale, dtype=float)
    return None if bool(np.all(s == 1.0)) else s


def _shown_scale(s: np.ndarray) -> str:
    return f"{float(s):g}" if s.ndim == 0 else f"array of shape {s.shape}"


def _same_scale(a: np.ndarray, b: np.ndarray) -> bool:
    try:
        x, y = np.broadcast_arrays(a, b)
    except ValueError:
        return False
    return bool(np.allclose(x, y, rtol=1e-6, atol=0.0))


def _distribution_params(fn) -> dict[str, np.ndarray]:
    """``{parameter: array}`` of the numeric parameters along ``fn``'s
    ``base_dist`` chain. Masked / Expanded / Independent wrappers declare no
    parameters of their own (``arg_constraints == {}``); the wrapped
    distribution does. The outermost occurrence of a name wins."""
    out: dict[str, np.ndarray] = {}
    for d in _distribution_chain(fn):
        for pname in getattr(d, "arg_constraints", None) or {}:
            if pname in out:
                continue
            value = getattr(d, pname, None)
            if value is None:
                continue
            try:
                out[pname] = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                continue
    return out


def _compare_site_laws(fn_data, fn_probe, probe: np.ndarray) -> dict:
    """How the law of one observed site differs between the data-conditioned
    trace (``fn_data``) and the probe trace (``fn_probe``: un-conditioned or
    response-permuted, same latent values) — F3.

    Returns ``{dependent, params, n_logprob, n_points}``: ``params`` names the
    distribution parameters that differ beyond float tolerance (shapes are
    broadcast against each other: a scalar rate expanded by the plate and a
    per-observation rate vector compare elementwise), ``n_logprob`` the
    positions of ``probe`` — the observed data, which lie in the support —
    whose log-probability differs (-1 when the two log-probabilities could not
    be evaluated on a common shape). A masked position cannot show in the
    log-probability (0 on both sides) but does show in the parameters.
    """
    p_data, p_probe = _distribution_params(fn_data), _distribution_params(fn_probe)
    changed: list[str] = []
    for pname in sorted(set(p_data) | set(p_probe)):
        if pname not in p_data or pname not in p_probe:
            changed.append(pname)
            continue
        try:
            a, b = np.broadcast_arrays(p_data[pname], p_probe[pname])
        except ValueError:
            changed.append(pname)
            continue
        if not np.allclose(a, b, rtol=1e-5, atol=1e-6, equal_nan=True):
            changed.append(pname)
    n_logprob, n_points = -1, int(probe.size)
    try:
        lp_data = np.asarray(fn_data.log_prob(probe), dtype=float)
        lp_probe = np.asarray(fn_probe.log_prob(probe), dtype=float)
        if lp_data.shape == lp_probe.shape:
            n_logprob = int(np.sum(~np.isclose(lp_data, lp_probe, rtol=1e-5, atol=1e-6,
                                               equal_nan=True)))
            n_points = int(lp_data.size)
    except Exception:
        n_logprob = -1
    return {"dependent": bool(changed) or n_logprob > 0, "params": changed,
            "n_logprob": n_logprob, "n_points": n_points}


def _permuted_response_kwargs(data: dict, tr, observed: list, seed: int) -> dict | None:
    """``data`` with every response array feeding an observed site replaced by
    a seeded permutation of itself along axis 0 — the fallback probe of F3 for
    a program whose un-conditioned trace cannot run. None when no response
    kwarg resolves (``_resolve_obs_kwarg``) or no response changes under the
    permutation (a constant response cannot be probed)."""
    out = dict(data)
    rng = np.random.default_rng(seed)
    found = False
    for name in observed:
        kwarg = _resolve_obs_kwarg(name, np.asarray(tr[name]["value"]), data)
        if kwarg is None or not isinstance(data.get(kwarg), np.ndarray | list | tuple):
            continue
        arr = np.asarray(data[kwarg])
        if arr.ndim == 0 or arr.shape[0] < 2:
            continue
        permuted = arr[rng.permutation(arr.shape[0])]
        if np.array_equal(permuted, arr):
            continue
        out[kwarg] = permuted
        found = True
    return out if found else None


def _response_dependence(model, data: dict, tr, observed: list, seed: int, *,
                         fallback_permutation: bool = False) -> dict | None:
    """Per observed site, whether its law changes when the response is withheld
    (the two-faced-likelihood check, F3).

    The program is re-traced with the SAME latent values
    (``handlers.substitute``) and the same seed, but with every response kwarg
    that feeds an observed site set to None — the un-conditioned call the PPC
    already requires (``_freed_obs_kwargs``). A distribution parameter or a
    log-probability at the data that differs between the two traces means the
    likelihood parameters depend on the observed values themselves: the
    ``rate = lam if y is None else jnp.where(y > 4, lam, 0.5 * y + 0.2 * lam)``
    probe is an honest Poisson when sampled (so it passes the PPC) and a soft
    clamp when scored.

    Returns None when the un-conditioned trace cannot run (sizes derived from
    ``y.shape``, masks computed from ``y``) — unless ``fallback_permutation``
    is set, in which case the response is PERMUTED along axis 0 instead of
    withheld. The fallback is used for the baseline, whose author never
    promised an un-conditioned call: an honest iid likelihood keeps its
    parameters under a permutation of its own data, a conditional one does
    not. Result: ``{site: {dependent, params, n_logprob, n_points, probe}}``
    with ``probe`` in ``{"unconditioned", "permuted"}``.
    """
    latents = {name: site["value"] for name, site in tr.items()
               if site["type"] == "sample" and not site.get("is_observed")}
    probe_kind = "unconditioned"
    try:
        call = _freed_obs_kwargs(model, data, tr, observed, seed)
    except Exception:
        if not fallback_permutation:
            return None
        call = _permuted_response_kwargs(data, tr, observed, seed)
        if call is None:
            return None
        probe_kind = "permuted"
    try:
        tr_probe = _trace_under(handlers.substitute(model, data=latents), call, seed)
    except Exception:
        return None
    out: dict[str, dict] = {}
    for name in observed:
        if name not in tr_probe or tr_probe[name].get("type") != "sample":
            continue
        info = _compare_site_laws(tr[name]["fn"], tr_probe[name]["fn"],
                                  np.asarray(tr[name]["value"]))
        info["probe"] = probe_kind
        out[name] = info
    return out


def _describe_dependence(info: dict) -> str:
    parts: list[str] = []
    if info["params"]:
        plural = "s" if len(info["params"]) > 1 else ""
        parts.append(f"parameter{plural} " + ", ".join(repr(p) for p in info["params"]))
    if info["n_logprob"] > 0:
        parts.append(f"the log-probability at the data at {info['n_logprob']} of "
                     f"{info['n_points']} positions")
    what = " and ".join(parts) if parts else "the distribution"
    against = ("the un-conditioned trace" if info.get("probe") == "unconditioned"
               else "a trace on a permutation of the response")
    return (f"{what} differ between the data-conditioned trace and {against} "
            "with the same latent values")


def _pair_observed_sites(refined: list[str], original: list[str]) -> dict[str, str | None]:
    """Refined observed site -> the original's it is compared with: by name,
    or by position when each program has exactly one observed site (a
    refinement may rename it); None when unpaired."""
    common = [n for n in refined if n in original]
    if not common and len(refined) == 1 and len(original) == 1:
        return {refined[0]: original[0]}
    return {n: (n if n in original else None) for n in refined}


def compare_observed_structure(refined_code: str, original_code: str | None, data: dict,
                               *, seed: int = 0) -> dict:
    """Trace the refined program — and the original, when given — on the data
    and judge the refined observed sites COMPARATIVELY (F2, F3).

    Returns ``{"reasons": [...], "notes": [...], "sites": {...}}``: ``reasons``
    are the structural refusals (the caller prefixes ``model comparison
    refused: ``), ``notes`` descriptive findings that do not refuse, ``sites``
    the per-site record (paired baseline site, excluded count, scale, response
    dependence) for reports. Rules, per refined observed site paired with the
    original's (``_pair_observed_sites``; an unpaired site, or every site when
    no original is given or the original cannot be traced, falls back to the
    single-program rules):

    * MASK (F2): refused only when the refined mask excludes observations the
      original scores. The standard missing-data design where BOTH programs
      apply the data's own mask (``handlers.mask(mask=data["mask"])``) passes;
      a candidate adding a mask of its own, or widening the shared one, is
      refused. Single-program rule: any mask that excludes a point.
    * SCALE (F2): refused when the refined site is scaled and the original is
      not, or when the two scales differ; equal scales pass. Single-program
      rule: any scale other than 1.
    * RESPONSE DEPENDENCE (F3, ``_response_dependence``): refused when the
      refined likelihood parameters depend on the observed response and the
      original's do not. When both do (an autoregressive / time-series
      conditional likelihood on both sides) a note is recorded instead; when
      the refined un-conditioned trace cannot run the check is skipped (the
      PPC reports that separately); when the original cannot be probed even by
      permutation a note is recorded and nothing is refused on this ground.

    Raises whatever the REFINED program raises when it cannot be traced — the
    caller decides (``run_full_validation`` refuses: a program the verifier
    cannot trace cannot be certified).
    """
    data = data or {}
    with _restored_x64():
        model = _exec_model(refined_code)
        tr, observed, _latent = _traced_site_split(model, data, seed)
        ref_dep = _response_dependence(model, data, tr, observed, seed)
        orig_tr, orig_observed, orig_dep, orig_note = None, [], None, None
        if original_code:
            try:
                orig_model = _exec_model(original_code)
                orig_tr, orig_observed, _ = _traced_site_split(orig_model, data, seed)
                orig_dep = _response_dependence(orig_model, data, orig_tr, orig_observed,
                                                seed, fallback_permutation=True)
            except Exception as exc:
                orig_tr, orig_observed, orig_dep = None, [], None
                orig_note = (f"the baseline program could not be traced on the data ({exc}); "
                             "the structural guards ran on the candidate alone")
    pairs = _pair_observed_sites(observed, orig_observed)
    reasons: list[str] = []
    notes: list[str] = [] if orig_note is None else [orig_note]
    sites: dict[str, dict] = {}
    for name in observed:
        site = tr[name]
        value = np.asarray(site["value"])
        mask, scale = _effective_mask(site), _site_scale(site)
        o_name = pairs.get(name)
        o_site = orig_tr[o_name] if (orig_tr is not None and o_name is not None) else None
        o_mask = _effective_mask(o_site) if o_site is not None else None
        o_scale = _site_scale(o_site) if o_site is not None else None
        record: dict[str, Any] = {"paired_with": o_name, "n": int(value.size),
                                  "excluded": 0, "scale": None, "response_dependent": None}

        if mask is not None:
            excluded = int(np.sum(~mask))
            record["excluded"] = excluded
            if o_site is not None and o_mask is not None:
                extra = (int(np.sum(~mask & o_mask)) if o_mask.shape == mask.shape
                         else excluded)
                if extra:
                    reasons.append(
                        f"observed site {name!r} is masked (numpyro mask on the likelihood): "
                        f"{extra} of {value.size} observations are excluded from its "
                        "log-likelihood but scored by the baseline, so the candidate would "
                        "be scored on fewer data than the baseline")
            else:
                reasons.append(
                    f"observed site {name!r} is masked (numpyro mask on the likelihood): "
                    f"{excluded} of {value.size} observations are excluded from its "
                    "log-likelihood, so the candidate would be scored on fewer data "
                    "than the baseline")

        if scale is not None:
            record["scale"] = _shown_scale(scale)
            if o_site is not None and o_scale is not None:
                if not _same_scale(scale, o_scale):
                    reasons.append(
                        f"observed site {name!r} carries a log-likelihood scale "
                        f"({_shown_scale(scale)}; numpyro.handlers.scale) that differs from "
                        f"the baseline's ({_shown_scale(o_scale)}): differently reweighted "
                        "observation sets are not comparable")
            else:
                reasons.append(
                    f"observed site {name!r} carries a log-likelihood scale "
                    f"({_shown_scale(scale)}; numpyro.handlers.scale): a reweighted "
                    "observation set is not comparable with the baseline's")

        info = (ref_dep or {}).get(name)
        if info is not None:
            record["response_dependent"] = bool(info["dependent"])
            if info["dependent"]:
                detail = _describe_dependence(info)
                o_info = (orig_dep or {}).get(o_name) if o_name is not None else None
                if o_info is not None and o_info["dependent"]:
                    notes.append(
                        f"observed site {name!r}: the likelihood parameters depend on the "
                        f"observed response in both the candidate and the baseline ({detail}); "
                        "a conditional (autoregressive / time-series) likelihood is compared "
                        "as written — check that the conditioning is the intended one")
                elif o_info is not None:
                    reasons.append(
                        f"observed site {name!r}: the likelihood parameters depend on the "
                        f"observed response (two-faced likelihood — {detail}) while the "
                        "baseline's do not; the candidate scores its own data with a "
                        "response-dependent law, so its pointwise log-likelihood is not "
                        "comparable with the baseline's")
                else:
                    why = ("the baseline could not be probed" if original_code
                           else "no baseline program was given")
                    notes.append(
                        f"observed site {name!r}: the candidate's likelihood parameters depend "
                        f"on the observed response ({detail}); {why}, so this was not refused "
                        "on structural grounds")
        sites[name] = record
    return {"reasons": reasons, "notes": notes, "sites": sites}


def check_observed_sites(model_code: str, data: dict, *, seed: int = 0,
                         original_code: str | None = None,
                         notes: list[str] | None = None) -> list[str]:
    """Reasons the refined program's observed sites cannot be compared on the
    full data; ``[]`` when the likelihood is clean (defect V1a; comparative
    since F2/F3 — this is ``compare_observed_structure``'s ``reasons`` list).

    Single-program form (``original_code=None``): traces ``model_code`` on
    ``data`` (seeded; same execution as the PPC) and refuses an observed
    sample site when

    * its distribution — or any distribution it wraps through ``base_dist`` —
      is a ``numpyro.distributions.MaskedDistribution`` whose mask is not
      identically True. This covers ``numpyro.handlers.mask(mask=...)``,
      ``dist.X(...).mask(m)`` and the ``obs_mask=`` keyword of
      ``numpyro.sample`` (numpyro 0.19 implements it with the same masked
      distribution on the ``<name>_observed`` site). Masked points contribute
      ``log_prob == 0`` exactly, so a candidate that masks the awkward
      observations "improves" elpd by their missing deficit. A mask that is
      identically True is harmless (numpyro does not even wrap the
      distribution for a scalar ``True``) and passes.
    * it carries a ``site["scale"]`` other than None / 1 (``handlers.scale``):
      a scaled log-likelihood is a reweighted observation set.

    Comparative form (``original_code`` given): a mask or scale the baseline
    shares passes, a wider mask or a different scale is refused, and a
    response-dependent likelihood (F3) is refused only when the baseline's is
    not. Descriptive notes (both programs response-dependent; a baseline that
    could not be traced or probed) are appended to ``notes`` when a list is
    passed.

    Latent sites are not examined (masking a prior changes the model, not the
    observation set). Raises whatever the refined model raises when it cannot
    be traced — the caller decides (``run_full_validation`` refuses: a program
    the verifier cannot trace cannot be certified).
    """
    result = compare_observed_structure(model_code, original_code, data or {}, seed=seed)
    if notes is not None:
        notes.extend(result["notes"])
    return list(result["reasons"])


def _ppc_numpyro_guarded(idata, model_code: str, data: dict, seed: int):
    model = _exec_model(model_code)
    data = data or {}

    # Trace once to learn which sites are observed vs latent.
    tr, observed_sites, latent_sites = _traced_site_split(model, data, seed)
    if not observed_sites:
        raise ValueError("model has no observed sites; PPC is undefined")

    # Posterior samples flattened over (chain, draw); latent sites only —
    # deterministic sites are recomputed by Predictive.
    posterior = {}
    for name in idata.posterior.data_vars:
        if name not in latent_sites:
            continue
        values = np.asarray(idata.posterior[name].values)
        posterior[name] = values.reshape((-1,) + values.shape[2:])

    predict_kwargs = _freed_obs_kwargs(model, data, tr, observed_sites, seed)

    predictive = Predictive(model, posterior_samples=posterior)
    replicated = predictive(jax.random.PRNGKey(seed), **predict_kwargs)

    all_passed = True
    details: list[dict] = []
    for name in observed_sites:
        observed = _observed_for_site(name, data, idata, tr)
        site_passed, site_details = _ppc_stat_details(replicated[name], observed, label=name)
        details.extend(site_details)
        all_passed = all_passed and site_passed
    return all_passed, details


def _observed_for_site(name: str, data: dict, idata, tr) -> np.ndarray:
    """The observed array of site ``name`` as floats: the data kwarg of the same
    name, else the fit's ``observed_data`` entry, else the value the
    data-conditioned trace recorded for the site (a renamed site observing a
    cast of its kwarg, F1 — exactly what the model scored)."""
    if name in data and data[name] is not None:
        return np.asarray(data[name], dtype=float)
    group = getattr(idata, "observed_data", None)
    if group is not None and name in group:
        return np.asarray(group[name].values, dtype=float)
    return np.asarray(tr[name]["value"], dtype=float)


def _freed_obs_kwargs(model, data: dict, tr, observed_sites: list, seed: int) -> dict:
    """Kwargs with every observed site un-conditioned (verified), for Predictive."""
    predict_kwargs = dict(data)
    for name in observed_sites:
        if name in predict_kwargs:
            predict_kwargs[name] = None  # let the model generate replicates
        else:
            # Site name differs from the data kwarg (e.g. site "obs" fed by
            # kwarg y=...). Trial-null value-matching kwargs ONE AT A TIME and
            # keep the first that frees the site while the model still traces
            # cleanly — nulling all matches at once could also null a
            # covariate that happens to equal the response (and crash the
            # model with a misattributed error).
            # Matched at the coarser precision (F1): the site value is a float32
            # copy of the kwarg when the program casts its response.
            site_value = np.asarray(tr[name]["value"])
            candidates = [
                kwarg for kwarg, value in data.items()
                if predict_kwargs.get(kwarg) is not None
                and isinstance(value, np.ndarray | list | tuple)
                and _same_observed_values(np.asarray(value), site_value)[0]
            ]
            for kwarg in candidates:
                trial = dict(predict_kwargs)
                trial[kwarg] = None
                try:
                    trial_tr = _trace_under(model, trial, seed)
                except Exception:
                    continue  # this kwarg is structural (covariate) — wrong match
                if not trial_tr[name].get("is_observed"):
                    predict_kwargs = trial
                    break

    # Verify the un-conditioning actually worked: re-trace under
    # predict_kwargs and require every observed site to now be latent.
    try:
        check_tr = _trace_under(model, predict_kwargs, seed)
    except Exception as exc:
        raise ValueError(
            f"un-conditioning broke the model (ambiguous kwarg match?): {exc}"
        ) from exc
    still_observed = [name for name in observed_sites
                      if check_tr[name].get("is_observed")]
    if still_observed:
        raise ValueError(
            f"cannot un-condition observed site(s) {still_observed}: no data "
            "kwarg matches; PPC would silently compare the data with itself"
        )
    return predict_kwargs


# ---------------------------------------------------------------------------
# Mixed-replication PPC + unit-aware statistics (v4 Component 2) — DESCRIPTIVE.
# The conditional PPC above remains the only PPC gate; everything here is
# recorded (ValidationResult.ppc_mixed) and feeds localize_misspecification,
# never failure_reasons. Gate promotion is an L-grid (Phase 2) decision.
# ---------------------------------------------------------------------------

def _kurtosis_rows(means: np.ndarray) -> np.ndarray:
    """Per-draw kurtosis of the unit means (population definition; Normal = 3).

    The SHAPE statistic of the unit-effect law: an inflated tau matches the
    spread and the max of contaminated or bimodal unit effects, but not their
    kurtosis — heavy-tailed laws push it above the replicated band, bimodal
    laws below it (platykurtic). Two-sided by construction."""
    c = means - means.mean(axis=1, keepdims=True)
    m2 = (c ** 2).mean(axis=1)
    m4 = (c ** 4).mean(axis=1)
    return m4 / np.maximum(m2 ** 2, 1e-300)


def _unit_stats(values, group_idx=None):
    """Unit-level statistics of one (n_draws, *obs_shape) array.

    Units are axis-0 rows of a >=2-D observation, or the groups of
    ``group_idx`` for a 1-D observation. Returns {name: (n_draws,) array} with
    within_unit_sd (mean over units of the within-unit sd; only units with
    >=2 obs), between_unit_spread (sd of unit means), max_unit_mean.
    Empty dict when no usable unit structure exists.
    """
    v = np.asarray(values, dtype=float)
    if group_idx is not None:
        if v.ndim != 2:
            return {}
        idx = np.asarray(group_idx).astype(int).ravel()
        if idx.shape[0] != v.shape[1]:
            return {}
        J = int(idx.max()) + 1
        onehot = np.zeros((idx.shape[0], J))
        onehot[np.arange(idx.shape[0]), idx] = 1.0
        counts = onehot.sum(axis=0)                          # (J,)
        means = (v @ onehot) / np.maximum(counts, 1.0)       # (S, J)
        ex2 = (v ** 2 @ onehot) / np.maximum(counts, 1.0)
        wsd = np.sqrt(np.maximum(ex2 - means ** 2, 0.0))     # ddof=0, consistent
        eligible = counts >= 2
        out = {}
        if eligible.any():
            out["within_unit_sd"] = wsd[:, eligible].mean(axis=1)
        out["between_unit_spread"] = means.std(axis=1)
        out["max_unit_mean"] = means.max(axis=1)
        if means.shape[1] >= 8:
            out["unit_mean_kurtosis"] = _kurtosis_rows(means)
        return out
    if v.ndim < 3:
        return {}                                            # (S, J, items...) required
    flat = v.reshape(v.shape[0], v.shape[1], -1)             # (S, J, m)
    means = flat.mean(axis=2)
    out = {"between_unit_spread": means.std(axis=1),
           "max_unit_mean": means.max(axis=1)}
    if flat.shape[2] >= 2:
        out["within_unit_sd"] = flat.std(axis=2).mean(axis=1)
    if means.shape[1] >= 8:
        out["unit_mean_kurtosis"] = _kurtosis_rows(means)
    return out


_UNIT_ONE_SIDED_HIGH = ("max_unit_mean",)


def _unit_stat_details(replicated, observed, *, label: str, group_idx=None):
    """Central-98% bands over the unit-level statistics; detail dicts share the
    ``_ppc_stat_details`` shape so localization and reports treat them alike."""
    rep = _unit_stats(np.asarray(replicated, dtype=float), group_idx=group_idx)
    if not rep:
        return True, []
    obs_arr = np.asarray(observed, dtype=float)[None, ...]
    obs = _unit_stats(obs_arr, group_idx=group_idx)
    details: list[dict] = []
    passed = True
    for stat, rep_s in rep.items():
        obs_s = float(obs[stat][0])
        lo, hi = (float(x) for x in np.percentile(np.asarray(rep_s), [1.0, 99.0]))
        in_band = lo <= obs_s <= hi
        side = "in" if in_band else ("high" if obs_s > hi else "low")
        ok = (obs_s <= hi) if stat in _UNIT_ONE_SIDED_HIGH else in_band
        details.append({"site": label, "stat": stat, "observed": obs_s,
                        "lo": lo, "hi": hi, "passed": bool(ok), "side": side})
        passed = passed and bool(ok)
    return passed, details


def run_mixed_ppc(idata, model_code: str, data: dict, *, local_sites,
                  group_idx_key: str | None = None, seed: int = 2):
    """Mixed-replication PPC (Marshall & Spiegelhalter 2007): redraw the LOCAL
    latents from p(t | theta_s) row-wise — Predictive conditioned on the
    GLOBAL posterior draws only — then replicate the observations, and score
    the usual statistics PLUS the unit-level ones. This is the check with
    power against latent-law misspecification and the sponge (a conditional
    PPC's replicates chase the fitted latents and are blind there).

    DESCRIPTIVE ONLY: the caller records the result; it never gates. Returns
    ``(passed | None, details)`` with detail labels prefixed ``mixed:``.
    """
    with _restored_x64():
        model = _exec_model(model_code)
        data = data or {}
        tr, observed_sites, latent_sites = _traced_site_split(model, data, seed)
        if not observed_sites:
            return None, []
        local = set(local_sites or [])
        posterior = {}
        for name in idata.posterior.data_vars:
            if name in latent_sites and name not in local:
                values = np.asarray(idata.posterior[name].values)
                posterior[name] = values.reshape((-1,) + values.shape[2:])
        if not posterior:
            return None, []                       # nothing global to condition on

        predict_kwargs = _freed_obs_kwargs(model, data, tr, observed_sites, seed)
        predictive = Predictive(model, posterior_samples=posterior)
        replicated = predictive(jax.random.PRNGKey(seed), **predict_kwargs)

        all_passed = True
        details: list[dict] = []
        for name in observed_sites:
            observed = _observed_for_site(name, data, idata, tr)
            gi = None
            if group_idx_key is not None and group_idx_key in data:
                gi = np.asarray(data[group_idx_key])
            p1, d1 = _ppc_stat_details(replicated[name], observed,
                                       label=f"mixed:{name}")
            p2, d2 = _unit_stat_details(replicated[name], observed,
                                        label=f"mixed:{name}", group_idx=gi)
            details.extend(d1 + d2)
            all_passed = all_passed and p1 and p2
    return all_passed, details


# Generated-quantity name suffixes that conventionally hold posterior-predictive
# replicates in a Stan program (e.g. ``y_rep``, ``y_hat``, ``y_tilde``).
_PPC_VAR_SUFFIXES = ("_rep", "_hat", "_pred", "_tilde", "_ppc", "_sim")


def _strip_pp_suffix(name: str) -> str:
    """``y_rep`` -> ``y``; map the no-underscore conventions too."""
    low = name.lower()
    for s in _PPC_VAR_SUFFIXES:
        if low.endswith(s):
            return name[: -len(s)]
    if low in ("yrep", "ypred", "ytilde"):
        return "y"
    return name


def _match_observed(name, event_shape, observed_arrays, obs_data_group):
    """Find the observed array a replicate variable should be compared against,
    by base-name first (``y_rep`` -> ``y``), then by matching element count."""
    target = int(np.prod(event_shape)) if event_shape else 1
    base = _strip_pp_suffix(name)
    candidates = [base, base.rstrip("_"), "y"]

    def _fits(arr):
        arr = np.asarray(arr, dtype=float)
        return arr if (arr.size == target or arr.shape == tuple(event_shape)) else None

    for cand in candidates:                       # 1) name match in the data dict
        if cand in observed_arrays and (hit := _fits(observed_arrays[cand])) is not None:
            return hit
    if obs_data_group is not None:                # 2) arviz observed_data group
        dv = getattr(obs_data_group, "data_vars", {})
        for cand in candidates:
            if cand in dv and (hit := _fits(obs_data_group[cand].values)) is not None:
                return hit
    # 3) a same-size observed array, but only when UNAMBIGUOUS: with several
    # size-matching arrays (e.g. a covariate alongside the response) a
    # first-match guess could score replicates against the wrong array and
    # emit a wrong blocking verdict — return None (unavailable, descriptive).
    hits = [hit for arr in observed_arrays.values()
            if (hit := _fits(arr)) is not None]
    return hits[0] if len(hits) == 1 else None


def _run_ppc_stan(idata, data: dict):
    """Stan PPC: compare the model's posterior-predictive generated quantities
    (e.g. ``y_rep``) against the observed data with the same central-98% mean+sd
    test as the numpyro path.

    ``az.from_cmdstanpy(posterior=fit)`` lands generated quantities like
    ``y_rep`` in ``idata.posterior``; we locate the replicate variable(s) by the
    conventional name suffixes, match each to its observed array, and check it.

    Returns True/False when a replicate variable is found and checked, or
    ``None`` when PPC could not be computed (no conventionally-named replicate,
    or none matched an observed array). The caller treats None as UNAVAILABLE
    (descriptive, non-blocking) — only an actual False records a PPC failure, so
    a converged + improved Stan model is never blocked merely because its
    replicate variable is unnamed/unconventionally named.
    """
    data = data or {}
    seen: set[str] = set()  # by VALUE: the same name may appear in both groups
    rep_vars = []
    for grp in ("posterior_predictive", "posterior"):
        ds = getattr(idata, grp, None)
        if ds is None:
            continue
        for vname in ds.data_vars:
            low = str(vname).lower()
            if (any(low.endswith(s) for s in _PPC_VAR_SUFFIXES)
                    or low in ("yrep", "ypred", "ytilde")) and low not in seen:
                seen.add(low)
                rep_vars.append((vname, ds))
    if not rep_vars:
        return None, []  # no posterior-predictive generated quantity -> unavailable

    observed_arrays = {k: np.asarray(v, dtype=float) for k, v in data.items()
                       if isinstance(v, (list, tuple, np.ndarray))}
    obs_data_group = getattr(idata, "observed_data", None)

    checked_any = False
    all_passed = True
    details: list[dict] = []
    for vname, ds in rep_vars:
        rep = np.asarray(ds[vname].values, dtype=float)
        # (chain, draw, *event) -> (n_draws, *event)
        rep = rep.reshape((-1,) + rep.shape[2:]) if rep.ndim >= 2 else rep[:, None]
        observed = _match_observed(vname, rep.shape[1:], observed_arrays, obs_data_group)
        if observed is None:
            continue
        checked_any = True
        var_passed, var_details = _ppc_stat_details(rep, observed, label=str(vname))
        details.extend(var_details)
        all_passed = all_passed and var_passed
    if not checked_any:
        return None, []  # replicate var(s) present but no observed array matched
    return all_passed, details


# ---------------------------------------------------------------------------
# Misspecification localization (Calibration-Not-Compilation [13] borrow) — map
# a failed diagnostic to a suspected bug type + fix class. Descriptive ONLY: a
# repair hint for the strategist's next round; never touches acceptance/routing.
# ---------------------------------------------------------------------------

def localize_misspecification(diagnostics: dict, ppc_details: list,
                              mixed_details: list | None = None,
                              comparison: dict | None = None,
                              audit: dict | None = None) -> list[str]:
    """Turn failed PPC statistics + sampler pathologies into repair hints.

    Mirrors the diagnostic→bug-type localization of [13]: an over-dispersed
    predictive spread points at a Negative-Binomial/Student-t likelihood, excess
    zeros at zero-inflation, a too-small predictive maximum at heavy tails, a
    mean miss at a link/covariate error, and sampler divergences/high R-hat at a
    bad (centered) parameterization. Every failed detail is visited (multiple
    observed sites can fail the SAME statistic in different directions), each
    hint names its site, and duplicate hint strings are emitted once. Returns
    human-readable strings; the caller surfaces them to the strategist but the
    MCMC verdict is unaffected.

    The latent-aware extension (v4, EVIDENCE hint channel — all optional, all
    evidence-triggered): ``mixed_details`` from ``run_mixed_ppc`` localizes
    misfit BETWEEN hierarchy levels (the sponge, the latent law);
    ``comparison`` contributes S-gate escalation provenance and per-unit
    delta concentration (crossval's pointwise deltas); ``audit``
    (resolution.audit_resolution) contributes the Mundlak within/between
    conflict. Fix-CLASS suggestions appear only when a measured diagnostic
    fired — the published hint discipline.
    """
    hints: list[str] = []

    def _add(hint: str) -> None:
        if hint not in hints:
            hints.append(hint)

    for d in list(ppc_details or []) + list(mixed_details or []):
        if d.get("passed"):
            continue
        site, stat, side = d.get("site", "y"), d.get("stat"), d.get("side")
        if stat == "sd" and side == "high":
            _add(
                f"site '{site}': the data are more dispersed than the model "
                "predicts (posterior-predictive sd too low) — suspect OVER-"
                "DISPERSION; consider a Negative-Binomial likelihood (counts) "
                "or a Student-t (continuous)"
            )
        elif stat == "zeros" and side == "high":
            _add(
                f"site '{site}': the data have more zeros than the model "
                "predicts — suspect ZERO-INFLATION; consider a zero-inflated "
                "or hurdle likelihood"
            )
        elif stat == "max" and side == "high":
            _add(
                f"site '{site}': the model cannot reach the largest observed "
                "values — suspect HEAVY TAILS; consider a heavier-tailed "
                "likelihood (Student-t)"
            )
        elif stat == "mean":
            _add(
                f"site '{site}': the posterior-predictive mean misses the data "
                "mean — suspect a LOCATION/LINK misfit or a missing covariate; "
                "check the link function"
            )
    # --- latent-aware rules (v4) — each fires only on measured evidence -----
    for d in mixed_details or []:
        if d.get("passed"):
            continue
        site, stat, side = d.get("site", "y"), d.get("stat"), d.get("side")
        if stat == "within_unit_sd" and side == "high":
            _add(
                f"site '{site}': WITHIN-unit dispersion exceeds what the unit "
                "effects explain (mixed-replication PPC) — the misfit lives in "
                "the CONDITIONAL likelihood inside units, not the latent law; "
                "consider a Negative-Binomial (counts) or Student-t (continuous) "
                "conditional likelihood"
            )
        elif stat in ("max_unit_mean", "between_unit_spread") and side == "high":
            _add(
                f"site '{site}': unit-LEVEL statistics escape the fitted "
                "group-effect distribution (mixed-replication PPC) — suspect a "
                "misspecified LATENT LAW; consider heavier-tailed (Student-t) "
                "or mixture group effects"
            )
        elif stat == "unit_mean_kurtosis":
            direction = ("HEAVIER-tailed than the fitted law (consider "
                         "Student-t group effects)" if side == "high" else
                         "SHORTER-tailed/multimodal (consider a mixture of "
                         "group effects)")
            _add(
                f"site '{site}': the SHAPE of the unit-effect distribution is "
                f"off — kurtosis of unit means outside the replicated band, "
                f"{direction} (mixed-replication PPC)"
            )
    if comparison:
        khat_esc = comparison.get("escalated_from_khat")
        if khat_esc is not None:
            _add(
                f"per-unit latent structure made PSIS-LOO unreliable (Pareto "
                f"k-hat {float(khat_esc):.2f}); the comparison was re-adjudicated "
                "by cross-validation on the NEW-UNIT estimand — refinements must "
                "improve prediction for unseen units, not in-sample fit"
            )
        per_unit = (comparison.get("_crossval") or {}).get("per_unit_delta")
        if per_unit:
            d_abs = np.sort(np.abs(np.asarray(per_unit, dtype=float)))[::-1]
            total = float(d_abs.sum())
            k = max(1, int(np.ceil(0.1 * d_abs.size)))
            if total > 0 and float(d_abs[:k].sum()) / total > 0.5:
                _add(
                    "the cross-validated comparison is decided by a few units "
                    f"(top {k} of {d_abs.size} carry >50% of the total "
                    "|delta-eLPD|) — inspect those units before further "
                    "structural changes"
                )
    for cov, c in sorted(((audit or {}).get("mundlak") or {}).items()):
        if c.get("conflict"):
            _add(
                f"covariate `{cov}`: within-group and between-group effects "
                f"disagree (within {c['within']:.3g} vs between {c['between']:.3g}, "
                f"z={c['z']:.1f}) — latent-covariate dependence; group-mean "
                "centering (a Mundlak term) is the canonical repair class"
            )

    rhat = diagnostics.get("rhat_max", 0.0)
    if diagnostics.get("divergences", 0) > 0 or not (rhat < RHAT_THRESHOLD):
        hints.append(
            "sampler pathologies (divergences / high R-hat) point at a difficult "
            "posterior GEOMETRY; consider a non-centered reparameterization"
        )
    return hints


# ---------------------------------------------------------------------------
# Held-out predictive check (Calibration-Not-Compilation [13] borrow) — a real
# out-of-sample log predictive density on the WINNER. Reported, NEVER a gate:
# the MCMC validator already accepted via in-sample LOO/WAIC; this is a cleaner
# out-of-sample confirmation and a refinement is kept even if it does not win it.
# NOTE sign convention: lppd is a LOG SCORE (HIGHER = better), so a POSITIVE
# ``lppd_delta`` (refined - original) means the refinement predicts held-out data
# better — the OPPOSITE sign to the deviance-scale WAIC/LOO deltas elsewhere.
# ---------------------------------------------------------------------------

def model_site_shapes(model_code: str, data: dict) -> dict[str, tuple[str, bool, tuple]]:
    """``{site: (type, is_observed, value_shape)}`` for the sample and
    deterministic sites of one seeded trace of the model on ``data``."""
    from numpyro import handlers

    with _restored_x64():
        model = _exec_model(model_code)
        tr = handlers.trace(handlers.seed(model, 0)).get_trace(**data)
    return {name: (s["type"], bool(s.get("is_observed", False)),
                   tuple(np.shape(s["value"])))
            for name, s in tr.items() if s["type"] in ("sample", "deterministic")}


def model_site_types(model_code: str, data: dict) -> dict[str, tuple[str, bool]]:
    """``{site: (type, is_observed)}``; see ``model_site_shapes``."""
    return {n: (t, obs) for n, (t, obs, _) in model_site_shapes(model_code, data).items()}


def _marginalized_pointwise(model_code: str, idata, test_data: dict, local: list[str],
                            sites: list[str] | None) -> np.ndarray:
    # Observation-level latents cannot be plugged into new observations; the
    # ladder's scorer draws them from their conditional prior per global draw.
    from compass import crossval

    shapes = model_site_shapes(model_code, test_data)
    obs_sites = [n for n, (t, obs, _) in shapes.items() if t == "sample" and obs]
    names = obs_sites if sites is None else list(sites)
    # The site name and the data kwarg feeding it may differ (site "obs" fed
    # by y=...). Resolve the kwarg the way infer_cv_spec does (defect V6): the
    # site name when it is a data key, else the data array equal to the traced
    # site value — the old ``obs_kwarg=site`` made the scorer raise KeyError.
    with _restored_x64():
        tr = handlers.trace(handlers.seed(_exec_model(model_code), 0)).get_trace(**test_data)
    parts: list[np.ndarray] = []
    for site in names:
        n_units = int(shapes[site][2][0]) if shapes[site][2] else 1
        per_unit = [k for k, v in test_data.items()
                    if hasattr(v, "shape") and np.ndim(v) >= 1 and np.shape(v)[0] == n_units]
        obs_kwarg = _resolve_obs_kwarg(site, np.asarray(tr[site]["value"]), test_data)
        if obs_kwarg is None:
            raise ValueError(
                f"cannot resolve the data kwarg feeding observed site {site!r} "
                "(no data array equals the traced site value)")
        spec = crossval.CVSpec(per_unit_keys=per_unit, local_latents=list(local),
                               obs_site=site, obs_kwarg=obs_kwarg)
        lppd, _gap = crossval.held_out_unit_lppd(model_code, idata, test_data, spec,
                                                 n_units=n_units)
        parts.append(np.asarray(lppd, dtype=float))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=float)


def _resolve_obs_kwarg(site: str, site_value: np.ndarray, data: dict) -> str | None:
    """Data kwarg feeding observed ``site``: the site name when it is a data
    key, else the first array-valued data entry equal to the traced value
    (``infer_cv_spec``'s rule, at the coarser precision — F1: a program that
    casts its response observes a float32 copy); None when nothing matches."""
    if site in data:
        return site
    for key, value in data.items():
        if not isinstance(value, np.ndarray | list | tuple):
            continue
        if _same_observed_values(np.asarray(value), site_value)[0]:
            return key
    return None


def observed_sites(model_code: str, data: dict) -> list[str]:
    """Names of the observed sample sites, in model order."""
    return [n for n, (t, obs) in model_site_types(model_code, data).items()
            if t == "sample" and obs]


def heldout_lppd_pointwise(model_code: str, idata, test_data: dict, *,
                           sites: list[str] | None = None) -> np.ndarray:
    """Per-observation held-out log predictive density under the posterior.

    lppd_i = log( (1/S) sum_s p(y_i | theta_s) ), computed with numpyro's
    ``log_likelihood`` over the test observations and concatenated across the
    observed sites ``sites`` (default: all, in model order; a site with more
    than two dims is flattened over its non-draw dims). Higher = better fit.
    Runs under ``_restored_x64`` (this path exec's the model source too).
    """
    from jax.scipy.special import logsumexp
    from numpyro.infer import log_likelihood

    # Only latent sample sites are substituted from the posterior. Deterministic
    # and observed sites (often per-observation, e.g. a recorded ``mu``) carry
    # the TRAINING fold's length and must be recomputed on the test fold.
    try:
        shapes = model_site_shapes(model_code, test_data)
        substitutable = {n for n, (t, obs, _) in shapes.items() if t == "sample" and not obs}
        local = [n for n in idata.posterior.data_vars if n in substitutable
                 and tuple(np.asarray(idata.posterior[n].values).shape[2:]) != shapes[n][2]]
    except Exception:
        substitutable, local = None, []
    if local:
        return _marginalized_pointwise(model_code, idata, test_data, local, sites)
    with _restored_x64():
        model = _exec_model(model_code)
        latent = {}
        for name in idata.posterior.data_vars:
            if substitutable is not None and name not in substitutable:
                continue
            vals = np.asarray(idata.posterior[name].values)
            latent[name] = vals.reshape((-1,) + vals.shape[2:])
        ll = log_likelihood(model, latent, **test_data)  # {site: (n_draws, n_test)}
        names = list(ll) if sites is None else list(sites)
        parts: list[np.ndarray] = []
        for name in names:
            a = jnp.asarray(np.asarray(ll[name], dtype=float))
            if a.ndim == 1:
                a = a[:, None]
            elif a.ndim > 2:
                a = a.reshape(a.shape[0], -1)
            n_draws = a.shape[0]
            lppd_i = logsumexp(a, axis=0) - np.log(n_draws)  # (n_test,)
            parts.append(np.asarray(lppd_i, dtype=float))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=float)


def _heldout_lppd(model_code: str, idata, test_data: dict) -> float:
    """Total held-out lppd: ``heldout_lppd_pointwise(...).sum()``."""
    return float(heldout_lppd_pointwise(model_code, idata, test_data).sum())


def heldout_predictive_check(refined_code: str, original_code: str, data: dict,
                             framework: str = "numpyro", *, split_frac: float = 0.25,
                             seed: int = 0, cache_dir=None, fast: bool = True,
                             num_warmup: int | None = None,
                             num_samples: int | None = None,
                             num_chains: int | None = None) -> dict:
    """Out-of-sample confirmation for the winner: split, fit on train, score the
    held-out log predictive density of refined vs original.

    REPORTED, never an arbiter — a refinement the MCMC validator accepted is
    kept regardless of the held-out outcome (``refined_better`` is descriptive).
    numpyro only; returns ``{"available": False, "reason": ...}`` otherwise or
    when there are too few observations to split.
    """
    if framework != "numpyro":
        return {"available": False, "reason": "held-out lppd is numpyro-only"}
    data = data or {}
    if isinstance(data.get("y"), (list, tuple, np.ndarray)):
        n = int(np.asarray(data["y"]).shape[0])
    else:
        lens = [int(np.asarray(v).shape[0]) for v in data.values()
                if isinstance(v, (list, tuple, np.ndarray)) and np.asarray(v).ndim >= 1]
        n = max(lens) if lens else 0
    if n < 8:
        return {"available": False, "reason": f"too few observations to split (n={n})"}

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(split_frac * n)))
    test_idx = np.sort(idx[:n_test])
    train_idx = np.sort(idx[n_test:])

    def _split(sel):
        out = {}
        for k, v in data.items():
            arr = np.asarray(v) if isinstance(v, (list, tuple, np.ndarray)) else v
            if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == n:
                out[k] = arr[sel]
            elif (isinstance(v, int | np.integer) and not isinstance(v, bool)
                  and int(v) == n):
                # A scalar size kwarg equal to n (``def model(n=None, y=None)``
                # sizing its plate from ``n``) must follow the fold — the same
                # rule infer_cv_spec applies to ``unit_size_keys`` (defect V5:
                # left at n, the plate no longer matched the sliced arrays and
                # the README's own model failed with a broadcasting error).
                out[k] = int(len(sel))
            else:
                out[k] = v  # a shared scalar / non-per-observation entry
        return out

    train, test = _split(train_idx), _split(test_idx)
    nw = num_warmup if num_warmup is not None else (200 if fast else 500)
    ns = num_samples if num_samples is not None else (500 if fast else 1000)
    nc = num_chains if num_chains is not None else 2
    try:
        _, ref_idata = run_mcmc(refined_code, train, "numpyro", nw, ns, nc,
                                seed=seed, cache_dir=cache_dir)
        _, orig_idata = run_mcmc(original_code, train, "numpyro", nw, ns, nc,
                                 seed=seed, cache_dir=cache_dir)
        refined_lppd = _heldout_lppd(refined_code, ref_idata, test)
        original_lppd = _heldout_lppd(original_code, orig_idata, test)
    except Exception as exc:
        return {"available": False, "reason": f"held-out fit/score failed: {exc}"}

    delta = refined_lppd - original_lppd  # POSITIVE = refined predicts better
    return {
        "available": True,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "refined_lppd": refined_lppd,
        "original_lppd": original_lppd,
        "lppd_delta": delta,
        "per_point_delta": delta / len(test_idx),
        "refined_better": bool(delta > 0),
    }


# ---------------------------------------------------------------------------
# Per-proposal prediction scoring (asta-theorizer) + margin-calibrated
# confidence (MEDEA) — both deterministic, scored from the MCMC result only.
# ---------------------------------------------------------------------------

def _score_metric(metric: str, diagnostics: dict, comparison: dict,
                  ppc_passed: bool, direction: str = "improve"):
    """(observed, passed) for one prediction metric vs the aggregate result.

    ``passed`` is None when the metric is unmeasured/unknown (excluded from the
    pass/fail tally rather than counted as a failure). ``direction`` matters
    for the delta metrics only: "increase" predicts a POSITIVE (deviance-scale,
    i.e. worse) delta; "improve"/"decrease"/"pass" all mean the canonical good
    direction (delta < 0). The threshold metrics (divergences, rhat_max,
    ess_min, ebfmi_min) pass iff the validator threshold is met — their
    prediction is always "reach the healthy side of the threshold".
    """
    m = (metric or "").lower().strip()
    if m == "divergences":
        obs = int(diagnostics.get("divergences", 0))
        return obs, obs == 0
    if m == "rhat_max":
        obs = float(diagnostics.get("rhat_max", float("nan")))
        return obs, bool(np.isfinite(obs) and obs < RHAT_THRESHOLD)
    if m == "ess_min":
        obs = float(diagnostics.get("ess_min", float("nan")))
        return obs, bool(np.isfinite(obs) and obs > ESS_THRESHOLD)
    if m == "ebfmi_min":
        obs = float(diagnostics.get("ebfmi_min", float("nan")))
        return obs, (None if not np.isfinite(obs) else bool(obs > EBFMI_THRESHOLD))
    if m in ("loo_delta", "waic_delta"):
        obs = comparison.get(m)
        if obs is None:
            return None, None
        if (direction or "").lower().strip() == "increase":
            return float(obs), bool(float(obs) > 0)
        return float(obs), bool(float(obs) < 0)
    if m == "ppc":
        return bool(ppc_passed), bool(ppc_passed)
    return None, None  # unknown metric -> unscorable


def score_proposal_predictions(proposals, *, diagnostics: dict,
                               comparison: dict, ppc_passed: bool) -> list[dict]:
    """Score each proposal's falsifiable prediction against the MCMC result.

    Each proposal carrying a ``prediction`` (schemas.FalsifiablePrediction) is
    scored against the AGGREGATE post-refinement diagnostics — i.e. the full
    refined model with all proposals applied, NOT a per-proposal ablation. A
    pass therefore means "the refinement that included this proposal achieved
    the predicted diagnostic", not strict causal attribution (that would need
    the separate ablation backlog item). The MCMC numbers — never an LLM — are
    the arbiter. Returns one dict per scored proposal.
    """
    scores: list[dict] = []
    for raw in proposals or []:
        p = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
        pred = p.get("prediction")
        if not pred:
            continue  # no falsifiable prediction attached -> nothing to score
        pred = pred if isinstance(pred, dict) else dict(pred)
        metric = str(pred.get("metric", "")).lower().strip()
        direction = str(pred.get("direction", "improve"))
        observed, passed = _score_metric(metric, diagnostics, comparison,
                                         ppc_passed, direction=direction)
        scores.append({
            "name": p.get("name", "?"),
            "metric": metric,
            "direction": direction,
            "predicted": f"{metric} should {direction}",
            "observed": observed,
            "passed": passed,
            "note": str(pred.get("failure_mode", "")) if passed is False else "",
        })
    return scores


def confidence_from_margin(*, converged: bool, significant_improvement: bool,
                           elpd_diff: float, elpd_diff_se: float, ess_min: float,
                           divergences: int, ppc_passed: bool | None) -> str:
    """Calibrate HIGH | MEDIUM | LOW from the validator margin (MEDEA borrow).

    Derived purely from MCMC ground truth (the predictive z-margin
    |elpd_diff|/SE, ESS headroom over the threshold, divergences, PPC) — NOT an
    LLM judge. DESCRIPTIVE ONLY: surfaced in the report; it does not change
    routing (a converged-but-not-better run is still routed by the validator's
    failure_reasons exactly as before).

    ``ppc_passed=None`` (PPC unavailable — e.g. a Stan model without a
    recognized replicate variable) is NEUTRAL, mirroring the routing semantics:
    it blocks HIGH (no PPC evidence) but does not force LOW like a real fail.
    """
    if not converged or not significant_improvement:
        return "LOW"
    z = (abs(elpd_diff) / elpd_diff_se
         if elpd_diff_se and elpd_diff_se > 0 else float("inf"))
    ess_head = ess_min / ESS_THRESHOLD if np.isfinite(ess_min) else 0.0
    if z >= 4.0 and ess_head >= 1.5 and divergences == 0 and ppc_passed is True:
        return "HIGH"
    if z < 2.5 or ess_head < 1.1 or ppc_passed is False:
        return "LOW"
    return "MEDIUM"


# ---------------------------------------------------------------------------
# Full validation
# ---------------------------------------------------------------------------

def run_full_validation(refined_code: str, original_code: str, data: dict,
                        framework: str = "numpyro", fast: bool = False,
                        cache_dir=None, seed: int = 0,
                        num_warmup: int | None = None,
                        num_samples: int | None = None,
                        num_chains: int | None = None,
                        target_accept: float = 0.8,
                        proposals: list | None = None,
                        cv_spec: dict | None = None,
                        auto_cv: bool = True,
                        repair_clause: bool = True) -> ValidationResult:
    """Sample both models, run diagnostics + comparison + PPC -> ValidationResult.

    Budgets: full 1000/2000/4, fast 200/500/2 (override per-argument when a
    specific test needs more draws for stable ESS). `converged` reflects
    convergence ONLY; a converged-but-not-better model keeps converged=True
    and records the improvement failure in `failure_reasons`.

    `proposals` (optional) are scored individually against their falsifiable
    predictions (`score_proposal_predictions`); the result also carries a
    margin-calibrated `confidence`. Both are descriptive — the routing-relevant
    verdict is still `converged` + `failure_reasons`.

    `cv_spec` switches the significance gate to real cross-validation
    (crossval.py) — required for models with LOCAL per-unit latents, where
    PSIS-LOO collapses: pass ``{"spec": CVSpec(...), "scheme": "kfold", ...}``
    (everything but "spec" is forwarded to `crossval_compare`). Without it,
    `auto_cv` (default True, numpyro only) escalates automatically when the
    refined model's Pareto k-hat exceeds 0.7 (or is inf) AND the unit
    structure is derivable from the program's plates; escalation can only
    re-adjudicate comparisons the k-hat gate would have blocked, never a
    PSIS-accepted one.

    Comparability (defect V1; F1–F4): before any S-gate instrument runs, the
    refined program is traced on the data next to the original
    (`check_observed_sites` with ``original_code``) and the comparison is
    REFUSED — failure reason ``model comparison refused: ...``, no escalation,
    no repair clause — when an observed site is masked beyond the baseline's
    mask, scaled differently, or governed by response-dependent likelihood
    parameters the baseline does not have; `compare_models` additionally
    refuses fits whose observation sets differ or whose refined pointwise
    log-likelihood has degenerate (constant / non-finite) columns the baseline
    does not share. Descriptive structural notes (both programs
    response-dependent) land in `misspecification_hints`. A refusal never
    flips `converged`.
    """
    budget = FAST_BUDGET if fast else FULL_BUDGET
    warmup = num_warmup if num_warmup is not None else budget[0]
    samples = num_samples if num_samples is not None else budget[1]
    chains = num_chains if num_chains is not None else budget[2]
    data = data or {}

    _, refined_idata = run_mcmc(
        refined_code, data, framework, warmup, samples, chains, seed,
        target_accept, cache_dir,
    )
    _, original_idata = run_mcmc(
        original_code, data, framework, warmup, samples, chains, seed,
        target_accept, cache_dir,
    )

    diagnostics = compute_diagnostics(refined_idata)
    failure_reasons: list[str] = []
    converged = True

    if diagnostics["divergences"] > 0:
        failure_reasons.append(f"{diagnostics['divergences']} divergent transitions")
        converged = False
    if not (diagnostics["rhat_max"] < RHAT_THRESHOLD):  # NaN also fails
        failure_reasons.append(f"R-hat too high: {diagnostics['rhat_max']:.3f}")
        converged = False
    if not (diagnostics["ess_min"] > ESS_THRESHOLD):
        failure_reasons.append(f"ESS too low: {diagnostics['ess_min']:.0f}")
        converged = False
    ebfmi_min = diagnostics["ebfmi_min"]
    if np.isfinite(ebfmi_min) and not (ebfmi_min > EBFMI_THRESHOLD):
        failure_reasons.append(f"E-BFMI low: {ebfmi_min:.2f}")
        converged = False

    # The BASELINE fit must be trustworthy too: a non-mixing original
    # UNDER-estimates elpd(M0) and would manufacture a false accept. Recorded
    # as a comparison-reliability failure (like no-significant-improvement) —
    # `converged` keeps describing the refined fit only. `orig_bad` is also
    # the trigger evidence for the repair-equivalence clause below.
    orig_bad: list[str] = []
    try:
        orig_diag = compute_diagnostics(original_idata)
        if orig_diag["divergences"] > 0:
            orig_bad.append(f"{orig_diag['divergences']} divergences")
        if not (orig_diag["rhat_max"] < RHAT_THRESHOLD):
            orig_bad.append(f"R-hat {orig_diag['rhat_max']:.3f}")
        if not (orig_diag["ess_min"] > ESS_THRESHOLD):
            orig_bad.append(f"ESS {orig_diag['ess_min']:.0f}")
        if orig_bad:
            failure_reasons.append(
                "original-model fit unreliable (comparison baseline): "
                + ", ".join(orig_bad))
    except Exception:
        pass  # diagnostics unavailable on the baseline -> comparison speaks

    # Structural guard BEFORE the S gate (defect V1a): every instrument below
    # (PSIS-LOO, K-fold CV, reloo) trusts the refined program's own pointwise
    # log-likelihood, so a candidate that masks or re-weights part of its
    # observed site would be scored on a different observation set than the
    # baseline and could "win" by the excluded points' deficit. Decided by
    # tracing the program on the data (numpyro only — Stan has no such
    # handlers); a program the verifier cannot trace is refused too, since
    # the comparison would rest on a likelihood nobody inspected.
    structural: list[str] = []
    structural_notes: list[str] = []
    if framework == "numpyro":
        try:
            structural = list(check_observed_sites(
                refined_code, data, seed=seed, original_code=original_code or None,
                notes=structural_notes))
        except Exception as exc:
            structural = [f"the refined program could not be traced on the data ({exc})"]

    # Comparison / significance (S) gate — a reliability LADDER:
    #   1. explicit cv_spec  -> real cross-validation (crossval.py), no PSIS.
    #   2. default           -> PSIS-LOO (compare_models).
    #   3. auto-escalation   -> if PSIS is untrustworthy on the REFINED side
    #      (Pareto k-hat > 0.7, the blocking condition below) and the unit/
    #      latent structure is derivable from the program's plates
    #      (infer_cv_spec), re-adjudicate by real K-fold CV. This can only
    #      REPLACE a verdict the k-hat gate would have blocked anyway — a
    #      PSIS-accepted comparison is never re-adjudicated, so published
    #      PSIS verdicts are preserved by construction. If no spec can be
    #      derived, the conservative k-hat block stands.
    # CV comparisons carry their own reliability guards (fold-fit R-hat,
    # marginalization Monte-Carlo error) surfaced via `reliability_note`.
    used_cv = cv_spec is not None
    no_comparison = {
        "waic": 0.0, "loo": 0.0, "waic_delta": None, "loo_delta": None,
        "elpd_diff": float("nan"), "elpd_diff_se": float("nan"),
        "significant_improvement": False, "khat_max": None,
        "khat_refined": None, "khat_original": None,
    }
    comparison_error: str | None = None
    comparison_refused: str | None = None
    try:
        if structural:
            raise ComparisonRefused("; ".join(structural))
        if cv_spec is not None:
            from compass.crossval import crossval_compare
            spec_opts = dict(cv_spec)
            spec = spec_opts.pop("spec")
            comparison = crossval_compare(
                refined_code, original_code, data, spec,
                seed=seed, cache_dir=cache_dir, **spec_opts)
        else:
            comparison = compare_models(refined_idata, original_idata)
    except ComparisonRefused as exc:
        # V1: structurally incomparable fits (masked/scaled observed site,
        # different observation sets, degenerate pointwise columns). A verdict
        # in its own right — recorded below as the failure reason — not a
        # tooling error; nothing downstream (escalation, repair) may run.
        comparison = dict(no_comparison)
        comparison_refused = str(exc)
    except Exception as exc:  # e.g. no observed sites -> no log_likelihood
        comparison = dict(no_comparison)
        comparison_error = str(exc)

    _khat = comparison.get("khat_refined")
    # k-hat = inf (degenerate importance weights) escalates like k > 0.7; NaN
    # (k-hat not estimable) does not — see _khat_unreliable (V2).
    if (auto_cv and not used_cv and comparison_error is None
            and comparison_refused is None and framework == "numpyro"
            and _khat_unreliable(_khat)):
        try:
            from compass.crossval import crossval_compare, infer_cv_spec, reloo_refined
            inferred = infer_cv_spec(refined_code, data, original_code, seed=seed)
            if inferred is not None:
                # Rung 2a — local latents tied to the obs structure: PSIS's
                # estimand itself is broken; re-adjudicate by real K-fold CV.
                cv_cmp = crossval_compare(
                    refined_code, original_code, data, inferred["spec"],
                    scheme="kfold", K=3 if fast else 5,
                    seed=seed, cache_dir=cache_dir)
                cv_cmp["escalated_from_khat"] = float(_khat)
                cv_cmp["s_gate"] = (
                    f"{cv_cmp['s_gate']} over plate '{inferred['plate']}' "
                    f"[auto-escalated: PSIS k-hat {_khat:.2f} > 0.7]")
                comparison = cv_cmp
                used_cv = True
            else:
                # Rung 2b — no local latents: a few highly influential points;
                # compute exactly the LOO contributions PSIS could not
                # approximate (refit-without-point), keep PSIS for the rest.
                rl = reloo_refined(refined_code, original_code, data,
                                   refined_idata, original_idata,
                                   budget=(warmup, samples, chains),
                                   seed=seed, cache_dir=cache_dir)
                if rl is not None:
                    rl["escalated_from_khat"] = float(_khat)
                    rl["s_gate"] = (f"{rl['s_gate']} "
                                    f"[auto-escalated: PSIS k-hat {_khat:.2f} > 0.7]")
                    comparison = rl
        except Exception:
            pass  # keep the PSIS comparison -> the k-hat gate blocks below

    ppc_passed = False  # True | False | None (None = could not be computed)
    ppc_details: list = []
    ppc_error: str | None = None
    if framework in ("numpyro", "stan"):
        try:
            ppc_passed, ppc_details = run_ppc(refined_idata, refined_code, data,
                                              framework=framework, seed=seed + 1,
                                              return_details=True)
        except Exception as exc:
            # Still blocking (conservative: a PPC that cannot run is not
            # evidence of fit) but the cause is surfaced in the failure reason
            # below instead of being silently discarded.
            ppc_passed, ppc_details = False, []
            ppc_error = str(exc)

    # Mixed-replication PPC + unit-aware statistics (v4, DESCRIPTIVE — never a
    # gate): only when the program has local latents (resolution.py classifier;
    # lazy import — resolution imports this module's helpers).
    ppc_mixed: dict | None = None
    audit: dict | None = None
    if framework == "numpyro":
        try:
            from compass.resolution import audit_resolution
            audit = audit_resolution(refined_code, data, seed=seed)
        except Exception:
            audit = None
        if audit and audit.get("has_local_latents"):
            local_sites = sorted({s for p in audit["plates"] for s in p["sites"]})
            gkey = next((p["index_key"] for p in audit["plates"]
                         if p["tie"] == "indexed"), None)
            try:
                mp, md = run_mixed_ppc(refined_idata, refined_code, data,
                                       local_sites=local_sites,
                                       group_idx_key=gkey, seed=seed + 2)
                ppc_mixed = {"passed": mp, "details": md,
                             "local_sites": local_sites}
            except Exception as exc:
                ppc_mixed = {"passed": None, "details": [], "error": str(exc)}

    # Improvement / PPC notes do NOT flip `converged` (orchestrator routes
    # converged-but-not-better runs to "search_more", spec §5.5).
    if comparison_refused is not None:
        failure_reasons.append(comparison_refused)  # "model comparison refused: ..."
    elif comparison_error is not None:
        failure_reasons.append(f"model comparison unavailable: {comparison_error}")
    elif not comparison["significant_improvement"] and not comparison.get("reliability_note"):
        # With a reliability_note the margin may well have been cleared — the
        # comparison was DEMOTED as non-certifiable; the note (appended below)
        # is the true reason and a generic "no improvement" would mislead.
        failure_reasons.append(
            "no significant improvement: delta-eLPD "
            f"{comparison['elpd_diff']:.2f} (se {comparison['elpd_diff_se']:.2f})"
        )
    # PSIS reliability gate — REFINED model only. High k-hat means PSIS tends
    # to OVER-estimate that model's elpd: on the refined candidate that
    # inflates the apparent gain (false-acceptance risk -> block); on the
    # original it inflates the baseline and biases AGAINST acceptance
    # (conservative -> recorded via khat_original/khat_max, never blocking —
    # a badly misspecified original, e.g. Poisson on overdispersed counts, is
    # the textbook high-k-hat case and must not be un-refinable). Like
    # no-significant-improvement this is a comparison-reliability failure:
    # recorded, never flips `converged`.
    # Skipped entirely under cross-validation (explicit cv_spec or auto-
    # escalation): no importance sampling was used, so there is no khat to
    # gate on (comparison["khat_refined"] is None) — the CV path carries its
    # own reliability guards, surfaced below via `reliability_note`.
    # k-hat = inf is arviz's verdict on DEGENERATE importance weights and must
    # gate like k > 0.7; only NaN (not estimable) is non-gating — see
    # _khat_unreliable (V2; the former `np.isfinite(k) and k > 0.7` test let
    # inf through and so disabled this gate on exactly the masked candidates).
    khat_refined = comparison.get("khat_refined")
    if not used_cv and _khat_unreliable(khat_refined):
        if np.isfinite(khat_refined):
            failure_reasons.append(
                f"PSIS unreliable: refined-model Pareto-k-hat {khat_refined:.2f} > 0.7"
            )
        else:
            # arviz returns inf whenever at most 4 importance weights exceed the
            # tail cutoff — constant columns, but also near-tied weights (F4).
            failure_reasons.append(
                f"PSIS unreliable: refined-model Pareto-k-hat is {float(khat_refined)} "
                "(degenerate or tied importance weights)"
            )
    reliability_note = comparison.get("reliability_note")
    if reliability_note:
        failure_reasons.append(str(reliability_note))
    # Only a genuine False blocks routing; None means PPC was unavailable
    # (e.g. a Stan model with no recognized replicate variable) and is
    # descriptive, not a failure — restoring the pre-Stan-PPC behavior of not
    # penalizing a converged, improved model for a missing convention.
    if framework in ("numpyro", "stan") and ppc_passed is False:
        failure_reasons.append(
            f"posterior predictive check failed to run: {ppc_error}"
            if ppc_error else "posterior predictive check failed"
        )

    # --- repair-equivalence clause (v4) --------------------------------------
    # A refinement whose value is inferential/computational rather than
    # predictive (identification constraints, reparameterizations) leaves
    # delta-eLPD ~ 0, so the S gate can never certify it — yet if the BASELINE
    # failed the convergence gates and the refinement passes them with
    # TOST-equivalent predictions, refusing it abandons the user inside a
    # broken fit. Preconditions (ALL required): refined converged; baseline
    # failed G (orig_bad, the trigger evidence); comparison reliable (finite
    # SE > 0, no reliability note, refined-side k-hat clean or CV-adjudicated);
    # TOST equivalence |diff| <= 2*SE AND |diff| + 2*SE <= 4 nats (the CI sits
    # inside ±4 — a NOISY comparison must never grant a repair); and nothing
    # else went wrong (every failure reason is one of the two expected kinds).
    # Fires only where the baseline fails G, so no published verdict — every
    # published baseline converges by construction — can flip (I2).
    repair: str | None = None
    if (repair_clause and converged and comparison_error is None
            and comparison_refused is None and orig_bad):
        _diff = comparison.get("elpd_diff")
        _se = comparison.get("elpd_diff_se")
        _khat_r = comparison.get("khat_refined")
        # The clause needs POSITIVE evidence that PSIS was reliable, so the
        # refined k-hat must be finite and <= 0.7 (or absent under CV): both
        # inf (degenerate weights, V2) and NaN (not estimable) refuse here —
        # stricter than the gate above, where NaN is merely non-gating.
        comparison_reliable = (
            _se is not None and np.isfinite(_se) and _se > 0
            and _diff is not None and np.isfinite(_diff)
            and not comparison.get("reliability_note")
            and (_khat_r is None or (np.isfinite(_khat_r)
                                     and not _khat_unreliable(_khat_r))))
        tost = bool(comparison_reliable
                    and abs(_diff) <= 2 * _se
                    and abs(_diff) + 2 * _se <= 4.0)
        _expected = ("no significant improvement:",
                     "original-model fit unreliable")
        only_expected = failure_reasons and all(
            str(r).startswith(_expected) for r in failure_reasons)
        if tost and only_expected:
            # Abuse guard (measured on the latent grid): a cosmetic rival that
            # merely out-mixes a MARGINALLY under-sampled baseline must not be
            # crowned a repair. The baseline gets one retry at doubled budget;
            # only if it STILL fails G is it treated as broken. True
            # pathologies (ridges, funnels) fail at any budget; thin-budget
            # ESS misses do not.
            retry_ok = False
            try:
                _, _orig_retry = run_mcmc(
                    original_code, data, framework, warmup * 2, samples * 2,
                    chains, seed, target_accept, cache_dir)
                _d2 = compute_diagnostics(_orig_retry)
                retry_ok = (_d2["divergences"] == 0
                            and _d2["rhat_max"] < RHAT_THRESHOLD
                            and _d2["ess_min"] > ESS_THRESHOLD)
            except Exception:
                retry_ok = False
            if not retry_ok:
                repair = (
                    "accepted as computational repair: predictive equivalence "
                    f"(|delta-eLPD| {abs(_diff):.2f} <= 2*SE {2 * _se:.2f}, CI "
                    f"within ±4 nats) while the baseline fit failed the "
                    f"convergence gates ({', '.join(orig_bad)}; still failing "
                    "at doubled budget)")
                failure_reasons = []

    # Descriptive enrichments (do not affect routing): per-proposal prediction
    # scoring (asta) and margin-calibrated confidence (MEDEA).
    proposal_scores = score_proposal_predictions(
        proposals, diagnostics=diagnostics, comparison=comparison,
        ppc_passed=ppc_passed,
    )
    confidence = confidence_from_margin(
        converged=converged,
        significant_improvement=comparison["significant_improvement"],
        elpd_diff=comparison["elpd_diff"],
        elpd_diff_se=comparison["elpd_diff_se"],
        ess_min=diagnostics["ess_min"],
        divergences=diagnostics["divergences"],
        ppc_passed=ppc_passed,
    )
    # Descriptive repair hints from the failed diagnostics (localization borrow);
    # surfaced to the strategist, never touching `converged`/`failure_reasons`.
    # v4: latent-aware evidence — mixed-PPC details, S-gate escalation
    # provenance + per-unit delta concentration, Mundlak contrast.
    misspecification_hints = localize_misspecification(
        diagnostics, ppc_details,
        mixed_details=(ppc_mixed or {}).get("details"),
        comparison=comparison, audit=audit)
    # Structural notes (F3: both programs condition their likelihood on the
    # response; a baseline that could not be probed) — descriptive, never a
    # failure reason.
    for note in structural_notes:
        if note not in misspecification_hints:
            misspecification_hints.append(note)

    return ValidationResult(
        converged=converged,
        rhat_max=diagnostics["rhat_max"],
        ess_min=diagnostics["ess_min"],
        divergences=diagnostics["divergences"],
        waic=comparison["waic"],
        loo=comparison["loo"],
        waic_delta=comparison["waic_delta"],
        loo_delta=comparison["loo_delta"],
        elpd_diff=(float(comparison["elpd_diff"])
                   if np.isfinite(comparison.get("elpd_diff", float("nan"))) else None),
        elpd_diff_se=(float(comparison["elpd_diff_se"])
                      if np.isfinite(comparison.get("elpd_diff_se", float("nan"))) else None),
        khat_max=comparison.get("khat_max"),
        s_gate=comparison.get("s_gate"),
        posterior_predictive_check=bool(ppc_passed),  # None (unavailable) -> False
        ppc_mixed=ppc_mixed,
        repair=repair,
        failure_reasons=failure_reasons,
        confidence=confidence,
        proposal_scores=proposal_scores,
        misspecification_hints=misspecification_hints,
    )


# ---------------------------------------------------------------------------
# Agent factory (LangGraph node), system convention #1
# ---------------------------------------------------------------------------

def create_validator_agent(llm=None, *, data: dict | None = None,
                           framework: str | None = None,
                           fast: bool = False, cache_dir=None, seed: int = 0,
                           validation_fn: Callable | None = None,
                           compute_heldout: bool = False,
                           cv_spec: dict | None = None,
                           auto_cv: bool = True,
                           repair_clause: bool = True):
    """No-LLM validator node. `llm` is accepted (and ignored) for signature
    uniformity with the other agent factories.

    `data` is the actual dataset dict (arrays) — WorkflowState carries only a
    textual dataset_summary, so the orchestrator supplies arrays here.
    `validation_fn` lets tests inject a stub instead of real MCMC.

    `compute_heldout=True` runs the descriptive out-of-sample held-out predictive
    check ON THE WINNER only — i.e. when this refinement actually passes (real
    numpyro validation, real data). It is attached to `result.heldout` and never
    changes acceptance: a passing refinement is kept even if the held-out check
    does not favor it. Off by default so the routine pipeline pays nothing.
    """
    fn = validation_fn or run_full_validation

    def run(state: WorkflowState) -> dict:
        fw = framework
        if fw is None and state.original_model is not None:
            fw = state.original_model.framework or "numpyro"
        fw = fw or "numpyro"
        original_code = (state.original_model.source_code
                         if state.original_model is not None else "")
        extra: dict = {}
        if fn is run_full_validation:
            if cv_spec is not None:
                extra["cv_spec"] = cv_spec
            if not auto_cv:
                extra["auto_cv"] = False
            if not repair_clause:
                extra["repair_clause"] = False
        try:
            result = fn(state.refined_model_code, original_code, data or {},
                        framework=fw, fast=fast, cache_dir=cache_dir, seed=seed,
                        proposals=state.proposals, **extra)
        except Exception as exc:
            result = ValidationResult(
                converged=False,
                failure_reasons=[f"validation error: {exc}"],
            )
            state.error_log.append(f"validator: {exc}")
            state.validation = result
            # Validator does NOT set status — orchestrator routes on result.
            return {"validation": result, "error_log": state.error_log}
        # WINNER-ONLY out-of-sample confirmation (descriptive; never rejects).
        # Runs only for a real (non-stub) numpyro validation that actually passed.
        if (compute_heldout and fn is run_full_validation and (data or {})
                and (fw or "numpyro") == "numpyro"
                and result.converged and not result.failure_reasons):
            if cv_spec is not None or (result.s_gate or "").strip():
                # The S gate WAS an out-of-sample comparison (cross-validated);
                # the naive row-split check would leak/fail on per-unit latents.
                result.heldout = {
                    "available": False,
                    "reason": "superseded by cross-validated significance gate "
                              "(out-of-sample by construction)",
                }
            else:
                try:
                    result.heldout = heldout_predictive_check(
                        state.refined_model_code, original_code, data or {},
                        framework="numpyro", seed=seed, cache_dir=cache_dir, fast=fast,
                    )
                except Exception as exc:  # descriptive only — never break the run
                    state.error_log.append(f"held-out check skipped: {exc}")
        state.validation = result
        return {"validation": result}

    return run
