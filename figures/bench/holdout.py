"""Proposer-hidden holdout scoring with a standard error.

Split the data BEFORE the pipeline runs (``split_visible_hidden``), fit every
candidate and the baseline on the visible fold only (``fit_visible``), and
score the paired per-observation log-predictive-density difference on the
hidden fold (``score_pair_on_hidden``). The hidden fold is never shown to the
proposer, so the Δ is an out-of-sample ground truth rather than the in-sample
LOO the pipeline optimises against.

Sign convention: positive ``delta`` = the candidate predicts the hidden fold
better (log score, higher = better) — the OPPOSITE of the deviance-scale
``loo_delta`` / ``waic_delta`` in ``agents.validator`` where negative favours
the refinement.

Limitation shared with ``agents.validator.heldout_predictive_check``: only
arrays with a leading dim of ``n`` are split; integer size scalars such as
``N`` or ``n_groups`` are passed through UNCHANGED, so models that take an
explicit observation count must not be scored this way. Unlike the validator,
"array" here means anything numpy can view (lists, numpy, jax, pandas), and
an object with a leading axis of ``n`` that cannot be sliced is an error
rather than a silent row misalignment.
"""

from __future__ import annotations

import math

import numpy as np

import compass.agents.validator as validator

LABELS = ("genuine", "confabulated", "harmful", "unavailable")
LPPD_SANITY_BOUND = 1e6  # per-observation |log density| beyond this is a broken model


def _as_array(v) -> np.ndarray | None:
    # Anything with a leading axis that numpy can view (lists, numpy, jax,
    # pandas); scalars, 0-d arrays and strings come back None.
    if isinstance(v, (str, bytes)) or not (isinstance(v, (list, tuple)) or hasattr(v, "__array__")):
        return None
    arr = np.asarray(v)
    return arr if arr.ndim >= 1 else None


def _infer_n(data: dict) -> int:
    # Mirrors validator.heldout_predictive_check: y's leading dim if y is
    # array-like, else the max leading dim over array-like entries.
    y = _as_array(data.get("y"))
    if y is not None:
        return int(y.shape[0])
    lens = [a.shape[0] for a in map(_as_array, data.values()) if a is not None]
    return int(max(lens)) if lens else 0


def split_visible_hidden(data: dict, *, hidden_frac: float = 0.25, seed: int = 0,
                         n: int | None = None) -> tuple[dict, dict, dict]:
    """Deterministic row split of ``data`` into (visible, hidden, info).

    Same permutation / rounding as ``heldout_predictive_check`` (so the same
    seed and fraction give the same fold). Arrays whose ``shape[0] == n`` are
    sliced along axis 0 with their dtype kept; everything else (scalars,
    strings, arrays with another leading dim) is passed through unchanged.
    """
    if not 0 < hidden_frac <= 0.5:
        raise ValueError(f"hidden_frac must be in (0, 0.5], got {hidden_frac}")
    data = data or {}
    n = _infer_n(data) if n is None else int(n)
    if n < 8:
        raise ValueError(f"too few observations to split (n={n})")

    arrays = {k: _as_array(v) for k, v in data.items()}
    for k, v in data.items():
        if arrays[k] is None:
            shape = getattr(v, "shape", None)
            if shape and shape[0] == n:
                raise ValueError(f"{k!r} has a leading dim of n={n} but cannot be sliced as an array")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_hidden = max(1, int(round(hidden_frac * n)))
    hidden_idx = np.sort(perm[:n_hidden])
    visible_idx = np.sort(perm[n_hidden:])

    def _take(sel: np.ndarray) -> dict:
        return {k: (arrays[k][sel] if arrays[k] is not None and arrays[k].shape[0] == n else v)
                for k, v in data.items()}

    info = {
        "n": int(n),
        "n_visible": int(len(visible_idx)),
        "n_hidden": int(len(hidden_idx)),
        "hidden_idx": [int(i) for i in hidden_idx],
        "seed": int(seed),
        "hidden_frac": float(hidden_frac),
    }
    return _take(visible_idx), _take(hidden_idx), info


def fit_visible(model_code: str, visible: dict, *, budget: dict, seed: int, cache_dir,
                framework: str = "numpyro"):
    """Sample ``model_code`` on the visible fold and return the ``InferenceData``.

    Goes through ``validator.run_mcmc`` with its trace cache, keyed on
    (model code, data, budget, seed, framework, x64): the baseline fit is a
    cache hit from the second candidate onwards, so scoring K candidates costs
    K + 1 fits, not 2K.
    """
    _, idata = validator.run_mcmc(
        model_code, visible, framework=framework,
        num_warmup=budget["num_warmup"], num_samples=budget["num_samples"],
        num_chains=budget["num_chains"],
        target_accept=budget.get("target_accept", 0.8),
        seed=seed, cache_dir=cache_dir,
    )
    return idata


def classify_delta(delta: float, se: float, k: float = 2.0) -> str:
    if not (math.isfinite(delta) and math.isfinite(se)):
        raise ValueError(f"non-finite delta/se: {delta}, {se}")
    if se > 0 and delta > k * se:
        return "genuine"
    if se > 0 and delta < -k * se:
        return "harmful"
    return "confabulated"


def score_pair_on_hidden(candidate_code: str, baseline_code: str, visible: dict,
                         hidden: dict, *, budget: dict, seed: int = 0, cache_dir=None,
                         framework: str = "numpyro") -> dict:
    """Paired hidden-fold Δlppd (candidate − baseline) with its SE and a label.

    ``se = sd(per-obs deltas, ddof=1) * sqrt(n)`` (the ``az.compare`` ``dse``
    convention on the sum scale, as in ``crossval._summarize``); 0.0 when only
    one observation is scored. Never raises: any failure (sampling, exec of
    LLM-written code, shape mismatch between the two models' observed sites,
    no observed site, a non-finite pointwise lppd such as a hidden value
    outside a model's support) returns
    ``{"available": False, "reason": ..., "label": "unavailable"}``.
    """
    try:
        cand_idata = fit_visible(candidate_code, visible, budget=budget, seed=seed,
                                 cache_dir=cache_dir, framework=framework)
        base_idata = fit_visible(baseline_code, visible, budget=budget, seed=seed,
                                 cache_dir=cache_dir, framework=framework)
        common = [s for s in validator.observed_sites(baseline_code, hidden)
                  if s in set(validator.observed_sites(candidate_code, hidden))]
        if not common:
            raise ValueError("no observed site shared by candidate and baseline")
        cand = np.asarray(validator.heldout_lppd_pointwise(candidate_code, cand_idata, hidden,
                                                            sites=common), dtype=float)
        base = np.asarray(validator.heldout_lppd_pointwise(baseline_code, base_idata, hidden,
                                                            sites=common), dtype=float)
        if cand.shape != base.shape:
            raise ValueError(
                f"pointwise lppd length mismatch: candidate {cand.shape} vs baseline {base.shape}"
            )
        if cand.size == 0:
            raise ValueError("no observed sites scored")
        bad_c, bad_b = ~np.isfinite(cand), ~np.isfinite(base)
        if bad_c.any() or bad_b.any():
            raise ValueError(
                f"non-finite pointwise lppd on {int((bad_c | bad_b).sum())} of {cand.size} "
                f"hidden observations (candidate {int(bad_c.sum())}, baseline {int(bad_b.sum())})"
            )
        # a finite but astronomical log density is a degenerate likelihood, not a score
        worst = float(max(np.abs(cand).max(), np.abs(base).max()))
        if worst > LPPD_SANITY_BOUND:
            raise ValueError(f"implausible pointwise lppd (|lppd| = {worst:.3g} > {LPPD_SANITY_BOUND:g})")
        d = cand - base
        delta = float(d.sum())
        se = float(d.std(ddof=1) * math.sqrt(len(d))) if len(d) > 1 else 0.0
        return {
            "available": True,
            "n_hidden": int(len(d)),
            "candidate_lppd": float(cand.sum()),
            "baseline_lppd": float(base.sum()),
            "delta": delta,
            "se": se,
            "label": classify_delta(delta, se),
            "sites_scored": common,
            "per_obs_delta": [float(x) for x in d],
        }
    except Exception as e:  # LLM-written candidate code can fail any way; the row records it
        return {"available": False, "reason": f"{type(e).__name__}: {e}", "label": "unavailable"}


def summarize_labels(rows: list[dict]) -> dict:
    """Counts and fractions of each label over ``score_pair_on_hidden`` outputs."""
    counts = dict.fromkeys(LABELS, 0)
    for row in rows:
        label = row.get("label", "unavailable")
        counts[label] = counts.get(label, 0) + 1
    n = len(rows)
    fractions = {k: (v / n if n else 0.0) for k, v in counts.items()}
    return {"n": n, "counts": counts, "fractions": fractions}


def fresh_draw(inst_type: str, seed: int, offset: int = 10_000) -> dict:
    """A fresh dataset from the SAME data-generating process as ``make_instance(inst_type, seed)``.

    Every ``bench.dgp.make_*`` hardcodes its DGP parameters (n, coefficients,
    dispersion, noise scale); only the RNG seed varies, so ``seed + offset`` is
    an independent draw from the identical process. ``score_pair_on_hidden(...,
    hidden=fresh_draw(...))`` therefore scores against the exact-DGP ground
    truth rather than a held-out subsample. A generator that accepts
    ``obs_seed`` (``missing_hierarchy``) keeps its group effects and redraws
    only the observations, so fitted per-group effects transfer to the fresh
    data as they would to new measurements of the same groups.
    """
    import inspect

    from bench.dgp import SPECS

    fn = SPECS[inst_type]
    if "obs_seed" in inspect.signature(fn).parameters:
        return fn(seed, obs_seed=seed + offset)["data"]
    return fn(seed + offset)["data"]
