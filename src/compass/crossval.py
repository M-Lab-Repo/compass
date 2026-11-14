"""crossval.py — cross-validated model comparison for LATENT-VARIABLE models.

Why this exists
---------------
`compare_models` (validator.py) certifies a refinement via PSIS-LOO. PSIS-LOO
APPROXIMATES leave-one-out by importance-reweighting the full-data posterior;
that approximation collapses when the model has a LOCAL latent per unit (one
latent variable tied to each exchangeable unit): dropping a unit reverts its
latent to the prior, importance weights blow up, Pareto-k-hat >> 0.7, and the
reliability gate refuses even a genuine win. Refinements that ADD unit-level
structure (overdispersion effects, hierarchical pooling, IRT abilities) are
among the most common real fixes, so a k-hat gate alone makes them
un-certifiable: the verifier needs a second instrument, not a lower bar.

This module replaces the importance-sampling APPROXIMATION with a real
out-of-sample fit, so k-hat never enters. It returns the SAME numbers
`compare_models` does — {loo_delta, elpd_diff, elpd_diff_se,
significant_improvement} — so it is a drop-in for the significance gate; the
acceptance predicate downstream is unchanged.

Statistical design (each point is load-bearing)
-----------------------------------------------
* Held-out units are scored by the log predictive density with the unit's
  local latent t marginalized under the model's own CONDITIONAL prior:
      p(y_u | theta_s) = E_{t ~ p(t | theta_s)} p(y_u | t, theta_s),
  estimated with R fresh draws of t PER posterior draw theta_s, obtained via
  `Predictive(model, posterior_samples=<globals repeated R times>)` — numpyro
  samples the missing sites conditioned row-wise on the provided ones. This is
  exact for centered AND non-centered parameterizations; sampling t from the
  MARGINAL prior instead (a plain prior-`Predictive` call) would be wrong
  whenever the local prior depends on hyperparameters — i.e. in every
  hierarchical model.
* The per-unit likelihood is the JOINT likelihood of all of the unit's
  observations (summed over trailing obs axes) inside the marginalization —
  marginalizing per item instead would score a different (wrong) predictive.
* log(1/R * sum_r p) is biased DOWNWARD for finite R, and the bias is smaller
  for the model whose conditional prior concentrates near the data — typically
  the refinement. Unchecked, that asymmetry manufactures false accepts. Every
  scoring call therefore also computes a half-R jackknife gap
  (lppd_R - lppd_{R/2} ~= remaining bias); `crossval_compare` doubles R until
  the summed gap of each model is small vs the decision SE (cheap: MCMC fold
  fits are trace-cached, only re-scoring reruns), and if it cannot get there it
  DEMOTES the comparison to not-significant with an explicit reliability note —
  the conservative direction, mirroring the PSIS k-hat gate.
* Both models are fit on the SAME folds with the SAME seeds, and compared by
  PAIRED per-unit deltas; the SE is the usual sqrt(n)*sd of the deltas (the
  az.compare `dse` convention). K-fold pointwise contributions are not fully
  independent across units (shared training fits — Bengio & Grandvalet 2004);
  the pairing cancels the shared fold-level component in the DIFFERENCE, which
  is the quantity gated on.
* Every fold fit's convergence is checked (R-hat over both models' fits); a
  non-mixing fold fit poisons the ELPD estimate, so it also demotes the
  comparison rather than silently passing through.

Generality
----------
Nothing here is tied to a particular application. A model opts in with a small
`CVSpec` describing (a) which data axis indexes the exchangeable unit, (b)
which sites are LOCAL latents tied to that unit. `infer_cv_spec` derives a
CVSpec automatically from the program's plate structure (numpyro
cond_indep_stack) and dry-run-validates it, which is what lets the validator
ESCALATE from PSIS-LOO to real CV without user input. Three resampling schemes
wrap the one scoring primitive:

  * `entry_mask_score`    — hold out a random subset of the (unit x item)
                            entries; the unit's latent stays informed by its
                            retained items, so NO marginalization is needed.
                            Needs a mask-aware model. Deltas are aggregated
                            per UNIT before the SE (held-out entries within a
                            unit share its latent and are not independent).
  * `heldout_unit_score`  — single train/test split of whole units; held-out
                            units are genuinely new, latents marginalized.
  * `kfold_unit_elpd`     — K-fold over units; accumulates a pointwise ELPD
                            over every unit with a proper SE. The
                            arbiter-grade version.

All three consume `held_out_unit_lppd`, the marginalization primitive.
Scoring assumes the obs array's UNIT axis is axis 0 (auto-inference verifies
this; hand-written specs must respect it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import arviz as az
import jax
import numpy as np
from jax.scipy.special import logsumexp
from numpyro import handlers
from numpyro.infer import Predictive, log_likelihood

# Reuse the validator's model exec + sampler + x64 guard + thresholds
# (no new infra, same convergence bar).
from compass.agents.validator import (
    RHAT_THRESHOLD,
    _exec_model,
    _restored_x64,
    _same_observed_values,
    compute_diagnostics,
    run_mcmc,
    shared_degenerate_columns,
)


# ---------------------------------------------------------------------------
# Descriptor: everything model-specific about the CV, in one place.
# ---------------------------------------------------------------------------
@dataclass
class CVSpec:
    """Declares the exchangeable unit and its local latents for CV.

    per_unit_keys  : data arrays sliced along the unit axis when folding.
    local_latents  : sites tied to the unit — marginalized when a unit is held
                     out. Listing a site a model does not have is fine (it is
                     intersected with the model's actual sample sites), so one
                     spec serves an original/refined pair where only one side
                     has the latent. Deterministic sites need not be listed —
                     they are recomputed, never substituted.
    obs_site       : the observation SITE name.
    obs_kwarg      : the data kwarg feeding that site (defaults to obs_site).
    unit_size_keys : scalar kwargs set to the fold's unit count (e.g. J).
    shared_keys    : data passed unchanged to every fold (documentation only —
                     anything not per-unit/size is passed through anyway).
    unit_axis      : default axis of per-unit arrays indexing the unit.
    per_unit_axes  : per-key override of unit_axis, e.g. {"x": 1}.
    plate          : unit plate name when known (provenance / reporting).
    """
    per_unit_keys: list[str]
    local_latents: list[str]
    obs_site: str = "y"
    obs_kwarg: str | None = None
    unit_size_keys: list[str] = field(default_factory=list)
    shared_keys: list[str] = field(default_factory=list)
    unit_axis: int = 0
    per_unit_axes: dict = field(default_factory=dict)
    plate: str | None = None

    @property
    def obs_data_key(self) -> str:
        return self.obs_kwarg or self.obs_site


def _slice_units(data: dict, idx: np.ndarray, spec: CVSpec) -> dict:
    """Build a fold's data dict: slice per-unit arrays, resize size scalars."""
    out = {}
    for k, v in data.items():
        if k in spec.per_unit_keys:
            axis = int(spec.per_unit_axes.get(k, spec.unit_axis))
            out[k] = np.take(np.asarray(v), idx, axis=axis)
        elif k in spec.unit_size_keys:
            out[k] = int(len(idx))
        else:
            out[k] = v
    return out


def _n_units(data: dict, spec: CVSpec) -> int:
    key = spec.per_unit_keys[0]
    axis = int(spec.per_unit_axes.get(key, spec.unit_axis))
    return int(np.asarray(data[key]).shape[axis])


def _model_sample_sites(model, call_kwargs: dict, seed: int) -> tuple[set, set]:
    """(latent sample sites, observed sample sites) of `model` under kwargs."""
    with handlers.trace() as tr:
        handlers.seed(model, jax.random.PRNGKey(seed))(**call_kwargs)
    latent = {n for n, s in tr.items()
              if s["type"] == "sample" and not s.get("is_observed")}
    observed = {n for n, s in tr.items()
                if s["type"] == "sample" and s.get("is_observed")}
    return latent, observed


def _flat_global_draws(idata, names: list[str], s_use: int, seed: int) -> dict:
    """Flatten (chain, draw, ...) -> (S, ...) and jointly subsample S_use rows.

    One shared index set across names: per-name subsampling would break the
    joint posterior.
    """
    draws = {}
    S = None
    for nm in names:
        a = np.asarray(idata.posterior[nm].values)
        a = a.reshape((-1,) + a.shape[2:])
        draws[nm] = a
        S = a.shape[0] if S is None else S
    if S is None:
        return {}
    rng = np.random.default_rng(seed)
    take = rng.choice(S, size=min(s_use, S), replace=False)
    return {nm: a[take] for nm, a in draws.items()}


# ---------------------------------------------------------------------------
# The scoring primitive: log predictive density of held-out units, with the
# local latent marginalized over the model's own CONDITIONAL prior p(t|theta).
# ---------------------------------------------------------------------------
def held_out_unit_lppd(model_code: str, train_idata, test_data: dict, spec: CVSpec,
                       *, n_units: int, s_use: int = 300, r_prior: int = 60,
                       seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Per-held-out-unit lppd and its Monte-Carlo bias gap.

    lppd_u = log (1/S) sum_s (1/R) sum_r p(y_u | t_{s,r}, theta_s), with
    theta_s S global posterior draws from the TRAIN fit and t_{s,r} ~
    p(t | theta_s) drawn via `Predictive(posterior_samples=...)` (conditional
    prior — correct for centered and non-centered parameterizations alike).
    p(y_u | ...) is the unit's JOINT likelihood (trailing obs axes summed
    before any log-mean-exp).

    Returns ``(lppd, mc_gap)``, both shape (n_units,): ``mc_gap`` is the
    half-R jackknife estimate of the remaining log-mean-exp bias
    (lppd at R minus lppd at R/2; ~c/R, >= 0 in expectation). Zero when the
    model has none of the spec's local latents (then R is forced to 1 and the
    score is an exact plug-in over the global posterior).
    """
    obs_key = spec.obs_data_key
    if obs_key not in test_data:
        raise KeyError(f"obs data kwarg {obs_key!r} not in data")
    freed = {**test_data, obs_key: None}

    with _restored_x64():
        model = _exec_model(model_code)
        latent_sites, observed_sites = _model_sample_sites(model, freed, seed)
        if spec.obs_site not in latent_sites and spec.obs_site not in observed_sites:
            raise ValueError(f"obs site {spec.obs_site!r} not found in model trace")

        local = [nm for nm in spec.local_latents if nm in latent_sites]
        gnames = [v for v in train_idata.posterior.data_vars
                  if v in latent_sites and v not in spec.local_latents]
        gdraws = _flat_global_draws(train_idata, gnames, s_use, seed)
        S = next(iter(gdraws.values())).shape[0] if gdraws else 1

        # obs was freed for the trace, so it shows up latent — not "missing"
        missing = latent_sites - set(gnames) - set(local) - {spec.obs_site}
        if missing:
            raise ValueError(
                f"latent sample site(s) {sorted(missing)} neither in the train "
                "posterior nor declared as local latents — cannot score")

        if local:
            R = int(max(2, r_prior))
            params = {nm: np.repeat(a, R, axis=0) for nm, a in gdraws.items()}
            cond = Predictive(model, posterior_samples=params, return_sites=local)
            draws = cond(jax.random.PRNGKey(seed + 7), **freed)
            for lat in local:
                params[lat] = np.asarray(draws[lat])
        else:
            R = 1
            params = gdraws

        ll = log_likelihood(model, params, **test_data)[spec.obs_site]
        ll = np.asarray(ll, dtype=float)
        if ll.shape[0] != S * R:
            raise ValueError(f"unexpected draw dim {ll.shape} (S={S}, R={R})")
        # (S*R, n_units, *items) -> per-unit JOINT log-likelihood (S, R, n_units)
        ll = ll.reshape(S, R, n_units, -1).sum(axis=-1)

        per_s = logsumexp(ll, axis=1) - np.log(R)        # marginalize local
        lppd = np.asarray(logsumexp(per_s, axis=0) - np.log(S), dtype=float)

        if R >= 2:
            R2 = R // 2
            per_s_h = logsumexp(ll[:, :R2], axis=1) - np.log(R2)
            lppd_h = np.asarray(logsumexp(per_s_h, axis=0) - np.log(S), dtype=float)
            mc_gap = lppd - lppd_h
        else:
            mc_gap = np.zeros_like(lppd)
        return lppd, mc_gap


# ---------------------------------------------------------------------------
# Small helpers to turn per-unit lppd pairs into the compare_models contract.
# ---------------------------------------------------------------------------
def _summarize(lppd_ref: np.ndarray, lppd_orig: np.ndarray) -> dict:
    """Per-unit log-score deltas -> {elpd_diff, se, significant, ...}.

    Log score: HIGHER = better, so delta = refined - original, POSITIVE favors
    refined. Reported in compare_models' ELPD-difference convention as well:
    `elpd_diff` NEGATIVE favors refined (= -sum delta), with `elpd_diff_se` its
    SE (sd of the paired per-unit deltas * sqrt(n), the az.compare `dse`
    convention on the sum scale).
    """
    delta = np.asarray(lppd_ref) - np.asarray(lppd_orig)     # (n_units,)
    n = delta.size
    total = float(delta.sum())
    se = float(delta.std(ddof=1) * np.sqrt(n)) if n > 1 else float("nan")
    significant = bool(total > 0 and np.isfinite(se) and se > 0 and total >= 2 * se)
    return {
        "n_units": int(n),
        "lppd_delta": total,            # >0 favors refined
        "lppd_delta_se": se,
        "per_unit_delta_mean": float(delta.mean()) if n else float("nan"),
        "significant_improvement": significant,
        # compare_models contract (negative elpd_diff favors refined):
        "elpd_diff": -total,
        "elpd_diff_se": se,
        "loo_delta": -2.0 * total,      # deviance scale, negative favors refined
        "khat_refined": None,           # no importance sampling used
        # pointwise paired deltas (>0 favors refined) — evidence for the
        # concentration hint (localize_misspecification) and for reports:
        "per_unit_delta": [float(d) for d in delta],
    }


def _mc_reliability(res: dict, gap_ref: float, gap_orig: float) -> None:
    """Annotate `res` with the Monte-Carlo marginalization reliability verdict.

    The decision margin is 2*SE; the log-mean-exp bias is model-asymmetric
    (smaller for the model whose conditional prior hugs the data — usually the
    refinement), so unchecked it pushes toward FALSE ACCEPTS. We require the
    two models' remaining-bias estimates to sum below SE/2 (i.e. <= 25% of the
    margin); above that the comparison is flagged for R-doubling / demotion.
    """
    se = res.get("lppd_delta_se")
    tot = abs(float(gap_ref)) + abs(float(gap_orig))
    if se is None or not np.isfinite(se) or se <= 0:
        unreliable = bool(tot > 1e-6)
    else:
        unreliable = bool(tot > 0.5 * se)
    res["mc_gap_refined"] = float(gap_ref)
    res["mc_gap_original"] = float(gap_orig)
    res["mc_unreliable"] = unreliable


def _fit_fold(code: str, data: dict, budget, seed: int, cache_dir,
              target_accept: float, retries: int = 1):
    """One fold fit with adaptive convergence retry.

    R-hat at small fold budgets is noisy around the 1.01 bar; a value of 1.012
    is as often sampling noise as real non-mixing. Rather than loosen the bar
    (unsound) or fail folds on noise (needless demotion), refit once with a
    doubled budget when R-hat misses, and gate on the fit actually used.
    Returns (idata, diagnostics).
    """
    warmup, samples, chains = budget
    _, idata = run_mcmc(code, data, "numpyro", warmup, samples, chains,
                        seed=seed, cache_dir=cache_dir, target_accept=target_accept)
    diag = compute_diagnostics(idata)
    while not (diag.get("rhat_max", float("nan")) < RHAT_THRESHOLD) and retries > 0:
        retries -= 1
        warmup, samples = warmup * 2, samples * 2
        _, idata = run_mcmc(code, data, "numpyro", warmup, samples, chains,
                            seed=seed, cache_dir=cache_dir,
                            target_accept=target_accept)
        diag = compute_diagnostics(idata)
    return idata, diag


def _fold_reliability(res: dict, diags: list[dict]) -> None:
    """Annotate `res` with the worst fold-fit convergence over BOTH models.

    A non-mixing fold fit biases its fold's lppd unpredictably; R-hat above the
    validator's own threshold (or undefined) marks the whole comparison
    non-certifiable. Divergences are recorded but not blocking here: the fold
    budget's target_accept=0.9 already leans conservative, and the full-data
    fits are divergence-gated separately by the validator.
    """
    if not diags:
        res["cv_fit_unreliable"] = False
        return
    rhats = [d.get("rhat_max", float("nan")) for d in diags]
    worst = float(np.nanmax(rhats)) if np.any(np.isfinite(rhats)) else float("nan")
    res["fold_rhat_worst"] = worst
    res["fold_divergences_total"] = int(sum(d.get("divergences", 0) for d in diags))
    res["cv_fit_unreliable"] = bool(not (worst < RHAT_THRESHOLD))


# ---------------------------------------------------------------------------
# Scheme 1 — entry masking (no marginalization; needs a mask-aware model).
# ---------------------------------------------------------------------------
def entry_mask_score(refined_code: str, original_code: str, data: dict, *,
                     obs_key: str = "y", obs_site: str | None = None,
                     mask_key: str = "mask", frac: float = 0.2,
                     seed: int = 0, budget=(400, 800, 2), cache_dir=None) -> dict:
    """Hold out a random subset of the (unit x item) matrix entries.

    The model must accept a boolean `mask_key` array (True = observed in
    training) and apply it via numpyro.handlers.mask. Each unit keeps most of
    its items, so its local latent stays informed and held-out entries are
    scored with the FULL posterior — no marginalization. Held-out entry
    log-scores are summed WITHIN each unit (axis-0 row) before the SE: entries
    sharing a unit share its latent and are not independent evidence.

    ``obs_key`` is the DATA kwarg holding the obs matrix; ``obs_site`` is the
    observation SITE name (defaults to ``obs_key``). numpyro keys
    ``log_likelihood`` by site name, so the two must be told apart (defect V7:
    indexing the log-likelihood by the data kwarg raised KeyError for every
    model whose site is not named after its kwarg, e.g. site "obs" fed by
    ``y=``). `crossval_compare` passes both from the CVSpec.
    """
    obs_site = obs_key if obs_site is None else obs_site
    y = np.asarray(data[obs_key])
    if y.ndim < 2:
        raise ValueError("entry_mask_score needs a (units x items) obs matrix")
    rng = np.random.default_rng(seed)
    held = rng.uniform(size=y.shape) < frac
    train_mask = ~held                              # True where kept for training
    score_mask = np.ones_like(y, dtype=bool)        # score everywhere, select held below

    diags: list[dict] = []

    def _fit_and_score(code):
        train_data = {**data, mask_key: train_mask}
        idata, diag = _fit_fold(code, train_data, budget, seed, cache_dir,
                                target_accept=0.9)
        diags.append(diag)
        with _restored_x64():
            model = _exec_model(code)
            latent_sites, _ = _model_sample_sites(
                model, {**data, mask_key: train_mask}, seed)
            latent = {nm: np.asarray(idata.posterior[nm].values).reshape(
                        (-1,) + idata.posterior[nm].values.shape[2:])
                      for nm in idata.posterior.data_vars if nm in latent_sites}
            score_data = {**data, mask_key: score_mask}
            ll = log_likelihood(model, latent, **score_data)[obs_site]  # (S, *y.shape)
        ll = np.asarray(ll, dtype=float)
        S = ll.shape[0]
        lppd_entry = logsumexp(ll.reshape(S, -1), axis=0) - np.log(S)  # (n_entries,)
        per_entry = np.asarray(lppd_entry, dtype=float).reshape(y.shape)
        # sum held-out entries within each unit (axis-0 row); drop empty units
        unit_has_held = held.reshape(y.shape[0], -1).any(axis=1)
        per_unit = np.where(held, per_entry, 0.0).reshape(y.shape[0], -1).sum(axis=1)
        return per_unit[unit_has_held]

    res = _summarize(_fit_and_score(refined_code), _fit_and_score(original_code))
    _mc_reliability(res, 0.0, 0.0)                  # full-posterior scoring: no MC gap
    _fold_reliability(res, diags)
    res["scheme"] = f"entry_mask(frac={frac})"
    res["n_held_out_entries"] = int(held.sum())
    return res


# ---------------------------------------------------------------------------
# Scheme 2 — single held-out-unit split (marginalized latent).
# ---------------------------------------------------------------------------
def heldout_unit_score(refined_code: str, original_code: str, data: dict, spec: CVSpec,
                       *, split_frac: float = 0.25, seed: int = 0,
                       budget=(400, 800, 2), s_use: int = 300, r_prior: int = 60,
                       cache_dir=None) -> dict:
    """One train/test split over units; held-out units' latents marginalized."""
    n = _n_units(data, spec)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(split_frac * n)))
    test_idx, train_idx = np.sort(idx[:n_test]), np.sort(idx[n_test:])
    train = _slice_units(data, train_idx, spec)
    test = _slice_units(data, test_idx, spec)

    diags: list[dict] = []

    def _score(code):
        idata, diag = _fit_fold(code, train, budget, seed, cache_dir,
                                target_accept=0.9)
        diags.append(diag)
        return held_out_unit_lppd(code, idata, test, spec, n_units=len(test_idx),
                                  s_use=s_use, r_prior=r_prior, seed=seed)

    lppd_ref, gap_ref = _score(refined_code)
    lppd_orig, gap_orig = _score(original_code)
    res = _summarize(lppd_ref, lppd_orig)
    _mc_reliability(res, float(gap_ref.sum()), float(gap_orig.sum()))
    _fold_reliability(res, diags)
    res["scheme"] = f"heldout_unit(split={split_frac})"
    res["n_train"], res["n_test"] = int(len(train_idx)), int(len(test_idx))
    return res


# ---------------------------------------------------------------------------
# Scheme 3 — K-fold over units (arbiter-grade pointwise ELPD).
# ---------------------------------------------------------------------------
def kfold_unit_elpd(refined_code: str, original_code: str, data: dict, spec: CVSpec,
                    *, K: int = 5, seed: int = 0, budget=(400, 800, 2),
                    s_use: int = 300, r_prior: int = 60, cache_dir=None) -> dict:
    """K-fold over units; accumulate a pointwise lppd for every unit.

    Both models are fit on the SAME folds with the SAME per-fold seeds, so the
    per-unit deltas are paired and fold-level fit noise largely cancels in the
    difference the gate tests.
    """
    n = _n_units(data, spec)
    K = int(min(K, n))
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), K)
    lppd_ref = np.full(n, np.nan)
    lppd_orig = np.full(n, np.nan)
    gap_ref = 0.0
    gap_orig = 0.0
    diags: list[dict] = []

    def _score(code, train, test, test_idx, fold_seed):
        idata, diag = _fit_fold(code, train, budget, fold_seed, cache_dir,
                                target_accept=0.9)
        diags.append(diag)
        return held_out_unit_lppd(code, idata, test, spec, n_units=len(test_idx),
                                  s_use=s_use, r_prior=r_prior, seed=fold_seed)

    for f, test_idx in enumerate(folds):
        if len(test_idx) == 0:
            continue
        test_idx = np.sort(test_idx)
        train_idx = np.sort(np.setdiff1d(np.arange(n), test_idx))
        train = _slice_units(data, train_idx, spec)
        test = _slice_units(data, test_idx, spec)

        lr, gr = _score(refined_code, train, test, test_idx, seed + f)
        lo, go = _score(original_code, train, test, test_idx, seed + f)
        lppd_ref[test_idx] = lr
        lppd_orig[test_idx] = lo
        gap_ref += float(gr.sum())
        gap_orig += float(go.sum())

    res = _summarize(lppd_ref, lppd_orig)
    _mc_reliability(res, gap_ref, gap_orig)
    _fold_reliability(res, diags)
    res["scheme"] = f"kfold_unit(K={K})"
    return res


# ---------------------------------------------------------------------------
# Automatic CVSpec inference from the program's plate structure.
# ---------------------------------------------------------------------------
def _obs_data_key(obs_name: str, obs_value, data: dict) -> str | None:
    """Data kwarg feeding the observed site: its name when that is a data key,
    else the first array entry equal to the traced value at the coarser
    precision (F1); None when nothing matches."""
    if obs_name in data:
        return obs_name
    value = np.asarray(obs_value)
    for k, v in data.items():
        if isinstance(v, np.ndarray | list | tuple) and _same_observed_values(np.asarray(v), value)[0]:
            return k
    return None


def infer_cv_spec(refined_code: str, data: dict, original_code: str | None = None,
                  *, seed: int = 0, min_units: int = 8) -> dict | None:
    """Derive a CVSpec from the refined model's plate structure, or None.

    Reads the numpyro trace: the exchangeable unit is a plate shared (by name,
    else by size) between the observed site and at least one latent sample
    site — exactly the structure that breaks PSIS-LOO. The result is
    dry-run-validated by slicing a 2-unit fold and tracing the model(s) on it;
    any failure returns None so the caller keeps the conservative PSIS block.
    Returns {"spec": CVSpec, "plate": name, "n_units": n} on success.

    The data kwarg feeding an observed site whose name is not a data key is
    found by value at the coarser precision (``_same_observed_values``, F1):
    a program that casts its response with ``jnp.asarray`` observes a float32
    copy, which exact equality never matched.
    """
    try:
        with _restored_x64():
            model = _exec_model(refined_code)
            with handlers.trace() as tr:
                handlers.seed(model, jax.random.PRNGKey(seed))(**data)

        obs_sites = [(n, s) for n, s in tr.items()
                     if s["type"] == "sample" and s.get("is_observed")]
        if len(obs_sites) != 1:
            return None                      # ambiguous/absent observation site
        obs_name, obs_site = obs_sites[0]
        obs_frames = list(obs_site.get("cond_indep_stack", []))
        if not obs_frames:
            return None

        latents = {n: list(s.get("cond_indep_stack", [])) for n, s in tr.items()
                   if s["type"] == "sample" and not s.get("is_observed")}

        # unit plate: obs frame shared with >=1 latent — by NAME, else by SIZE.
        def _candidates(match):
            out = []
            for f in obs_frames:
                shared = [n for n, fr in latents.items()
                          if any(match(f, g) for g in fr)]
                if shared:
                    out.append((f, shared))
            return out

        cands = _candidates(lambda a, b: a.name == b.name)
        if not cands:
            cands = _candidates(lambda a, b: a.size == b.size)
        if len(cands) != 1:
            return None                      # none, or ambiguous across plates
        frame, local = cands[0]
        n = int(frame.size)
        if n < min_units:
            return None

        # the scoring primitive assumes the obs UNIT axis is axis 0
        obs_arr_key = _obs_data_key(obs_name, obs_site["value"], data)
        if obs_arr_key is None:
            return None
        obs_ndim = np.asarray(data[obs_arr_key]).ndim
        if obs_ndim + int(frame.dim) != 0:
            return None                      # unit plate is not obs axis 0

        # per-unit data arrays: any array with a size-n axis (first match)
        per_unit, axes, size_keys = [], {}, []
        for k, v in data.items():
            if isinstance(v, np.ndarray | list | tuple):
                shape = np.asarray(v).shape
                if n in shape:
                    per_unit.append(k)
                    axes[k] = int(shape.index(n))
            elif isinstance(v, int | np.integer) and int(v) == n:
                size_keys.append(k)

        if obs_arr_key not in per_unit or axes.get(obs_arr_key) != 0:
            return None

        spec = CVSpec(
            per_unit_keys=per_unit,
            local_latents=sorted(local),
            obs_site=obs_name,
            obs_kwarg=obs_arr_key,
            unit_size_keys=size_keys,
            per_unit_axes=axes,
            plate=str(frame.name),
        )

        # dry-run: a 2-unit fold must slice and trace cleanly on both models
        fold = _slice_units(data, np.arange(min(2, n)), spec)
        for code in filter(None, [refined_code, original_code]):
            with _restored_x64():
                m = _exec_model(code)
                with handlers.trace() as tr2:
                    handlers.seed(m, jax.random.PRNGKey(seed + 1))(**fold)
                v = np.asarray(tr2[obs_name]["value"])
                if v.shape[0] != min(2, n):
                    return None
        return {"spec": spec, "plate": str(frame.name), "n_units": n}
    except Exception:
        return None                          # never escalate on a broken inference


# ---------------------------------------------------------------------------
# Second escalation rung — exact refits for flagged points (reloo).
# For models WITHOUT local latents, a Pareto k-hat > 0.7 usually means a few
# highly influential observations, not a broken estimand: the canonical remedy
# is to compute those points' leave-one-out elpd EXACTLY (refit without the
# point, score it out of sample) and keep PSIS for the well-behaved rest.
# ---------------------------------------------------------------------------
def infer_obs_spec(model_code: str, data: dict, *, seed: int = 0) -> dict | None:
    """Derive a per-OBSERVATION slicing spec for exact leave-one-out refits.

    Same trace introspection as `infer_cv_spec`, but the "unit" is the obs
    plate itself and there must be NO latent sample site tied to it (that case
    belongs to the CV rung — a per-obs latent makes single-point refits just as
    hopeless as PSIS). Requires a single observed site with a 1-D data array
    (so az.loo's pointwise order equals the slicing order); the array is the
    data key named after the site, else the entry equal to the traced value at
    the coarser precision (F1). Dry-run-validated; None on any doubt.
    """
    try:
        with _restored_x64():
            model = _exec_model(model_code)
            with handlers.trace() as tr:
                handlers.seed(model, jax.random.PRNGKey(seed))(**data)

        obs_sites = [(n, s) for n, s in tr.items()
                     if s["type"] == "sample" and s.get("is_observed")]
        if len(obs_sites) != 1:
            return None
        obs_name, obs_site = obs_sites[0]
        obs_frames = list(obs_site.get("cond_indep_stack", []))
        if not obs_frames:
            return None
        latents = {n: list(s.get("cond_indep_stack", [])) for n, s in tr.items()
                   if s["type"] == "sample" and not s.get("is_observed")}

        obs_arr_key = _obs_data_key(obs_name, obs_site["value"], data)
        if obs_arr_key is None:
            return None
        obs_arr = np.asarray(data[obs_arr_key])
        if obs_arr.ndim != 1:
            return None                       # pointwise order must equal row order
        n = int(obs_arr.shape[0])
        frame = next((f for f in obs_frames if int(f.size) == n), None)
        if frame is None or obs_arr.ndim + int(frame.dim) != 0:
            return None
        # a latent tied to the obs plate (name or size) -> the CV rung's case
        for fr_list in latents.values():
            if any(g.name == frame.name or int(g.size) == n for g in fr_list):
                return None

        per_unit, axes, size_keys = [], {}, []
        for k, v in data.items():
            if isinstance(v, np.ndarray | list | tuple):
                shape = np.asarray(v).shape
                if n in shape:
                    per_unit.append(k)
                    axes[k] = int(shape.index(n))
            elif isinstance(v, int | np.integer) and int(v) == n:
                size_keys.append(k)
        if obs_arr_key not in per_unit or axes.get(obs_arr_key) != 0:
            return None

        spec = CVSpec(per_unit_keys=per_unit, local_latents=[],
                      obs_site=obs_name, obs_kwarg=obs_arr_key,
                      unit_size_keys=size_keys, per_unit_axes=axes,
                      plate=str(frame.name))
        fold = _slice_units(data, np.arange(min(2, n)), spec)
        with _restored_x64():
            m = _exec_model(model_code)
            with handlers.trace() as tr2:
                handlers.seed(m, jax.random.PRNGKey(seed + 1))(**fold)
            if np.asarray(tr2[obs_name]["value"]).shape[0] != min(2, n):
                return None
        return {"spec": spec, "plate": str(frame.name), "n_units": n}
    except Exception:
        return None


def reloo_refined(refined_code: str, original_code: str, data: dict,
                  refined_idata, original_idata, *, budget=(400, 800, 2),
                  seed: int = 0, cache_dir=None, khat_threshold: float = 0.7,
                  max_refits: int = 8) -> dict | None:
    """PSIS-LOO with the refined model's flagged points computed exactly.

    Replaces the flagged points' elpd contributions with genuine held-out
    values (refit without the point, plug-in log predictive; the exact LOO
    estimand PSIS was approximating) and re-tests the same paired 2*SE margin
    over ALL points. Only the REFINED side is corrected: high k-hat on the
    original inflates the baseline, which biases AGAINST acceptance and stays
    conservative. Refit convergence is gated like a CV fold (retry once at
    doubled budget, then demote). Returns a compare_models-shaped dict, or
    None when this rung does not apply (too many flagged points, latent tied
    to the obs plate, non-1-D obs, underivable slicing).

    Positions where BOTH fits' pointwise log-likelihood is the same constant —
    the missing cells of a shared data mask (F2) — carry a Pareto k-hat of inf
    by construction and no information: they are neither flagged for a refit
    nor counted in the reported k-hat values (``shared_degenerate_columns``).
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo_ref = az.loo(refined_idata, pointwise=True, scale="log")
        loo_orig = az.loo(original_idata, pointwise=True, scale="log")
    k = np.asarray(loo_ref.pareto_k, dtype=float).ravel()
    k_orig = np.asarray(loo_orig.pareto_k, dtype=float).ravel()
    exempt = shared_degenerate_columns(refined_idata, original_idata)
    if exempt is not None and exempt.shape == k.shape:
        k = np.where(exempt, 0.0, k)          # shared missing cells: no information
        if k_orig.shape == exempt.shape:
            k_orig = np.where(exempt, 0.0, k_orig)
    flagged = np.where(k > khat_threshold)[0]
    if flagged.size == 0 or flagged.size > max_refits:
        return None
    info = infer_obs_spec(refined_code, data, seed=seed)
    if info is None or info["n_units"] != k.size:
        return None
    spec, n = info["spec"], info["n_units"]
    ref_i = np.asarray(loo_ref.loo_i, dtype=float).ravel().copy()
    orig_i = np.asarray(loo_orig.loo_i, dtype=float).ravel()
    if ref_i.size != n or orig_i.size != n:
        return None

    diags: list[dict] = []
    for i in flagged:
        keep = np.delete(np.arange(n), int(i))
        train = _slice_units(data, keep, spec)
        test = _slice_units(data, np.array([int(i)]), spec)
        idata_i, diag = _fit_fold(refined_code, train, budget,
                                  seed + 1000 + int(i), cache_dir, 0.9)
        diags.append(diag)
        lppd_i, _ = held_out_unit_lppd(refined_code, idata_i, test, spec,
                                       n_units=1, seed=seed)
        ref_i[int(i)] = float(lppd_i[0])

    res = _summarize(ref_i, orig_i)
    _mc_reliability(res, 0.0, 0.0)            # plug-in scoring: no MC gap
    _fold_reliability(res, diags)
    res["scheme"] = f"psis+reloo({flagged.size} exact refit{'s' if flagged.size > 1 else ''})"
    note = None
    if res.get("cv_fit_unreliable"):
        res["significant_improvement"] = False
        note = (f"exact-refit (reloo) fit unreliable (worst R-hat "
                f"{res.get('fold_rhat_worst', float('nan')):.3f}); "
                "comparison not certifiable")
    residual_khat = float(np.max(np.delete(k, flagged))) if k.size > flagged.size else 0.0
    return {
        "waic": float("nan"), "loo": float(ref_i.sum()),
        "waic_delta": res["loo_delta"], "loo_delta": res["loo_delta"],
        "elpd_diff": res["elpd_diff"], "elpd_diff_se": res["elpd_diff_se"],
        "significant_improvement": res["significant_improvement"],
        # flagged points are now exact; the residual (well-behaved) k-hat is
        # what remains subject to the PSIS gate — below threshold by
        # construction, reported honestly rather than as None.
        "khat_max": float(np.nanmax(k_orig.tolist() + [residual_khat])),
        "khat_refined": residual_khat,
        "khat_original": float(np.nanmax(k_orig)),
        "s_gate": res["scheme"],
        "reliability_note": note,
        "_crossval": res,
    }


# ---------------------------------------------------------------------------
# Adapter — run a scheme and return compare_models' dict (drop-in for the S gate).
# ---------------------------------------------------------------------------
def crossval_compare(refined_code: str, original_code: str, data: dict, spec: CVSpec,
                     *, scheme: str = "kfold", K: int = 5, split_frac: float = 0.25,
                     budget=(400, 800, 2), s_use: int = 300, r_prior: int = 60,
                     seed: int = 0, cache_dir=None) -> dict:
    """Cross-validated comparison, shaped exactly like `compare_models`' return
    so `run_full_validation` can consume it with no other change. No PSIS, so
    every khat field is None (the PSIS reliability gate is skipped upstream —
    this scheme carries its OWN reliability guards instead):

    * fold-fit convergence (R-hat over every fold fit of both models), and
    * marginalization Monte-Carlo error (half-R jackknife gap vs the SE), with
      R doubled (up to 8x) until reliable — cheap when `cache_dir` is set,
      because the fold MCMC re-runs are trace-cache hits and only the scoring
      recomputes.

    A comparison that stays unreliable is DEMOTED: significant_improvement is
    forced False and `reliability_note` says why (conservative, mirroring the
    PSIS k-hat gate's direction).
    """
    def _run(r_p):
        if scheme == "kfold":
            return kfold_unit_elpd(refined_code, original_code, data, spec, K=K,
                                   seed=seed, budget=budget, s_use=s_use,
                                   r_prior=r_p, cache_dir=cache_dir)
        if scheme == "split":
            return heldout_unit_score(refined_code, original_code, data, spec,
                                      split_frac=split_frac, seed=seed, budget=budget,
                                      s_use=s_use, r_prior=r_p, cache_dir=cache_dir)
        if scheme == "entry_mask":
            return entry_mask_score(refined_code, original_code, data, seed=seed,
                                    budget=budget, cache_dir=cache_dir,
                                    obs_key=spec.obs_data_key, obs_site=spec.obs_site)
        raise ValueError(f"unknown crossval scheme: {scheme!r}")

    r_used = int(r_prior)
    r = _run(r_used)
    note = None
    if r.get("cv_fit_unreliable"):
        r["significant_improvement"] = False
        note = (f"CV fold fit unreliable (worst R-hat "
                f"{r.get('fold_rhat_worst', float('nan')):.3f}); "
                "comparison not certifiable")
    else:
        while r.get("mc_unreliable") and r_used < int(r_prior) * 8:
            r_used *= 2
            r = _run(r_used)
        if r.get("mc_unreliable"):
            r["significant_improvement"] = False
            note = (f"CV marginalization Monte-Carlo error too large at "
                    f"r_prior={r_used} (bias gaps {r.get('mc_gap_refined', 0.0):.2f} + "
                    f"{r.get('mc_gap_original', 0.0):.2f} vs SE "
                    f"{r.get('lppd_delta_se', float('nan')):.2f}); "
                    "comparison not certifiable")
    r["r_prior_used"] = r_used

    return {
        "waic": float("nan"), "loo": float("nan"),
        "waic_delta": r["loo_delta"], "loo_delta": r["loo_delta"],
        "elpd_diff": r["elpd_diff"], "elpd_diff_se": r["elpd_diff_se"],
        "significant_improvement": r["significant_improvement"],
        "khat_max": None, "khat_refined": None, "khat_original": None,
        "s_gate": r["scheme"],
        "reliability_note": note,
        "_crossval": r,
    }


__all__ = [
    "CVSpec", "held_out_unit_lppd", "entry_mask_score",
    "heldout_unit_score", "kfold_unit_elpd", "crossval_compare",
    "infer_cv_spec",
]
