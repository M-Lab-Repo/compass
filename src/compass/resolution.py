"""resolution.py — design-resolution introspection for the strategist loop.

Everything here is TRACE-ONLY (no MCMC, no LLM) and ABSTAINING: any doubt
returns None / tie="free" rather than a guess, because downstream consumers
steer prompts and reports — a wrong structural claim is worse than none.

Three layers (v4 plan, Components 0 + 1 + 1b):

* `latent_structure` — the shared classifier: which latent sites are local,
  tied to what plate, with what replication. Ties:
    - "obs-plate":  the latent's plate is one of the observation site's own
                    plates (matched by name, else size) — one latent per
                    observation row; the PSIS-risk case.
    - "indexed":    an integer data array of length n_obs with values in
                    [0, J) and max+1 == J indexes the latent's plate — the
                    classic group-effect structure (8-schools style, where
                    the obs site often carries NO plate at all).
    - "free":       everything else — including plates over predictors
                    (coefficient vectors), which must NEVER be treated as
                    local latents.
* `audit_resolution` — the resolution ladder read off the design: per latent
  plate J and M, which estimands the design identifies, prior-sensitivity at
  small J, PSIS risk, and a cheap OLS within/between (Mundlak) contrast per
  covariate when a group index exists.
* `format_resolution_block` — the deterministic DESIGN RESOLUTION text for
  the strategist prompt and the report, governed by `hint_level`:
    - "none":            no block at all.
    - "structure":       audit facts only.
    - "structure+ladder" (default): audit facts + the failure-mode LADDER for
      the detected structure class. HONESTY DISCIPLINE: the ladder text is
      conditioned only on STRUCTURE (every model with the same plates gets
      the same words — the instance's data never select the message) and is
      phrased at the failure-MODE level, never as model edits; evidence-
      triggered fix-class hints remain the job of localize_misspecification.
"""

from __future__ import annotations

import numpy as np

from compass.agents.validator import _exec_model, _restored_x64


def _trace_sites(model_code: str, data: dict, seed: int):
    import jax
    from numpyro import handlers

    with _restored_x64():
        model = _exec_model(model_code)
        with handlers.trace() as tr:
            handlers.seed(model, jax.random.PRNGKey(seed))(**data)
    return tr


def _is_int_array(v) -> bool:
    try:
        a = np.asarray(v)
    except Exception:
        return False
    return a.ndim >= 1 and np.issubdtype(a.dtype, np.integer)


# ---------------------------------------------------------------------------
# Component 0 — the classifier
# ---------------------------------------------------------------------------
def latent_structure(model_code: str, data: dict, *, seed: int = 0) -> dict | None:
    """Classify the model's latent anatomy from one prior trace. None on doubt."""
    try:
        tr = _trace_sites(model_code, data, seed)

        obs_sites = [(n, s) for n, s in tr.items()
                     if s["type"] == "sample" and s.get("is_observed")]
        if len(obs_sites) != 1:
            return None
        obs_name, obs_site = obs_sites[0]
        obs_val = np.asarray(obs_site["value"])
        obs_frames = [(str(f.name), int(f.dim), int(f.size))
                      for f in obs_site.get("cond_indep_stack", [])]
        obs_kwarg = obs_name if obs_name in data else None
        if obs_kwarg is None:
            for k, v in data.items():
                try:
                    a = np.asarray(v)
                except Exception:
                    continue
                if a.shape == obs_val.shape and np.array_equal(a, obs_val):
                    obs_kwarg = k
                    break
        n_obs = int(obs_val.shape[0]) if obs_val.ndim else 1

        latents = {n: s for n, s in tr.items()
                   if s["type"] == "sample" and not s.get("is_observed")}
        globals_ = sorted(n for n, s in latents.items()
                          if not s.get("cond_indep_stack"))

        # candidate group-index arrays: int, length n_obs, min>=0
        index_arrays = {}
        for k, v in data.items():
            if _is_int_array(v):
                a = np.asarray(v)
                if a.ndim == 1 and a.shape[0] == n_obs and a.size and a.min() >= 0:
                    index_arrays[k] = int(a.max()) + 1

        obs_frame_names = {nm for nm, _, _ in obs_frames}
        obs_frame_sizes = {sz for _, _, sz in obs_frames}

        plates: dict[tuple, dict] = {}
        for name, site in latents.items():
            for f in site.get("cond_indep_stack", []):
                key = (str(f.name), int(f.size))
                plates.setdefault(key, {"plate": key[0], "size": key[1],
                                        "sites": []})["sites"].append(name)

        latent_plates = []
        for (pname, psize), info in sorted(plates.items()):
            tie, index_key = "free", None
            m_min = m_median = float("nan")
            if pname in obs_frame_names or psize in obs_frame_sizes:
                tie = "obs-plate"
                per_unit = obs_val.size / psize if psize else float("nan")
                m_min = m_median = float(per_unit)
            else:
                for k, j in index_arrays.items():
                    if j == psize:
                        tie, index_key = "indexed", k
                        counts = np.bincount(np.asarray(data[k]).astype(int),
                                             minlength=psize)
                        m_min = float(counts.min())
                        m_median = float(np.median(counts))
                        break
            info.update(tie=tie, index_key=index_key,
                        m_min=m_min, m_median=m_median,
                        sites=sorted(info["sites"]))
            latent_plates.append(info)

        return {
            "obs": {"site": obs_name, "kwarg": obs_kwarg, "n": n_obs,
                    "size": int(obs_val.size), "ndim": int(obs_val.ndim),
                    "frames": obs_frames},
            "latent_plates": latent_plates,
            "globals": globals_,
        }
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Component 1 — the resolution audit (the identifiability ladder, read off
# the design) + a cheap OLS within/between (Mundlak) contrast per covariate.
# ---------------------------------------------------------------------------
def _ols_slope(x: np.ndarray, y: np.ndarray, df: int) -> tuple[float, float] | None:
    """Slope + SE of y ~ 1 + x with the given residual df. None if degenerate."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < df + 2:
        return None
    xc, yc = x - x.mean(), y - y.mean()
    sxx = float(xc @ xc)
    if sxx <= 0 or df <= 0:
        return None
    b = float(xc @ yc) / sxx
    resid = yc - b * xc
    s2 = float(resid @ resid) / df
    return b, float(np.sqrt(s2 / sxx))


def _mundlak_contrast(y: np.ndarray, x: np.ndarray, idx: np.ndarray, J: int) -> dict | None:
    """Descriptive within- vs between-group slope contrast (no MCMC).

    between: group-mean y on group-mean x (J points, df J-2);
    within:  group-demeaned y on group-demeaned x (n points, df n-J-1).
    A large gap is the classic signature of latent-covariate dependence
    (Mundlak 1978); reported as evidence, never a gate.
    """
    y = np.asarray(y, float).ravel()
    x = np.asarray(x, float).ravel()
    idx = np.asarray(idx).astype(int)
    if y.shape != x.shape or y.shape != idx.shape or J < 3:
        return None
    xbar = np.array([x[idx == g].mean() if np.any(idx == g) else np.nan
                     for g in range(J)])
    ybar = np.array([y[idx == g].mean() if np.any(idx == g) else np.nan
                     for g in range(J)])
    ok = np.isfinite(xbar) & np.isfinite(ybar)
    between = _ols_slope(xbar[ok], ybar[ok], df=int(ok.sum()) - 2)
    within = _ols_slope(x - xbar[idx], y - ybar[idx], df=y.size - J - 1)
    if between is None or within is None:
        return None
    b_b, se_b = between
    b_w, se_w = within
    denom = float(np.sqrt(se_b ** 2 + se_w ** 2))
    z = abs(b_w - b_b) / denom if denom > 0 else float("inf")
    total_var = float(np.var(x))
    within_frac = float(np.var(x - xbar[idx]) / total_var) if total_var > 0 else 0.0
    return {"between": b_b, "between_se": se_b, "within": b_w, "within_se": se_w,
            "z": float(z), "conflict": bool(z > 2.0), "within_var_frac": within_frac}


def audit_resolution(model_code: str, data: dict, *, seed: int = 0) -> dict | None:
    """The design's resolution ladder: which estimands this data CAN identify."""
    struct = latent_structure(model_code, data, seed=seed)
    if struct is None:
        return None
    tied = [p for p in struct["latent_plates"] if p["tie"] in ("obs-plate", "indexed")]
    plates = []
    for p in tied:
        j = p["size"]
        law = "none" if j < 8 else ("moments" if j < 50 else "shape")
        plates.append({**p, "law_resolution": law, "prior_sensitive": bool(j < 8)})

    mundlak: dict[str, dict] = {}
    idx_plate = next((p for p in tied if p["tie"] == "indexed"), None)
    obs_kwarg = struct["obs"]["kwarg"]
    if idx_plate is not None and obs_kwarg is not None and struct["obs"]["ndim"] == 1:
        idx = data[idx_plate["index_key"]]
        yarr = np.asarray(data[obs_kwarg], float)
        for k, v in data.items():
            if k in (obs_kwarg, idx_plate["index_key"]):
                continue
            try:
                a = np.asarray(v)
            except Exception:
                continue
            if (a.ndim == 1 and a.shape[0] == struct["obs"]["n"]
                    and np.issubdtype(a.dtype, np.floating)):
                c = _mundlak_contrast(yarr, a, idx, idx_plate["size"])
                if c is not None:
                    mundlak[k] = c

    return {
        "struct": struct,
        "n_obs": struct["obs"]["n"],
        "plates": plates,
        "has_local_latents": bool(plates),
        "psis_risk": any(p["tie"] == "obs-plate" for p in plates),
        "within_between_identified": any(p["m_median"] >= 2 for p in plates),
        "mundlak": mundlak,
    }


# ---------------------------------------------------------------------------
# Component 1b — the DESIGN RESOLUTION block (strategist prompt + report).
# ---------------------------------------------------------------------------
HINT_LEVELS = ("none", "structure", "structure+ladder")

_LADDER_OBS_PLATE = (
    "- per-observation latent structure — characteristic failure modes (symptom level):\n"
    "    tails/shape of the effect law · conditional-likelihood misfit ABSORBED by\n"
    "    the latents (marginal checks pass while the latents change meaning) ·\n"
    "    residual dependence beyond the latents")
_LADDER_INDEXED = (
    "- group-effect structure — characteristic failure modes (symptom level):\n"
    "    tails/shape of the group-effect law · a missing level or crossed factor ·\n"
    "    group-covariate dependence (within vs between effects) · informative group size")
_ANTI_SHOTGUN = (
    "The ladder is a search ORDER, not a bundle: propose the single most\n"
    "evidence-supported rung this iteration; later iterations walk the rest.")


def format_resolution_block(audit: dict | None, hint_level: str = "structure+ladder") -> str:
    """Deterministic DESIGN RESOLUTION text. '' when disabled or unavailable."""
    if hint_level not in HINT_LEVELS:
        raise ValueError(f"unknown hint_level {hint_level!r}")
    if hint_level == "none" or audit is None:
        return ""
    lines = ["DESIGN RESOLUTION (structural facts, computed before any sampling):",
             f"- observations: n={audit['n_obs']}"]
    for p in audit["plates"]:
        how = (f"indexed by `{p['index_key']}`" if p["tie"] == "indexed"
               else "one latent per observation row")
        m = (f"{p['m_min']:.0f}-{p['m_median']:.0f}" if p["m_min"] != p["m_median"]
             else f"{p['m_median']:.0f}")
        lines.append(f"- latent plate '{p['plate']}' ({how}): J={p['size']} units, "
                     f"replication ~{m} obs/unit; sites: {', '.join(p['sites'])}")
        lines.append(f"    latent-law resolution at J={p['size']}: "
                     f"{p['law_resolution'].upper()}"
                     + (" — shape claims are not adjudicable here"
                        if p["law_resolution"] != "shape" else ""))
        if p["prior_sensitive"]:
            lines.append(f"    J={p['size']} < 8: variance-component (tau-scale) claims "
                         "are prior-dominated — treat as unadjudicable")
    if audit["has_local_latents"]:
        lines.append("- within/between decomposition: "
                     + ("identified (replication >= 2)" if audit["within_between_identified"]
                        else "NOT identified (single observation per unit)"))
        if audit["psis_risk"]:
            lines.append("- PSIS-LOO is unreliable for this structure: the S gate will "
                         "escalate to K-fold CV — refinements are judged on NEW-UNIT "
                         "prediction")
    else:
        lines.append("- no local latent structure detected (flat / global-parameter model)")
    for k, c in sorted(audit["mundlak"].items()):
        verdict = ("CONFLICT — within and between effects disagree (latent-covariate "
                   "dependence)" if c["conflict"] else "no within/between conflict")
        lines.append(f"- covariate `{k}`: within-slope {c['within']:.3g}±{c['within_se']:.2g} "
                     f"vs between-slope {c['between']:.3g}±{c['between_se']:.2g} "
                     f"(z={c['z']:.1f}) — {verdict}")
    if hint_level == "structure+ladder" and audit["has_local_latents"]:
        lines.append("")
        lines.append("FAILURE-MODE LADDER for the detected structure "
                     "(symptom level — the fix is yours to argue from evidence):")
        if any(p["tie"] == "obs-plate" for p in audit["plates"]):
            lines.append(_LADDER_OBS_PLATE)
        if any(p["tie"] == "indexed" for p in audit["plates"]):
            lines.append(_LADDER_INDEXED)
        lines.append(_ANTI_SHOTGUN)
    lines.append(f"(hint_level={hint_level} — recorded in the report)")
    return "\n".join(lines)
