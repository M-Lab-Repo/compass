"""DGP registry — the misspecification-type grid for the paper's breadth study.

Each *type* is a curated, code-invisible misspecification (in the sense of
Calibration-Not-Compilation): an original model that compiles, runs, and
converges, yet is statistically wrong for the data. For every type we ship:

  * a TRUE data-generating process (numpy only; deterministic per seed),
  * the misspecified ``original`` model,
  * a ``gold`` reference refinement (the correct fix),
  * a ``decoy`` — a plausible-LOOKING refinement that a code reader tends to
    accept but that real MCMC rejects (either statistically neutered, e.g. a
    Negative-Binomial with dispersion pinned huge, or a cosmetic prior-widen
    that does not change the fit).

The eight types are all genuine *model-data* misspecifications with a real
predictive signal the MCMC validator credits as a win (overdispersion,
zero-inflation, heavy tails, binomial overdispersion, positive skew,
heteroscedasticity, missing nonlinearity, missing hierarchy). Pure-geometry
fixes (centered->non-centered funnel) live in the §9.A case studies instead,
because a reparameterisation has ~zero ELPD delta and so is not a "win" under
our tightened success criterion — an honest distinction the paper makes.

No LLM, no MCMC here: this module only generates data and holds code strings,
so it is fully unit-testable in milliseconds.
"""

from __future__ import annotations

import re

import numpy as np

# ---------------------------------------------------------------------------
# Recovery detection — did the pipeline's refined model land on the gold class?
# ---------------------------------------------------------------------------
# The robust, class-agnostic metric is "refinement-wins" = the MCMC validator
# accepted the refined model (validation_passed). On top of that we report a
# best-effort CLASS-recovery flag: for likelihood-swap types we read the actual
# ``y`` observation distribution out of the refined code; for structural types
# (heteroscedasticity, nonlinearity, hierarchy) we pattern-match the structure.

_Y_DIST_RE = re.compile(r"""sample\(\s*["']y["']\s*,\s*dist\.(\w+)""")
# two-step idiom: obs_dist = dist.X(...); numpyro.sample("y", obs_dist, obs=y)
_Y_VAR_RE = re.compile(r"""sample\(\s*["']y["']\s*,\s*(\w+)\s*[,)]""")


def y_likelihood(code: str) -> str:
    """The distribution class name used for the ``y`` observation, or "".

    Resolves both the inline idiom (``sample("y", dist.X(...))``) and the
    two-step idiom (``obs_dist = dist.X(...); sample("y", obs_dist)``). The
    generic ``dist.ZeroInflatedDistribution(dist.X(...), ...)`` wrapper is
    canonicalised to ``ZeroInflatedX`` so it matches the gold class names.
    """
    code = code or ""
    m = _Y_DIST_RE.search(code)
    name = m.group(1) if m else ""
    if not name:
        mv = _Y_VAR_RE.search(code)
        if mv:
            md = re.search(rf"\b{re.escape(mv.group(1))}\s*=\s*dist\.(\w+)", code)
            name = md.group(1) if md else ""
    if name == "ZeroInflatedDistribution":
        mi = (re.search(r"ZeroInflatedDistribution\(\s*dist\.(\w+)", code)
              or re.search(r"base_dist\s*=\s*dist\.(\w+)", code))
        if mi:
            name = "ZeroInflated" + mi.group(1)
    return name


def _hetero_structural(code: str) -> bool:
    """True iff the scale is modelled as a (non-pinned) function of x."""
    code = code or ""
    for m in re.finditer(r"(?:sigma|scale)\s*=\s*([^\n]+)", code):
        expr = m.group(1)
        # \bx\b: the identifier x itself — a bare "x" in expr would also match
        # the 'x' inside e.g. ``jnp.exp(log_sigma)`` (a constant variance).
        if re.search(r"\bx\b", expr) and not re.search(r"0\.?0?\s*\*\s*x\b", expr):
            return True
    return False


def _nonlinear_structural(code: str) -> bool:
    return bool(re.search(r"x\s*\*\s*x|x\s*\*\*\s*2|square\(x\)|power\(\s*x", code or ""))


def _hierarchy_structural(code: str) -> bool:
    code = code or ""
    return ("[group]" in code or "group]" in code
            or bool(re.search(r"plate\(\s*[\"'][^\"']*group", code)))


# ---------------------------------------------------------------------------
# Model source strings (fixed per type; only the data varies with the seed)
# ---------------------------------------------------------------------------
# Convention shared with the rest of the codebase: the obs plate is sized from a
# PREDICTOR (x / n_groups), never from ``y``, so the validator can null ``y`` and
# regenerate replicates for the posterior predictive check.

# --- 1. overdispersion: Poisson -> Negative-Binomial ------------------------
_ORIG_COUNT = '''
def model(x=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    mu = jnp.exp(a + b * x)
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.Poisson(mu), obs=y)
'''
_GOLD_NEGBIN = '''
def model(x=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    phi = numpyro.sample("phi", dist.Exponential(1.0))
    mu = jnp.exp(a + b * x)
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.NegativeBinomial2(mu, phi), obs=y)
'''
_DECOY_NEGBIN = '''
def model(x=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    mu = jnp.exp(a + b * x)
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.NegativeBinomial2(mu, 1000.0), obs=y)
'''

# --- 2. zero-inflation: Poisson -> Zero-Inflated Poisson --------------------
_GOLD_ZIP = '''
def model(x=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    gate = numpyro.sample("gate", dist.Beta(1.0, 1.0))
    lam = jnp.exp(a + b * x)
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.ZeroInflatedPoisson(gate, lam), obs=y)
'''
_DECOY_ZIP = '''
def model(x=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    lam = jnp.exp(a + b * x)
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.ZeroInflatedPoisson(0.001, lam), obs=y)
'''

# --- 3. heavy tails: Normal -> Student-t ------------------------------------
_ORIG_GAUSS = '''
def model(x=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5.0))
    mu = a + b * x
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
'''
_GOLD_STUDENTT = '''
def model(x=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5.0))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    mu = a + b * x
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.StudentT(nu, mu, sigma), obs=y)
'''
_DECOY_STUDENTT = '''
def model(x=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5.0))
    mu = a + b * x
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.StudentT(100.0, mu, sigma), obs=y)
'''

# --- 4. binomial overdispersion: Binomial -> Beta-Binomial ------------------
_ORIG_BINOM = '''
def model(x=None, n=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    p = jax.nn.sigmoid(a + b * x)
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.Binomial(total_count=n, probs=p), obs=y)
'''
_GOLD_BETABINOM = '''
def model(x=None, n=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    kappa = numpyro.sample("kappa", dist.Exponential(0.1))
    p = jax.nn.sigmoid(a + b * x)
    c1 = p * kappa
    c0 = (1.0 - p) * kappa
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.BetaBinomial(c1, c0, total_count=n), obs=y)
'''
_DECOY_BETABINOM = '''
def model(x=None, n=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    p = jax.nn.sigmoid(a + b * x)
    c1 = p * 1000.0
    c0 = (1.0 - p) * 1000.0
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.BetaBinomial(c1, c0, total_count=n), obs=y)
'''

# --- 5. positive skew: Normal -> Gamma --------------------------------------
_GOLD_GAMMA = '''
def model(x=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    shape = numpyro.sample("shape", dist.Exponential(1.0))
    mu = jnp.exp(a + b * x)
    rate = shape / mu
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.Gamma(shape, rate), obs=y)
'''
_DECOY_GAMMA = '''
def model(x=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 10.0))
    b = numpyro.sample("b", dist.Normal(0.0, 10.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(10.0))
    mu = a + b * x
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
'''

# --- 6. heteroscedasticity: constant sigma -> sigma(x) ----------------------
_GOLD_HETERO = '''
def model(x=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    c = numpyro.sample("c", dist.Normal(0.0, 5.0))
    d = numpyro.sample("d", dist.Normal(0.0, 5.0))
    mu = a + b * x
    sigma = jnp.exp(c + d * x)
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
'''
_DECOY_HETERO = '''
def model(x=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    c = numpyro.sample("c", dist.Normal(0.0, 5.0))
    mu = a + b * x
    sigma = jnp.exp(c + 0.0 * x)
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
'''

# --- 7. missing nonlinearity: linear -> quadratic ---------------------------
_GOLD_QUAD = '''
def model(x=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    q = numpyro.sample("q", dist.Normal(0.0, 5.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5.0))
    mu = a + b * x + q * x * x
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
'''
_DECOY_QUAD = '''
def model(x=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 20.0))
    b = numpyro.sample("b", dist.Normal(0.0, 20.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(20.0))
    mu = a + b * x
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
'''

# --- 8. missing hierarchy: complete pooling -> varying intercept ------------
_ORIG_POOLED = '''
def model(x=None, group=None, n_groups=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5.0))
    mu = a + b * x
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
'''
_GOLD_HIER = '''
def model(x=None, group=None, n_groups=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))
    tau = numpyro.sample("tau", dist.HalfNormal(2.0))
    with numpyro.plate("groups", n_groups):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0))
    intercept = a + tau * z
    mu = intercept[group] + b * x
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
'''
_DECOY_HIER = '''
def model(x=None, group=None, n_groups=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 1.0))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))
    mu = a + b * x
    with numpyro.plate("obs", x.shape[0]):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
'''


# ---------------------------------------------------------------------------
# Instance factories (data varies with seed; code is fixed per type)
# ---------------------------------------------------------------------------

def _f32(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def _instance(**kw) -> dict:
    """Assemble an instance dict, defaulting the recovery predicate."""
    gold_dists = kw.pop("gold_dists", ())
    structural = kw.pop("structural", None)

    def recovers(refined_code: str, passed: bool) -> bool:
        if not passed:
            return False
        if gold_dists and y_likelihood(refined_code) in gold_dists:
            return True
        if structural is not None and structural(refined_code):
            return True
        return False

    kw["recovers"] = recovers
    return kw


def make_overdispersion(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = 150
    x = rng.normal(0.0, 1.0, n)
    mu = np.exp(1.0 + 0.6 * x)
    phi = 1.5  # small dispersion -> variance = mu + mu^2/phi >> mu
    lam = rng.gamma(shape=phi, scale=mu / phi)
    y = rng.poisson(lam)
    return _instance(
        type="overdispersion", family="counts", seed=seed,
        data={"x": _f32(x), "y": _f32(y)},
        original_code=_ORIG_COUNT, gold_code=_GOLD_NEGBIN, decoy_code=_DECOY_NEGBIN,
        gold_class="likelihood: Negative-Binomial", decoy_label="dispersion pinned huge",
        problem="Regression of an overdispersed non-negative integer count on a "
                "continuous covariate. The response variance greatly exceeds its mean.",
        gold_dists=("NegativeBinomial2", "NegativeBinomial", "GammaPoisson"),
    )


def make_zero_inflation(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = 150
    x = rng.normal(0.0, 1.0, n)
    lam = np.exp(1.5 + 0.4 * x)
    y = rng.poisson(lam)
    structural_zero = rng.random(n) < 0.45
    y = np.where(structural_zero, 0, y)
    return _instance(
        type="zero_inflation", family="counts", seed=seed,
        data={"x": _f32(x), "y": _f32(y)},
        original_code=_ORIG_COUNT, gold_code=_GOLD_ZIP, decoy_code=_DECOY_ZIP,
        gold_class="likelihood: Zero-Inflated Poisson", decoy_label="inflation gate pinned ~0",
        problem="Regression of a count response on a covariate. The response has a "
                "large spike of exact zeros — many more than a Poisson mean would predict.",
        gold_dists=("ZeroInflatedPoisson", "ZeroInflatedNegativeBinomial2"),
    )


def make_heavy_tails(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = 150
    x = rng.normal(0.0, 1.0, n)
    y = 0.0 + 1.0 * x + 1.0 * rng.standard_t(df=3, size=n)
    return _instance(
        type="heavy_tails", family="continuous", seed=seed,
        data={"x": _f32(x), "y": _f32(y)},
        original_code=_ORIG_GAUSS, gold_code=_GOLD_STUDENTT, decoy_code=_DECOY_STUDENTT,
        gold_class="likelihood: Student-t", decoy_label="degrees-of-freedom pinned high",
        problem="Linear regression of a continuous response on a covariate. The "
                "residuals carry occasional gross outliers (heavy tails).",
        gold_dists=("StudentT",),
    )


def make_binomial_overdispersion(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_obs = 120
    trials = 20
    x = rng.normal(0.0, 1.0, n_obs)
    p_mean = 1.0 / (1.0 + np.exp(-(0.0 + 0.8 * x)))
    kappa = 2.0  # low precision -> strong beta-binomial overdispersion
    pi = rng.beta(p_mean * kappa, (1.0 - p_mean) * kappa)
    y = rng.binomial(trials, pi)
    return _instance(
        type="binomial_overdispersion", family="binomial", seed=seed,
        data={"x": _f32(x), "n": np.full(n_obs, trials, dtype=np.int32), "y": _f32(y)},
        original_code=_ORIG_BINOM, gold_code=_GOLD_BETABINOM, decoy_code=_DECOY_BETABINOM,
        gold_class="likelihood: Beta-Binomial", decoy_label="precision pinned huge",
        problem="Regression of a success count out of a fixed number of trials on a "
                "covariate. The counts are far more variable than a Binomial allows.",
        gold_dists=("BetaBinomial", "DirichletMultinomial"),
    )


def make_positive_skew(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = 150
    x = rng.normal(0.0, 1.0, n)
    mu = np.exp(1.2 + 0.5 * x)
    shape = 2.0  # right-skewed positive response
    y = rng.gamma(shape=shape, scale=mu / shape)
    return _instance(
        type="positive_skew", family="continuous", seed=seed,
        data={"x": _f32(x), "y": _f32(y)},
        original_code=_ORIG_GAUSS, gold_code=_GOLD_GAMMA, decoy_code=_DECOY_GAMMA,
        gold_class="likelihood: Gamma (log link)", decoy_label="wider Gaussian priors",
        problem="Regression of a strictly positive, right-skewed continuous response "
                "(e.g. a waiting time) on a covariate.",
        gold_dists=("Gamma", "LogNormal", "InverseGamma", "Weibull"),
    )


def make_heteroscedasticity(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = 150
    x = rng.normal(0.0, 1.0, n)
    mu = 0.0 + 1.0 * x
    sigma = np.exp(-0.5 + 0.8 * x)  # spread grows with x
    y = mu + sigma * rng.normal(0.0, 1.0, n)
    return _instance(
        type="heteroscedasticity", family="continuous", seed=seed,
        data={"x": _f32(x), "y": _f32(y)},
        original_code=_ORIG_GAUSS, gold_code=_GOLD_HETERO, decoy_code=_DECOY_HETERO,
        gold_class="variance model: sigma depends on x", decoy_label="variance-slope pinned 0",
        problem="Linear regression of a continuous response on a covariate where the "
                "residual spread clearly grows with the covariate (non-constant variance).",
        structural=_hetero_structural,
    )


def make_missing_nonlinearity(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = 150
    x = rng.normal(0.0, 1.5, n)
    mu = 0.0 + 0.0 * x + 1.0 * x * x  # pure curvature; a line misses it entirely
    y = mu + rng.normal(0.0, 1.0, n)
    return _instance(
        type="missing_nonlinearity", family="regression", seed=seed,
        data={"x": _f32(x), "y": _f32(y)},
        original_code=_ORIG_GAUSS, gold_code=_GOLD_QUAD, decoy_code=_DECOY_QUAD,
        gold_class="mean structure: add quadratic term", decoy_label="wider priors only",
        problem="Regression of a continuous response on a covariate. A straight-line "
                "mean leaves a strong systematic U-shaped residual pattern.",
        structural=_nonlinear_structural,
    )


def make_missing_hierarchy(seed: int, obs_seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n_groups = 10          # enough groups to identify tau and clear the funnel
    per = 20
    n = n_groups * per
    group = np.repeat(np.arange(n_groups), per)
    x = rng.normal(0.0, 1.0, n)
    u = rng.normal(0.0, 1.5, n_groups)  # group intercept deviations
    if obs_seed is not None:
        # fresh observations from the SAME groups: a model's fitted group effects
        # only transfer to new data when the groups are the ones it saw
        rng = np.random.default_rng(obs_seed)
        x = rng.normal(0.0, 1.0, n)
    mu = 0.5 + u[group] + 0.7 * x
    y = mu + rng.normal(0.0, 0.6, n)
    return _instance(
        type="missing_hierarchy", family="hierarchical", seed=seed,
        data={"x": _f32(x), "group": group.astype(np.int32),
              "n_groups": int(n_groups), "y": _f32(y)},
        original_code=_ORIG_POOLED, gold_code=_GOLD_HIER, decoy_code=_DECOY_HIER,
        gold_class="structure: varying (group) intercept", decoy_label="tighter pooled priors",
        problem="Regression of a continuous response on a covariate, with "
                "observations nested in groups. A single pooled intercept ignores "
                "clear between-group differences in level.",
        structural=_hierarchy_structural,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SPECS = {
    "overdispersion": make_overdispersion,
    "zero_inflation": make_zero_inflation,
    "heavy_tails": make_heavy_tails,
    "binomial_overdispersion": make_binomial_overdispersion,
    "positive_skew": make_positive_skew,
    "heteroscedasticity": make_heteroscedasticity,
    "missing_nonlinearity": make_missing_nonlinearity,
    "missing_hierarchy": make_missing_hierarchy,
}

TYPES = list(SPECS)

# Symptom-free problem statements for the discovery arm: the named strings
# above tell the proposer what is wrong; these describe only the data.
_NEUTRAL_COUNT = "Regression of a non-negative integer count on a continuous covariate."
_NEUTRAL_CONTINUOUS = "Regression of a continuous response on a continuous covariate."
NEUTRAL_PROBLEMS: dict[str, str] = {
    "overdispersion": _NEUTRAL_COUNT,
    "zero_inflation": _NEUTRAL_COUNT,
    "heavy_tails": _NEUTRAL_CONTINUOUS,
    "binomial_overdispersion": ("Regression of the number of successes out of a fixed "
                                "number of trials on a continuous covariate."),
    "positive_skew": _NEUTRAL_CONTINUOUS,
    "heteroscedasticity": _NEUTRAL_CONTINUOUS,
    "missing_nonlinearity": _NEUTRAL_CONTINUOUS,
    "missing_hierarchy": ("Regression of a continuous response on a continuous covariate; "
                          "each observation carries a group index."),
}


def neutralize(inst: dict) -> dict:
    """Shallow copy of ``inst`` whose ``problem`` names no symptom; the named
    statement moves to ``problem_named``."""
    out = dict(inst)
    out["problem_named"] = inst["problem"]
    out["problem"] = NEUTRAL_PROBLEMS[inst["type"]]
    return out


def make_instance(type_name: str, seed: int) -> dict:
    return SPECS[type_name](seed)


def all_instances(seeds=(0, 1, 2)) -> list[dict]:
    """The full pilot grid: every type × every seed (8 × 3 = 24 by default)."""
    return [SPECS[t](s) for t in TYPES for s in seeds]


__all__ = ["SPECS", "TYPES", "NEUTRAL_PROBLEMS", "make_instance", "all_instances",
           "neutralize", "y_likelihood"]
