"""Real-data arm of the benchmark: seven literature datasets, each with its
textbook misspecified starting model.

Every dataset is vendored under ``tutorials/demos/data/`` (provenance, md5 and
retrieval date in that directory's README) and the loaders read only those
files. Each entry carries two problem statements: ``problem_neutral`` describes
the data and the scientific question without naming the symptom, while
``problem_named`` states the textbook diagnosis for the named-symptom arm.
``make_real_instance`` returns a dict shaped like a ``bench.dgp`` instance minus
the gold/decoy pair, so ``bench.run`` can drive it unchanged.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("COMPASS_DATA_DIR",
                                Path(__file__).resolve().parents[3] / "tutorials" / "demos" / "data"))  # research: parent.parent / "tutorials" / "demos" / "data"

DEFAULT_BUDGET = dict(num_warmup=500, num_samples=1000, num_chains=2, target_accept=0.9)
RADON_BUDGET = dict(num_warmup=1000, num_samples=2000, num_chains=4, target_accept=0.95)

# Convention shared with bench.dgp: integer-valued responses are float32 and a
# Binomial's total_count is int32; index arrays are int32 and 0-based; group
# counts are plain ints (a plate size must be static under jit).
DTYPE_NOTE = ("y float32 (integer-valued where applicable), Binomial total_count int32, "
              "indices int32 0-based, group counts python int — the bench.dgp convention; "
              "each baseline is sampled once in tests/test_realdata.py (slow lane).")


@dataclass(frozen=True)
class RealDataset:
    key: str
    title: str
    field: str
    family: str
    citation: str
    source_url: str
    n: int
    loader: Callable[[], dict]
    original_code: str
    problem_neutral: str
    problem_named: str
    budget: dict
    notes: str


# ---------------------------------------------------------------------------
# Loaders (vendored files only; no network)
# ---------------------------------------------------------------------------

def _f32(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def _i32(a) -> np.ndarray:
    return np.asarray(a, dtype=np.int32)


def _csv_columns(name: str, *cols: str) -> list[np.ndarray]:
    with open(DATA_DIR / name, newline="") as f:
        rows = list(csv.DictReader(f))
    return [np.array([float(r[c]) for r in rows]) for c in cols]


def _json(name: str) -> dict:
    with open(DATA_DIR / name) as f:
        return json.load(f)


def load_roaches() -> dict:
    y, roach1, treatment, senior, exposure2 = _csv_columns(
        "roaches.csv", "y", "roach1", "treatment", "senior", "exposure2")
    # roach1 / 100 puts the log-scale slope at O(1), as in demo_9.
    return {"roach1": _f32(roach1 / 100.0), "treatment": _f32(treatment),
            "senior": _f32(senior), "log_expo": _f32(np.log(exposure2)), "y": _f32(y)}


def load_ucbadmit() -> dict:
    dept, male, applications, admit = _csv_columns(
        "ucbadmit.csv", "dept", "male", "applications", "admit")
    return {"male": _f32(male), "dept": _i32(dept), "n_dept": int(dept.max()) + 1,
            "applications": _i32(applications), "y": _f32(admit)}


def load_baseball() -> dict:
    at_bats, hits = _csv_columns("baseball_efron_morris.csv", "at_bats", "hits")
    return {"at_bats": _i32(at_bats), "y": _f32(hits)}


def load_seeds() -> dict:
    d = _json("seeds_data.json")
    return {"x1": _f32(d["x1"]), "x2": _f32(d["x2"]),
            "n_total": _i32(d["N"]), "y": _f32(d["n"])}


def load_radon() -> dict:
    d = _json("radon_mn.json")
    return {"floor": _f32(d["floor_measure"]),
            "county_idx": _i32(np.asarray(d["county_idx"]) - 1),  # posteriordb is 1-based
            "n_counties": int(d["J"]), "log_uppm": _f32(d["log_uppm"]),
            "y": _f32(d["log_radon"])}


def load_kidiq() -> dict:
    d = _json("kidiq.json")
    mom_iq = np.asarray(d["mom_iq"], dtype=np.float64)
    return {"mom_hs": _f32(d["mom_hs"]), "mom_iq_c": _f32((mom_iq - 100.0) / 15.0),
            "y": _f32(d["kid_score"])}


def load_golf() -> dict:
    distance, tries, successes = _csv_columns(
        "golf_putting.csv", "distance", "tries", "successes")
    return {"distance": _f32(distance), "tries": _i32(tries), "y": _f32(successes)}


# ---------------------------------------------------------------------------
# Literature-standard misspecified baselines
# ---------------------------------------------------------------------------
# The obs plate is sized from a PREDICTOR, never from ``y``, so the validator can
# null ``y`` and regenerate replicates for the posterior predictive check.

ROACHES_ORIGINAL = '''
def model(roach1=None, treatment=None, senior=None, log_expo=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b_roach = numpyro.sample("b_roach", dist.Normal(0.0, 5.0))
    b_treat = numpyro.sample("b_treat", dist.Normal(0.0, 5.0))
    b_senior = numpyro.sample("b_senior", dist.Normal(0.0, 5.0))
    eta = a + b_roach * roach1 + b_treat * treatment + b_senior * senior + log_expo
    with numpyro.plate("obs", roach1.shape[0]):
        numpyro.sample("y", dist.Poisson(jnp.exp(eta)), obs=y)
'''

UCBADMIT_ORIGINAL = '''
def model(male=None, dept=None, n_dept=None, applications=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    p = jax.nn.sigmoid(a + b * male)
    with numpyro.plate("obs", male.shape[0]):
        numpyro.sample("y", dist.Binomial(total_count=applications, probs=p), obs=y)
'''

BASEBALL_ORIGINAL = '''
def model(at_bats=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    p = jax.nn.sigmoid(a)
    with numpyro.plate("obs", at_bats.shape[0]):
        numpyro.sample("y", dist.Binomial(total_count=at_bats, probs=p), obs=y)
'''

SEEDS_ORIGINAL = '''
def model(x1=None, x2=None, n_total=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b1 = numpyro.sample("b1", dist.Normal(0.0, 5.0))
    b2 = numpyro.sample("b2", dist.Normal(0.0, 5.0))
    b12 = numpyro.sample("b12", dist.Normal(0.0, 5.0))
    p = jax.nn.sigmoid(a + b1 * x1 + b2 * x2 + b12 * x1 * x2)
    with numpyro.plate("obs", x1.shape[0]):
        numpyro.sample("y", dist.Binomial(total_count=n_total, probs=p), obs=y)
'''

RADON_ORIGINAL = '''
def model(floor=None, county_idx=None, n_counties=None, log_uppm=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5.0))
    mu = a + b * floor
    with numpyro.plate("obs", floor.shape[0]):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
'''

KIDIQ_ORIGINAL = '''
def model(mom_hs=None, mom_iq_c=None, y=None):
    a = numpyro.sample("a", dist.Normal(100.0, 50.0))
    b_hs = numpyro.sample("b_hs", dist.Normal(0.0, 20.0))
    b_iq = numpyro.sample("b_iq", dist.Normal(0.0, 20.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(50.0))
    mu = a + b_hs * mom_hs + b_iq * mom_iq_c
    with numpyro.plate("obs", mom_hs.shape[0]):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
'''

GOLF_ORIGINAL = '''
def model(distance=None, tries=None, y=None):
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    p = jax.nn.sigmoid(a + b * distance)
    with numpyro.plate("obs", distance.shape[0]):
        numpyro.sample("y", dist.Binomial(total_count=tries, probs=p), obs=y)
'''


# ---------------------------------------------------------------------------
# Registry (fixed order)
# ---------------------------------------------------------------------------

_DATASETS = (
    RealDataset(
        key="roaches",
        title="Roaches: pest-management count regression",
        field="public health / pest management",
        family="counts",
        citation="Gelman & Hill (2007) ch. 8; Vehtari, Gelman & Gabry (2017), roaches case study",
        source_url="https://raw.githubusercontent.com/avehtari/ROS-Examples/master/Roaches/data/roaches.csv",
        n=262,
        loader=load_roaches,
        original_code=ROACHES_ORIGINAL,
        problem_neutral=(
            "Pest-management field study of 262 apartments. The response y is the number "
            "of roaches caught in traps after a treatment period; predictors are the "
            "pre-treatment roach level roach1 (scaled by 1/100), a treatment indicator and "
            "an indicator for buildings restricted to senior residents; log_expo, the log "
            "of trap-days of exposure, is an offset. Goal: estimate the effect of the "
            "treatment on the roach count."),
        problem_named=(
            "The roach counts are grossly overdispersed (observed sd 50.7 against a "
            "Poisson-implied replicated-sd band near 32-38; variance >> mean). A Poisson "
            "likelihood cannot reproduce this; the textbook fix is a Negative-Binomial "
            "likelihood with a learned dispersion (Gelman & Hill 2007; Vehtari, Gelman & "
            "Gabry 2017)."),
        budget=DEFAULT_BUDGET,
        notes=("Same CSV, loader transform and ORIGINAL text as demo_9_roaches_real.py "
               "(roach1 / 100, log offset). " + DTYPE_NOTE),
    ),
    RealDataset(
        key="ucbadmit",
        title="UC Berkeley 1973 graduate admissions",
        field="education / admissions",
        family="binomial",
        citation="Bickel, Hammel & O'Connell (1975), Science 187:398-404; McElreath (2020) ch. 11",
        source_url="https://github.com/pyro-ppl/datasets/blob/master/UCBadmit.csv?raw=true",
        n=12,
        loader=load_ucbadmit,
        original_code=UCBADMIT_ORIGINAL,
        problem_neutral=(
            "UC Berkeley graduate admissions, fall 1973: 12 rows, one per department (dept, "
            "index 0-5; n_dept = 6) and applicant gender (male = 1 for male applicants), "
            "each with the number of applications and the number admitted (y). Goal: "
            "estimate the effect of applicant gender on the probability of admission."),
        problem_named=(
            "Simpson's paradox: admission rates differ sharply across departments and the "
            "two genders apply to different departments, so a logistic regression on gender "
            "alone attributes the department-composition effect to gender. The textbook fix "
            "is a varying intercept per department, i.e. a hierarchical (partially pooled) "
            "logistic model (Bickel, Hammel & O'Connell 1975; McElreath 2020 ch. 11)."),
        budget=DEFAULT_BUDGET,
        notes=("Only 12 aggregated rows (4,526 applicants); row-wise LOO is high-leverage, "
               "so expect wide PSIS k-hat. male vendored as 0/1 from numpyro's bool. "
               + DTYPE_NOTE),
    ),
    RealDataset(
        key="baseball",
        title="Efron-Morris 1970 batting averages",
        field="sports statistics",
        family="binomial",
        citation="Efron & Morris (1975), JASA 70:311-319 (Table 1); NumPyro baseball example",
        source_url="https://github.com/pyro-ppl/datasets/blob/master/EfronMorrisBB.txt?raw=true",
        n=18,
        loader=load_baseball,
        original_code=BASEBALL_ORIGINAL,
        problem_neutral=(
            "Efron & Morris (1975) batting data: 18 Major League players, each with y hits "
            "out of at_bats = 45 at-bats early in the 1970 season. Goal: estimate each "
            "player's probability of getting a hit."),
        problem_named=(
            "Complete pooling (one shared hit probability) ignores real between-player "
            "heterogeneity, while no pooling overfits 45 at-bats; the textbook fix is "
            "partial pooling - a hierarchical Binomial with player-level abilities shrunk "
            "toward a common mean (Efron & Morris 1975; the NumPyro baseball example)."),
        budget=DEFAULT_BUDGET,
        notes=("N = 18 players, one row each. Season totals (season_at_bats, season_hits) "
               "are kept in the CSV as the classic held-out target but are not in the data "
               "dict. " + DTYPE_NOTE),
    ),
    RealDataset(
        key="seeds",
        title="Crowder seed germination (2x2 factorial)",
        field="agriculture / biostatistics",
        family="binomial",
        citation="Crowder (1978), Appl. Statist. 27:34-37; Breslow & Clayton (1993); BUGS 'Seeds'",
        source_url="https://raw.githubusercontent.com/stan-dev/posteriordb/master/posterior_database/data/data/seeds_data.json.zip",
        n=21,
        loader=load_seeds,
        original_code=SEEDS_ORIGINAL,
        problem_neutral=(
            "Crowder (1978) seed germination experiment: 21 plates in a 2 x 2 factorial of "
            "seed variety (x1) and root extract (x2); each plate has n_total seeds of which y "
            "germinated. Goal: estimate the effects of variety, extract and their "
            "combination on the germination probability."),
        problem_named=(
            "Binomial overdispersion: plate-to-plate variation in germination exceeds what "
            "a Binomial with the factorial predictors allows. The textbook fix is a "
            "plate-level random effect on the logit scale (Breslow & Clayton 1993; the BUGS "
            "'Seeds' example) or equivalently a Beta-Binomial likelihood."),
        budget=DEFAULT_BUDGET,
        notes=("N = 21 plates, some with as few as 4 seeds; the extra-binomial signal is "
               "modest, so a verdict may be a tie. " + DTYPE_NOTE),
    ),
    RealDataset(
        key="radon",
        title="Minnesota radon survey (919 homes, 85 counties)",
        field="environmental health",
        family="continuous",
        citation="Gelman & Hill (2007) ch. 12; posteriordb radon_mn",
        source_url="https://raw.githubusercontent.com/stan-dev/posteriordb/master/posterior_database/data/data/radon_mn.json.zip",
        n=919,
        loader=load_radon,
        original_code=RADON_ORIGINAL,
        problem_neutral=(
            "Minnesota radon survey (EPA): 919 homes in 85 counties. The response y is log "
            "radon concentration; floor is the measurement floor (0 basement, 1 first "
            "floor), county_idx is the 0-based county of each home (n_counties = 85) and "
            "log_uppm is the county's log soil-uranium level. Goal: estimate the effect of "
            "the measurement floor on log radon."),
        problem_named=(
            "Complete pooling across counties ignores large between-county differences in "
            "radon level (and the county-level uranium predictor). The textbook fix is a "
            "varying-intercept multilevel model by county, optionally with log_uppm as a "
            "county-level predictor (Gelman & Hill 2007 ch. 12)."),
        budget=RADON_BUDGET,
        notes=("Budget raised like bench.run's HIER_BUDGET: a varying-intercept refinement "
               "carries a county-scale funnel whose tau ESS needs more draws. county_idx "
               "converted from posteriordb's 1-based 1..85 to 0-based. " + DTYPE_NOTE),
    ),
    RealDataset(
        key="kidiq",
        title="Children's test scores and maternal IQ (NLSY)",
        field="developmental psychology / education",
        family="continuous",
        citation="Gelman & Hill (2007) ch. 3; Gelman, Hill & Vehtari (2020) ch. 10; posteriordb kidiq",
        source_url="https://raw.githubusercontent.com/stan-dev/posteriordb/master/posterior_database/data/data/kidiq.json.zip",
        n=434,
        loader=load_kidiq,
        original_code=KIDIQ_ORIGINAL,
        problem_neutral=(
            "Children's cognitive test scores (National Longitudinal Survey of Youth): 434 "
            "children. The response y is the child's test score at age 3-4; predictors are "
            "whether the mother completed high school (mom_hs) and the mother's IQ, "
            "standardized as mom_iq_c = (mom_iq - 100) / 15. Goal: estimate how maternal "
            "education and IQ relate to the child's score."),
        problem_named=(
            "The additive model misses the mom_hs x mom_iq interaction: the slope on "
            "maternal IQ is much steeper for mothers without a high-school degree (Gelman & "
            "Hill 2007 ch. 3; Gelman, Hill & Vehtari 2020 ch. 10). The textbook fix adds the "
            "product term."),
        budget=DEFAULT_BUDGET,
        notes=("posteriordb's prediction scalars mom_hs_new / mom_iq_new are dropped. Priors "
               "are on the test-score scale (mean ~87, sd ~20) rather than the dgp's unit "
               "scale. The interaction's predictive gain is modest (ΔLOO of a few nats), so "
               "this is a hard, tie-prone case. " + DTYPE_NOTE),
    ),
    RealDataset(
        key="golf",
        title="Professional golf putting by distance",
        field="sports / biomechanics",
        family="binomial",
        citation="Berry (1996), Statistics: A Bayesian Perspective; Gelman, 'Model building and expansion for golf putting' (Stan case study)",
        source_url="https://mc-stan.org/users/documentation/case-studies/golf.html",
        n=19,
        loader=load_golf,
        original_code=GOLF_ORIGINAL,
        problem_neutral=(
            "Professional golf putting (Berry 1996): 19 distances from 2 to 20 feet; at "
            "each distance the number of putts tried and the number holed (y). Goal: model "
            "the probability of holing a putt as a function of distance."),
        problem_named=(
            "A logit-linear-in-distance success probability is the wrong functional form: "
            "the data follow the geometry of the shot - a putt succeeds when the angular "
            "error is within asin((R - r) / distance), giving p = 2 Phi(asin((R - r) / x) / "
            "sigma) - 1 with R the hole radius (2.125 in) and r the ball radius (0.84 in) "
            "(Gelman's golf putting case study). The logistic fit misses the curvature "
            "systematically."),
        budget=DEFAULT_BUDGET,
        notes=("N = 19 aggregated rows over 5,988 putts; the misfit is systematic, not "
               "dispersion, so a PPC on the mean alone may pass. Rows transcribed from the "
               "Stan case study (PyMC's golf example agrees). " + DTYPE_NOTE),
    ),
)

DATASETS: dict[str, RealDataset] = {d.key: d for d in _DATASETS}
KEYS = list(DATASETS)


def make_real_instance(key: str) -> dict:
    ds = DATASETS[key]
    return {
        "type": ds.key,
        "family": ds.family,
        "seed": 0,
        "data": ds.loader(),
        "original_code": ds.original_code,
        "problem": ds.problem_neutral,
        "problem_named": ds.problem_named,
        "budget": dict(ds.budget),
        "recovers": lambda code, passed: bool(passed),
        "source": ds.source_url,
        "notes": ds.notes,
    }


def all_real_instances() -> list[dict]:
    return [make_real_instance(k) for k in KEYS]


__all__ = ["DATASETS", "KEYS", "RealDataset", "all_real_instances", "make_real_instance",
           "DEFAULT_BUDGET", "RADON_BUDGET"]
