"""Shared by the figure3*_create.py scripts: reading drift trajectories, the Δ a round is drawn
at, counting latent sample sites, and the order of the real datasets. Rule names, colours,
record reading and the label rule come from figure2/figure2_common.py (imported here as
``common``, so every Figure 3 script reads the same rule once). Standard library only.

A trajectory is one JSON file under <arm>/drift/, named <type>_seed<s>_<rule>.json: one
starting model, one acceptance rule, and a list of rounds. Each round holds the candidate the
agent proposed, the rule's verdict, the model deployed after the round and, inline in the
committed records or in scores/<file>.json for re-runs, the held-out scores of that model
against the starting one ("hidden" on the hidden fold, "fresh" on a fresh draw from the true
process) and of the candidate against the model it was proposed for ("step").
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the repo root, so "from figure2 import" works when run as a file
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from figure2 import figure2_common as common  # noqa: E402

TRAJECTORY_NAME = re.compile(r"^(?P<type>.+)_seed(?P<seed>\d+)_(?P<rule>.+)\.json$")
REAL_ORDER = ("roaches", "ucbadmit", "baseball", "seeds", "radon", "kidiq", "golf")  # bench.realdata order
DATASET_NAME = {"ucbadmit": "UCBadmit", "kidiq": "KidIQ"}
_SAMPLE = re.compile(r"numpyro\.sample\(")
_OBS = re.compile(r"\bobs\s*=")


def load_trajectories(drift_dir) -> list[dict]:
    """Every trajectory of a drift directory, sorted by file name. Files that are not
    trajectories (no rounds list, or another name) are skipped. Re-runs keep the held-out
    scores in scores/<file>.json; when that exists its "hidden", "step" and "fresh" entries
    are joined onto the rounds, so every panel script reads one round shape."""
    out = []
    for path in sorted(Path(drift_dir).glob("*.json")):
        m = TRAJECTORY_NAME.match(path.name)
        try:
            rec = json.loads(path.read_text())
        except ValueError:
            continue
        if not m or not isinstance(rec, dict) or not isinstance(rec.get("rounds"), list):
            continue
        rec.setdefault("type", m["type"])
        rec.setdefault("seed", int(m["seed"]))
        rec.setdefault("rule", m["rule"])
        side = path.parent / "scores" / path.name
        if side.exists():
            scores = json.loads(side.read_text()).get("rounds") or {}
            for r in rec["rounds"]:
                for key, value in (scores.get(str(r.get("round"))) or {}).items():
                    if key in ("hidden", "step", "fresh"):
                        r[key] = value
        out.append(rec)
    return out


def round_delta(rnd: dict, rec: dict) -> float | None:
    """Δ (nats) of the model deployed after this round against the starting model: the
    fresh-draw score for a synthetic record when it exists, the hidden-fold score otherwise;
    None when that score is unavailable (the round is then left out, never drawn as zero)."""
    fresh = rnd.get("fresh") if isinstance(rnd.get("fresh"), dict) else {}
    hidden = rnd.get("hidden") if isinstance(rnd.get("hidden"), dict) else {}
    synthetic = rec.get("type") in common.SYNTHETIC_TYPES
    score = fresh if synthetic and fresh.get("label") not in (None, "unavailable") else hidden
    delta = score.get("delta") if score.get("available") else None
    return float(delta) if isinstance(delta, (int, float)) else None


def series(rec: dict) -> dict[int, float]:
    """{round: Δ} of one trajectory with round 0 at 0.0; rounds without a score are left out."""
    pts = {0: 0.0}
    for r in rec["rounds"]:
        d = round_delta(r, rec)
        if d is not None:
            pts[int(r["round"])] = d
    return pts


def count_latent_sites(code: str) -> int:
    """``numpyro.sample(`` calls whose arguments carry no ``obs=``: a regex count, good enough
    to follow how a model grows over rounds (same rule as bench.drift.count_sample_sites)."""
    code = code or ""
    n = 0
    for m in _SAMPLE.finditer(code):
        start, depth, span = m.end() - 1, 0, None
        for i in range(start, len(code)):
            depth += (code[i] == "(") - (code[i] == ")")
            if depth == 0:
                span = code[start:i + 1]
                break
        if span is None:  # unbalanced (a parenthesis inside a string): the rest of the line
            end = code.find("\n", start)
            span = code[start:] if end < 0 else code[start:end]
        if not _OBS.search(span):
            n += 1
    return n


def legend_name(rule: str) -> str:
    """One-line name of a rule for a legend."""
    return common.rule_name(rule).split("\n")[0]


def dataset_order(types) -> list[str]:
    """The registered real datasets in their reference order, then any other sorted."""
    return sorted(set(types), key=lambda t: (REAL_ORDER.index(t) if t in REAL_ORDER else len(REAL_ORDER), t))


def dataset_name(ds: str) -> str:
    return DATASET_NAME.get(ds, ds)
