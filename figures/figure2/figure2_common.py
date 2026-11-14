"""The only code shared by the figure2*_create.py scripts: reading record files, the label
rule, reading a verdict, and the names and colours of the acceptance rules.

A record is one JSON file under bench/arms/<arm>/, one per pipeline run. The rows a panel
counts are the run's pipeline candidates (the agent's own revisions), its template controls
and, on misspecified-baseline arms, its arbiter cases (gold fix, decoy, unchanged copy).
Standard library only.
"""

from __future__ import annotations

import json
from pathlib import Path

# bench/dgp.py TYPES: a record whose "type" is one of these is synthetic (the true
# data-generating process is known, so a fresh draw from it exists); any other type is a
# real dataset, for which only the hidden fold exists.
SYNTHETIC_TYPES = ("overdispersion", "zero_inflation", "heavy_tails", "binomial_overdispersion",
                   "positive_skew", "heteroscedasticity", "missing_nonlinearity",
                   "missing_hierarchy")
SMALL_FOLD = 10  # a hidden fold under this many points cannot separate a candidate at 2 SE

LABELS = ("genuine", "confabulated", "harmful", "underpowered", "unavailable")  # stacking order
LABEL_COLOR = {"genuine": "#0ca30c", "confabulated": "#898781", "harmful": "#d03b3b",
               "underpowered": "#dcdbd3", "unavailable": "#ffffff"}
RULE_COLOR = {"execution": "#eb6834", "self": "#eda100", "control": "#eda100",
              "independent": "#1baf7a", "expert": "#e87ba4", "referee": "#2a78d6"}
BOUND_PCT = 2.3  # the referee's false-acceptance bound, Phi(-2), in percent

MODEL_SHORT = {
    "claude-haiku-4-5-20251001": "Haiku 4.5", "claude-sonnet-4-6": "Sonnet 4.6",
    "claude-opus-4-5-20251101": "Opus 4.5", "claude-opus-4-6": "Opus 4.6",
    "gpt-5": "GPT-5", "gpt-5-mini": "GPT-5 mini", "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro", "gemini-3.8-flash": "Gemini 3.8 Flash",
}


def load_records(arm_dir) -> list[dict]:
    """Every record of an arm directory, sorted by file name. A record is a JSON dict with a
    "type" key; the arm summary (<arm>_summary.json) and unreadable files are skipped.

    Committed arms carry their labels and reader verdicts inside the record. Re-runs keep
    them in side files (scores/<run>.json, readers/<run>.json, written by figure2_score.py);
    when those exist their values are joined onto the rows here, so every panel script reads
    one row shape.
    """
    records = []
    for path in sorted(Path(arm_dir).glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except ValueError:
            continue
        if isinstance(rec, dict) and "type" in rec:
            _join_side_files(rec, path)
            records.append(rec)
    return records


def _rows_by_key(rec: dict) -> dict[str, dict[str, dict]]:
    """{"pipeline": {iteration: row}, "control": {name: row}, "arbiter": {name: row}}."""
    candidates = (rec.get("pipeline") or {}).get("candidates") or []
    arbiter_rows = ((rec.get("arbiter") or {}).get("rules") or {}).get("rows") or []
    return {"pipeline": {str(c.get("iteration")): c for c in candidates},
            "control": {c.get("name"): c for c in rec.get("control") or []},
            "arbiter": {r.get("name"): r for r in arbiter_rows}}


def _join_side_files(rec: dict, path: Path) -> None:
    rows = _rows_by_key(rec)
    scores_path = path.parent / "scores" / path.name
    readers_path = path.parent / "readers" / path.name
    if scores_path.exists():
        scores = json.loads(scores_path.read_text())
        for where, by_key in rows.items():
            for key, row in by_key.items():
                for fold in ("hidden", "fresh"):
                    if fold in (scores.get(where) or {}).get(key, {}):
                        row[fold] = scores[where][key][fold]
    if readers_path.exists():
        readers = json.loads(readers_path.read_text())
        for where, by_key in rows.items():
            for key, row in by_key.items():
                for model, verdict in (readers.get(where) or {}).get(key, {}).items():
                    row.setdefault("judges", {})[f"independent:{model}"] = verdict


def dataset_summary(n: int, problem: str) -> str:
    """The one-line dataset summary every judge reads, exactly as the live runs built it."""
    return f"N={n}, {problem}"


def hidden_score(row: dict) -> dict:
    """The hidden-fold score of a row: the "hidden" dict, or the flat hidden_label /
    hidden_delta / hidden_se keys that older arbiter rows carry instead; {} when absent."""
    if isinstance(row.get("hidden"), dict):
        return row["hidden"]
    if row.get("hidden_label") is not None:
        return {"label": row["hidden_label"], "delta": row.get("hidden_delta"),
                "se": row.get("hidden_se")}
    return {}


def label_of(row: dict, record: dict) -> str:
    """The label the figure colours a row by.

    Synthetic record: the fresh-draw label (150 new points from the true process) whenever
    that score exists; otherwise, and always for real data, the hidden-fold label (the 25%
    the proposer never saw). A hidden "confabulated" on a fold under SMALL_FOLD points is
    reported as "underpowered". A missing or failed score is "unavailable".
    """
    fresh = row.get("fresh") if isinstance(row.get("fresh"), dict) else {}
    synthetic = record.get("type") in SYNTHETIC_TYPES
    if synthetic and fresh.get("label") and fresh.get("label") != "unavailable":
        score = fresh
    else:
        score = hidden_score(row)
    label = score.get("label") or "unavailable"
    n = score.get("n_hidden")
    if label == "confabulated" and isinstance(n, (int, float)) and n < SMALL_FOLD:
        label = "underpowered"
    return label


def verdict(row: dict, rule: str):
    """True / False / None (no verdict) of one rule on one row. "mcmc" is the referee:
    the in-run MCMC verdict of a candidate or control, mcmc_passed of an arbiter case."""
    if rule == "mcmc":
        m = row.get("mcmc")
        return m.get("passed") if isinstance(m, dict) else row.get("mcmc_passed")
    j = (row.get("judges") or {}).get(rule)
    return j.get("passed") if isinstance(j, dict) else None


def rule_family(rule: str) -> str | None:
    if rule == "execution":
        return "execution"
    if rule in ("self", "control"):
        return rule
    if rule.startswith("independent"):
        return "independent"
    if rule.startswith("expert"):
        return "expert"
    if rule in ("mcmc", "referee"):
        return "referee"
    return None


def rule_order(rules) -> list[str]:
    """Display order: execution, self, control, independent:<model> (alphabetical), expert,
    referee. A key of no known family is reported and left out."""
    families = ("execution", "self", "control", "independent", "expert", "referee")
    known, unknown = [], []
    for r in set(rules):
        (known if rule_family(r) else unknown).append(r)
    if unknown:
        print(f"note: ignoring judge keys of no known family: {sorted(unknown)}")
    return sorted(known, key=lambda r: (families.index(rule_family(r)), r))


def model_short(model_id: str) -> str:
    return MODEL_SHORT.get(model_id, model_id)


def rule_name(rule: str) -> str:
    """Axis label of a rule."""
    if rule.startswith("independent:"):
        return f"Independent model\n({model_short(rule.split(':', 1)[1])})"
    return {"execution": "Execution check", "self": "Self-evaluation",
            "control": "Control judge\n(self colour, no rationale)", "mcmc": "Referee",
            "referee": "Referee"}.get(rule, rule)


def rule_color(rule: str) -> str:
    return RULE_COLOR[rule_family(rule) or "referee"]


CONDITION_ORDER = ("E5", "E1", "real data")  # display order of the experiments
CONDITION_TITLE = {                            # what each experiment is, in plain words
    "E5": "Nothing is wrong: the agent revises the correct model",
    "E1": "A fix exists: the agent revises a misspecified model",
    "real data": "Real datasets: the agent revises the textbook model",
}


def condition(name: str, records: list[dict]) -> str | None:
    """The experiment an arm belongs to, in the paper's codes: "E5" (every run starts from the
    correct model), "E1" (from a misspecified one) or "real data". Understands both the
    explanatory arm names of figure2_protocols and the committed directory names; None when
    the name is not recognisable."""
    if records and all(r.get("type") not in SYNTHETIC_TYPES for r in records):
        return "real data"
    if name.startswith("correct-baseline") or name.startswith("e5"):
        return "E5"
    if name.startswith("misspecified-baseline") or name.startswith("e1"):
        return "E1"
    return None


def arm_model(records: list[dict]) -> str | None:
    """The one proposer model of an arm's records as a short name; None when they mix models."""
    models = sorted({r.get("model") for r in records if r.get("model")})
    return model_short(models[0]) if len(models) == 1 else None


def arm_label(name: str, records: list[dict]) -> str:
    """Short name of an arm for a stem label: "Sonnet 4.6, E5" (model, experiment code); the
    directory name itself when the arm mixes models or its condition is not recognisable."""
    cond, model = condition(name, records), arm_model(records)
    return name if cond is None or model is None else f"{model}, {cond}"
