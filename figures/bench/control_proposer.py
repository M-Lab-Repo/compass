"""Template control proposer — deterministic, LLM-free perturbations of a NumPyro program.

The "simpler proposal mechanism" baseline and the control arm for the
fluency-is-the-hazard study. Each transform is a conservative regex edit that
fires only when its exact idiom is present (the ``bench.dgp`` originals: literal
prior scales, an inline ``numpyro.sample("y", dist.X(...), obs=y)`` site, a
``+ b * x`` linear predictor, ``jnp.exp`` links). Every candidate carries an
empty rationale.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from typing import NamedTuple

import numpy as np


class Transform(NamedTuple):
    name: str
    applies: Callable[[str], bool]
    apply: Callable[[str, np.random.Generator], str | None]


_NUM = r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?"
_Y_SITE_RE = re.compile(r"""sample\(\s*(["'])y\1\s*,""")
_PLATE_WITH_RE = re.compile(r"^\s*with\b.*\bplate\(")
_NORMAL_PRIOR_RE = re.compile(rf"dist\.Normal\(\s*[^,()]+?\s*,\s*({_NUM})\s*\)")
_HALFNORMAL_RE = re.compile(rf"dist\.HalfNormal\(\s*({_NUM})\s*\)")
_PLUS_BX_RE = re.compile(r"\+\s*[A-Za-z_]\w*\s*\*\s*x(?![\w.\[])")
_COEF_X_RE = re.compile(r"\b[A-Za-z_]\w*\s*\*\s*x(?![\w.\[])")
_QUADRATIC_RE = re.compile(r"x\s*\*\s*x|x\s*\*\*\s*2")


# ---------------------------------------------------------------------------
# Source helpers
# ---------------------------------------------------------------------------

def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _y_line_index(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        if _Y_SITE_RE.search(line):
            return i
    return None


def _quote(code: str) -> str:
    m = _Y_SITE_RE.search(code)
    return m.group(1) if m else '"'


def _name_free(code: str, name: str) -> bool:
    return re.search(rf"\b{name}\b", code) is None


def _call_span(text: str, head: str) -> tuple[int, int, int] | None:
    """(start, args_start, close) of the first balanced ``head...)`` call; head ends with "("."""
    start = text.find(head)
    if start < 0:
        return None
    depth = 1
    for i in range(start + len(head), len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return start, start + len(head), i
    return None


def _split_top(args: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for c in args:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _kwargs(parts: list[str]) -> dict[str, str] | None:
    out: dict[str, str] = {}
    for p in parts:
        m = re.fullmatch(r"(\w+)\s*=\s*(.+)", p, re.S)
        if not m:
            return None
        out[m.group(1)] = m.group(2).strip()
    return out


def _insertion_point(lines: list[str], idx: int) -> tuple[int, str]:
    """Where a new global parameter goes: hoisted above any enclosing obs plate.

    A sample statement placed inside ``with numpyro.plate(...)`` would be
    broadcast to one parameter per observation, not the scalar intended.
    """
    indent = _indent(lines[idx])
    at = idx
    while True:
        k = at - 1
        while k >= 0 and (not lines[k].strip() or len(_indent(lines[k])) >= len(indent)):
            k -= 1
        if k >= 0 and _PLATE_WITH_RE.match(lines[k]):
            at, indent = k, _indent(lines[k])
            continue
        return at, indent


def _edit_first_literal(code: str, pattern: re.Pattern, scale: float) -> str | None:
    lines = code.split("\n")
    yi = _y_line_index(lines)
    for i, line in enumerate(lines):
        if i == yi:
            continue
        m = pattern.search(line)
        if m:
            # round: a float-noise literal like 0.30000000000000004 would mark the
            # emitted program as machine-edited to any reader
            new = repr(round(float(m.group(1)) * scale, 12))
            lines[i] = line[: m.start(1)] + new + line[m.end(1) :]
            return "\n".join(lines)
    return None


def _has_literal_off_y(code: str, pattern: re.Pattern) -> bool:
    lines = code.split("\n")
    yi = _y_line_index(lines)
    return any(pattern.search(line) for i, line in enumerate(lines) if i != yi)


def _y_has_head(code: str, head: str) -> bool:
    lines = code.split("\n")
    yi = _y_line_index(lines)
    return yi is not None and head in lines[yi]


def _swap_y_likelihood(
    code: str, head: str,
    rebuild: Callable[[str, str], tuple[str, list[str]] | None],
) -> str | None:
    lines = code.split("\n")
    yi = _y_line_index(lines)
    if yi is None:
        return None
    line = lines[yi]
    span = _call_span(line, head)
    if span is None:
        return None
    start, args_start, close = span
    built = rebuild(line[args_start:close], _quote(code))
    if built is None:
        return None
    new_call, prelude = built
    lines[yi] = line[:start] + new_call + line[close + 1 :]
    at, indent = _insertion_point(lines, yi)
    lines[at:at] = [indent + s for s in prelude]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _prior_widen(code: str, rng: np.random.Generator) -> str | None:
    return _edit_first_literal(code, _NORMAL_PRIOR_RE, 10.0)


def _prior_narrow(code: str, rng: np.random.Generator) -> str | None:
    return _edit_first_literal(code, _NORMAL_PRIOR_RE, 0.1)


def _scale_prior_widen(code: str, rng: np.random.Generator) -> str | None:
    return _edit_first_literal(code, _HALFNORMAL_RE, 2.0)


def _swap_count(code: str, rng: np.random.Generator) -> str | None:
    if not _name_free(code, "conc"):
        return None

    def rebuild(args: str, q: str) -> tuple[str, list[str]] | None:
        rate = args.strip()
        if not rate or "=" in rate:
            return None
        return (f"dist.NegativeBinomial2({rate}, conc)",
                [f"conc = numpyro.sample({q}conc{q}, dist.HalfNormal(5.0))"])

    return _swap_y_likelihood(code, "dist.Poisson(", rebuild)


def _swap_gauss(code: str, rng: np.random.Generator) -> str | None:
    if not _name_free(code, "nu"):
        return None

    def rebuild(args: str, q: str) -> tuple[str, list[str]] | None:
        parts = _split_top(args)
        if len(parts) == 2 and not any("=" in p for p in parts):
            mu, sigma = parts
        else:
            kw = _kwargs(parts)
            if kw is None or "loc" not in kw or set(kw) - {"loc", "scale"}:
                return None
            mu, sigma = kw["loc"], kw.get("scale", "1.0")
        return (f"dist.StudentT(nu, {mu}, {sigma})",
                [f"nu = numpyro.sample({q}nu{q}, dist.Gamma(2.0, 0.1))"])

    return _swap_y_likelihood(code, "dist.Normal(", rebuild)


def _swap_binom(code: str, rng: np.random.Generator) -> str | None:
    if not _name_free(code, "kappa"):
        return None

    def rebuild(args: str, q: str) -> tuple[str, list[str]] | None:
        kw = _kwargs(_split_top(args))
        if kw is None or "total_count" not in kw or set(kw) - {"total_count", "probs", "logits"}:
            return None
        if "probs" in kw and "logits" not in kw:
            p = kw["probs"]
        elif "logits" in kw and "probs" not in kw:
            p = f"jax.nn.sigmoid({kw['logits']})"
        else:
            return None
        p = p if re.fullmatch(r"\w+", p) else f"({p})"
        return (f"dist.BetaBinomial(concentration1={p} * kappa, "
                f"concentration0=(1.0 - {p}) * kappa, total_count={kw['total_count']})",
                [f"kappa = numpyro.sample({q}kappa{q}, dist.HalfNormal(10.0))"])

    return _swap_y_likelihood(code, "dist.Binomial(", rebuild)


def _add_quadratic(code: str, rng: np.random.Generator) -> str | None:
    if not _name_free(code, "c2") or _QUADRATIC_RE.search(code):
        return None
    lines = code.split("\n")
    q = _quote(code)
    for i, line in enumerate(lines):
        m = _PLUS_BX_RE.search(line)
        if m and re.match(r"^\s*\w+\s*=", line):
            lines[i] = line[: m.end()] + " + c2 * x**2" + line[m.end() :]
            at, indent = _insertion_point(lines, i)
            lines[at:at] = [f"{indent}c2 = numpyro.sample({q}c2{q}, dist.Normal(0.0, 1.0))"]
            return "\n".join(lines)
    return None


def _link_swap(code: str, rng: np.random.Generator) -> str | None:
    return code.replace("jnp.exp(", "jax.nn.softplus(", 1)


def _drop_covariate(code: str, rng: np.random.Generator) -> str | None:
    return _COEF_X_RE.sub("0.0 * x", code, count=1)


MENU: tuple[Transform, ...] = (
    Transform("prior_widen",
              lambda c: _has_literal_off_y(c, _NORMAL_PRIOR_RE), _prior_widen),
    Transform("prior_narrow",
              lambda c: _has_literal_off_y(c, _NORMAL_PRIOR_RE), _prior_narrow),
    Transform("scale_prior_widen",
              lambda c: _has_literal_off_y(c, _HALFNORMAL_RE), _scale_prior_widen),
    Transform("likelihood_swap_count",
              lambda c: _y_has_head(c, "dist.Poisson("), _swap_count),
    Transform("likelihood_swap_gauss",
              lambda c: _y_has_head(c, "dist.Normal("), _swap_gauss),
    Transform("likelihood_swap_binom",
              lambda c: _y_has_head(c, "dist.Binomial("), _swap_binom),
    Transform("add_quadratic",
              lambda c: bool(_PLUS_BX_RE.search(c)) and not _QUADRATIC_RE.search(c),
              _add_quadratic),
    Transform("link_swap", lambda c: "jnp.exp(" in c, _link_swap),
    Transform("drop_covariate", lambda c: bool(_COEF_X_RE.search(c)), _drop_covariate),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _parses(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def applicable_transforms(code: str) -> list[str]:
    return [t.name for t in MENU if t.applies(code)]


def propose(code: str, *, k: int = 3, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    menu = [t for t in MENU if t.applies(code)]
    out: list[dict] = []
    seen = {code}
    for i in rng.permutation(len(menu)):
        if len(out) >= k:
            break
        t = menu[i]
        new = t.apply(code, rng)
        if new is None or new in seen or not _parses(new):
            continue
        seen.add(new)
        out.append({"name": t.name, "code": new, "rationale": "", "source": "template"})
    return out


__all__ = ["MENU", "Transform", "applicable_transforms", "propose"]
