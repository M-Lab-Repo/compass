#!/usr/bin/env python3
"""Verify that a copied code tree (``figures/``) is a faithful copy of the research code.

    python -I tools/check_provenance.py --research /path/to/code --manifest DIR/provenance.json [--diff]

The manifest names, for every Python module in its directory that was copied from the
research repository, the research source file and the names whose definition is allowed to
differ. This script parses both sides with ``ast``, collects every
top-level definition (functions, classes and assignments, keyed by name) and compares the
source text of each definition line by line. Every name of a copy ends up in one status:

IDENTICAL                 same source text (trailing whitespace ignored)
IDENTICAL_MODULO_IMPORTS  differs only in import statements whose module name the copy
                          rewrote, as ``import_map`` describes (``from llm import``
                          became ``from compass.llm import``); also the module-level
                          statement block of a full copy when its statements differ only in
                          such import lines
DERIVED                   a derived constant (``TYPES``, ``KEYS``) whose value equals the one
                          the research module computes (imported in a ``python -I`` subprocess)
REPOINTED                 allowlisted as legitimately different (path constants, output
                          defaults); the diff is printed with ``--diff``
DROPPED                   a research name allowlisted as intentionally absent from a full copy
                          (by name in ``dropped``, or by a regular expression in
                          ``dropped_patterns``, each of which must match at least one such name)
NEW                       allowlisted as existing only in the copy
DIFFERENT                 FAIL: differs and is not allowlisted
MISSING                   FAIL: a research name absent from a full copy without being
                          allowlisted as dropped, or an allowlisted name absent from the copy
UNCOVERED                 FAIL: a copy name that no research source defines and that is not
                          allowlisted as new
STALE                     FAIL: an allowlist or ``not_covered`` entry that no longer describes
                          the files
UNLISTED                  FAIL: a ``*.py`` file under the manifest's directory (outside
                          ``tests/``) that the manifest names neither in ``files`` nor in
                          ``not_covered``

The module docstring, the module-level import block and the remaining module-level
statements (``matplotlib.use("Agg")``, ``plt.rcParams.update(...)``, the ``__main__`` guard)
are compared as blocks. The statement block fails a full copy when it differs, except for the
exact statements a file lists under ``dropped_statements`` (reported DROPPED) and for
statements that differ only in import lines mapped by ``import_map`` (reported
IDENTICAL_MODULO_IMPORTS, pairwise and in order). The import block
is reported for information unless the manifest says ``"imports": "enforce"``: then, for a full
copy, the copy's import statements minus its ``new_imports`` must equal, after normalisation,
the research statements minus its ``dropped_imports`` (as multisets; a mismatch is DIFFERENT,
an allowlist entry that no longer applies is STALE or MISSING).

``import_map`` maps module prefixes of the copy back to research names: a key matches a module
token equal to it or starting with the key and a dot, the value replaces the prefix, and an
empty value strips it (``compass.agents.validator`` -> ``agents.validator``; ``from compass
import X`` -> ``import X``). Only import lines are touched. A blank line directly after an
import is ignored when comparing a definition modulo imports. The research HEAD and the cleanliness of the listed
source files are reported for information; when HEAD is not the manifest commit, the listed
sources are diffed between the two (``git diff --quiet``) and any source that changed since the
manifest commit is a STALE failure, so a manifest pinned to an old commit cannot pass once its
sources moved. Every ``*.py`` under the manifest's directory
(``__init__.py``, ``tests/`` and ``__pycache__`` excepted) must be a key of ``files`` or of
``not_covered`` (a key ending in ``/`` admits every file under that directory), so a stray
module cannot escape the check. The exit status is 0 only when no item failed.

Standard library only.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

PASS_STATUSES = ("IDENTICAL", "IDENTICAL_MODULO_IMPORTS", "DERIVED", "REPOINTED", "DROPPED", "NEW")
FAIL_STATUSES = ("DIFFERENT", "MISSING", "UNCOVERED", "UNLISTED", "STALE")
STATUS_ORDER = PASS_STATUSES + FAIL_STATUSES

# Runs in a ``python -I`` subprocess with the research root on ``sys.path``; argv[1] is the
# root, argv[2] a JSON object {key: [module, attribute]}. Prints {key: {"value": ...} |
# {"error": ...}} as JSON.
DERIVED_PROBE = r"""
import importlib, json, sys
root, specs = sys.argv[1], json.loads(sys.argv[2])
sys.path.insert(0, root)
out = {}
for key, (module, name) in specs.items():
    try:
        out[key] = {"value": getattr(importlib.import_module(module), name)}
    except Exception as exc:
        out[key] = {"error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(out, default=list))
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

@dataclass
class Definition:
    name: str
    kind: str  # "function", "class" or "assign"
    lineno: int
    text: str  # source lines of the statement, each right-stripped
    node: ast.stmt = field(repr=False)


@dataclass
class Module:
    path: Path
    docstring: str | None
    imports: list[str]  # source text of every module-level import statement
    statements: list[str]  # source text of every other non-definition statement
    defs: dict[str, Definition]
    duplicates: list[str]


def _def_name(node: ast.stmt) -> tuple[str, str] | None:
    """(name, kind) for a definition statement, else None."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name, "function"
    if isinstance(node, ast.ClassDef):
        return node.name, "class"
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id, "assign"
    if isinstance(node, ast.Assign):
        parts = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                parts.append(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                parts.append(ast.unparse(target).strip("()[]"))
            else:
                return None  # attribute or subscript target: not a module-level name
        return " = ".join(parts), "assign"
    return None


def normalise_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _segment(lines: list[str], node: ast.stmt) -> str:
    start = node.lineno
    for deco in getattr(node, "decorator_list", []):
        start = min(start, deco.lineno)
    return normalise_text("\n".join(lines[start - 1:node.end_lineno]))


def parse_module(path: Path) -> Module:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines()
    docstring = ast.get_docstring(tree, clean=False)
    body = tree.body
    if docstring is not None:
        body = body[1:]
    imports: list[str] = []
    statements: list[str] = []
    defs: dict[str, Definition] = {}
    duplicates: list[str] = []
    for node in body:
        named = _def_name(node)
        if named is not None:
            name, kind = named
            if name in defs:
                duplicates.append(name)
            defs[name] = Definition(name, kind, node.lineno, _segment(lines, node), node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(_segment(lines, node))
        else:
            statements.append(_segment(lines, node))
    return Module(path, docstring, imports, statements, defs, duplicates)


_IMPORT_LINE = re.compile(r"^(?P<indent>\s*)(?:from (?P<frm>[\w.]+) import (?P<names>.*)"
                          r"|import (?P<mod>[\w.]+)(?P<rest>.*))$")


def map_module(token: str, import_map: dict[str, str]) -> str | None:
    """The research name of a copy module ``token`` under ``import_map`` (longest key first; a
    key matches the token itself or any ``key.`` prefix; an empty value strips the prefix, so
    the token equal to such a key maps to ``""``). None when no key applies."""
    for key in sorted(import_map, key=len, reverse=True):
        if token == key or token.startswith(key + "."):
            tail = token[len(key):]
            value = import_map[key]
            return value + tail if value else tail[1:]
    return None


def normalise_imports(text: str, import_map: dict[str, str]) -> str:
    """``text`` with the module names of its import lines mapped back to the research
    package names through :func:`map_module`. Only lines that are import statements are
    touched; ``from P import X`` becomes ``import X`` when ``P`` maps to the empty string."""
    out = []
    for line in text.splitlines():
        m = _IMPORT_LINE.match(line)
        if m and m.group("frm"):
            new = map_module(m.group("frm"), import_map)
            if new == "" and not m.group("names").lstrip().startswith("("):
                line = f"{m.group('indent')}import {m.group('names')}"
            elif new:
                line = f"{m.group('indent')}from {new} import {m.group('names')}"
        elif m:
            new = map_module(m.group("mod"), import_map)
            if new:
                line = f"{m.group('indent')}import {new}{m.group('rest')}"
        out.append(line)
    return "\n".join(out)


def _drop_blank_after_imports(text: str) -> str:
    """``text`` without the blank lines that directly follow an import line."""
    out: list[str] = []
    after_import = False
    for line in text.splitlines():
        if after_import and not line.strip():
            continue
        after_import = bool(_IMPORT_LINE.match(line))
        out.append(line)
    return "\n".join(out)


def same_modulo_imports(research: str, copy: str, import_map: dict[str, str]) -> str | None:
    """``""`` when ``copy`` equals ``research`` once its import lines are normalised, a note
    when it does so only after ignoring blank lines that follow an import, else None."""
    normalised = normalise_imports(copy, import_map)
    if normalised == research:
        return ""
    if _drop_blank_after_imports(normalised) == _drop_blank_after_imports(research):
        return "a blank line after an import is ignored"
    return None


def statements_modulo_imports(research: list[str], copy: list[str],
                              import_map: dict[str, str]) -> int | None:
    """The number of statements of ``copy`` that differ from their counterpart in ``research``
    only in import lines (see :func:`same_modulo_imports`), when the two lists match pairwise
    and in order; None when they differ otherwise (a different count included)."""
    if len(research) != len(copy):
        return None
    n = 0
    for r, c in zip(research, copy, strict=True):
        if r == c:
            continue
        if same_modulo_imports(r, c, import_map) is None:
            return None
        n += 1
    return n


def unified_diff(research: str, copy: str, label_research: str, label_copy: str) -> str:
    diff = difflib.unified_diff(research.splitlines(), copy.splitlines(),
                                fromfile=label_research, tofile=label_copy, lineterm="", n=2)
    return "\n".join(diff)


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------

@dataclass
class Item:
    file: str
    name: str
    status: str
    note: str = ""
    diff: str = ""
    category: str = "definition"  # or "statement", "import", "file"

    @property
    def failed(self) -> bool:
        return self.status in FAIL_STATUSES


@dataclass
class FileReport:
    copy: str
    source: str | None
    kind: str
    info: list[str] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)


@dataclass
class Report:
    manifest: Path
    research_root: Path
    manifest_commit: str | None
    files: list[FileReport] = field(default_factory=list)
    git: dict = field(default_factory=dict)
    sweep: dict = field(default_factory=dict)

    @property
    def items(self) -> list[Item]:
        return [item for fr in self.files for item in fr.items]

    @property
    def failures(self) -> list[Item]:
        return [item for item in self.items if item.failed]

    def counts(self) -> dict[str, int]:
        counts = dict.fromkeys(STATUS_ORDER, 0)
        for item in self.items:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    @property
    def ok(self) -> bool:
        return not self.failures


# ---------------------------------------------------------------------------
# Derived constants and git state
# ---------------------------------------------------------------------------

def derived_values(research_root: Path, specs: dict[str, tuple[str, str]]) -> dict[str, dict]:
    """{key: {"value": ...} | {"error": ...}} by importing each ``(module, attribute)`` of
    ``specs`` in an isolated subprocess with ``research_root`` first on ``sys.path``."""
    if not specs:
        return {}
    cmd = [sys.executable, "-I", "-c", DERIVED_PROBE, str(research_root),
           json.dumps({k: list(v) for k, v in specs.items()})]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {k: {"error": f"subprocess failed: {exc}"} for k in specs}
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip().splitlines()
        tail = err[-1] if err else f"exit {proc.returncode}"
        return {k: {"error": f"subprocess exit {proc.returncode}: {tail}"} for k in specs}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        return {k: {"error": f"unreadable probe output: {exc}"} for k in specs}


def git_info(research_root: Path, sources: list[str], manifest_commit: str | None = None) -> dict:
    """HEAD of the research checkout, the ``git status --porcelain`` of ``sources`` and, when
    HEAD is not ``manifest_commit``, whether any of ``sources`` changed between the two
    (``sources_unchanged``; ``sources_changed`` lists them; ``diff_error`` when the manifest
    commit cannot be compared)."""
    info: dict = {"head": None, "clean": None, "dirty": [], "error": None,
                  "sources_unchanged": None, "sources_changed": [], "diff_error": None}

    def git(*args: str) -> subprocess.CompletedProcess | None:
        try:
            return subprocess.run(["git", "-C", str(research_root), *args], capture_output=True,
                                  text=True, timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            info["error"] = f"git unavailable: {exc}"
            return None

    head = git("rev-parse", "HEAD")
    if head is None:
        return info
    if head.returncode != 0:
        info["error"] = head.stderr.strip() or "git rev-parse HEAD failed"
        return info
    info["head"] = head.stdout.strip()
    status = git("status", "--porcelain", "--", *sources)
    if status is None:
        return info
    if status.returncode != 0:
        info["error"] = status.stderr.strip() or "git status failed"
        return info
    info["dirty"] = [line for line in status.stdout.splitlines() if line.strip()]
    info["clean"] = not info["dirty"]
    if not manifest_commit:
        return info
    if info["head"] == manifest_commit:
        info["sources_unchanged"] = True
        return info
    diff = git("diff", "--quiet", manifest_commit, "HEAD", "--", *sources)
    if diff is None:
        return info
    if diff.returncode == 0:
        info["sources_unchanged"] = True
    elif diff.returncode == 1:
        names = git("diff", "--name-only", "--relative", manifest_commit, "HEAD", "--", *sources)
        changed = (names.stdout.split("\n") if names is not None and names.returncode == 0
                   else ["(could not list the changed sources)"])
        info["sources_unchanged"] = False
        info["sources_changed"] = [n for n in changed if n.strip()]
    else:
        err = diff.stderr.strip().splitlines()
        info["diff_error"] = err[-1] if err else f"git diff exit {diff.returncode}"
    return info


# ---------------------------------------------------------------------------
# Per-file check
# ---------------------------------------------------------------------------

def _block_info(label: str, research: list[str], copy: list[str]) -> str:
    if research == copy:
        return f"{label}: identical ({len(copy)})"
    if not copy:
        return f"{label}: none in the copy (research has {len(research)})"
    r_lines = "\n".join(research).splitlines()
    c_lines = "\n".join(copy).splitlines()
    changed = sum(1 for line in difflib.ndiff(r_lines, c_lines) if line[:2] in ("+ ", "- "))
    return f"{label}: rewritten ({changed} lines differ)"


def _enforce_imports(copy_key: str, source: str, research: Module, copy: Module,
                     import_map: dict[str, str], new_imports: list[str],
                     dropped_imports: list[str], info: list[str]) -> list[Item]:
    """Items of the enforced import-block comparison of a full copy (see the module doc)."""
    items: list[Item] = []
    copy_imports = list(copy.imports)
    expected = list(research.imports)
    for text in new_imports:
        if text in copy_imports:
            copy_imports.remove(text)
        else:
            items.append(Item(copy_key, text, "MISSING", "allowlisted as a new import but absent "
                              "from the copy", category="import"))
    for text in dropped_imports:
        if text in copy.imports:
            items.append(Item(copy_key, text, "STALE", "allowlisted as a dropped import but still "
                              "present in the copy", category="import"))
        if text in expected:
            expected.remove(text)
        else:
            items.append(Item(copy_key, text, "STALE", "allowlisted as a dropped import but the "
                              "research file has no such import", category="import"))
    normalised = sorted(normalise_imports(text, import_map) for text in copy_imports)
    expected.sort()
    if normalised == expected:
        info.append(f"imports: enforced, {len(expected)} statements match after normalisation "
                    f"({len(new_imports)} new, {len(dropped_imports)} dropped)")
    else:
        items.append(Item(copy_key, "<imports>", "DIFFERENT",
                          "module-level imports differ from the research file after "
                          "normalisation (new_imports and dropped_imports applied)",
                          unified_diff("\n".join(expected), "\n".join(normalised),
                                       f"research {source} <imports> minus dropped_imports",
                                       f"copy {copy_key} <imports> normalised, minus new_imports"),
                          category="import"))
    return items


def check_file(copy_key: str, spec: dict, copy_root: Path, research_root: Path,
               import_map: dict[str, str], derived: dict[str, dict],
               import_mode: str = "report") -> FileReport:
    kind = spec.get("kind", "full")
    source = spec.get("source")
    report = FileReport(copy_key, source, kind)
    copy_path = copy_root / copy_key
    if not copy_path.is_file():
        report.items.append(Item(copy_key, "<module>", "MISSING", f"copy not found: {copy_path}"))
        return report
    copy = parse_module(copy_path)
    for name in copy.duplicates:
        report.info.append(f"note: {name} is defined more than once in the copy; the last wins")

    repointed = list(spec.get("repointed", []))
    dropped = list(spec.get("dropped", []))
    dropped_patterns = [re.compile(p) for p in spec.get("dropped_patterns", [])]
    patterns_used: set[str] = set()
    new = list(spec.get("new", []))
    extra_sources: dict[str, str] = dict(spec.get("extra_sources", {}))
    derived_spec: dict[str, dict] = dict(spec.get("derived", {}))
    dropped_statements = list(spec.get("dropped_statements", []))
    new_imports = list(spec.get("new_imports", []))
    dropped_imports = list(spec.get("dropped_imports", []))

    # research definitions: primary source plus the extra sources of inlined names
    research: Module | None = None
    extra_modules: dict[str, Module] = {}
    if source is not None:
        src_path = research_root / source
        if not src_path.is_file():
            report.items.append(Item(copy_key, "<module>", "MISSING",
                                     f"research source not found: {src_path}"))
            return report
        research = parse_module(src_path)
    for rel in sorted(set(extra_sources.values())):
        path = research_root / rel
        if not path.is_file():
            report.items.append(Item(copy_key, "<module>", "MISSING",
                                     f"extra research source not found: {path}"))
            return report
        extra_modules[rel] = parse_module(path)

    def research_def(name: str) -> tuple[Definition | None, str | None]:
        rel = extra_sources.get(name)
        if rel is not None:
            return extra_modules[rel].defs.get(name), rel
        if research is not None and name in research.defs:
            return research.defs[name], source
        return None, None

    if research is not None:
        report.info.append("docstring: identical" if copy.docstring == research.docstring
                           else "docstring: rewritten")
        if import_mode == "enforce" and kind == "full":
            report.items.extend(_enforce_imports(copy_key, source, research, copy, import_map,
                                                 new_imports, dropped_imports, report.info))
        else:
            report.info.append(_block_info("imports", research.imports, copy.imports))
        if kind == "full":
            expected = list(research.statements)
            for text in dropped_statements:
                first = text.splitlines()[0][:80] if text.strip() else repr(text)
                if text not in expected:
                    report.items.append(Item(copy_key, first, "STALE",
                                             "allowlisted as a dropped statement but the research "
                                             "file has no such module-level statement",
                                             category="statement"))
                    continue
                expected.remove(text)
                if text in copy.statements:
                    report.items.append(Item(copy_key, first, "STALE",
                                             "allowlisted as a dropped statement but still "
                                             "present in the copy", category="statement"))
                else:
                    report.items.append(Item(copy_key, first, "DROPPED",
                                             f"module-level statement of {source} absent from "
                                             "the copy", text, category="statement"))
            modulo = statements_modulo_imports(expected, copy.statements, import_map)
            if expected == copy.statements:
                report.info.append(f"statements: identical ({len(copy.statements)}"
                                   + (f", {len(dropped_statements)} dropped)" if dropped_statements
                                      else ")"))
            elif modulo is not None:
                report.items.append(Item(
                    copy_key, "<statements>", "IDENTICAL_MODULO_IMPORTS",
                    f"{modulo} of {len(copy.statements)} module-level statement(s) differ only "
                    "in import lines", category="statement"))
            else:
                report.items.append(Item(
                    copy_key, "<statements>", "DIFFERENT",
                    "module-level statements differ from the research file",
                    unified_diff("\n\n".join(expected), "\n\n".join(copy.statements),
                                 f"research {source} <statements>",
                                 f"copy {copy_key} <statements>"), category="statement"))
        else:
            foreign = [s for s in copy.statements
                       if s not in research.statements
                       and not any(same_modulo_imports(r, s, import_map) is not None
                                   for r in research.statements)]
            report.info.append(f"statements: {len(copy.statements)} in copy, "
                               f"{len(foreign)} not found in research")
            for stmt in foreign:
                report.items.append(Item(copy_key, "<statement>", "UNCOVERED",
                                         stmt.splitlines()[0][:80]))

    # every name of the copy, in file order
    for name, cdef in sorted(copy.defs.items(), key=lambda kv: kv[1].lineno):
        if name in derived_spec:
            d = derived_spec[name]
            key = f"{d['module']}:{d['name']}"
            probe = derived.get(key, {"error": "not probed"})
            try:
                copy_value = ast.literal_eval(cdef.node.value)  # type: ignore[attr-defined]
            except (ValueError, AttributeError, SyntaxError):
                report.items.append(Item(copy_key, name, "DIFFERENT",
                                         "the copied value is not a literal"))
                continue
            if "error" in probe:
                report.items.append(Item(copy_key, name, "DIFFERENT",
                                         f"research {key} could not be imported: {probe['error']}"))
            elif probe["value"] == copy_value:
                report.items.append(Item(copy_key, name, "DERIVED",
                                         f"equals research {key} ({len(copy_value)} entries)"
                                         if hasattr(copy_value, "__len__") else
                                         f"equals research {key}"))
            else:
                report.items.append(Item(
                    copy_key, name, "DIFFERENT", f"value differs from research {key}",
                    unified_diff(json.dumps(probe["value"], indent=1, default=str),
                                 json.dumps(copy_value, indent=1, default=str),
                                 f"research {key}", f"copy {copy_key}:{name}")))
            continue
        rdef, rel = research_def(name)
        if name in new:
            note = "" if rdef is None else f"not compared; {rel}:{rdef.lineno} also defines this name"
            report.items.append(Item(copy_key, name, "NEW", note))
            continue
        if rdef is None:
            report.items.append(Item(copy_key, name, "UNCOVERED",
                                     "no research definition and not allowlisted as new"))
            continue
        where = f"{rel}:{rdef.lineno}"
        if rdef.text == cdef.text:
            note = where
            if name in repointed:
                note += "; allowlisted as repointed but identical"
            report.items.append(Item(copy_key, name, "IDENTICAL", note))
        elif (modulo := same_modulo_imports(rdef.text, cdef.text, import_map)) is not None:
            report.items.append(Item(copy_key, name, "IDENTICAL_MODULO_IMPORTS",
                                     where + (f"; {modulo}" if modulo else "")))
        else:
            diff = unified_diff(rdef.text, cdef.text, f"research {where}",
                                f"copy {copy_key}:{cdef.lineno}")
            status = "REPOINTED" if name in repointed else "DIFFERENT"
            report.items.append(Item(copy_key, name, status, where, diff))

    # research names absent from the copy
    if research is not None:
        for name, rdef in sorted(research.defs.items(), key=lambda kv: kv[1].lineno):
            if name in copy.defs:
                continue
            if name in dropped:
                report.items.append(Item(copy_key, name, "DROPPED", f"{source}:{rdef.lineno}",
                                         rdef.text))
            elif any(p.search(name) for p in dropped_patterns):
                patterns_used.update(p.pattern for p in dropped_patterns if p.search(name))
                report.items.append(Item(copy_key, name, "DROPPED",
                                         f"{source}:{rdef.lineno}; matches dropped_patterns",
                                         rdef.text))
            elif kind == "full":
                report.items.append(Item(copy_key, name, "MISSING",
                                         f"{source}:{rdef.lineno} has no counterpart in the copy"))

    # allowlist hygiene
    for name in repointed + new:
        if name not in copy.defs:
            report.items.append(Item(copy_key, name, "MISSING",
                                     "allowlisted in the manifest but absent from the copy"))
    for name in dropped:
        if research is None or name not in research.defs:
            report.items.append(Item(copy_key, name, "STALE",
                                     "allowlisted as dropped but the research file does not define it"))
        elif name in copy.defs:
            report.items.append(Item(copy_key, name, "STALE",
                                     "allowlisted as dropped but present in the copy"))
    for name in derived_spec:
        if name not in copy.defs:
            report.items.append(Item(copy_key, name, "MISSING",
                                     "derived constant listed in the manifest but absent from the copy"))
    for p in dropped_patterns:
        if p.pattern not in patterns_used:
            report.items.append(Item(copy_key, p.pattern, "STALE",
                                     "allowlisted dropped pattern matches no research definition "
                                     "absent from the copy"))
    return report


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def sweep_unlisted(copy_root: Path, files: dict, not_covered: dict | list,
                   sweep: dict) -> list[FileReport]:
    """One failing FileReport per ``*.py`` under ``copy_root`` that the manifest names
    neither in ``files`` nor in ``not_covered`` (UNLISTED), and one per ``not_covered`` entry
    whose file does not exist (STALE). ``__init__.py``, ``tests/`` and ``__pycache__`` are
    skipped. ``sweep`` receives the counts for the report."""
    dirs = [k for k in not_covered if k.endswith("/")]
    listed = set(files) | {k for k in not_covered if not k.endswith("/")}
    swept: list[str] = []
    reports: list[FileReport] = []
    for path in sorted(copy_root.rglob("*.py")):
        rel = path.relative_to(copy_root).as_posix()
        parts = path.relative_to(copy_root).parts
        if path.name == "__init__.py" or "tests" in parts[:-1] or "__pycache__" in parts:
            continue
        swept.append(rel)
        if rel not in listed and not any(rel.startswith(d) for d in dirs):
            fr = FileReport(rel, None, "unlisted")
            fr.items.append(Item(rel, "<module>", "UNLISTED",
                                 "a Python file under the manifest's directory that it names "
                                 "neither in files nor in not_covered", category="file"))
            reports.append(fr)
    for rel in not_covered:
        if rel.endswith("/"):
            if not (copy_root / rel).is_dir():
                fr = FileReport(rel, None, "not_covered")
                fr.items.append(Item(rel, "<directory>", "STALE",
                                     "listed in not_covered but the directory does not exist",
                                     category="file"))
                reports.append(fr)
        elif not (copy_root / rel).is_file():
            fr = FileReport(rel, None, "not_covered")
            fr.items.append(Item(rel, "<module>", "STALE",
                                 "listed in not_covered but the file does not exist",
                                 category="file"))
            reports.append(fr)
    sweep.update(swept=swept, unlisted=[fr.copy for fr in reports if fr.kind == "unlisted"],
                 stale=[fr.copy for fr in reports if fr.kind == "not_covered"])
    return reports


def run(manifest_path: Path, research_root: Path,
        derived_fn: Callable[[Path, dict[str, tuple[str, str]]], dict[str, dict]] | None = None,
        ) -> Report:
    manifest_path = Path(manifest_path).resolve()
    research_root = Path(research_root).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    copy_root = manifest_path.parent  # the copied tree the manifest describes
    import_map = dict(manifest.get("import_map") or {})
    import_mode = manifest.get("imports", "report")
    if import_mode not in ("report", "enforce"):
        raise SystemExit(f"{manifest_path}: \"imports\" must be \"report\" or \"enforce\", "
                         f"not {import_mode!r}")
    files: dict[str, dict] = manifest.get("files", {})
    report = Report(manifest_path, research_root, manifest.get("research_commit"))

    specs: dict[str, tuple[str, str]] = {}
    for spec in files.values():
        for d in spec.get("derived", {}).values():
            specs[f"{d['module']}:{d['name']}"] = (d["module"], d["name"])
    derived = (derived_fn or derived_values)(research_root, specs) if specs else {}

    for copy_key, spec in files.items():
        report.files.append(check_file(copy_key, spec, copy_root, research_root, import_map,
                                       derived, import_mode))
    report.files.extend(sweep_unlisted(copy_root, files, manifest.get("not_covered") or {},
                                       report.sweep))

    sources: list[str] = []
    for spec in files.values():
        for rel in [spec.get("source"), *spec.get("extra_sources", {}).values()]:
            if rel and rel not in sources:
                sources.append(rel)
    report.git = git_info(research_root, sources, report.manifest_commit)
    report.git["sources"] = sources
    commit = (report.manifest_commit or "")[:7]
    if report.git.get("sources_changed"):
        fr = FileReport("<research>", None, "research")
        for rel in report.git["sources_changed"]:
            fr.items.append(Item("<research>", rel, "STALE",
                                 f"changed in the research tree since manifest commit {commit}; "
                                 "the copies were taken from that commit"))
        report.files.append(fr)
    elif report.git.get("diff_error"):
        fr = FileReport("<research>", None, "research")
        fr.items.append(Item("<research>", f"research_commit {commit}", "STALE",
                             f"cannot be compared with HEAD: {report.git['diff_error']}"))
        report.files.append(fr)
    return report


def format_report(report: Report, show_diff: bool = False) -> str:
    out: list[str] = []
    out.append(f"manifest  {report.manifest}")
    out.append(f"research  {report.research_root}")
    kinds = {"full": "full copy", "subset": "subset", "derived": "derived constants",
             "new": "new code", "unlisted": "NOT IN THE MANIFEST",
             "not_covered": "not_covered entry", "research": "sources since the manifest commit"}
    for fr in report.files:
        src = (f"<- {fr.source}" if fr.source
               else "" if fr.kind == "research" else "(no research source)")
        out.append("")
        out.append("  ".join(part for part in (f"== {fr.copy}", src,
                                               f"[{kinds.get(fr.kind, fr.kind)}]") if part))
        for line in fr.info:
            out.append(f"   {line}")
        for item in fr.items:
            note = f"  ({item.note})" if item.note else ""
            out.append(f"   {item.status:<24} {item.name}{note}")
            if item.diff and (show_diff or item.status in FAIL_STATUSES):
                out.extend(f"      | {line}" for line in item.diff.splitlines())

    out.append("")
    sweep = report.sweep
    if sweep:
        n_bad = len(sweep.get("unlisted", [])) + len(sweep.get("stale", []))
        out.append(f"copy sweep         {len(sweep.get('swept', []))} *.py files outside tests/; "
                   + ("all listed in files or not_covered" if not n_bad
                      else f"{len(sweep.get('unlisted', []))} unlisted, "
                           f"{len(sweep.get('stale', []))} stale not_covered entries"))
    git = report.git
    head, commit = git.get("head"), report.manifest_commit
    if not head:
        out.append("research HEAD      unknown" + (f"  ({git['error']})" if git.get("error") else ""))
        if commit:
            out.append(f"manifest commit    {commit}  (not compared)")
    elif not commit:
        out.append(f"research HEAD      {head}  (the manifest names no research_commit)")
    elif head == commit:
        out.append(f"research HEAD      {head}  (equals the manifest commit)")
    elif git.get("sources_unchanged"):
        out.append(f"research HEAD      {head}  (sources unchanged since manifest commit {commit})")
    elif git.get("sources_changed"):
        out.append(f"research HEAD      {head}  ({len(git['sources_changed'])} listed source(s) "
                   f"changed since manifest commit {commit}; see STALE)")
    else:
        out.append(f"research HEAD      {head}  (manifest commit {commit}: "
                   f"{git.get('diff_error') or git.get('error') or 'not compared'})")
    if git.get("clean") is None:
        out.append("research tree      status unknown")
    elif git["clean"]:
        out.append(f"research tree      clean for the {len(git.get('sources', []))} listed sources")
    else:
        out.append(f"research tree      {len(git['dirty'])} listed source(s) modified:")
        out.extend(f"   {line}" for line in git["dirty"])

    out.append("")
    counts = report.counts()
    width = max(len(s) for s in STATUS_ORDER)
    out.append("summary")
    for status in STATUS_ORDER:
        flag = "  FAIL" if status in FAIL_STATUSES and counts[status] else ""
        cats: dict[str, int] = {}
        for item in report.items:
            if item.status == status:
                cats[item.category] = cats.get(item.category, 0) + 1
        breakdown = ""
        if len(cats) > 1:
            breakdown = "  (" + ", ".join(f"{n} {c}s" for c, n in sorted(cats.items())) + ")"
        out.append(f"   {status:<{width}}  {counts[status]:>4}{flag}{breakdown}")
    for status, n in counts.items():
        if status not in STATUS_ORDER:
            out.append(f"   {status:<{width}}  {n:>4}")
    n_fail = len(report.failures)
    out.append(f"   {'total':<{width}}  {len(report.items):>4}")
    out.append("")
    out.append("PASS: every definition is accounted for" if n_fail == 0
               else f"FAIL: {n_fail} item(s) are DIFFERENT, MISSING, UNCOVERED, UNLISTED or STALE")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--research", required=True, type=Path,
                    help="root of the research checkout (the directory holding bench/ and llm.py)")
    ap.add_argument("--manifest", type=Path, required=True,
                    help="the provenance manifest of the copied tree (figures/provenance.json)")
    ap.add_argument("--diff", action="store_true",
                    help="print the diff of every REPOINTED and DROPPED definition "
                         "(failing items always show theirs)")
    args = ap.parse_args(argv)
    if not args.manifest.is_file():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    if not args.research.is_dir():
        print(f"research root not found: {args.research}", file=sys.stderr)
        return 2
    report = run(args.manifest, args.research)
    print(format_report(report, show_diff=args.diff))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
