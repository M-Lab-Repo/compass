#!/usr/bin/env python3
"""Copy the paper's figure pipeline into ``figures/``: the per-panel scripts of the research
``figure2/``, ``figure3/`` and ``figure4/`` directories (run, score, create, protocols, the
engines behind them) and the four benchmark modules they import from the research ``bench/``.

The research scripts import each other by sibling name when run as files (``import
figure2_common as common``) and as package modules otherwise (``from figure2 import
figure2_common``), and the benchmark helpers as ``from bench import dgp``. The copies live under
``figures/`` in the same packages, so with ``figures/`` standing in for the research root every
such import resolves unchanged; only the imports of the pipeline itself (``llm``, ``graph``,
``judge``, ``agents.*``, ``state``) are rewritten to the installed ``compass`` package.

    python tools/sync_figures.py --research /path/to/research/code          # write
    python tools/sync_figures.py --research /path/to/research/code --check  # verify
    python tools/sync_figures.py --hash FILE START END                      # the hash of a line window

Rules, in order:

1. Import rewrite, guided by the AST (only the line of an ``import`` or ``from ... import``
   node is edited; indentation, aliases, trailing comments and ``# noqa`` markers are kept; text
   inside strings is never touched): a pipeline module becomes ``compass.<module>``; ``bench``,
   ``figure2``, ``figure3`` and ``figure4`` stay as they are, and every such import must name a
   module that is copied; an import of any other module of the research tree is an error.
2. Hashed line windows (``HASHED_PATCHES``): a few lines of four scripts describe or register an
   arm on the research tree's eighth real dataset, the one entry beyond the seven textbook
   datasets of the paper (``TEXTBOOK_KEYS``), which this repository does not distribute. Each
   rule names the file, the number of lines, the sha256 of the research lines and the text that
   replaces them; the window must occur exactly once, so a research edit stops the tool and the
   rule is re-pinned with ``--hash``.
3. Derived rules for the same dataset: its key is read off ``REAL_ORDER`` in
   ``figure3/figure3_common.py`` and off the registry of ``bench/realdata.py`` (the two must
   agree) and removed from ``REAL_ORDER`` and ``DATASET_NAME``, from the usage line of
   ``figure3/figure3c_create.py``, and from ``bench/realdata.py`` (its constants, loader and
   registry entry, each with the comment block directly above; the name in ``__all__``; the
   docstring lines that announce the entry). The arms run on that dataset are read off
   ``ARM_PROTOCOLS`` in ``figure2/figure2_protocols.py``, and no spelling of the key, nor an
   arm name, may survive in any copy.
4. Roots (``PATCHES``): the MCMC trace cache of ``figure2/figure2_run.py`` and the data
   directory of ``bench/realdata.py`` honour ``COMPASS_CACHE_DIR`` and ``COMPASS_DATA_DIR``,
   with the research layout as the default.
5. ``figures/provenance.json`` records the research commit, whether the sources were clean, the
   date, the sha256 of every source and of every copy, every rewritten window (line number, the
   copied text, the sha256 of the research text) and the allowlists that
   ``tools/check_provenance.py`` honours:

       python tools/check_provenance.py --research /path/to/code --manifest figures/provenance.json

The per-figure ``README.md`` and ``flow.svg`` files, ``render.py``, ``compare_pdf.py`` and the
two manifests are maintained in this repository and are not copies. ``--check`` writes nothing
and exits 1 when any copy or manifest entry (the date and commit fields aside) differs from what
the research tree would produce now, or when a file under ``figures/`` is neither a copy nor a
package-only file.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

# research path == path under figures/ (file names and packages are kept)
FIGURE2 = ("__init__", "figure2_common", "figure2_compose", "figure2_protocols", "figure2_run",
           "figure2_score", "figure2a_create", "figure2a_run", "figure2a_score", "figure2b_create",
           "figure2b_run", "figure2b_score", "figure2c_create", "figure2c_run", "figure2c_score",
           "figure2d_create", "figure2d_run", "figure2d_score")
FIGURE3 = ("__init__", "figure3_common", "figure3_compose", "figure3_protocols", "figure3_run",
           "figure3_score", "figure3a_create", "figure3a_run", "figure3a_score", "figure3b_create",
           "figure3b_run", "figure3c_create", "figure3c_run", "figure3c_score", "figure3e_run",
           "figure3e_score")
FIGURE4 = ("__init__", "figure4_create", "figure4_run", "figure4_score")
BENCH = ("__init__", "dgp", "realdata", "holdout", "control_proposer")
MODULES: tuple[str, ...] = (
    *(f"figure2/{m}.py" for m in FIGURE2), *(f"figure3/{m}.py" for m in FIGURE3),
    *(f"figure4/{m}.py" for m in FIGURE4), *(f"bench/{m}.py" for m in BENCH),
)
PIPELINE_ROOTS = ("ablation", "crossval", "graph", "judge", "llm", "resolution", "schemas",
                  "state", "tournament", "agents", "fixtures")
KEEP_ROOTS = ("bench", "figure2", "figure3", "figure4")
SIBLINGS = frozenset(Path(m).stem for m in MODULES) - {"__init__"}  # bare sibling imports stay

PACKAGE_ONLY = ("render.py", "compare_pdf.py", "reference_manifest.json", "provenance.json",
                "requirements.txt", "README.md",
                "figure2/README.md", "figure2/flow.svg", "figure3/README.md", "figure3/flow.svg",
                "figure4/README.md", "figure4/flow.svg")
PACKAGE_ONLY_DIRS = ("out/", ".test_cache/", "runs/")

# the seven textbook datasets of the paper (research bench/realdata.py order); anything else in
# REAL_ORDER or in the registry is the eighth dataset this repository does not distribute
TEXTBOOK_KEYS: tuple[str, ...] = ("roaches", "ucbadmit", "baseball", "seeds", "radon", "kidiq", "golf")
F3_COMMON = "figure3/figure3_common.py"
F3C = "figure3/figure3c_create.py"
F2_RUN = "figure2/figure2_run.py"
F2_PROTOCOLS = "figure2/figure2_protocols.py"
REALDATA = "bench/realdata.py"
KEPT_ARM = "e1_anthropic_realdata_x2"  # the arm the usage line of figure3c_create.py keeps
EIGHTH_DATASET_REMOVALS = 7  # five uppercase constants, the loader and the registry entry
# the module docstring of realdata.py announces the eighth entry after the seven textbook datasets
REALDATA_DOC = re.compile(r"(textbook misspecified starting model), plus the .*? as the eighth entry\.\n",
                          re.DOTALL)

# rule 2: (number of research lines, sha256 of those lines, the copy's text), per file
HASHED_PATCHES: dict[str, list[tuple[int, str, str]]] = {
    F2_PROTOCOLS: [
        (2, "a3d56feb91a72456125ca5a66e7a8d3ca675aae5cb02f4fa522883b825911513",
         "Figure 2 reads the first ten arms (CENSUS_ARMS); the eleventh, run by the same engine, is read\n"
         "by Figure 4 (real data at the default budget). The\n"),
        (2, "8170d2a2c30fd3903504594617c9df37b81e6d492cbcb899a0ac761c5a088d91", ""),
        (3, "4971984b9d41e60efcba387053c9f1ff230014e1ffb69a205e5b54792fda3826",
         "    # one more census arm, same engine, read by Figure 4 (not by Figure 2)\n"),
        (1, "1378fc7ee56afbada9915c9a873f3404e9984a867375ef103d92c35933161a9c",
         '    if name.startswith("real-data") or "realdata" in name:\n'),
    ],
    "figure3/figure3_protocols.py": [
        (2, "4a6ae755dbdf25ba5bdda9ce7a105a520b6f2aa506557504049e49c277f1bc5f",
         "nothing heavy. Panel c's arm is a census arm and lives in figure2/figure2_protocols.py\n"
         "(MATRIX_ARMS names it).\n"),
        (1, "1f0ddcc9a906326606d5f5fe489404ce08ea87c237e7fd3f2cf63911ed4050bd",
         'MATRIX_ARMS = ("real-data_sonnet-4.6",)  # panel c: a census arm, figure2\'s engine\n'),
    ],
    "figure3/figure3c_run.py": [
        (1, "1c36e6ca6034cc338929e18fb42fd1c06e0dacfef07e65d4e4386a6d64330d70",
         "  --arm ARM        run only this arm; default: the arm of the panel\n"),
        (6, "8f38815c4f8a381406c2b1b557f400f41bef100e0ad77a4a6efb370fd95f04bb",
         "  ARM      the one name in ARMS below:\n"
         "             real-data_sonnet-4.6        seven textbook datasets, three hidden splits, double MCMC budget\n"
         "                                         (the same arm Figure 2 draws; committed as e1_anthropic_realdata_x2)\n"),
        (3, "703232481061e7c5f94b2a80873ed5f097f15a39114c4dfd118a14a19acfbec5",
         "  PREFIX   <dataset>_seed0[_s<split>]_r1 with <dataset> one of roaches, ucbadmit, baseball, seeds, radon,\n"
         "           kidiq, golf, and <split> 1 or 2 (split 0 has no suffix).\n"
         "           Examples: --only roaches (3 runs), --only golf_seed0_s1 (1 run).\n"),
        (4, "aa996560b197f9157abd308f793bd8dd5edf4bd6c14ad230e3c410aa0a4f59cb",
         "This is a census arm: each run is the Figure 2 protocol (the agent revises the textbook model,\n"
         "the live rules judge every candidate, template controls are fitted and judged), run by\n"
         "figure2/figure2_run.py. 21 runs, about 15 minutes and $0.80 each. A run whose record exists is skipped.\n"),
        (1, "5245c12ca2ba28bea9a7c4cbfa998fb28ea38a4813c99ab49035b0bff8d71bf5",
         'ARMS = ("real-data_sonnet-4.6",)\n'),
    ],
    "figure3/figure3c_score.py": [
        (1, "7b749eb8cb23ca42042986bdbc4250bec2a8e69d7778e3d9e6bc2ae0a89225d1",
         "  --arm ARM        score only this arm; default: the arm of the panel\n"),
        (3, "1cbcd2206a7f1753baa19105c54bcf309ea3659199895e64f01cd564c6d2c2c8",
         "  ARM      real-data_sonnet-4.6 (figure3c_run.py says what it is).\n"
         "  LIST     provider:model pairs separated by commas, e.g. openai:gpt-5. Without --readers the arm gets its\n"
         '           GPT-5 reader (the matrix\'s "Independent model" column).\n'),
        (1, "5245c12ca2ba28bea9a7c4cbfa998fb28ea38a4813c99ab49035b0bff8d71bf5",
         'ARMS = ("real-data_sonnet-4.6",)\n'),
    ],
}

# rule 4: exact research line -> copy line (every anchor must occur exactly once)
CACHE_RESEARCH = 'CACHE = CODE_DIR / ".test_cache"          # MCMC trace cache shared with bench.run\n'
CACHE_COPY = ('CACHE = Path(os.environ.get("COMPASS_CACHE_DIR", CODE_DIR / ".test_cache"))'
              '  # MCMC trace cache; research: CODE_DIR / ".test_cache"\n')
DATA_DIR_RESEARCH = 'DATA_DIR = Path(__file__).resolve().parent.parent / "tutorials" / "demos" / "data"\n'
DATA_DIR_COPY = ('DATA_DIR = Path(os.environ.get("COMPASS_DATA_DIR",\n'
                 '                                Path(__file__).resolve().parents[3] / "tutorials" / "demos" / "data"))'
                 '  # research: parent.parent / "tutorials" / "demos" / "data"\n')
PATCHES: dict[str, list[tuple[str, str]]] = {F2_RUN: [(CACHE_RESEARCH, CACHE_COPY)],
                                              REALDATA: [(DATA_DIR_RESEARCH, DATA_DIR_COPY)]}
NEW_IMPORTS: dict[str, list[str]] = {F2_RUN: ["import os"], REALDATA: ["import os"]}

# the definitions tools/check_provenance.py reports REPOINTED, DROPPED (by name or pattern)
REPOINTED: dict[str, list[str]] = {
    F3_COMMON: ["REAL_ORDER", "DATASET_NAME"],
    F2_PROTOCOLS: ["ARM_PROTOCOLS", "condition"],
    "figure3/figure3_protocols.py": ["MATRIX_ARMS"],
    "figure3/figure3c_run.py": ["ARMS"],
    "figure3/figure3c_score.py": ["ARMS"],
    F2_RUN: ["CACHE"],
    REALDATA: ["DATA_DIR", "_DATASETS", "__all__"],
}
DROPPED: dict[str, list[str]] = {F2_PROTOCOLS: ["REAL_PUBLISHED"]}
DROPPED_PATTERNS: dict[str, list[str]] = {
    REALDATA: ["^[A-Z]+_[A-Z]+_(BUDGET|FILE|CHANNELS|INT_SCALE|ORIGINAL)$", "^load_[a-z_]+$"]}

RE_FROM = re.compile(r"^(?P<indent>\s*)from (?P<mod>[\w.]+) import ")
RE_IMPORT_AS = re.compile(r"^(?P<indent>\s*)import (?P<mod>[\w.]+) as (?P<alias>\w+)(?P<tail>[ \t]*(?:#.*)?)$")
RE_IMPORT = re.compile(r"^(?P<indent>\s*)import (?P<mod>\w+)(?P<tail>[ \t]*(?:#.*)?)$")
RE_REAL_ORDER = re.compile(r"^REAL_ORDER = \((?P<items>.*)\)(?P<tail>.*)$", re.MULTILINE)
RE_DATASET_NAME = re.compile(r"^DATASET_NAME = \{(?P<items>.*)\}(?P<tail>.*)$", re.MULTILINE)
RE_USAGE = re.compile(r"^(?P<head>[ \t]*python figure3/figure3c_create\.py bench/arms/" + re.escape(KEPT_ARM)
                      + r") bench/arms/[A-Za-z0-9_]+ (?P<tail>--out \S+)$", re.MULTILINE)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- rule 1: imports

def map_module(mod: str) -> str | None:
    """The copy's dotted module name for a research dotted name: ``compass.`` for the pipeline,
    unchanged for the kept roots, None when no rule applies."""
    root = mod.split(".")[0]
    if root in PIPELINE_ROOTS:
        return "compass." + mod
    if root in KEEP_ROOTS:
        return mod
    return None


def _resolves_in_research(root: str, research: Path) -> bool:
    return (research / f"{root}.py").is_file() or (research / root / "__init__.py").is_file()


def _destination_exists(mod: str, compass: Path) -> bool:
    parts = mod.split(".")
    if parts[0] == "compass":
        rel = "/".join(parts[1:])
        return (compass / f"{rel}.py").is_file() or (compass / rel / "__init__.py").is_file()
    if parts[0] in KEEP_ROOTS:
        return len(parts) == 1 or "/".join(parts) + ".py" in MODULES
    return False


def rewrite_imports(source: str, rel: str, research: Path, compass: Path) -> tuple[str, int]:
    """Return (rewritten source, number of import nodes rewritten)."""
    tree = ast.parse(source, filename=rel)
    lines = source.splitlines(keepends=True)
    touched: set[int] = set()
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                continue
            mod = node.module
            new_mod = map_module(mod)
            if new_mod is None:
                if _resolves_in_research(mod.split(".")[0], research):
                    raise SystemExit(f"{rel}:{node.lineno}: import of research module {mod!r} has no rule")
                continue
            if not _destination_exists(new_mod, compass):
                raise SystemExit(f"{rel}:{node.lineno}: import target {new_mod!r} is not copied")
            if new_mod in KEEP_ROOTS:  # from bench import dgp: the named module must be copied too
                for alias in node.names:
                    if f"{new_mod}/{alias.name}.py" not in MODULES:
                        raise SystemExit(f"{rel}:{node.lineno}: {new_mod!r} is copied but "
                                         f"{alias.name!r} is not")
            if new_mod == mod:
                continue
            i = node.lineno - 1
            m = RE_FROM.match(lines[i])
            if not m or m.group("mod") != mod:
                raise SystemExit(f"{rel}:{node.lineno}: unrecognised import line: {lines[i].rstrip()}")
            new_line = f"{m.group('indent')}from {new_mod} import " + lines[i][m.end():]
        elif isinstance(node, ast.Import):
            hits = [a for a in node.names if map_module(a.name) is not None]
            if not hits:
                for a in node.names:
                    if _resolves_in_research(a.name.split(".")[0], research):
                        raise SystemExit(f"{rel}:{node.lineno}: import of research module {a.name!r} has no rule")
                continue
            if len(node.names) != 1:
                raise SystemExit(f"{rel}:{node.lineno}: several names on one import line (needs a hand rule)")
            alias = node.names[0]
            new_mod = map_module(alias.name)
            if not _destination_exists(new_mod, compass):
                raise SystemExit(f"{rel}:{node.lineno}: import target {new_mod!r} is not copied")
            if new_mod == alias.name:
                continue
            i = node.lineno - 1
            if alias.asname:
                m = RE_IMPORT_AS.match(lines[i])
                if not m or m.group("mod") != alias.name:
                    raise SystemExit(f"{rel}:{node.lineno}: unrecognised import line: {lines[i].rstrip()}")
                new_line = f"{m.group('indent')}import {new_mod} as {m.group('alias')}{m.group('tail')}\n"
            else:
                if "." in alias.name:
                    raise SystemExit(f"{rel}:{node.lineno}: dotted import without alias (needs a hand rule)")
                m = RE_IMPORT.match(lines[i])
                if not m or m.group("mod") != alias.name:
                    raise SystemExit(f"{rel}:{node.lineno}: unrecognised import line: {lines[i].rstrip()}")
                head, _, leaf = new_mod.rpartition(".")
                new_line = f"{m.group('indent')}from {head} import {leaf}{m.group('tail')}\n"
        else:
            continue
        if i in touched:
            raise SystemExit(f"{rel}:{node.lineno}: two project imports on one line")
        if not lines[i].endswith("\n"):
            new_line = new_line.rstrip("\n")
        lines[i] = new_line
        touched.add(i)
        n += 1
    return "".join(lines), n


# --------------------------------------------------------------------------- rule 2: hashed windows

def window_hash(lines: list[str]) -> str:
    return sha256_text("".join(lines))


def apply_hashed(text: str, rel: str) -> tuple[str, list[dict]]:
    """Replace every window of ``HASHED_PATCHES[rel]``; each must occur exactly once."""
    rewritten: list[dict] = []
    for n_lines, sha, replacement in HASHED_PATCHES.get(rel, []):
        lines = text.splitlines(keepends=True)
        hits = [i for i in range(len(lines) - n_lines + 1) if window_hash(lines[i:i + n_lines]) == sha]
        if len(hits) != 1:
            raise SystemExit(f"{rel}: hashed window {sha[:12]} ({n_lines} line(s)) found {len(hits)} times, "
                             f"expected 1; re-pin the rule with --hash")
        i = hits[0]
        rewritten.append({"line": i + 1, "research_lines": n_lines, "copy": replacement.rstrip("\n"),
                          "research_sha256": sha})
        text = "".join(lines[:i]) + replacement + "".join(lines[i + n_lines:])
    return text, rewritten


# --------------------------------------------------------------------------- rule 3: the eighth dataset

def extra_keys(text: str, rel: str = F3_COMMON) -> list[str]:
    """The keys of ``REAL_ORDER`` beyond ``TEXTBOOK_KEYS``: exactly one, the eighth dataset."""
    tree = ast.parse(text, filename=rel)
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "REAL_ORDER"):
            if not isinstance(node.value, ast.Tuple) or not all(
                    isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.value.elts):
                raise SystemExit(f"{rel}: REAL_ORDER is not a tuple of strings")
            keys = [e.value for e in node.value.elts if e.value not in TEXTBOOK_KEYS]
            if len(keys) != 1:
                raise SystemExit(f"{rel}: REAL_ORDER names {len(keys)} dataset(s) beyond the "
                                 f"textbook seven, expected 1")
            return keys
    raise SystemExit(f"{rel}: REAL_ORDER not found")


def removed_arms(text: str, keys: list[str], rel: str = F2_PROTOCOLS) -> list[str]:
    """The names of the ``ARM_PROTOCOLS`` entries whose ``datasets=`` name one of ``keys``."""
    tree = ast.parse(text, filename=rel)
    for node in tree.body:
        if _names_of(node) == ["ARM_PROTOCOLS"] and isinstance(node, ast.Assign) \
                and isinstance(node.value, ast.Dict):
            arms = []
            for k, v in zip(node.value.keys, node.value.values, strict=True):
                if not (isinstance(k, ast.Constant) and isinstance(v, ast.Call)):
                    continue
                for kw in v.keywords:
                    if kw.arg == "datasets" and isinstance(kw.value, ast.Tuple) and any(
                            isinstance(e, ast.Constant) and e.value in keys for e in kw.value.elts):
                        arms.append(k.value)
            return arms
    raise SystemExit(f"{rel}: the ARM_PROTOCOLS dict was not found")


def mention_regex(keys: list[str], arms: list[str] = ()) -> re.Pattern:
    """Every spelling of the keys a copy must not carry (as written, with hyphens, without
    separators) and the names of the removed arms (and their first word, the baseline part);
    matched without regard to case."""
    forms: set[str] = set()
    for arm in arms:
        forms.update({arm, arm.split("_")[0]})
    for key in keys:
        forms.update({key, key.replace("_", "-"), key.replace("_", "")})
    return re.compile("|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True)), re.IGNORECASE)


def _rewritten(text: str, start: int, old: str, new: str) -> dict:
    return {"line": text[:start].count("\n") + 1, "research_lines": old.count("\n") or 1,
            "copy": new.rstrip("\n"), "research_sha256": sha256_text(old)}


def patch_figure3_common(text: str, rel: str, keys: list[str]) -> tuple[str, list[dict]]:
    """``REAL_ORDER`` and ``DATASET_NAME`` without the eighth dataset."""
    rewritten: list[dict] = []
    hits = RE_REAL_ORDER.findall(text)
    m = RE_REAL_ORDER.search(text)
    if m is None or len(hits) != 1:
        raise SystemExit(f"{rel}: REAL_ORDER line found {len(hits)} times, expected 1")
    items = m.group("items")
    for key in keys:
        token = f', "{key}"'
        if items.count(token) != 1:
            raise SystemExit(f"{rel}: REAL_ORDER does not end with the eighth dataset's key")
        items = items.replace(token, "")
    new_line = f"REAL_ORDER = ({items}){m.group('tail')}"
    rewritten.append(_rewritten(text, m.start(), m.group(0) + "\n", new_line + "\n"))
    text = text[:m.start()] + new_line + text[m.end():]

    hits = RE_DATASET_NAME.findall(text)
    m = RE_DATASET_NAME.search(text)
    if m is None or len(hits) != 1:
        raise SystemExit(f"{rel}: DATASET_NAME line found {len(hits)} times, expected 1")
    items = m.group("items")
    for key in keys:
        entry = re.compile(r'"' + re.escape(key) + r'": "[^"]*"(?:, )?')
        if len(entry.findall(items)) != 1:
            raise SystemExit(f"{rel}: DATASET_NAME has no entry for the eighth dataset")
        items = entry.sub("", items).rstrip(", ")
    new_line = f"DATASET_NAME = {{{items}}}{m.group('tail')}"
    rewritten.append(_rewritten(text, m.start(), m.group(0) + "\n", new_line + "\n"))
    text = text[:m.start()] + new_line + text[m.end():]
    return text, rewritten


def patch_figure3c(text: str, rel: str, keys: list[str]) -> tuple[str, list[dict]]:
    """The usage line of the docstring without its second arm directory."""
    hits = list(RE_USAGE.finditer(text))
    if len(hits) != 1:
        raise SystemExit(f"{rel}: usage line with a second arm directory found {len(hits)} times, expected 1")
    m = hits[0]
    new_line = f"{m.group('head')} {m.group('tail')}"
    rewritten = [_rewritten(text, m.start(), m.group(0) + "\n", new_line + "\n")]
    return text[:m.start()] + new_line + text[m.end():], rewritten


def _names_of(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    return []


def _registry(tree: ast.Module, rel: str) -> ast.Tuple:
    for node in tree.body:
        if _names_of(node) == ["_DATASETS"] and isinstance(node, ast.Assign) \
                and isinstance(node.value, ast.Tuple):
            return node.value
    raise SystemExit(f"{rel}: the _DATASETS tuple was not found")


def _entry_key(elt: ast.expr) -> str | None:
    if not isinstance(elt, ast.Call):
        return None
    for kw in elt.keywords:
        if kw.arg == "key" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def eighth_dataset_key(tree: ast.Module, rel: str) -> str:
    """The key of the one ``_DATASETS`` entry beyond ``TEXTBOOK_KEYS``."""
    keys = [k for k in (_entry_key(e) for e in _registry(tree, rel).elts) if k not in TEXTBOOK_KEYS]
    if len(keys) != 1 or keys[0] is None:
        raise SystemExit(f"{rel}: _DATASETS has {len(keys)} entr(ies) beyond the textbook seven, expected 1")
    return keys[0]


def _drop_lines(lines: list[str], drop: set[int]) -> str:
    """``lines`` without the 0-based indices in ``drop``; where a removed block leaves more
    than two consecutive blank lines, the surplus blank lines are removed as well."""
    out: list[str] = []
    seams: list[int] = []
    i = 0
    while i < len(lines):
        if i in drop:
            while i < len(lines) and i in drop:
                i += 1
            seams.append(len(out))
            continue
        out.append(lines[i])
        i += 1
    for pos in reversed(sorted(set(seams))):
        start = pos
        while start > 0 and not out[start - 1].strip():
            start -= 1
        end = pos
        while end < len(out) and not out[end].strip():
            end += 1
        if end - start > 2:
            del out[start + 2:end]
    return "".join(out)


def _replace_once(text: str, old: str, new: str, rel: str) -> str:
    k = text.count(old)
    if k != 1:
        raise SystemExit(f"{rel}: anchor found {k} times, expected 1: {old[:70]!r}")
    return text.replace(old, new)


def remove_eighth_dataset(text: str, rel: str, keys: list[str], source_sha: str = "") -> tuple[str, list[dict]]:
    """The copy of ``bench/realdata.py`` without its eighth dataset: the key is read off the
    registry (and must be the one ``REAL_ORDER`` names); the constants named after it, its loader
    and its registry entry go, each with the comment block directly above; the name leaves
    ``__all__`` and the docstring loses the lines that announce the entry."""
    tree = ast.parse(text, filename=rel)
    key = eighth_dataset_key(tree, rel)
    if keys and key != keys[0]:
        raise SystemExit(f"{rel}: the registry's eighth dataset is not the one {F3_COMMON} names")
    prefix, loader = key.upper() + "_", "load_" + key
    lines = text.splitlines(keepends=True)
    drop: set[int] = set()
    removed = 0

    def drop_block(first: int, last: int) -> None:
        i = first - 1
        while i >= 0 and lines[i].lstrip().startswith("#"):
            i -= 1
        drop.update(range(i + 1, last + 1))

    for node in tree.body:
        names = _names_of(node)
        if any(n.startswith(prefix) or n == loader for n in names):
            drop_block(node.lineno - 1, node.end_lineno - 1)
            removed += 1
    for elt in _registry(tree, rel).elts:
        if _entry_key(elt) == key:
            drop_block(elt.lineno - 1, elt.end_lineno - 1)
            removed += 1
    if removed != EIGHTH_DATASET_REMOVALS:
        raise SystemExit(f"{rel}: the eighth-dataset rule removed {removed} definition(s), "
                         f"expected {EIGHTH_DATASET_REMOVALS}; the research module changed shape")
    text = _drop_lines(lines, drop)
    text = _replace_once(text, f', "{prefix}BUDGET"]', "]", rel)
    text, k = REALDATA_DOC.subn(r"\1.\n", text, count=1)
    if k != 1:
        raise SystemExit(f"{rel}: the module docstring does not announce the eighth entry as expected")
    rewritten = [{"line": 1, "research_lines": len(lines), "copy": "(the module without its eighth dataset)",
                  "research_sha256": source_sha or sha256_text("".join(lines))}]
    return text, rewritten


# --------------------------------------------------------------------------- rule 4: roots

def _insert_import(text: str, stmt: str, rel: str) -> str:
    """Add the one-line ``import X`` among the module's top-level bare imports, in sorted order."""
    tree = ast.parse(text, filename=rel)
    lines = text.splitlines(keepends=True)
    name = stmt.split()[1]
    bare = [n for n in tree.body if isinstance(n, ast.Import) and len(n.names) == 1
            and not n.names[0].asname and "." not in n.names[0].name]
    if not bare:
        raise SystemExit(f"{rel}: no top-level import to anchor {stmt.strip()!r} on")
    if any(n.names[0].name == name for n in bare):
        return text
    later = [n for n in bare if n.names[0].name > name]
    at = (later[0].lineno - 1) if later else bare[-1].end_lineno
    return "".join(lines[:at]) + stmt + "".join(lines[at:])


def apply_patches(text: str, rel: str) -> tuple[str, list[dict]]:
    rewritten: list[dict] = []
    for old, new in PATCHES.get(rel, []):
        k = text.count(old)
        if k != 1:
            raise SystemExit(f"{rel}: patch anchor found {k} times, expected 1: {old[:70]!r}")
        rewritten.append(_rewritten(text, text.index(old), old, new))
        text = text.replace(old, new)
    return text, rewritten


# --------------------------------------------------------------------------- generate

def generate(research: Path, compass: Path, rel: str, keys: list[str], sweep: re.Pattern
             ) -> tuple[str, list[dict], str, int]:
    """(copy text, rewritten windows sorted by line, sha256 of the source, import nodes rewritten)."""
    src_path = research / rel
    if not src_path.is_file():
        raise SystemExit(f"missing research module: {src_path}")
    source = src_path.read_bytes()
    source_text = source.decode("utf-8")
    text, n_imports = rewrite_imports(source_text, rel, research, compass)
    for stmt in NEW_IMPORTS.get(rel, []):  # first, so that every recorded line number is final
        text = _insert_import(text, stmt + "\n", rel)
    text, rewritten = apply_hashed(text, rel)
    if rel == F3_COMMON:
        text, more = patch_figure3_common(text, rel, keys)
        rewritten += more
    elif rel == F3C:
        text, more = patch_figure3c(text, rel, keys)
        rewritten += more
    elif rel == REALDATA:
        text, more = remove_eighth_dataset(text, rel, keys, sha256_text(source_text))
        rewritten += more
    text, more = apply_patches(text, rel)
    rewritten += more
    m = sweep.search(text)
    if m:
        lineno = text[:m.start()].count("\n") + 1
        raise SystemExit(f"{rel}:{lineno}: the copy still mentions the eighth dataset or its arm "
                         f"({m.group(0)!r}); add a rule")
    try:
        ast.parse(text, filename=rel)
    except SyntaxError as exc:  # pragma: no cover - guards the rules
        raise SystemExit(f"{rel}: result does not parse: {exc}") from exc
    return text, sorted(rewritten, key=lambda r: r["line"]), sha256_text(source_text), n_imports


def research_state(research: Path) -> tuple[str, list[str]]:
    try:
        head = subprocess.run(["git", "-C", str(research), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(research), "status", "--porcelain", "--", *MODULES],
                                capture_output=True, text=True, check=True).stdout
        return head, [line for line in status.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)", []


COMMENT = (
    "Provenance of figures/: copies of the research repository's figure pipeline (figure2/, figure3/, "
    "figure4/) and of the four benchmark modules it imports (bench/), written by tools/sync_figures.py "
    "(regenerate with --research; verify with --check). Every entry of `files` names a copy (path under "
    "figures/), its research source (the same path under the research root), the sha256 of both, and "
    "the windows the tool rewrote (`rewritten`: line number, the copied text, the sha256 of the research "
    "text; they drop the research tree's eighth real dataset and the arm run on it, which this "
    "repository does not distribute, and make two roots configurable). Import lines of the pipeline "
    "are rewritten to the compass package (`import_map` maps them back). tools/check_provenance.py "
    "--manifest figures/provenance.json compares every top-level definition and statement with the "
    "research file, enforces the import block, and fails on any difference the manifest does not "
    "allow (`repointed`, `dropped`, `dropped_patterns`, `new_imports`). `research_commit` is the "
    "research HEAD the copies were generated from; `research_sources_dirty` lists uncommitted changes "
    "to the sources at that time (empty when the sources were clean)."
)
VOLATILE = ("copied_on", "research_commit", "research_commit_short", "research_sources_dirty")


def build_manifest(entries: dict[str, tuple[str, list[dict], str]], head: str, dirty: list[str],
                   today: str) -> dict:
    files: dict[str, dict] = {}
    for rel, (text, rewritten, sha_src) in entries.items():
        lines = ", ".join(str(r["line"]) for r in rewritten)
        files[rel] = {
            "source": rel, "kind": "full",
            "repointed": REPOINTED.get(rel, []), "dropped": DROPPED.get(rel, []),
            "dropped_patterns": DROPPED_PATTERNS.get(rel, []), "new": [],
            "new_imports": NEW_IMPORTS.get(rel, []), "dropped_imports": [],
            "sha256_source": sha_src, "sha256_copy": sha256_text(text),
            "rewritten": rewritten,
            "notes": (f"window(s) at line(s) {lines} rewritten; everything else verbatim modulo imports")
                     if rewritten else "verbatim modulo imports",
        }
    return {
        "_comment": COMMENT,
        "generator": "tools/sync_figures.py",
        "research_commit": head,
        "research_commit_short": head[:7],
        "research_sources_dirty": dirty,
        "copied_on": today,
        "imports": "enforce",
        "import_map": {"compass": ""},
        "files": files,
        "not_covered": {
            "render.py": "new driver (no research source): renders the manuscript's PDFs",
            "compare_pdf.py": "new tool (no research source): date-neutral comparison of PDFs",
        },
    }


def unexpected_package_files(package: Path) -> list[str]:
    found = []
    for p in sorted(package.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(package).as_posix()
        if rel in MODULES or rel in PACKAGE_ONLY or rel.startswith(PACKAGE_ONLY_DIRS):
            continue
        found.append(rel)
    return found


def _diff(current: str | None, new: str, rel: str, src: str) -> None:
    sys.stdout.writelines(difflib.unified_diff(
        (current or "").splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"package/{rel}", tofile=src))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--research", type=Path,
                    help="research code directory holding figure2/, figure3/, figure4/ and bench/")
    ap.add_argument("--package", type=Path,
                    default=Path(__file__).resolve().parents[1] / "figures",
                    help="destination directory (default: figures/ next to tools/)")
    ap.add_argument("--compass", type=Path, default=None,
                    help="the compass package the rewritten imports must resolve in "
                         "(default: src/compass next to --package)")
    ap.add_argument("--check", action="store_true",
                    help="write nothing; exit 1 if any file differs from the result")
    ap.add_argument("--hash", nargs=3, metavar=("FILE", "START", "END"),
                    help="print the sha256 of lines START..END (1-based, inclusive) of FILE and exit")
    args = ap.parse_args(argv)
    if args.hash:
        path, start, end = Path(args.hash[0]), int(args.hash[1]), int(args.hash[2])
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)[start - 1:end]
        print(f"{len(lines)} {window_hash(lines)}")
        return 0
    if args.research is None:
        ap.error("--research is required")
    research, package = args.research.resolve(), args.package.resolve()
    compass = args.compass.resolve() if args.compass else package.parent / "src" / "compass"
    if not research.is_dir():
        print(f"research root not found: {research}", file=sys.stderr)
        return 2
    f3_common = research / F3_COMMON
    if not f3_common.is_file():
        raise SystemExit(f"missing research module: {f3_common}")
    keys = extra_keys(f3_common.read_text(encoding="utf-8"))
    arms = removed_arms((research / F2_PROTOCOLS).read_text(encoding="utf-8"), keys)
    sweep = mention_regex(keys, arms)

    entries: dict[str, tuple[str, list[dict], str]] = {}
    stale = 0
    n_windows = n_imports_total = 0
    for rel in MODULES:
        text, rewritten, sha_src, n_imports = generate(research, compass, rel, keys, sweep)
        entries[rel] = (text, rewritten, sha_src)
        n_windows += len(rewritten)
        n_imports_total += n_imports
        target = package / rel
        current = target.read_text(encoding="utf-8") if target.is_file() else None
        changed = current != text
        print(f"{rel:32s} rewrote {n_imports:2d} import node(s), {len(rewritten)} window(s), "
              f"{'changed' if changed else 'unchanged'}")
        if not changed:
            continue
        if args.check:
            stale += 1
            _diff(current, text, rel, f"research/{rel} (rewritten)")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")

    head, dirty = research_state(research)
    manifest = build_manifest(entries, head, dirty, date.today().isoformat())
    manifest_path = package / "provenance.json"
    current_manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                        if manifest_path.is_file() else None)
    if args.check:
        if current_manifest is None:
            stale += 1
            print("provenance.json                  missing")
        else:
            expected = dict(manifest)
            for key in VOLATILE:
                expected[key] = current_manifest.get(key)
            if expected != current_manifest:
                stale += 1
                _diff(json.dumps(current_manifest, indent=1) + "\n",
                      json.dumps(expected, indent=1) + "\n", "provenance.json",
                      "research (regenerated manifest, dates and commit as recorded)")
            else:
                print("provenance.json                  unchanged (dates and commit as recorded)")
    else:
        package.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        print(f"provenance.json                  written ({len(manifest['files'])} files, "
              f"research {head[:7]}{' DIRTY' if dirty else ''})")

    extra = unexpected_package_files(package)
    for rel in extra:
        print(f"{rel:32s} NOT MIRRORED: not a copy and not a package-only file")
    print(f"totals: {n_imports_total} import nodes rewritten, {n_windows} windows rewritten in "
          f"{len(MODULES)} modules")
    state = head + (" (research sources have uncommitted changes)" if dirty else "")
    if args.check:
        stale += len(extra)
        if stale:
            print(f"{stale} file(s) differ from research {state}")
            return 1
        print(f"{len(MODULES)} copies up to date with research {state}")
        return 0
    print(f"synced {len(MODULES)} copies from research {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
