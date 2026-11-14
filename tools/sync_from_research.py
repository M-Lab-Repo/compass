#!/usr/bin/env python3
"""Mirror the COMPASS research modules into this package.

The research code lives as flat top-level modules (``llm.py``, ``agents/``, ``fixtures/`` ...)
in the research repository. This package ships the same files under ``src/compass/`` with
every project import prefixed ``compass.`` and a handful of package-specific wording patches.
This tool regenerates the package tree from a research checkout so that the two never drift
by hand:

    python tools/sync_from_research.py --research /path/to/research/code          # write
    python tools/sync_from_research.py --research /path/to/research/code --check  # verify

Rules, in order:
1. Import rewrite, guided by the AST so that import-looking text inside strings and
   docstrings is never touched. Only the line of an ``import`` or ``from ... import`` node whose
   root is a project module is rewritten, keeping indentation and trailing comments:
   ``from llm import x``  ->  ``from compass.llm import x``;
   ``from agents.scout import y``  ->  ``from compass.agents.scout import y``;
   ``import crossval``  ->  ``from compass import crossval`` (single, un-aliased name only).
   The rewrite is idempotent on already-prefixed text, which is what makes ``--check`` meaningful.
2. Package-only patches applied as exact string replacements. Each anchor must occur exactly
   once or the tool stops with an error; nothing is ever skipped silently.

Anything the tool cannot handle (a missing source file, an import form it does not recognise,
a missing patch anchor) is a hard failure that names the file and line.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import subprocess
import sys
from pathlib import Path

ROOT_MODULES = ("ablation", "crossval", "graph", "judge", "llm", "resolution", "schemas",
                "state", "tournament")
PACKAGES = ("agents", "fixtures")
PROJECT_ROOTS = frozenset(ROOT_MODULES) | frozenset(PACKAGES)

# every file mirrored from the research tree (src/compass/__init__.py is package-only)
MODULES = [
    *(f"{m}.py" for m in ROOT_MODULES),
    "agents/__init__.py", "agents/analyst.py", "agents/retrieval.py", "agents/scout.py",
    "agents/strategist.py", "agents/synthesizer.py", "agents/validator.py",
    "fixtures/__init__.py", "fixtures/frozen_outputs.py", "fixtures/golden_data.py",
    "fixtures/golden_model.py", "fixtures/mock_corpus.py",
]

_ROOTS = "|".join(ROOT_MODULES)
RE_FROM = re.compile(
    rf"^(?P<indent>\s*)from (?P<mod>(?:{_ROOTS})|agents(?:\.\w+)*|fixtures(?:\.\w+)*) import "
)
RE_IMPORT = re.compile(rf"^(?P<indent>\s*)import (?P<mod>{_ROOTS})(?P<tail>\s*(?:#.*)?)$")

# package-only wording: (anchor in the research text, replacement). Anchors must be unique.
PATCHES: dict[str, list[tuple[str, str]]] = {
    "agents/validator.py": [
        ("\n    import cmdstanpy\n",
         "\n    try:\n"
         "        import cmdstanpy\n"
         "    except ImportError as exc:  # pragma: no cover - env-dependent\n"
         "        raise ImportError(\n"
         '            "The Stan backend requires cmdstanpy: "\n'
         "            \"pip install 'compass-refine[stan]' (plus a CmdStan toolchain)\"\n"
         "        ) from exc\n"),
    ],
    "agents/retrieval.py": [
        ("\"pip install 'bayesian-refinement[retrieval]' \"",
         "\"pip install 'compass-refine[retrieval]' \""),
    ],
}


def rewrite_imports(source: str, rel: str) -> tuple[str, int]:
    """Return (rewritten source, number of import lines rewritten)."""
    tree = ast.parse(source, filename=rel)
    lines = source.splitlines(keepends=True)
    touched: set[int] = set()
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level or not node.module or node.module.split(".")[0] not in PROJECT_ROOTS:
                continue
            pattern, repl = RE_FROM, r"\g<indent>from compass.\g<mod> import "
        elif isinstance(node, ast.Import):
            hits = [a for a in node.names if a.name.split(".")[0] in PROJECT_ROOTS]
            if not hits:
                continue
            if len(node.names) != 1 or hits[0].asname or "." in hits[0].name:
                raise SystemExit(f"{rel}:{node.lineno}: unsupported import form "
                                 f"(needs a hand rule): {lines[node.lineno - 1].rstrip()}")
            pattern, repl = RE_IMPORT, r"\g<indent>from compass import \g<mod>\g<tail>"
        else:
            continue
        i = node.lineno - 1
        if i in touched:
            raise SystemExit(f"{rel}:{node.lineno}: two project imports on one line")
        new, k = pattern.subn(repl, lines[i], count=1)
        if k != 1:
            raise SystemExit(f"{rel}:{node.lineno}: unrecognised import line: {lines[i].rstrip()}")
        lines[i] = new
        touched.add(i)
        n += 1
    return "".join(lines), n


def apply_patches(source: str, rel: str) -> tuple[str, int]:
    """Apply the package-only patches for ``rel``; every anchor must occur exactly once."""
    n = 0
    for old, new in PATCHES.get(rel, []):
        count = source.count(old)
        if count != 1:
            raise SystemExit(f"{rel}: patch anchor found {count} times, expected 1: {old!r}")
        source = source.replace(old, new)
        n += 1
    return source, n


def generate(research: Path, rel: str) -> tuple[str, int, int]:
    src_path = research / rel
    if not src_path.is_file():
        raise SystemExit(f"missing research module: {src_path}")
    text, n_imports = rewrite_imports(src_path.read_text(), rel)
    text, n_patches = apply_patches(text, rel)
    return text, n_imports, n_patches


PACKAGE_ONLY = ("__init__.py",)  # files that live only in the package, by design


def unexpected_package_files(package: Path) -> list[str]:
    """Python files under the package that are neither mirrored modules nor package-only
    files; such a file would never be checked by this tool, so it is reported."""
    found = []
    for p in sorted(package.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(package).as_posix()
        if rel not in MODULES and rel not in PACKAGE_ONLY:
            found.append(rel)
    return found


def research_head(research: Path) -> str:
    try:
        head = subprocess.run(["git", "-C", str(research), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(research), "status", "--porcelain", "--",
                                *MODULES], capture_output=True, text=True, check=True).stdout
        return head + (" (research modules have uncommitted changes)" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--research", required=True, type=Path,
                    help="research code directory holding llm.py, agents/, fixtures/ ...")
    ap.add_argument("--package", type=Path,
                    default=Path(__file__).resolve().parents[1] / "src" / "compass",
                    help="package directory to write (default: src/compass next to tools/)")
    ap.add_argument("--check", action="store_true",
                    help="write nothing; exit 1 if any package file differs from the result")
    args = ap.parse_args(argv)
    research, package = args.research.resolve(), args.package.resolve()

    stale = 0
    for rel in MODULES:
        text, n_imports, n_patches = generate(research, rel)
        target = package / rel
        current = target.read_text() if target.is_file() else None
        changed = current != text
        status = "changed" if changed else "unchanged"
        print(f"{rel:32s} rewrote {n_imports:2d} import line(s), patched {n_patches} anchor(s), {status}")
        if not changed:
            continue
        if args.check:
            stale += 1
            sys.stdout.writelines(difflib.unified_diff(
                (current or "").splitlines(keepends=True), text.splitlines(keepends=True),
                fromfile=f"package/{rel}", tofile=f"research/{rel} (rewritten)"))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
    extra = unexpected_package_files(package)
    for rel in extra:
        print(f"{rel:32s} NOT MIRRORED: not in MODULES and not a package-only file")
    head = research_head(research)
    if args.check:
        stale += len(extra)
        if stale:
            print(f"{stale} file(s) differ from research {head}")
            return 1
        print(f"{len(MODULES)} files up to date with research {head}")
        return 0
    print(f"synced {len(MODULES)} files from research {head}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
