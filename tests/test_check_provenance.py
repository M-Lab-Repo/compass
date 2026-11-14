"""Tests of ``tools/check_provenance.py`` on tiny synthetic research and copy trees.

The checker is imported by path, so these tests need neither a research checkout nor the
``conftest.py`` of this directory. Every tree is written to ``tmp_path``; the only
subprocesses are one ``python -I`` probe of a two-line synthetic module and one ``git init``.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CHECKER_PATH = Path(__file__).resolve().parents[1] / "tools" / "check_provenance.py"
COMMIT = "0123456789abcdef0123456789abcdef01234567"

RESEARCH_X = '''"""Research module."""

from __future__ import annotations

import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
OUT = CODE_DIR / "out"
A, B = (1, 2)


def f(a):
    return a + 1


def g():
    from bench.x import f
    return f(1)


def h():
    return "research"


if __name__ == "__main__":
    print(h())
'''

RESEARCH_Y = '''"""Second research module holding an inlined name."""

INLINED = ("p", "q")
OTHER = 3
'''

COPY_X = '''"""Research module."""

from __future__ import annotations

from pathlib import Path

from mirror import paths

OUT = paths.out_dir()
A, B = (1, 2)
INLINED = ("p", "q")


def f(a):
    return a + 1


def g():
    from mirror.x import f
    return f(1)


def h():
    return "{h_body}"
{extra}

if __name__ == "__main__":
    {guard}
'''


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_provenance_under_test", CHECKER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return _load_checker()


@pytest.fixture
def real_git():
    """Requested by the tests that exercise ``git_info`` itself (see ``_no_git``)."""
    return True


@pytest.fixture(autouse=True)
def _no_git(request, monkeypatch, checker):
    """Keep the runs offline and environment independent: no git subprocess unless a test
    opts in by requesting the ``real_git`` fixture."""
    if "real_git" in request.fixturenames:
        return
    monkeypatch.setattr(checker, "git_info",
                        lambda root, sources, commit=None: {
                            "head": None, "clean": None, "dirty": [], "error": "stubbed",
                            "sources_unchanged": None, "sources_changed": [],
                            "diff_error": None})


def base_manifest() -> dict:
    return {
        "research_commit": COMMIT,
        "import_map": {"mirror.x": "bench.x"},
        "files": {
            "mirror/x.py": {
                "source": "bench/x.py",
                "kind": "full",
                "repointed": ["OUT"],
                "dropped": ["CODE_DIR"],
                "new": [],
                "extra_sources": {"INLINED": "bench/y.py"},
            },
        },
    }


def make_trees(tmp_path: Path, *, h_body: str = "research", extra: str = "",
               guard: str = "print(h())", manifest: dict | None = None,
               copy_x: str | None = None) -> tuple[Path, Path]:
    """Write a research tree and a figures tree; returns (manifest path, research root)."""
    research = tmp_path / "research"
    (research / "bench").mkdir(parents=True)
    (research / "bench" / "__init__.py").write_text("")
    (research / "bench" / "x.py").write_text(RESEARCH_X)
    (research / "bench" / "y.py").write_text(RESEARCH_Y)
    (research / "bench" / "dgp.py").write_text('TYPES = ["a", "b"]\n')
    figures = tmp_path / "figures"
    (figures / "mirror").mkdir(parents=True)
    text = copy_x if copy_x is not None else COPY_X.format(h_body=h_body, extra=extra,
                                                            guard=guard)
    (figures / "mirror" / "x.py").write_text(text)
    manifest_path = figures / "provenance.json"
    manifest_path.write_text(json.dumps(manifest if manifest is not None else base_manifest()))
    return manifest_path, research


def statuses(report, file: str = "mirror/x.py") -> dict[str, str]:
    return {item.name: item.status for fr in report.files if fr.copy == file
            for item in fr.items}


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def test_identical_repointed_dropped_and_inlined_pass(checker, tmp_path):
    manifest, research = make_trees(tmp_path)
    report = checker.run(manifest, research)
    got = statuses(report)
    assert got["f"] == "IDENTICAL"
    assert got["A, B"] == "IDENTICAL", "tuple assignments are compared too"
    assert got["INLINED"] == "IDENTICAL", "inlined names are read from extra_sources"
    assert got["h"] == "IDENTICAL"
    assert got["OUT"] == "REPOINTED"
    assert got["CODE_DIR"] == "DROPPED"
    assert report.ok
    assert report.failures == []
    assert report.counts()["DIFFERENT"] == 0
    info = report.files[0].info
    assert "docstring: identical" in info
    assert "statements: identical (1)" in info
    assert any(line.startswith("imports: rewritten (") for line in info)


def test_import_only_change_is_identical_modulo_imports(checker, tmp_path):
    manifest, research = make_trees(tmp_path)
    report = checker.run(manifest, research)
    assert statuses(report)["g"] == "IDENTICAL_MODULO_IMPORTS"


def test_module_name_change_outside_an_import_line_is_different(checker, tmp_path):
    # the same textual change on a non-import line must not be forgiven
    copy = COPY_X.format(h_body="research", extra="", guard="print(h())").replace(
        "    from mirror.x import f\n    return f(1)",
        "    from bench.x import f\n    return f(1) if 'bench.x' else 'mirror.x'")
    manifest, research = make_trees(tmp_path, copy_x=copy)
    report = checker.run(manifest, research)
    assert statuses(report)["g"] == "DIFFERENT"
    assert not report.ok


def test_changed_function_not_allowlisted_fails(checker, tmp_path, capsys):
    manifest, research = make_trees(tmp_path, h_body="copy")
    report = checker.run(manifest, research)
    assert statuses(report)["h"] == "DIFFERENT"
    assert not report.ok
    assert [item.name for item in report.failures] == ["h"]
    rc = checker.main(["--research", str(research), "--manifest", str(manifest)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "DIFFERENT                h" in out
    assert '-    return "research"' in out and '+    return "copy"' in out, \
        "failing items print their diff even without --diff"
    assert out.rstrip().endswith(
        "FAIL: 1 item(s) are DIFFERENT, MISSING, UNCOVERED, UNLISTED or STALE")


def test_body_edit_in_a_function_that_only_repoints_an_import_is_different(checker, tmp_path):
    # g legitimately differs in its import line (IDENTICAL_MODULO_IMPORTS); an edit to its
    # body must not ride along on that allowance
    copy = COPY_X.format(h_body="research", extra="", guard="print(h())").replace(
        "    from mirror.x import f\n    return f(1)", "    from mirror.x import f\n    return f(2)")
    manifest_path, research = make_trees(tmp_path, copy_x=copy)
    report = checker.run(manifest_path, research)
    assert statuses(report)["g"] == "DIFFERENT"
    assert not report.ok
    assert checker.main(["--research", str(research), "--manifest", str(manifest_path)]) == 1


def test_allowlisted_repointed_function_passes(checker, tmp_path):
    manifest = base_manifest()
    manifest["files"]["mirror/x.py"]["repointed"].append("h")
    manifest_path, research = make_trees(tmp_path, h_body="copy", manifest=manifest)
    report = checker.run(manifest_path, research)
    assert statuses(report)["h"] == "REPOINTED"
    assert report.ok


def test_uncovered_extra_name_fails_unless_allowlisted_as_new(checker, tmp_path):
    extra = "\n\ndef extra():\n    return 0\n"
    manifest_path, research = make_trees(tmp_path, extra=extra)
    report = checker.run(manifest_path, research)
    assert statuses(report)["extra"] == "UNCOVERED"
    assert not report.ok

    manifest = base_manifest()
    manifest["files"]["mirror/x.py"]["new"] = ["extra"]
    manifest_path, research = make_trees(tmp_path / "second", extra=extra, manifest=manifest)
    report = checker.run(manifest_path, research)
    assert statuses(report)["extra"] == "NEW"
    assert report.ok


def test_research_name_absent_from_a_full_copy_is_missing_but_fine_in_a_subset(checker,
                                                                                  tmp_path):
    copy = COPY_X.format(h_body="research", extra="", guard="print(h())").replace(
        "def f(a):\n    return a + 1\n\n\n", "")
    manifest_path, research = make_trees(tmp_path, copy_x=copy)
    report = checker.run(manifest_path, research)
    assert statuses(report)["f"] == "MISSING"
    assert not report.ok

    manifest = base_manifest()
    manifest["files"]["mirror/x.py"]["kind"] = "subset"
    manifest_path, research = make_trees(tmp_path / "subset", copy_x=copy, manifest=manifest)
    report = checker.run(manifest_path, research)
    assert "f" not in statuses(report)
    assert report.ok


def test_divergent_module_level_statement_fails_a_full_copy(checker, tmp_path):
    manifest_path, research = make_trees(tmp_path, guard="print('changed')")
    report = checker.run(manifest_path, research)
    assert statuses(report)["<statements>"] == "DIFFERENT"
    assert not report.ok


def test_stale_and_absent_allowlist_entries_fail(checker, tmp_path):
    manifest = base_manifest()
    spec = manifest["files"]["mirror/x.py"]
    spec["dropped"].append("NEVER_EXISTED")  # not in research: stale
    spec["dropped"].append("f")  # present in the copy: stale
    spec["repointed"].append("GHOST")  # absent from the copy: missing
    spec["new"].append("PHANTOM")  # absent from the copy: missing
    manifest_path, research = make_trees(tmp_path, manifest=manifest)
    report = checker.run(manifest_path, research)
    got = statuses(report)
    assert got["NEVER_EXISTED"] == "STALE"
    assert got["GHOST"] == "MISSING"
    assert got["PHANTOM"] == "MISSING"
    stale_f = [i for i in report.files[0].items if i.name == "f" and i.status == "STALE"]
    assert len(stale_f) == 1
    assert len(report.failures) == 4


def test_new_code_file_without_source(checker, tmp_path):
    manifest = base_manifest()
    manifest["files"]["mirror/paths.py"] = {"source": None, "kind": "new", "new": ["ROOT"]}
    manifest_path, research = make_trees(tmp_path, manifest=manifest)
    (manifest_path.parent / "mirror" / "paths.py").write_text(
        "ROOT = 1\n\n\ndef helper():\n    return ROOT\n")
    report = checker.run(manifest_path, research)
    got = statuses(report, "mirror/paths.py")
    assert got == {"ROOT": "NEW", "helper": "UNCOVERED"}


def test_missing_copy_or_source_is_reported_not_raised(checker, tmp_path):
    manifest = base_manifest()
    manifest["files"]["mirror/absent.py"] = {"source": "bench/x.py", "kind": "full"}
    manifest["files"]["mirror/x.py"]["source"] = "bench/nowhere.py"
    manifest_path, research = make_trees(tmp_path, manifest=manifest)
    report = checker.run(manifest_path, research)
    assert statuses(report, "mirror/absent.py") == {"<module>": "MISSING"}
    assert statuses(report, "mirror/x.py") == {"<module>": "MISSING"}
    assert not report.ok


# ---------------------------------------------------------------------------
# sweep of the figures directory
# ---------------------------------------------------------------------------

def test_stray_module_is_unlisted_and_fails(checker, tmp_path, capsys):
    manifest_path, research = make_trees(tmp_path)
    figures = manifest_path.parent
    (figures / "mirror" / "stray.py").write_text("def evil():\n    return 1\n")
    # excluded from the sweep: package markers, tests and bytecode caches
    (figures / "mirror" / "__init__.py").write_text("")
    (figures / "tests").mkdir()
    (figures / "tests" / "test_something.py").write_text("def test_x():\n    pass\n")
    (figures / "mirror" / "__pycache__").mkdir()
    (figures / "mirror" / "__pycache__" / "x.cpython-310.py").write_text("")
    report = checker.run(manifest_path, research)
    assert statuses(report, "mirror/stray.py") == {"<module>": "UNLISTED"}
    assert statuses(report, "mirror/x.py")["f"] == "IDENTICAL"
    assert report.sweep["swept"] == ["mirror/stray.py", "mirror/x.py"]
    assert report.sweep["unlisted"] == ["mirror/stray.py"] and report.sweep["stale"] == []
    assert [i.name for i in report.failures] == ["<module>"] and not report.ok
    rc = checker.main(["--research", str(research), "--manifest", str(manifest_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "== mirror/stray.py  (no research source)  [NOT IN THE MANIFEST]" in out
    assert "copy sweep         2 *.py files outside tests/; 1 unlisted, 0 stale" in out
    assert "UNLISTED                     1  FAIL" in out
    assert out.rstrip().endswith(
        "FAIL: 1 item(s) are DIFFERENT, MISSING, UNCOVERED, UNLISTED or STALE")


@pytest.mark.parametrize("shape", ["dict", "list"])
def test_not_covered_entry_admits_a_file_that_is_not_a_copy(checker, tmp_path, shape):
    manifest = base_manifest()
    manifest["not_covered"] = ({"mirror/stray.py": "new tool"} if shape == "dict"
                               else ["mirror/stray.py"])
    manifest_path, research = make_trees(tmp_path, manifest=manifest)
    (manifest_path.parent / "mirror" / "stray.py").write_text("def tool():\n    return 1\n")
    report = checker.run(manifest_path, research)
    assert report.ok
    assert report.sweep == {"swept": ["mirror/stray.py", "mirror/x.py"], "unlisted": [],
                            "stale": []}
    assert "copy sweep         2 *.py files outside tests/; all listed" in \
        checker.format_report(report)


def test_not_covered_entry_for_a_missing_file_is_stale(checker, tmp_path):
    manifest = base_manifest()
    manifest["not_covered"] = {"mirror/gone.py": "was here once"}
    manifest_path, research = make_trees(tmp_path, manifest=manifest)
    report = checker.run(manifest_path, research)
    assert statuses(report, "mirror/gone.py") == {"<module>": "STALE"}
    assert report.sweep["stale"] == ["mirror/gone.py"]
    assert not report.ok


def test_pristine_tree_passes_the_sweep(checker, tmp_path):
    manifest_path, research = make_trees(tmp_path)
    report = checker.run(manifest_path, research)
    assert report.ok
    assert report.sweep == {"swept": ["mirror/x.py"], "unlisted": [], "stale": []}


# ---------------------------------------------------------------------------
# prefix import_map, enforced import block, dropped statements, directory not_covered
# ---------------------------------------------------------------------------

PREFIX_MAP = {"experiments": "bench", "compass": "", "experiments.extra.probe": "probe"}


@pytest.mark.parametrize("copy_line, research_line", [
    ("from experiments.holdout import x", "from bench.holdout import x"),
    ("from experiments import dgp", "from bench import dgp"),
    ("    from experiments import run as brun", "    from bench import run as brun"),
    ("import experiments.run as brun", "import bench.run as brun"),
    ("from experiments.extra.probe import y  # noqa: E402", "from probe import y  # noqa: E402"),
    ("from experiments.extra.probe.sub import y", "from probe.sub import y"),
    ("    from compass.agents.validator import z  # noqa: deferred",
     "    from agents.validator import z  # noqa: deferred"),
    ("    import compass.judge as judge_mod", "    import judge as judge_mod"),
    ("from compass import llm", "import llm"),
    ("from compass import llm as L", "import llm as L"),
    ("from compass.llm import a, b", "from llm import a, b"),
    ("from experimentsx.y import q", "from experimentsx.y import q"),  # no prefix match
    ("import compass", "import compass"),  # degenerate: left alone
    ("from compass import (", "from compass import ("),  # cannot become an import statement
    ("x = 'from experiments.holdout import x'", "x = 'from experiments.holdout import x'"),
])
def test_prefix_import_map_normalisation(checker, copy_line, research_line):
    assert checker.normalise_imports(copy_line, PREFIX_MAP) == research_line


def test_map_module_prefers_the_longest_key(checker):
    assert checker.map_module("experiments.extra.probe", PREFIX_MAP) == "probe"
    assert checker.map_module("experiments.extra.other", PREFIX_MAP) == "bench.extra.other"
    assert checker.map_module("compass", PREFIX_MAP) == ""
    assert checker.map_module("compassx", PREFIX_MAP) is None
    assert checker.map_module("mirror.plots", {}) is None, "no default import map"


def test_blank_line_after_a_rewritten_import_is_tolerated_elsewhere_not(checker, tmp_path):
    base = COPY_X.format(h_body="research", extra="", guard="print(h())")
    manifest_path, research = make_trees(
        tmp_path, copy_x=base.replace("    from mirror.x import f\n", "    from mirror.x import f\n\n"))
    report = checker.run(manifest_path, research)
    item = next(i for i in report.items if i.name == "g")
    assert item.status == "IDENTICAL_MODULO_IMPORTS"
    assert "blank line after an import is ignored" in item.note
    assert report.ok

    # a blank line that does not follow an import (inside f) is a real difference
    manifest_path, research = make_trees(
        tmp_path / "elsewhere", copy_x=base.replace("def f(a):\n    return a + 1\n",
                                                    "def f(a):\n\n    return a + 1\n"))
    report = checker.run(manifest_path, research)
    assert statuses(report)["f"] == "DIFFERENT"


RESEARCH_Z = '''"""Driver script."""

import os
import sys
from pathlib import Path

import judge as judge_mod
from bench.y import INLINED

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CODE_DIR = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_suffix(".results.json")


def run():
    return judge_mod.x(INLINED, OUT)


if __name__ == "__main__":
    run()
'''

COPY_Z = '''"""Driver script."""

from experiments import paths

import compass.judge as judge_mod
from experiments.y import INLINED

OUT = paths.results_file("z.results.json")


def run():
    return judge_mod.x(INLINED, OUT)


if __name__ == "__main__":
    run()
'''

SYS_PATH_1 = "sys.path.insert(0, str(Path(__file__).resolve().parent))"
SYS_PATH_2 = "sys.path.insert(0, str(Path(__file__).resolve().parents[2]))"


def z_manifest(**overrides) -> dict:
    spec = {
        "source": "bench/z.py", "kind": "full", "repointed": ["OUT"], "dropped": ["CODE_DIR"],
        "new": [], "dropped_statements": [SYS_PATH_1, SYS_PATH_2],
        "dropped_imports": ["import os", "import sys", "from pathlib import Path"],
        "new_imports": ["from experiments import paths"],
    }
    spec.update(overrides)
    return {"research_commit": COMMIT, "imports": "enforce",
            "import_map": {"experiments": "bench", "compass": ""},
            "files": {"mirror/z.py": spec}}


def make_z(tmp_path: Path, manifest: dict, copy_z: str = COPY_Z) -> tuple[Path, Path]:
    manifest_path, research = make_trees(tmp_path, manifest=manifest)
    (research / "bench" / "z.py").write_text(RESEARCH_Z)
    (manifest_path.parent / "mirror" / "x.py").unlink()  # the manifest names only z.py
    (manifest_path.parent / "mirror" / "z.py").write_text(copy_z)
    return manifest_path, research


def test_enforced_imports_and_dropped_statements_pass_on_the_documented_changes(checker,
                                                                                tmp_path):
    manifest_path, research = make_z(tmp_path, z_manifest())
    report = checker.run(manifest_path, research)
    got = statuses(report, "mirror/z.py")
    assert got["run"] == "IDENTICAL" and got["OUT"] == "REPOINTED" and got["CODE_DIR"] == "DROPPED"
    assert got[SYS_PATH_1] == "DROPPED" and got[SYS_PATH_2] == "DROPPED"
    assert report.ok, [(i.name, i.status, i.note) for i in report.failures]
    info = report.files[0].info
    assert "imports: enforced, 2 statements match after normalisation (1 new, 3 dropped)" in info
    assert "statements: identical (1, 2 dropped)" in info
    stmt_items = [i for i in report.files[0].items if i.category == "statement"]
    assert [i.status for i in stmt_items] == ["DROPPED", "DROPPED"]
    out = checker.format_report(report)
    assert "DROPPED                      3  (1 definitions, 2 statements)" in out


def test_enforced_imports_fail_on_an_undeclared_or_changed_import(checker, tmp_path):
    # the dropped research imports are not declared: the multisets differ
    manifest_path, research = make_z(tmp_path, z_manifest(dropped_imports=[]))
    report = checker.run(manifest_path, research)
    item = next(i for i in report.items if i.name == "<imports>")
    assert item.status == "DIFFERENT" and item.category == "import"
    assert "-import os" in item.diff and "-import sys" in item.diff
    assert not report.ok

    # a research import line changed in the copy (another module imported)
    manifest_path, research = make_z(tmp_path / "changed", z_manifest(),
                                     COPY_Z.replace("from experiments.y import INLINED",
                                                    "from experiments.w import INLINED"))
    report = checker.run(manifest_path, research)
    item = next(i for i in report.items if i.name == "<imports>")
    assert item.status == "DIFFERENT"
    assert "-from bench.y import INLINED" in item.diff and "+from bench.w import INLINED" in item.diff
    assert checker.main(["--research", str(research), "--manifest", str(manifest_path)]) == 1


def test_enforced_imports_allowlist_hygiene(checker, tmp_path):
    manifest = z_manifest(dropped_imports=["import os", "import sys", "from pathlib import Path",
                                           "import nothing"],
                          new_imports=["from experiments import paths", "import ghost"])
    manifest_path, research = make_z(tmp_path, manifest)
    report = checker.run(manifest_path, research)
    got = statuses(report, "mirror/z.py")
    assert got["import nothing"] == "STALE"
    assert got["import ghost"] == "MISSING"
    assert "<imports>" not in got, "the remaining multisets still match"
    assert len(report.failures) == 2

    # a dropped import that the copy still carries
    manifest_path, research = make_z(
        tmp_path / "still", z_manifest(),
        COPY_Z.replace("from experiments import paths\n", "import os\nfrom experiments import paths\n"))
    report = checker.run(manifest_path, research)
    items = [i for i in report.files[0].items if i.name == "import os"]
    assert [i.status for i in items] == ["STALE"]
    assert not report.ok


def test_report_mode_keeps_the_import_block_informational(checker, tmp_path):
    manifest = z_manifest(dropped_imports=[], new_imports=[])
    del manifest["imports"]
    manifest_path, research = make_z(tmp_path, manifest)
    report = checker.run(manifest_path, research)
    assert report.ok
    assert any(line.startswith("imports: rewritten (") for line in report.files[0].info)


def test_invalid_import_mode_is_rejected(checker, tmp_path):
    manifest = z_manifest()
    manifest["imports"] = "sometimes"
    manifest_path, research = make_z(tmp_path, manifest)
    with pytest.raises(SystemExit, match="imports"):
        checker.run(manifest_path, research)


def test_dropped_statements_hygiene_and_undeclared_statement(checker, tmp_path):
    manifest_path, research = make_z(
        tmp_path, z_manifest(dropped_statements=[SYS_PATH_1, SYS_PATH_2, "os.getcwd()"]))
    report = checker.run(manifest_path, research)
    assert statuses(report, "mirror/z.py")["os.getcwd()"] == "STALE"
    assert len(report.failures) == 1

    # the second sys.path line is undeclared: the statement blocks differ
    manifest_path, research = make_z(tmp_path / "undeclared", z_manifest(dropped_statements=[SYS_PATH_1]))
    report = checker.run(manifest_path, research)
    item = next(i for i in report.items if i.name == "<statements>")
    assert item.status == "DIFFERENT" and SYS_PATH_2 in item.diff

    # a dropped statement the copy still carries
    manifest_path, research = make_z(
        tmp_path / "still", z_manifest(),
        COPY_Z.replace("OUT = paths", SYS_PATH_1 + "\nOUT = paths"))
    report = checker.run(manifest_path, research)
    items = [i for i in report.files[0].items if i.name == SYS_PATH_1]
    assert [i.status for i in items] == ["STALE"]


def test_directory_not_covered_admits_its_files_and_is_stale_when_missing(checker, tmp_path):
    manifest = base_manifest()
    manifest["not_covered"] = {"tools/": "filled by another agent"}
    manifest_path, research = make_trees(tmp_path, manifest=manifest)
    figures = manifest_path.parent
    (figures / "tools" / "deep").mkdir(parents=True)
    (figures / "tools" / "helper.py").write_text("def h():\n    return 1\n")
    (figures / "tools" / "deep" / "nested.py").write_text("def n():\n    return 2\n")
    report = checker.run(manifest_path, research)
    assert report.ok
    assert report.sweep["swept"] == ["mirror/x.py", "tools/deep/nested.py", "tools/helper.py"]
    assert report.sweep["unlisted"] == []

    manifest["not_covered"] = {"gone/": "not here"}
    manifest_path, research = make_trees(tmp_path / "missing", manifest=manifest)
    report = checker.run(manifest_path, research)
    assert statuses(report, "gone/") == {"<directory>": "STALE"}
    assert not report.ok

    # a prefix match needs the slash: "tools" alone must not admit tools_extra.py
    manifest["not_covered"] = {"tools/": "x"}
    manifest_path, research = make_trees(tmp_path / "prefix", manifest=manifest)
    (manifest_path.parent / "tools").mkdir()
    (manifest_path.parent / "tools_extra.py").write_text("y = 1\n")
    report = checker.run(manifest_path, research)
    assert statuses(report, "tools_extra.py") == {"<module>": "UNLISTED"}


# ---------------------------------------------------------------------------
# derived constants
# ---------------------------------------------------------------------------

def _derived_manifest() -> dict:
    manifest = base_manifest()
    manifest["files"]["mirror/t.py"] = {
        "source": "bench/dgp.py", "kind": "derived",
        "derived": {"TYPES": {"module": "bench.dgp", "name": "TYPES"}},
    }
    return manifest


def _write_t(manifest_path: Path, text: str = 'TYPES = ["a", "b"]\n') -> None:
    (manifest_path.parent / "mirror" / "t.py").write_text(text)


def test_derived_constant_matches_through_injected_probe(checker, tmp_path):
    manifest_path, research = make_trees(tmp_path, manifest=_derived_manifest())
    _write_t(manifest_path)
    seen = {}

    def probe(root, specs):
        seen.update(specs)
        return {"bench.dgp:TYPES": {"value": ["a", "b"]}}

    report = checker.run(manifest_path, research, derived_fn=probe)
    assert seen == {"bench.dgp:TYPES": ("bench.dgp", "TYPES")}
    assert statuses(report, "mirror/t.py") == {"TYPES": "DERIVED"}
    assert report.ok


def test_derived_constant_mismatch_fails_via_monkeypatched_module_function(checker, tmp_path,
                                                                            monkeypatch):
    manifest_path, research = make_trees(tmp_path, manifest=_derived_manifest())
    _write_t(manifest_path)
    monkeypatch.setattr(checker, "derived_values",
                        lambda root, specs: {"bench.dgp:TYPES": {"value": ["a"]}})
    report = checker.run(manifest_path, research)
    item = next(i for i in report.items if i.name == "TYPES")
    assert item.status == "DIFFERENT"
    assert "value differs" in item.note
    assert item.diff, "the value diff is attached"
    assert not report.ok


def test_derived_constant_import_error_and_non_literal_fail(checker, tmp_path):
    manifest_path, research = make_trees(tmp_path, manifest=_derived_manifest())
    _write_t(manifest_path)
    report = checker.run(manifest_path, research,
                         derived_fn=lambda r, s: {"bench.dgp:TYPES": {"error": "boom"}})
    item = next(i for i in report.items if i.name == "TYPES")
    assert item.status == "DIFFERENT" and "boom" in item.note

    _write_t(manifest_path, "TYPES = list(range(2))\n")
    report = checker.run(manifest_path, research,
                         derived_fn=lambda r, s: {"bench.dgp:TYPES": {"value": [0, 1]}})
    item = next(i for i in report.items if i.name == "TYPES")
    assert item.status == "DIFFERENT" and "not a literal" in item.note


def test_derived_values_subprocess_imports_the_research_module(checker, tmp_path):
    _, research = make_trees(tmp_path)
    values = checker.derived_values(research, {"bench.dgp:TYPES": ("bench.dgp", "TYPES"),
                                               "bench.nope:X": ("bench.nope", "X")})
    assert values["bench.dgp:TYPES"] == {"value": ["a", "b"]}
    assert "ModuleNotFoundError" in values["bench.nope:X"]["error"]
    assert checker.derived_values(research, {}) == {}


# ---------------------------------------------------------------------------
# helpers, git state and the command line
# ---------------------------------------------------------------------------

def test_normalise_imports(checker):
    imap = {"mirror.dgp_types": "bench.dgp", "mirror.aggregate": "bench.aggregate"}
    text = ("    from mirror.aggregate import (\n        a, b)\n"
            "    from mirror.dgp_types import TYPES\n"
            "    from mirror.other import z\n"
            "    import mirror.other\n"
            "    x = 'mirror.aggregate'\n")
    expect = ("    from bench.aggregate import (\n        a, b)\n"
              "    from bench.dgp import TYPES\n"
              "    from mirror.other import z\n"  # no key covers it: left alone
              "    import mirror.other\n"
              "    x = 'mirror.aggregate'")
    assert checker.normalise_imports(text, imap) == expect


def test_parse_module_collects_definitions_imports_and_statements(checker, tmp_path):
    path = tmp_path / "m.py"
    path.write_text('"""Doc."""\nimport os\n\n@dec\ndef f():\n    pass   \n\nX: int = 1\n'
                    'a = b = 2\nos.getcwd()\n')
    mod = checker.parse_module(path)
    assert mod.docstring == "Doc."
    assert mod.imports == ["import os"]
    assert mod.statements == ["os.getcwd()"]
    assert set(mod.defs) == {"f", "X", "a = b"}
    assert mod.defs["f"].text == "@dec\ndef f():\n    pass", "decorators kept, trailing blanks dropped"
    assert mod.defs["f"].kind == "function" and mod.defs["X"].kind == "assign"


def _git(research: Path, home: Path, *args: str) -> str:
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@x", "HOME": str(home), "PATH": "/usr/bin:/bin"}
    return subprocess.run(["git", "-C", str(research), "-c", "commit.gpgsign=false", *args],
                          check=True, env=env, capture_output=True, text=True).stdout.strip()


def _git_repo(research: Path, home: Path) -> str:
    """Turn the synthetic research tree into a repository with one commit; returns HEAD."""
    _git(research, home, "init", "-q")
    _git(research, home, "add", "bench")
    _git(research, home, "commit", "-q", "-m", "init")
    return _git(research, home, "rev-parse", "HEAD")


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_info_reports_head_and_dirty_sources(checker, real_git, tmp_path):
    manifest_path, research = make_trees(tmp_path)
    head = _git_repo(research, tmp_path)
    info = checker.git_info(research, ["bench/x.py", "bench/y.py"])
    assert info == {"head": head, "clean": True, "dirty": [], "error": None,
                    "sources_unchanged": None, "sources_changed": [], "diff_error": None}
    info = checker.git_info(research, ["bench/x.py", "bench/y.py"], head)
    assert info["sources_unchanged"] is True and info["sources_changed"] == []

    (research / "bench" / "y.py").write_text("INLINED = ('p', 'q')\n")
    info = checker.git_info(research, ["bench/x.py", "bench/y.py"])
    assert info["clean"] is False and info["dirty"] == [" M bench/y.py"]

    manifest = base_manifest()
    manifest["research_commit"] = head
    manifest_path.write_text(json.dumps(manifest))
    report = checker.run(manifest_path, research)
    out = checker.format_report(report)
    assert f"research HEAD      {head}  (equals the manifest commit)" in out
    assert "1 listed source(s) modified:" in out and " M bench/y.py" in out


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_sources_changed_since_the_manifest_commit_are_stale(checker, real_git, tmp_path):
    manifest_path, research = make_trees(tmp_path)
    base = _git_repo(research, tmp_path)
    manifest = base_manifest()
    manifest["research_commit"] = base
    manifest_path.write_text(json.dumps(manifest))

    # a later commit that touches no listed source: information only, the run passes
    (research / "README.md").write_text("notes only\n")
    _git(research, tmp_path, "add", "README.md")
    _git(research, tmp_path, "commit", "-q", "-m", "notes")
    head = _git(research, tmp_path, "rev-parse", "HEAD")
    assert head != base
    report = checker.run(manifest_path, research)
    assert report.ok
    assert report.git["sources_unchanged"] is True and report.git["sources_changed"] == []
    assert (f"research HEAD      {head}  (sources unchanged since manifest commit {base})"
            in checker.format_report(report))

    # a commit that touches a listed source (bench/y.py via extra_sources) without altering
    # the copied name: the copy still matches, but the manifest commit is stale
    (research / "bench" / "y.py").write_text(RESEARCH_Y.replace("OTHER = 3", "OTHER = 4"))
    _git(research, tmp_path, "add", "bench/y.py")
    _git(research, tmp_path, "commit", "-q", "-m", "touch a source")
    head2 = _git(research, tmp_path, "rev-parse", "HEAD")
    report = checker.run(manifest_path, research)
    assert statuses(report)["INLINED"] == "IDENTICAL"
    assert report.git["sources_unchanged"] is False
    assert report.git["sources_changed"] == ["bench/y.py"]
    assert statuses(report, "<research>") == {"bench/y.py": "STALE"}
    assert [i.name for i in report.failures] == ["bench/y.py"]
    assert not report.ok
    out = checker.format_report(report)
    assert (f"research HEAD      {head2}  (1 listed source(s) changed since manifest commit "
            f"{base}; see STALE)") in out
    assert "== <research>  [sources since the manifest commit]" in out
    assert "STALE                        1  FAIL" in out

    # a manifest commit git cannot resolve is stale as well
    manifest["research_commit"] = "f" * 40
    manifest_path.write_text(json.dumps(manifest))
    report = checker.run(manifest_path, research)
    assert report.git["diff_error"] and report.git["sources_unchanged"] is None
    assert statuses(report, "<research>") == {f"research_commit {'f' * 7}": "STALE"}
    assert not report.ok


def test_git_info_without_git_is_reported_not_raised(checker, real_git, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no git here")
    monkeypatch.setattr(checker.subprocess, "run", boom)
    info = checker.git_info(Path("/nonexistent"), ["x.py"])
    assert info["head"] is None and "no git here" in info["error"]


def test_main_exit_codes_and_report_text(checker, tmp_path, capsys):
    manifest_path, research = make_trees(tmp_path)
    rc = checker.main(["--research", str(research), "--manifest", str(manifest_path), "--diff"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PASS: every definition is accounted for" in out
    assert "REPOINTED                OUT" in out
    assert '+OUT = paths.out_dir()' in out, "--diff prints the REPOINTED diff"
    assert "DROPPED                  CODE_DIR" in out
    assert "research HEAD      unknown  (stubbed)" in out
    assert "manifest commit    " + COMMIT + "  (not compared)" in out
    assert "research tree      status unknown" in out
    summary = out[out.index("summary"):]
    assert "IDENTICAL                    4" in summary
    assert "IDENTICAL_MODULO_IMPORTS     1" in summary

    rc = checker.main(["--research", str(research), "--manifest", str(manifest_path)])
    assert rc == 0
    assert "+OUT = paths.out_dir()" not in capsys.readouterr().out

    assert checker.main(["--research", str(research), "--manifest", str(tmp_path / "no.json")]) == 2
    assert checker.main(["--research", str(tmp_path / "nowhere"), "--manifest",
                         str(manifest_path)]) == 2


def test_dropped_patterns_match_absent_research_names_and_are_stale_when_unused(checker, tmp_path):
    manifest = base_manifest()
    manifest["files"]["mirror/x.py"]["dropped"] = []
    manifest["files"]["mirror/x.py"]["dropped_patterns"] = ["^CODE_", "^NOTHING_LIKE_THIS"]
    manifest_path, research = make_trees(tmp_path, manifest=manifest)
    report = checker.run(manifest_path, research)
    got = statuses(report)
    assert got["CODE_DIR"] == "DROPPED"
    assert got["^NOTHING_LIKE_THIS"] == "STALE" and not report.ok
    item = next(i for i in report.items if i.name == "CODE_DIR")
    assert "matches dropped_patterns" in item.note
    manifest["files"]["mirror/x.py"]["dropped_patterns"] = ["^CODE_"]
    manifest_path.write_text(json.dumps(manifest))
    report = checker.run(manifest_path, research)
    assert statuses(report)["CODE_DIR"] == "DROPPED" and report.ok
