"""``tools/sync_from_research.py``: the AST-guided import rewrite and the patch step.

The tool is a script, not a package module, so it is loaded by path."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "sync_from_research.py"


@pytest.fixture(scope="module")
def sync():
    spec = importlib.util.spec_from_file_location("sync_from_research", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_from_import_of_a_root_module(sync):
    out, n = sync.rewrite_imports("from llm import invoke_structured\n", "x.py")
    assert out == "from compass.llm import invoke_structured\n" and n == 1


def test_indented_bare_import_becomes_from_compass_import(sync):
    src = "def f():\n    import crossval\n    return crossval\n"
    out, n = sync.rewrite_imports(src, "x.py")
    assert out == "def f():\n    from compass import crossval\n    return crossval\n" and n == 1


def test_multiline_from_import_keeps_the_parenthesis(sync):
    src = "from agents.scout import (\n    create_scout_agent,\n    DEFAULT_SCORE_DROP,\n)\n"
    out, n = sync.rewrite_imports(src, "x.py")
    assert out == ("from compass.agents.scout import (\n    create_scout_agent,\n"
                   "    DEFAULT_SCORE_DROP,\n)\n")
    assert n == 1


def test_trailing_noqa_comment_is_preserved(sync):
    out, n = sync.rewrite_imports("import crossval  # noqa: F401\n", "x.py")
    assert out == "from compass import crossval  # noqa: F401\n" and n == 1
    out, n = sync.rewrite_imports("from llm import STRUCTURED_RETRIES  # noqa: F401\n", "x.py")
    assert out == "from compass.llm import STRUCTURED_RETRIES  # noqa: F401\n" and n == 1


def test_import_looking_text_inside_strings_is_untouched(sync):
    src = ('"""Usage:\n\n    from llm import get_llm\n"""\n'
           'HINT = "import crossval"\n'
           "code = '''\nfrom agents.scout import x\n'''\n")
    out, n = sync.rewrite_imports(src, "x.py")
    assert out == src and n == 0


def test_rewrite_is_idempotent(sync):
    src = "import os\nfrom llm import x\nimport crossval\nfrom fixtures.golden_model import G\n"
    once, n1 = sync.rewrite_imports(src, "x.py")
    twice, n2 = sync.rewrite_imports(once, "x.py")
    assert n1 == 3 and n2 == 0 and once == twice
    assert "import os\n" in once and "from compass.llm import x\n" in once
    assert "from compass.fixtures.golden_model import G\n" in once


def test_non_project_and_relative_imports_are_left_alone(sync):
    src = "import os\nfrom pathlib import Path\nfrom .sibling import y\nimport numpy as np\n"
    out, n = sync.rewrite_imports(src, "x.py")
    assert out == src and n == 0


@pytest.mark.parametrize("line", ["import llm as l", "import llm, os", "import agents.scout"])
def test_unsupported_import_forms_are_hard_failures(sync, line):
    with pytest.raises(SystemExit, match="unsupported import form"):
        sync.rewrite_imports(line + "\n", "x.py")


def test_apply_patches_rewrites_each_anchor_exactly_once(sync):
    anchors = sync.PATCHES["agents/validator.py"]
    src = "A\n" + "\n".join(old for old, _ in anchors) + "\nB\n"
    out, n = sync.apply_patches(src, "agents/validator.py")
    assert n == len(anchors)
    for old, new in anchors:
        assert old not in out and new in out
    assert "compass-refine[stan]" in out


def test_llm_module_needs_no_package_only_wording(sync):
    # 0.3.0: get_llm searches the .env from the current working directory, so the
    # research wording is right for the package too and llm.py has no patch entry
    assert "llm.py" not in sync.PATCHES


def test_apply_patches_missing_anchor_is_a_hard_failure(sync):
    with pytest.raises(SystemExit, match="patch anchor found 0 times, expected 1"):
        sync.apply_patches("nothing to anchor on\n", "agents/validator.py")


def test_apply_patches_duplicate_anchor_is_a_hard_failure(sync):
    old, _ = sync.PATCHES["agents/retrieval.py"][0]
    with pytest.raises(SystemExit, match="found 2 times"):
        sync.apply_patches(old + "\n" + old + "\n", "agents/retrieval.py")


def test_files_without_patches_pass_through(sync):
    assert sync.apply_patches("x = 1\n", "schemas.py") == ("x = 1\n", 0)


def test_module_list_covers_the_package_tree(sync):
    package = Path(__file__).resolve().parents[1] / "src" / "compass"
    for rel in sync.MODULES:
        assert (package / rel).is_file(), rel


def test_unexpected_package_files_reports_strays(sync, tmp_path):
    pkg = tmp_path / "compass"
    (pkg / "agents").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "llm.py").write_text("x = 1\n")
    (pkg / "agents" / "scout.py").write_text("x = 1\n")
    assert sync.unexpected_package_files(pkg) == []
    (pkg / "stray.py").write_text("def evil():\n    pass\n")
    (pkg / "agents" / "__pycache__").mkdir()
    (pkg / "agents" / "__pycache__" / "scout.cpython-310.pyc").write_bytes(b"")
    assert sync.unexpected_package_files(pkg) == ["stray.py"]
    # the real package holds exactly the mirrored modules plus __init__.py
    assert sync.unexpected_package_files(TOOL.resolve().parents[1] / "src" / "compass") == []
