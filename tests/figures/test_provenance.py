"""``tools/sync_figures.py`` and the manifest it writes, ``figures/provenance.json``, checked by
``tools/check_provenance.py``: on a tiny synthetic research tree that uses the research scripts'
import idioms (the ``try``/``except ImportError`` sibling import, the ``ROOT`` path constant put on
``sys.path``, pipeline imports by bare name) and carries the lines the tool rewrites, and, when
the research checkout sits next to this repository, against the real tree.

The tool and the checker are loaded by path; the checker's git probe is stubbed so that the
synthetic trees (written under ``tmp_path``, inside whatever repository holds it) are judged on
their content alone. The tool reads the key of the eighth dataset off ``REAL_ORDER`` and off the
registry of ``bench/realdata.py``, so the synthetic tree uses a key of its own (``zz_extra``), and
its hashed windows are pinned on the synthetic text by the tests themselves.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_DIR / "figures"
TOOL = REPO_DIR / "tools" / "sync_figures.py"
CHECKER = REPO_DIR / "tools" / "check_provenance.py"
RESEARCH = REPO_DIR.parent  # the research layout: compass-git/ sits inside code/

F3_COMMON = "figure3/figure3_common.py"
F3C = "figure3/figure3c_create.py"
F2_RUN = "figure2/figure2_run.py"
F2_PROTOCOLS = "figure2/figure2_protocols.py"
REALDATA = "bench/realdata.py"
EXTRA = "zz_extra"  # the synthetic tree's eighth dataset
EXTRA_ARM = "e1_anthropic_zzextra"

REAL_ORDER_OLD = ('REAL_ORDER = ("roaches", "ucbadmit", "baseball", "seeds", "radon", "kidiq", "golf", '
                  f'"{EXTRA}")  # bench.realdata order\n')
REAL_ORDER_NEW = ('REAL_ORDER = ("roaches", "ucbadmit", "baseball", "seeds", "radon", "kidiq", "golf")'
                  '  # bench.realdata order\n')
DATASET_NAME_OLD = f'DATASET_NAME = {{"{EXTRA}": "ZZ Extra", "ucbadmit": "UCBadmit", "kidiq": "KidIQ"}}\n'
DATASET_NAME_NEW = 'DATASET_NAME = {"ucbadmit": "UCBadmit", "kidiq": "KidIQ"}\n'
USAGE_OLD = ("    python figure3/figure3c_create.py bench/arms/e1_anthropic_realdata_x2 "
             f"bench/arms/{EXTRA_ARM} --out fig3c.png\n")
USAGE_NEW = "    python figure3/figure3c_create.py bench/arms/e1_anthropic_realdata_x2 --out fig3c.png\n"
# a hashed window of the synthetic figure2_protocols.py: two research lines become one
WINDOW_OLD = ('    # one more arm on the extra dataset\n'
              f'    "extra-model_sonnet-4.6": _real("anthropic", (), "{EXTRA_ARM}", datasets=("{EXTRA}",)),\n')
WINDOW_NEW = "    # the arms end here\n"

TRY_IMPORT = '''try:
    import {stem} as {alias}  # run as a file from the repo root
except ImportError:
    from {pkg} import {stem} as {alias}  # run as a module / imported by tests
'''
ROOT_BLOCK = '''ROOT = Path(__file__).resolve().parents[1]  # the repo root, so "from figure2 import" works when run as a file
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from figure2 import figure2_common as common  # noqa: E402
'''
CACHE_OLD = 'CACHE = CODE_DIR / ".test_cache"          # MCMC trace cache shared with bench.run\n'
REALDATA_RESEARCH = (
    '"""Real-data arm of the benchmark: seven literature datasets, each with its\n'
    "textbook misspecified starting model, plus the extra application\n"
    "(a later study) as the eighth entry.\n"
    '"""\n\nfrom __future__ import annotations\n\nimport json\nfrom pathlib import Path\n\nimport numpy as np\n\n'
    'DATA_DIR = Path(__file__).resolve().parent.parent / "tutorials" / "demos" / "data"\n\n'
    "DEFAULT_BUDGET = dict(num_warmup=500)\n"
    "# the eighth entry's budget\n"
    "ZZ_EXTRA_BUDGET = dict(num_warmup=3000)\n\n\n"
    "def load_golf() -> dict:\n    return {\"y\": np.ones(19)}\n\n\n"
    "# the extra columns\n"
    'ZZ_EXTRA_FILE = "eighth.csv"\n'
    'ZZ_EXTRA_CHANNELS = ("a", "b")\n'
    "ZZ_EXTRA_INT_SCALE = 1000\n\n\n"
    "def load_zz_extra() -> dict:\n    return {\"y\": np.ones(4)}\n\n\n"
    "GOLF_ORIGINAL = '''\ndef model(y=None):\n    pass\n'''\n\n"
    "# the extra baseline\n"
    "ZZ_EXTRA_ORIGINAL = '''\ndef model(y=None):\n    pass\n'''\n\n"
    "_DATASETS = (\n"
    '    dict(key="golf", loader=load_golf, original_code=GOLF_ORIGINAL, budget=DEFAULT_BUDGET),\n'
    f'    dict(key="{EXTRA}", loader=load_zz_extra, original_code=ZZ_EXTRA_ORIGINAL, budget=ZZ_EXTRA_BUDGET),\n'
    ")\n"
    'DATASETS = {d["key"]: d for d in _DATASETS}\n'
    'KEYS = list(DATASETS)\n\n'
    '__all__ = ["DATASETS", "KEYS", "DEFAULT_BUDGET", "ZZ_EXTRA_BUDGET"]\n')


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def tool():
    return _load(TOOL, "sync_figures_under_test")


@pytest.fixture(scope="module")
def checker():
    return _load(CHECKER, "check_provenance_under_figures_test")


@pytest.fixture(autouse=True)
def _no_git(monkeypatch, checker):
    monkeypatch.setattr(checker, "git_info",
                        lambda root, sources, commit=None: {
                            "head": None, "clean": None, "dirty": [], "error": "stubbed",
                            "sources_unchanged": None, "sources_changed": [],
                            "diff_error": None})


# ---------------------------------------------------------------------------
# A synthetic research tree with the scripts' idioms and the rewritten lines
# ---------------------------------------------------------------------------

def _stub(rel: str) -> str:
    """The research text of module ``rel``."""
    pkg, stem = rel.split("/")[0], Path(rel).stem
    if stem == "__init__":
        return f'"""One stub per panel of {pkg}."""\n' if pkg == "figure2" else ""
    head = f'"""Stub of {rel}."""\n\nfrom __future__ import annotations\n\n'
    if rel == REALDATA:
        return REALDATA_RESEARCH
    if rel == "bench/dgp.py":
        return head + 'TYPES = ["overdispersion"]\n\n\ndef make_instance(t, s):\n    return {"type": t, "seed": s}\n'
    if rel == "bench/holdout.py":
        return (head + "import agents.validator as validator\n\n\n"
                "def split(data):\n    return validator, data\n")
    if rel == "bench/control_proposer.py":
        return head + "MENU = ()\n"
    if stem == "figure2_common":
        return (head + "import json\nfrom pathlib import Path\n\n"
                'SYNTHETIC_TYPES = ("a", "b")\n\n\n'
                "def load_records(d):\n"
                '    return [json.loads(p.read_text()) for p in sorted(Path(d).glob("*.json"))]\n')
    if stem == "figure2_protocols":
        return (head + TRY_IMPORT.format(stem="figure2_common", alias="common", pkg="figure2")
                + "\n\ndef _real(provider, readers, legacy_dir, *, datasets=()):\n"
                "    return {\"provider\": provider, \"datasets\": datasets, \"legacy_dir\": legacy_dir}\n\n\n"
                "ARM_PROTOCOLS = {\n"
                '    "real-data_sonnet-4.6": _real("anthropic", (), "e1_anthropic_realdata_x2", datasets=("golf",)),\n'
                + WINDOW_OLD + "}\n")
    if stem == "figure2_run":
        return (head + "import json\nimport time\nfrom pathlib import Path\n\n"
                + TRY_IMPORT.format(stem="figure2_common", alias="common", pkg="figure2")
                + "\nCODE_DIR = Path(__file__).resolve().parents[1]\n" + CACHE_OLD
                + "\n\ndef run_arm(arm):\n    from bench import dgp, realdata\n    from llm import get_llm\n"
                "    return dgp, realdata, get_llm, json, time\n")
    if stem == "figure2_score":
        return (head + TRY_IMPORT.format(stem="figure2_run", alias="engine", pkg="figure2")
                + "\n\ndef score_arm(d):\n    import judge\n    from graph import infer_data_shapes\n"
                "    return engine, judge, infer_data_shapes\n")
    if stem == "figure3_common":
        return (head + "import sys\nfrom pathlib import Path\n\n" + ROOT_BLOCK + "\n"
                + REAL_ORDER_OLD + DATASET_NAME_OLD
                + "\n\ndef series(rec):\n    return common.load_records(rec)\n")
    if stem == "figure3c_create":
        return (f'"""Stub of {rel}.\n\n' + USAGE_OLD + '"""\n\nfrom __future__ import annotations\n\n'
                + TRY_IMPORT.format(stem="figure3_common", alias="f3", pkg="figure3")
                + "\n\ndef count(d):\n    return f3\n")
    if stem == "figure3_run":
        return (head + TRY_IMPORT.format(stem="figure3_common", alias="f3", pkg="figure3")
                + "from figure2 import figure2_run as census  # noqa: E402\n\n\n"
                "def run_arm(arm):\n    from state import ValidationResult\n    return census, f3, ValidationResult\n")
    if stem == "figure4_create":
        return (head + "import sys\nfrom pathlib import Path\n\n" + ROOT_BLOCK
                + "\n\ndef count(dirs):\n    return [common.load_records(d) for d in dirs]\n")
    if stem in ("figure3_compose",):
        return (head + "try:\n    import figure3_common  # noqa: F401\nexcept ImportError:\n    pass\n"
                "from figure2.figure2_compose import compose  # noqa: E402\n\n\n"
                "def main():\n    return compose\n")
    if stem == "figure2_compose":
        return head + "def compose(*panels):\n    return len(panels)\n"
    alias = "common" if pkg == "figure2" else ("f3" if pkg == "figure3" else "common")
    common_stem = "figure2_common" if pkg in ("figure2", "figure4") else "figure3_common"
    return (head + TRY_IMPORT.format(stem=common_stem, alias=alias, pkg="figure2" if pkg == "figure4" else pkg)
            + f"\n\ndef count(d):\n    return {alias}\n")


def _expected_copy(rel: str) -> str:
    """What the tool writes for ``rel``: the stub with its imports and windows rewritten."""
    text = _stub(rel)
    if rel == F3_COMMON:
        text = text.replace(REAL_ORDER_OLD, REAL_ORDER_NEW).replace(DATASET_NAME_OLD, DATASET_NAME_NEW)
    elif rel == F3C:
        text = text.replace(USAGE_OLD, USAGE_NEW)
    elif rel == F2_PROTOCOLS:
        text = text.replace(WINDOW_OLD, WINDOW_NEW)
    elif rel == F2_RUN:
        text = (text.replace(CACHE_OLD, _tool_cache_copy())
                .replace("import json\nimport time\n", "import json\nimport os\nimport time\n")
                .replace("    from llm import get_llm\n", "    from compass.llm import get_llm\n"))
    elif rel == "figure2/figure2_score.py":
        text = text.replace("    import judge\n", "    from compass import judge\n").replace(
            "    from graph import infer_data_shapes\n", "    from compass.graph import infer_data_shapes\n")
    elif rel == "figure3/figure3_run.py":
        text = text.replace("    from state import ValidationResult\n", "    from compass.state import ValidationResult\n")
    elif rel == "bench/holdout.py":
        text = text.replace("import agents.validator as validator\n", "import compass.agents.validator as validator\n")
    return text


_TOOL_CACHE_COPY: list[str] = []


def _tool_cache_copy() -> str:
    return _TOOL_CACHE_COPY[0]


@pytest.fixture
def tree(tmp_path, tool, monkeypatch):
    """The synthetic research tree, with the tool's hashed windows pinned on it."""
    _TOOL_CACHE_COPY[:] = [tool.CACHE_COPY]
    research = tmp_path / "research"
    for rel in tool.MODULES:
        p = research / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_stub(rel))
    # research modules the copies must not import by bare name without a rule
    for rel in ("llm.py", "graph.py", "judge.py", "state.py", "agents/__init__.py", "agents/validator.py",
                "bench/drift.py"):
        (research / rel).parent.mkdir(parents=True, exist_ok=True)
        (research / rel).write_text("")
    monkeypatch.setattr(tool, "HASHED_PATCHES", {
        F2_PROTOCOLS: [(2, tool.window_hash(WINDOW_OLD.splitlines(keepends=True)), WINDOW_NEW)]})
    monkeypatch.setattr(tool, "REPOINTED", {F3_COMMON: tool.REPOINTED[F3_COMMON], F2_RUN: tool.REPOINTED[F2_RUN],
                                             REALDATA: tool.REPOINTED[REALDATA], F2_PROTOCOLS: ["ARM_PROTOCOLS"]})
    monkeypatch.setattr(tool, "DROPPED", {})
    return research


@pytest.fixture
def compass(tmp_path):
    """A stand-in for src/compass: the modules the rewritten imports must resolve in."""
    root = tmp_path / "pkg" / "src" / "compass"
    for rel in ("__init__.py", "llm.py", "graph.py", "judge.py", "state.py", "agents/__init__.py",
                "agents/validator.py"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("")
    return root


def _run_tool(tool, research: Path, package: Path, compass: Path, *extra: str, capsys) -> tuple[int, str]:
    rc = tool.main(["--research", str(research), "--package", str(package), "--compass", str(compass), *extra])
    return rc, capsys.readouterr().out


def _write_package_only(package: Path) -> None:
    for name in ("render.py", "compare_pdf.py"):  # the tools the manifest lists as not_covered
        (package / name).write_text(f"# {name}\n")


def test_mention_regex_and_extra_keys(tool):
    arms = tool.removed_arms(_stub(F2_PROTOCOLS), [EXTRA])
    assert arms == ["extra-model_sonnet-4.6"]
    rx = tool.mention_regex([EXTRA], arms)
    for form in ("zz_extra", "ZZ_EXTRA_BUDGET", "zz-extra", "ZZ-EXTRA", "zzextra", "extra-model_sonnet-4.6",
                 "EXTRA-MODEL"):
        assert rx.search(form), form
    assert rx.search("extra") is None and rx.search("zz_ex") is None
    with pytest.raises(SystemExit, match="ARM_PROTOCOLS dict was not found"):
        tool.removed_arms("X = 1\n", [EXTRA])
    stub = _stub(F3_COMMON)
    assert tool.extra_keys(stub) == [EXTRA]
    with pytest.raises(SystemExit, match="names 0 dataset\\(s\\) beyond the textbook seven"):
        tool.extra_keys(stub.replace(f', "{EXTRA}"', ""))
    with pytest.raises(SystemExit, match="names 2 dataset\\(s\\) beyond the textbook seven"):
        tool.extra_keys(stub.replace(f', "{EXTRA}"', f', "{EXTRA}", "darts"'))
    with pytest.raises(SystemExit, match="REAL_ORDER not found"):
        tool.extra_keys(stub.replace("REAL_ORDER", "REAL_ORDERING"))


def test_sync_copies_rewrites_imports_and_windows(tool, tree, compass, tmp_path, capsys):
    package = tmp_path / "figures"
    rc, out = _run_tool(tool, tree, package, compass, capsys=capsys)
    assert rc == 0 and f"synced {len(tool.MODULES)} copies" in out
    sweep = tool.mention_regex([EXTRA], ["extra-model_sonnet-4.6"])
    for rel in tool.MODULES:
        copy = (package / rel).read_text()
        if rel != REALDATA:  # the realdata copy is checked by the next test
            assert copy == _expected_copy(rel), rel
        assert not sweep.search(copy), rel
    manifest = json.loads((package / "provenance.json").read_text())
    assert set(manifest["files"]) == set(tool.MODULES) and manifest["imports"] == "enforce"
    assert manifest["import_map"] == {"compass": ""}
    assert set(manifest["not_covered"]) == {"render.py", "compare_pdf.py"}
    for rel, entry in manifest["files"].items():
        assert entry["source"] == rel and entry["kind"] == "full"
        assert entry["sha256_source"] == _sha256(tree / rel), rel
        assert entry["sha256_copy"] == _sha256(package / rel), rel
        for r in entry["rewritten"]:  # the manifest never quotes the research text
            assert set(r) == {"line", "research_lines", "copy", "research_sha256"}
            assert not sweep.search(json.dumps(r))
            assert len(r["research_sha256"]) == 64
    assert manifest["files"][F3_COMMON]["repointed"] == ["REAL_ORDER", "DATASET_NAME"]
    assert [r["line"] for r in manifest["files"][F3_COMMON]["rewritten"]] == [
        _stub(F3_COMMON).splitlines().index(REAL_ORDER_OLD.rstrip("\n")) + 1,
        _stub(F3_COMMON).splitlines().index(DATASET_NAME_OLD.rstrip("\n")) + 1]
    assert manifest["files"][F3C]["rewritten"][0]["copy"] == USAGE_NEW.rstrip("\n")
    assert manifest["files"][F2_PROTOCOLS]["rewritten"] == [
        {"line": _stub(F2_PROTOCOLS).splitlines().index(WINDOW_OLD.splitlines()[0]) + 1, "research_lines": 2,
         "copy": WINDOW_NEW.rstrip("\n"), "research_sha256": tool.window_hash(WINDOW_OLD.splitlines(keepends=True))}]
    assert manifest["files"][F2_RUN]["new_imports"] == ["import os"]
    assert manifest["files"][REALDATA]["new_imports"] == ["import os"]
    assert manifest["files"][REALDATA]["dropped_patterns"] == tool.DROPPED_PATTERNS[REALDATA]
    assert manifest["files"]["bench/dgp.py"]["notes"] == "verbatim modulo imports"
    assert manifest["files"]["bench/dgp.py"]["sha256_source"] == manifest["files"]["bench/dgp.py"]["sha256_copy"]


def test_realdata_copy_loses_the_eighth_dataset(tool, tree, compass, tmp_path, capsys):
    package = tmp_path / "figures"
    _run_tool(tool, tree, package, compass, capsys=capsys)
    copy = (package / REALDATA).read_text()
    assert copy.startswith('"""Real-data arm of the benchmark: seven literature datasets, each with its\n'
                           'textbook misspecified starting model.\n"""')
    for gone in ("ZZ_EXTRA", "load_zz_extra", "the eighth entry's budget", "the extra columns", "the extra baseline"):
        assert gone not in copy
    for kept in ("DEFAULT_BUDGET", "def load_golf", "GOLF_ORIGINAL", 'key="golf"', 'KEYS = list(DATASETS)',
                 '__all__ = ["DATASETS", "KEYS", "DEFAULT_BUDGET"]'):
        assert kept in copy, kept
    assert "import os\n" in copy and 'os.environ.get("COMPASS_DATA_DIR"' in copy
    assert "\n\n\n\n" not in copy
    # the registry's eighth entry must be the one REAL_ORDER names
    (tree / REALDATA).write_text(REALDATA_RESEARCH.replace(f'key="{EXTRA}"', 'key="other"'))
    with pytest.raises(SystemExit, match="is not the one figure3/figure3_common.py names"):
        tool.main(["--research", str(tree), "--package", str(tmp_path / "second"), "--compass", str(compass)])
    # a research edit that adds a sixth constant stops the rule
    (tree / REALDATA).write_text(REALDATA_RESEARCH.replace("ZZ_EXTRA_INT_SCALE = 1000\n",
                                                            "ZZ_EXTRA_INT_SCALE = 1000\nZZ_EXTRA_MORE = 1\n"))
    with pytest.raises(SystemExit, match="removed 8 definition\\(s\\), expected 7"):
        tool.main(["--research", str(tree), "--package", str(tmp_path / "third"), "--compass", str(compass)])


def test_check_mode_is_idempotent_and_detects_drift_and_stray_files(tool, tree, compass, tmp_path, capsys):
    package = tmp_path / "figures"
    _run_tool(tool, tree, package, compass, capsys=capsys)
    rc, out = _run_tool(tool, tree, package, compass, "--check", capsys=capsys)
    assert rc == 0 and f"{len(tool.MODULES)} copies up to date" in out
    assert "provenance.json                  unchanged (dates and commit as recorded)" in out
    # a hand edit of a verbatim copy
    target = package / "figure2" / "figure2b_create.py"
    target.write_text(target.read_text().replace("return common", "return None"))
    rc, out = _run_tool(tool, tree, package, compass, "--check", capsys=capsys)
    assert rc == 1 and "figure2/figure2b_create.py" in out and "-    return None" in out
    _run_tool(tool, tree, package, compass, capsys=capsys)  # restores it
    # a research edit changes the copy and the recorded hashes
    src = tree / "figure2" / "figure2b_create.py"
    src.write_text(src.read_text() + "\n\ndef extra():\n    return 1\n")
    rc, out = _run_tool(tool, tree, package, compass, "--check", capsys=capsys)
    assert rc == 1 and "+def extra" in out and "sha256_source" in out
    _run_tool(tool, tree, package, compass, capsys=capsys)
    # a hand edit of the manifest, and a missing manifest
    manifest_path = package / "provenance.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][F3_COMMON]["repointed"] = ["REAL_ORDER"]
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    rc, out = _run_tool(tool, tree, package, compass, "--check", capsys=capsys)
    assert rc == 1 and "provenance.json" in out and '"DATASET_NAME"' in out
    manifest_path.unlink()
    rc, out = _run_tool(tool, tree, package, compass, "--check", capsys=capsys)
    assert rc == 1 and "provenance.json                  missing" in out
    _run_tool(tool, tree, package, compass, capsys=capsys)
    # a file that is neither a copy nor package-only
    (package / "notes.txt").write_text("stray")
    (package / "bench" / "rescore.py").write_text("# not copied\n")
    rc, out = _run_tool(tool, tree, package, compass, "--check", capsys=capsys)
    assert rc == 1 and "NOT MIRRORED" in out and "notes.txt" in out and "bench/rescore.py" in out
    (package / "notes.txt").unlink()
    (package / "bench" / "rescore.py").unlink()
    # outputs, caches and the package-only files are allowed
    for d in ("out", ".test_cache", "runs"):
        (package / d).mkdir()
        (package / d / "x.json").write_text("{}")
    _write_package_only(package)
    for name in ("reference_manifest.json", "requirements.txt", "README.md", "figure2/README.md",
                 "figure2/flow.svg", "figure3/README.md", "figure3/flow.svg", "figure4/README.md", "figure4/flow.svg"):
        (package / name).write_text("x\n")
    rc, out = _run_tool(tool, tree, package, compass, "--check", capsys=capsys)
    assert rc == 0 and f"{len(tool.MODULES)} copies up to date" in out


def test_sync_refuses_imports_without_a_rule_or_without_a_copy(tool, tree, compass, tmp_path):
    package = tmp_path / "figures"
    src = tree / "figure2" / "figure2c_create.py"
    base = _stub("figure2/figure2c_create.py")
    for line, message in (("from bench.drift import DriftConfig  # noqa: E402\n", "'bench.drift' is not copied"),
                          ("from bench import drift  # noqa: E402\n", "'bench' is copied but 'drift' is not"),
                          ("import figure3.figure3d_create  # noqa: E402\n", "'figure3.figure3d_create' is not copied"),
                          ("from tournament import run  # noqa: E402\n", "'compass.tournament' is not copied")):
        src.write_text(base + "\n" + line)
        with pytest.raises(SystemExit, match=message.replace("(", "\\(").replace(")", "\\)")):
            tool.main(["--research", str(tree), "--package", str(package), "--compass", str(compass)])
    (tree / "plots.py").write_text("")
    src.write_text(base + "\nfrom plots import draw  # noqa: E402\n")
    with pytest.raises(SystemExit, match="import of research module 'plots' has no rule"):
        tool.main(["--research", str(tree), "--package", str(package), "--compass", str(compass)])
    # the kept roots, their siblings and modules outside the research tree need no rule
    src.write_text(base + "\nimport figure2.figure2_common  # noqa: E402\n"
                   "from figure3 import figure3_common  # noqa: E402\n"
                   "from bench import dgp  # noqa: E402\n"
                   "import figure2_compose  # noqa: E402\n"
                   "import numpy as np  # noqa: E402\n")
    rc = tool.main(["--research", str(tree), "--package", str(package), "--compass", str(compass)])
    assert rc == 0
    assert (package / "figure2" / "figure2c_create.py").read_text() == src.read_text()


def test_sync_stops_when_a_window_or_an_anchor_is_missing(tool, tree, compass, tmp_path):
    package = tmp_path / "figures"
    # a hashed window that the research text no longer carries
    src = tree / F2_PROTOCOLS
    stub = _stub(F2_PROTOCOLS)
    src.write_text(stub.replace("    # one more arm on the extra dataset\n", "    # one more arm\n"))
    with pytest.raises(SystemExit, match="hashed window .* found 0 times, expected 1; re-pin the rule with --hash"):
        tool.main(["--research", str(tree), "--package", str(package), "--compass", str(compass)])
    src.write_text(stub)
    # the eighth key without its DATASET_NAME entry, and the usage line without a second arm
    f3 = tree / F3_COMMON
    f3.write_text(_stub(F3_COMMON).replace(DATASET_NAME_OLD, DATASET_NAME_NEW))
    with pytest.raises(SystemExit, match="DATASET_NAME has no entry for the eighth dataset"):
        tool.main(["--research", str(tree), "--package", str(package), "--compass", str(compass)])
    f3.write_text(_stub(F3_COMMON))
    (tree / F3C).write_text(_stub(F3C).replace(USAGE_OLD, USAGE_NEW))
    with pytest.raises(SystemExit, match="usage line with a second arm directory found 0 times"):
        tool.main(["--research", str(tree), "--package", str(package), "--compass", str(compass)])
    (tree / F3C).write_text(_stub(F3C))
    # the cache anchor of figure2_run.py
    (tree / F2_RUN).write_text(_stub(F2_RUN).replace(CACHE_OLD, 'CACHE = CODE_DIR / "cache"\n'))
    with pytest.raises(SystemExit, match="patch anchor found 0 times, expected 1"):
        tool.main(["--research", str(tree), "--package", str(package), "--compass", str(compass)])
    (tree / F2_RUN).write_text(_stub(F2_RUN))
    rc = tool.main(["--research", str(tree), "--package", str(package), "--compass", str(compass)])
    assert rc == 0


def test_sync_stops_when_a_mention_survives(tool, tree, compass, tmp_path):
    package = tmp_path / "figures"
    src = tree / "figure3" / "figure3a_create.py"
    stub = _stub("figure3/figure3a_create.py")
    for tail in (f"\n# {EXTRA} records are read elsewhere\n", f"\nEXTRA = {EXTRA.upper().replace('_', '-')!r}\n",
                 '\nARM = "extra-model_sonnet-4.6"\n'):
        src.write_text(stub + tail)
        with pytest.raises(SystemExit, match="figure3/figure3a_create.py:.*still mentions the eighth dataset or its arm"):
            tool.main(["--research", str(tree), "--package", str(package), "--compass", str(compass)])
        assert not (package / "figure3" / "figure3a_create.py").is_file()
    src.write_text(stub)


def test_hash_helper_prints_the_window_hash(tool, tmp_path, capsys):
    f = tmp_path / "x.py"
    f.write_text("a\nb\nc\n")
    assert tool.main(["--hash", str(f), "2", "3"]) == 0
    assert capsys.readouterr().out.strip() == f"2 {_sha256_text('b' + chr(10) + 'c' + chr(10))}"


# ---------------------------------------------------------------------------
# The checker on the generated package
# ---------------------------------------------------------------------------

def _statuses(report, file: str) -> dict[str, str]:
    return {item.name: item.status for item in report.items if item.file == file}


def _info(report, file: str) -> list[str]:
    return next(fr.info for fr in report.files if fr.copy == file)


def test_checker_passes_the_generated_package(tool, checker, tree, compass, tmp_path, capsys):
    package = tmp_path / "figures"
    _run_tool(tool, tree, package, compass, capsys=capsys)
    _write_package_only(package)
    report = checker.run(package / "provenance.json", tree)
    assert report.ok, checker.format_report(report)
    assert _statuses(report, F3_COMMON) == {"ROOT": "IDENTICAL", "REAL_ORDER": "REPOINTED",
                                            "DATASET_NAME": "REPOINTED", "series": "IDENTICAL"}
    assert _statuses(report, F2_PROTOCOLS)["ARM_PROTOCOLS"] == "REPOINTED"
    assert _statuses(report, F2_RUN) == {"CODE_DIR": "IDENTICAL", "CACHE": "REPOINTED", "run_arm": "IDENTICAL_MODULO_IMPORTS"}
    got = _statuses(report, REALDATA)
    assert got["DATA_DIR"] == "REPOINTED" and got["_DATASETS"] == "REPOINTED" and got["__all__"] == "REPOINTED"
    assert got["ZZ_EXTRA_BUDGET"] == got["load_zz_extra"] == got["ZZ_EXTRA_ORIGINAL"] == "DROPPED"
    assert "docstring: rewritten" in _info(report, REALDATA) and "docstring: rewritten" in _info(report, F3C)
    assert "imports: enforced, 4 statements match after normalisation (1 new, 0 dropped)" in _info(report, F2_RUN)
    assert _statuses(report, "bench/holdout.py") == {"split": "IDENTICAL"}
    assert "imports: enforced, 2 statements match after normalisation (0 new, 0 dropped)" in _info(report, "bench/holdout.py")
    counts = report.counts()
    assert counts["DIFFERENT"] == counts["UNCOVERED"] == counts["MISSING"] == counts["STALE"] == 0
    text = checker.format_report(report)
    assert "PASS: every definition is accounted for" in text
    assert f"copy sweep         {len(tool.MODULES) - 4 + 2} *.py files outside tests/; all listed" in text


def test_checker_fails_a_body_edit_and_an_unlisted_name(tool, checker, tree, compass, tmp_path, capsys):
    package = tmp_path / "figures"
    _run_tool(tool, tree, package, compass, capsys=capsys)
    _write_package_only(package)
    manifest = package / "provenance.json"
    target = package / "figure2" / "figure2a_create.py"
    original = target.read_text()
    target.write_text(original.replace("return common", "return None"))
    report = checker.run(manifest, tree)
    assert _statuses(report, "figure2/figure2a_create.py")["count"] == "DIFFERENT" and not report.ok
    target.write_text(original)
    f3 = package / F3_COMMON
    f3.write_text(f3.read_text() + "\nEXTRA = 1\n")
    report = checker.run(manifest, tree)
    assert _statuses(report, F3_COMMON)["EXTRA"] == "UNCOVERED" and not report.ok


# ---------------------------------------------------------------------------
# The committed copies and manifest
# ---------------------------------------------------------------------------

def test_committed_manifest_records_the_copies(tool):
    manifest = json.loads((FIGURES_DIR / "provenance.json").read_text())
    assert set(manifest["files"]) == set(tool.MODULES) and len(manifest["files"]) == 43
    assert manifest["generator"] == "tools/sync_figures.py"
    assert manifest["research_commit_short"] == manifest["research_commit"][:7]
    assert manifest["imports"] == "enforce" and manifest["import_map"] == {"compass": ""}
    assert set(manifest["not_covered"]) == {"render.py", "compare_pdf.py"}
    for rel, entry in manifest["files"].items():
        copy = FIGURES_DIR / rel
        assert entry["source"] == rel and entry["kind"] == "full", rel
        assert entry["sha256_copy"] == _sha256(copy), rel
        text = copy.read_text()
        lines = text.splitlines()
        for r in entry["rewritten"]:
            assert set(r) == {"line", "research_lines", "copy", "research_sha256"}
            assert len(r["research_sha256"]) == 64
            if "\n" not in r["copy"] and r["copy"] and not r["copy"].startswith("("):
                assert lines[r["line"] - 1] == r["copy"], (rel, r["line"])
        if rel in tool.REPOINTED or rel in tool.HASHED_PATCHES or rel in (F3C, REALDATA):
            assert entry["rewritten"], rel
        else:
            assert entry["rewritten"] == [], rel
            assert entry["notes"] == "verbatim modulo imports", rel
    assert {rel for rel, e in manifest["files"].items() if e["repointed"]} == set(tool.REPOINTED)
    assert manifest["files"][F2_PROTOCOLS]["dropped"] == ["REAL_PUBLISHED"]
    text = (FIGURES_DIR / F3_COMMON).read_text()
    with pytest.raises(SystemExit, match="names 0 dataset\\(s\\) beyond the textbook seven"):
        tool.extra_keys(text)


def _research_or_skip() -> Path:
    if not (RESEARCH / "figure2" / "figure2_run.py").is_file():
        pytest.skip("no research checkout next to this repository")
    return RESEARCH


def test_committed_copies_are_up_to_date_with_the_research_tree():
    research = _research_or_skip()
    proc = subprocess.run([sys.executable, "-I", str(TOOL), "--research", str(research), "--check"],
                          capture_output=True, text=True, timeout=300, cwd=REPO_DIR)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-800:]
    assert "43 copies up to date" in proc.stdout
    assert "totals: 21 import nodes rewritten, 20 windows rewritten in 43 modules" in proc.stdout


def test_committed_manifest_passes_the_checker_against_the_research_tree():
    research = _research_or_skip()
    proc = subprocess.run([sys.executable, "-I", str(CHECKER), "--research", str(research),
                           "--manifest", str(FIGURES_DIR / "provenance.json")],
                          capture_output=True, text=True, timeout=300, cwd=REPO_DIR)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-800:]
    assert "PASS: every definition is accounted for" in proc.stdout
    assert "REPOINTED                   11" in proc.stdout
    assert "DROPPED                      7" in proc.stdout
    assert "copy sweep         41 *.py files outside tests/; all listed" in proc.stdout
