#!/usr/bin/env python3
"""Render the manuscript's figures from the benchmark records: eleven vector PDFs.

    python figures/render.py [--arms DIR] [--out DIR] [--only a,b] [--list] [--png]

Every entry of the table is one documented command line of the research ``figure2/``,
``figure3/`` and ``figure4/`` READMEs, run through the copied script's ``main`` with the arm
directories taken from the data root: ``--arms DIR``, else ``$COMPASS_ARMS_DIR``, else
``../bench/arms`` next to this repository (the research layout). The files carry the names the
manuscript uses (``<name>_editable.pdf``: vector graphics, TrueType text that stays editable);
``--png`` writes a 200 dpi raster of every selected file as well (``<name>.png``, through the
same scripts). The driver stops before drawing, naming the missing directories, when an arm
directory a selected file needs is absent or holds no record; a composite needs its panels in
``--out`` (render them first or select them too). Every rendered file is announced with a
``rendered <name> -> <file>`` line, and the exit status is 0 only when every selected file was
written.

Reproducible bytes. matplotlib stamps every PDF with the time it was written, in the local
timezone (``D:YYYYMMDDhhmmssZ`` under UTC, ``D:YYYYMMDDhhmmss-05'00'`` elsewhere, six bytes
longer), so two renders of the same records would differ in that stamp and, across timezones,
in the cross-reference offsets after it. This driver sets ``SOURCE_DATE_EPOCH=0`` before
matplotlib is first imported unless the caller has set it, which fixes the stamp to
``D:19700101000000Z``: its renders are byte-identical on every machine in every timezone, and
``compare_pdf.py`` tells a stamp-only difference from a real one for files rendered without the
variable (a script run on its own does not set it).

  name             file                          script                               records
  fig2a            fig2a_editable.pdf            figure2/figure2a_create.py           the ten census arms
  fig2b            fig2b_editable.pdf            figure2/figure2b_create.py           the ten census arms
  fig2c            fig2c_editable.pdf            figure2/figure2c_create.py           e1_anthropic
  fig2d            fig2d_editable.pdf            figure2/figure2d_create.py           e5_anthropic
  fig2_composite   fig2_composite_editable.pdf   figure2/figure2_compose.py           the four panels above
  fig3a            fig3a_editable.pdf            figure3/figure3a_create.py           e2_anthropic/drift
  fig3b            fig3b_editable.pdf            figure3/figure3b_create.py           e2_anthropic/drift
  fig3c            fig3c_editable.pdf            figure3/figure3a_create.py --symlog  e2_anthropic_real/drift
  fig3d            fig3d_editable.pdf            figure3/figure3c_create.py           e1_anthropic_realdata_x2
  fig3_composite   fig3_composite_editable.pdf   figure3/figure3_compose.py           the four panels above, two row titles
  fig4_cost        fig4_cost_editable.pdf        figure4/figure4_create.py --grouped  the five cost arms

fig2a to fig2_composite are Figure 2 of the paper, fig3a to fig3_composite Figure 3, and
fig4_cost the cost figure (Supplementary Fig. 1). See figures/README.md.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

# Before matplotlib is first imported (the figureN modules below import it): the date stamp of
# every PDF written in this process is the epoch, in UTC, unless the caller chose one.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

FIGURES_DIR = Path(__file__).resolve().parent
if str(FIGURES_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURES_DIR))

from figure2 import (  # noqa: E402
    figure2_compose,
    figure2a_create,
    figure2b_create,
    figure2c_create,
    figure2d_create,
)
from figure3 import (  # noqa: E402
    figure3_compose,
    figure3a_create,
    figure3b_create,
    figure3c_create,
)
from figure4 import figure4_create  # noqa: E402

# ---------------------------------------------------------------------------
# The data root
# ---------------------------------------------------------------------------

ENV_VAR = "COMPASS_ARMS_DIR"
# the research layout: this repository sits inside the research checkout's code/ directory
DEFAULT_ARMS_DIR = FIGURES_DIR.parents[1] / "bench" / "arms"
DEFAULT_OUT = FIGURES_DIR / "out"
SUFFIX = "_editable"  # the manuscript's file names: <name>_editable.pdf


def arms_source(override: str | Path | None = None) -> str:
    """How the data root is chosen: ``flag``, ``env`` or ``default``."""
    if override:
        return "flag"
    if os.environ.get(ENV_VAR):
        return "env"
    return "default"


def arms_dir(override: str | Path | None = None) -> Path:
    """The directory holding the arm directories (no existence check)."""
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_ARMS_DIR


def require(root: Path, arms: Iterable[str], *, source: str = "default") -> None:
    """Stop with a clear message unless every arm directory exists under ``root`` and holds at
    least one record."""
    missing = []
    for arm in arms:
        d = root / arm
        if not d.is_dir():
            missing.append(f"{d}  (directory missing)")
        elif not any(d.glob("*.json")):
            missing.append(f"{d}  (no *.json records)")
    if not missing:
        return
    lines = [f"data root {root} (chosen by {source}) is missing {len(missing)} item(s):",
             *(f"  {m}" for m in missing),
             "Point the figure code at the arms directory of the benchmark records with",
             "  --arms /path/to/arms, or", f"  {ENV_VAR}=/path/to/arms, or",
             f"  keep this repository inside the research checkout (default {DEFAULT_ARMS_DIR}).",
             "Nothing was rendered."]
    raise SystemExit("\n".join(lines))


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

# research figure2/README.md, A=...: the ten census arms in display order
CENSUS_ARMS = ("e5_anthropic", "e1_anthropic", "e1_anthropic_realdata_x2", "e5_haiku", "e1_haiku",
               "e5_gpt", "e1_gpt", "e1_gpt_realdata_x2", "e5_gemini_flash", "e1_gemini_flash")
# research figure4/README.md, A=...: the misspecified-baseline arm of each proposer family and
# the Sonnet real-data arm at default budget
COST_ARMS = ("e1_anthropic", "e1_anthropic_realdata", "e1_haiku", "e1_gpt", "e1_gemini_flash")
# research figure3/README.md: the two row titles of the Figure 3 sheet
ROW_TITLES = ("When nothing is wrong: eight synthetic problems, each started from its correct model",
              "When a fix exists: seven real datasets, each started from its textbook model")
FIG2_PANELS = ("fig2a", "fig2b", "fig2c", "fig2d")
FIG3_PANELS = ("fig3a", "fig3b", "fig3c", "fig3d")


def output(out: Path, name: str, ext: str) -> Path:
    """The file a name is written to: ``<name>_editable.pdf`` or ``<name>.png``."""
    return out / (f"{name}{SUFFIX}.pdf" if ext == "pdf" else f"{name}.{ext}")


@dataclass(frozen=True)
class Entry:
    name: str
    figure: str  # the paper's figure or panel
    script: str  # the copied script (relative to figures/) and its flag, if any
    arms: tuple[str, ...]  # arm directories (relative to the data root) that must hold records
    inputs: tuple[str, ...]  # names of the files a composite reads from --out
    argv: Callable[[Path, Path, str], list[str]]  # (data root, out, ext) -> the script's argv
    main: Callable[[list[str]], object]


def _dirs(root: Path, arms: tuple[str, ...]) -> list[str]:
    return [str(root / a) for a in arms]


def _out(out: Path, name: str, ext: str) -> list[str]:
    return ["--out", str(output(out, name, ext))]


ENTRIES: tuple[Entry, ...] = (
    Entry("fig2a", "Fig. 2a", "figure2/figure2a_create.py", CENSUS_ARMS, (),
          lambda root, out, ext: [*_dirs(root, CENSUS_ARMS), *_out(out, "fig2a", ext)],
          figure2a_create.main),
    Entry("fig2b", "Fig. 2b", "figure2/figure2b_create.py", CENSUS_ARMS, (),
          lambda root, out, ext: [*_dirs(root, CENSUS_ARMS), *_out(out, "fig2b", ext)],
          figure2b_create.main),
    Entry("fig2c", "Fig. 2c", "figure2/figure2c_create.py", ("e1_anthropic",), (),
          lambda root, out, ext: [str(root / "e1_anthropic"), *_out(out, "fig2c", ext)],
          figure2c_create.main),
    Entry("fig2d", "Fig. 2d", "figure2/figure2d_create.py", ("e5_anthropic",), (),
          lambda root, out, ext: [str(root / "e5_anthropic"), *_out(out, "fig2d", ext)],
          figure2d_create.main),
    Entry("fig2_composite", "Fig. 2", "figure2/figure2_compose.py", (), FIG2_PANELS,
          lambda root, out, ext: [*(str(output(out, n, ext)) for n in FIG2_PANELS),
                                  *_out(out, "fig2_composite", ext)],
          figure2_compose.main),
    Entry("fig3a", "Fig. 3a", "figure3/figure3a_create.py", ("e2_anthropic/drift",), (),
          lambda root, out, ext: [str(root / "e2_anthropic" / "drift"), *_out(out, "fig3a", ext)],
          figure3a_create.main),
    Entry("fig3b", "Fig. 3b", "figure3/figure3b_create.py", ("e2_anthropic/drift",), (),
          lambda root, out, ext: [str(root / "e2_anthropic" / "drift"), *_out(out, "fig3b", ext)],
          figure3b_create.main),
    Entry("fig3c", "Fig. 3c", "figure3/figure3a_create.py --symlog", ("e2_anthropic_real/drift",), (),
          lambda root, out, ext: [str(root / "e2_anthropic_real" / "drift"),
                                  *_out(out, "fig3c", ext), "--symlog"],
          figure3a_create.main),
    Entry("fig3d", "Fig. 3d", "figure3/figure3c_create.py", ("e1_anthropic_realdata_x2",), (),
          lambda root, out, ext: [str(root / "e1_anthropic_realdata_x2"), *_out(out, "fig3d", ext)],
          figure3c_create.main),
    Entry("fig3_composite", "Fig. 3", "figure3/figure3_compose.py --row-titles ...", (), FIG3_PANELS,
          lambda root, out, ext: [*(str(output(out, n, ext)) for n in FIG3_PANELS),
                                  *_out(out, "fig3_composite", ext), "--row-titles", *ROW_TITLES],
          figure3_compose.main),
    Entry("fig4_cost", "Supplementary Fig. 1", "figure4/figure4_create.py --grouped", COST_ARMS, (),
          lambda root, out, ext: [*_dirs(root, COST_ARMS), *_out(out, "fig4_cost", ext), "--grouped"],
          figure4_create.main),
)
BY_NAME: dict[str, Entry] = {e.name: e for e in ENTRIES}
NAMES: tuple[str, ...] = tuple(BY_NAME)


def required_arms(names: list[str]) -> list[str]:
    """The arm directories the selected files read, in first-use order, each once."""
    seen: list[str] = []
    for name in names:
        for arm in BY_NAME[name].arms:
            if arm not in seen:
                seen.append(arm)
    return seen


def render(names: list[str], root: Path, out: Path, exts: tuple[str, ...]) -> list[str]:
    """Render the selected names into ``out`` for every extension; returns the failures."""
    failures: list[str] = []
    for name in names:
        entry = BY_NAME[name]
        for ext in exts:
            what = f"{name} ({ext})"
            target = output(out, name, ext)
            missing = [output(out, i, ext).name for i in entry.inputs
                       if not output(out, i, ext).is_file()]
            if missing:
                print(f"FAILED {what}: input(s) not in {out}: {missing}; render "
                      f"{', '.join(entry.inputs)} first, or select them too")
                failures.append(what)
                continue
            try:
                entry.main(entry.argv(root, out, ext))
            except SystemExit as exc:  # a script refusing its input ("no scored trajectories in ...")
                print(f"FAILED {what}: {exc}")
                failures.append(what)
                continue
            except Exception as exc:  # one file's failure must not hide the others
                traceback.print_exc()
                print(f"FAILED {what}: {exc!r}")
                failures.append(what)
                continue
            if not target.is_file():
                print(f"FAILED {what}: expected output not written: {target}")
                failures.append(what)
                continue
            print(f"rendered {name} -> {target}")
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=None,
                    help=f"directory holding the arm directories (default: ${ENV_VAR} "
                         f"or {DEFAULT_ARMS_DIR})")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output directory, created (default: figures/out, ignored by git)")
    ap.add_argument("--only", help="comma-separated names (default: all eleven)")
    ap.add_argument("--list", action="store_true", help="list the names and exit")
    ap.add_argument("--png", action="store_true",
                    help="also write <name>.png (200 dpi) for every selected name")
    args = ap.parse_args(argv)
    if args.list:
        print("\n".join(NAMES))
        return 0
    names = [s.strip() for s in args.only.split(",")] if args.only else list(NAMES)
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        print(f"unknown name(s) {unknown}; known: {list(NAMES)}", file=sys.stderr)
        return 2
    root = arms_dir(args.arms)
    source = arms_source(args.arms)
    require(root, required_arms(names), source=source)  # stops before drawing
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    print(f"data root {root} (chosen by {source}); output {out}")
    exts = ("pdf", "png") if args.png else ("pdf",)
    failures = render(names, root, out, exts)
    total = len(names) * len(exts)
    if failures:
        print(f"{len(failures)} of {total} file(s) failed: {failures}", file=sys.stderr)
        return 1
    print(f"{total} file(s) rendered into {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
