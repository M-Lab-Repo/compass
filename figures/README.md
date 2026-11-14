# Figure code of the paper

This directory is the single folder that reproduces the paper's figures: from scratch, with the
run, score and create scripts of every panel, and from the committed benchmark records, with the
driver that redraws the manuscript's PDFs. The per-panel scripts under `figure2/`, `figure3/` and
`figure4/` are copies of the research repository's scripts for Figure 2, Figure 3 and the cost
figure (Supplementary Fig. 1), written by `tools/sync_figures.py` and checked definition by
definition against their research sources (`tools/check_provenance.py` with `provenance.json`);
`bench/` holds the four benchmark modules the run and score steps import; `render.py` runs the
create scripts in the manuscript's order and writes its eleven vector PDFs; `compare_pdf.py`
checks a render against `reference_manifest.json`. The create scripts import nothing from the
pipeline; the run and score steps import the installed `compass` package. No benchmark record
ships with the repository: every panel reads the arm directories of the benchmark through a
configurable data root (see "Data"). Figure 1 is a conceptual schematic drawn by hand, not
rendered from records, so it has no code here.

| Path | Content |
|---|---|
| `figure2/` | Figure 2: `figure2_protocols.py` (the eleven census arms as data), `figure2_run.py` (the census engine) and `figure2_score.py` (labels by MCMC, independent readers), the thin `figure2a_run.py` to `figure2d_run.py` and `figure2a_score.py` to `figure2d_score.py`, the drawing scripts `figure2a_create.py` to `figure2d_create.py` and `figure2_compose.py`, `figure2_common.py` (reading records, the label rule, verdicts, rule names and colours); `README.md` and `flow.svg` |
| `figure3/` | Figure 3: `figure3_protocols.py` (the two drift arms), `figure3_run.py` (the drift engine) and `figure3_score.py`, the thin run and score scripts of panels a, b, c and e, `figure3a_create.py`, `figure3b_create.py`, `figure3c_create.py` and `figure3_compose.py`, `figure3_common.py` (trajectories, the Δ of a round, latent-site counts); `README.md` and `flow.svg` |
| `figure4/` | the cost figure: `figure4_run.py` and `figure4_score.py` (the five cost arms through Figure 2's engine) and `figure4_create.py`; `README.md` and `flow.svg` |
| `bench/` | the benchmark modules the engines import: `dgp.py` (the eight synthetic data-generating processes), `realdata.py` (the seven textbook datasets), `holdout.py` (the proposer-hidden split and the hidden-fold score), `control_proposer.py` (the template controls) |
| `render.py` | the driver: the eleven names, the script and the arm directories behind each, the data root and the output directory |
| `compare_pdf.py`, `reference_manifest.json` | the date-neutral comparison of rendered PDFs with the reference PDFs, and the hashes, sizes and page sizes of the eleven reference files |
| `provenance.json` | the manifest of the copies: the research commit, the sha256 of every source and copy, and the rewritten windows |
| `requirements.txt` | the library versions of the reference renders |

The scripts were copied from the research tree at commit `8e04643` (2026-11-12; `provenance.json`
records it as `research_commit`, with `copied_on` 2026-11-14), where the drawing scripts were
last changed at `835b750` (2026-11-12). The records every panel reads are the benchmark records
of the research repository at commit `2c66219` (2026-10-26: every independent reader re-judged
with the symptom-free problem statement, `e1_gpt_realdata_x2` rescored at its double budget),
and the reference PDFs of `reference_manifest.json` are renders of those records with the
scripts at `835b750` (see "Verifying a render").

The re-judge is what the reader numbers of Fig. 2d rest on. On the 72 runs of `e5_anthropic`, in
which the baseline is the correct model, the three independent readers (Haiku 4.5, Opus 4.5 and
GPT-5) would have replaced it in 93%, 81% and 92% of the runs (67, 58 and 66 of 72), the
proposer itself in 92% (66), the execution check in 96% (69) and the rationale-withheld control
in 76% (55), against the referee's 4% (3 of 72). `figure2d_create.py` prints these counts before
drawing.

## Panels, scripts and data

| Paper | Name in `render.py` | Script | Records read | Output file |
|---|---|---|---|---|
| Fig. 2a, what the agent proposed | `fig2a` | `figure2/figure2a_create.py` | the ten census arms | `fig2a_editable.pdf` |
| Fig. 2b, who accepts them | `fig2b` | `figure2/figure2b_create.py` | the ten census arms | `fig2b_editable.pdf` |
| Fig. 2c, the theorem in one picture | `fig2c` | `figure2/figure2c_create.py` | `e1_anthropic` for the two measured flat lines; the curves are simulated (4,000 replicates per n, seed 0) | `fig2c_editable.pdf` |
| Fig. 2d, when nothing is wrong | `fig2d` | `figure2/figure2d_create.py` | `e5_anthropic` | `fig2d_editable.pdf` |
| Fig. 2, the sheet | `fig2_composite` | `figure2/figure2_compose.py` | the four PDFs above | `fig2_composite_editable.pdf` |
| Fig. 3a, deployed model over successive rounds | `fig3a` | `figure3/figure3a_create.py` | `e2_anthropic/drift` | `fig3a_editable.pdf` |
| Fig. 3b, why the deployed model drifts | `fig3b` | `figure3/figure3b_create.py` | `e2_anthropic/drift` | `fig3b_editable.pdf` |
| Fig. 3c, drift on real data | `fig3c` | `figure3/figure3a_create.py --symlog` | `e2_anthropic_real/drift` | `fig3c_editable.pdf` |
| Fig. 3d, outcome of the accepted revision | `fig3d` | `figure3/figure3c_create.py` | `e1_anthropic_realdata_x2` | `fig3d_editable.pdf` |
| Fig. 3, the sheet with its two row titles | `fig3_composite` | `figure3/figure3_compose.py --row-titles ...` | the four PDFs above | `fig3_composite_editable.pdf` |
| Supplementary Fig. 1, what a verdict costs | `fig4_cost` | `figure4/figure4_create.py --grouped` | the five cost arms | `fig4_cost_editable.pdf` |

The ten census arms, in display order: `e5_anthropic`, `e1_anthropic`,
`e1_anthropic_realdata_x2`, `e5_haiku`, `e1_haiku`, `e5_gpt`, `e1_gpt`, `e1_gpt_realdata_x2`,
`e5_gemini_flash`, `e1_gemini_flash`. The five cost arms: `e1_anthropic`,
`e1_anthropic_realdata`, `e1_haiku`, `e1_gpt`, `e1_gemini_flash` (the misspecified-baseline arm
of each proposer family and the Sonnet real-data arm at default budget). The script names do not
follow the paper's panel letters everywhere (Fig. 3c is drawn by `figure3a_create.py --symlog`,
Fig. 3d by `figure3c_create.py`); the table is the mapping. The drawing scripts share only
`figure2/figure2_common.py` and `figure3/figure3_common.py`; they import matplotlib, numpy,
Pillow and pypdf and nothing from the pipeline. Each panel script prints the numbers it is about
to draw before writing its file (`--json PATH` writes them as well); an `--out` ending in `.png`
gives a 200 dpi raster instead of the vector PDF. `--help` prints a script's options.

## Redrawing the figures from the committed records

```bash
pip install -e '.[figures]'          # or pip install -r figures/requirements.txt for the reference versions
python figures/render.py --list                                              # the eleven names
python figures/render.py --arms /path/to/arms --out /path/to/out             # all eleven files
python figures/render.py --arms /path/to/arms --out /path/to/out --only fig3a,fig3b,fig3c,fig3d,fig3_composite
python figures/render.py --arms /path/to/arms --out /path/to/out --png       # plus <name>.png rasters
```

The driver runs each script's `main` with the arm directories of the data root (flag,
`COMPASS_ARMS_DIR`, default; see "Data"), in the order of the table, and prints
`rendered <name> -> <file>` per file. It stops before drawing, naming the missing directories,
when an arm directory a selected file needs is absent or holds no `*.json` record; a composite
needs its four panels in `--out` (render them first or select them too). The default output
directory is `figures/out/`, which git ignores. The exit status is 0 only when every selected
file was written. Before matplotlib is first imported the driver sets `SOURCE_DATE_EPOCH=0`
unless the caller has set it, so that the date stamp of every PDF is `D:19700101000000Z` and its
renders are byte-identical on every machine in every timezone (see "Verifying a render"). A
single script can also be run on its own, as a file from any directory
(`python figures/figure3/figure3a_create.py DIR --out x.pdf --symlog`) or as a module from
`figures/` (`cd figures && python -m figure3.figure3a_create DIR --out x.pdf --symlog`); the
copies keep the research scripts' sibling import lines because `figures/` mirrors the research
layout, with `figure2/`, `figure3/`, `figure4/` and `bench/` side by side.

## Verifying a render

```bash
python figures/compare_pdf.py --rendered /path/to/out --reference figures/reference_manifest.json
python figures/compare_pdf.py --rendered /path/to/out --reference /path/to/reference/pdfs
```

A matplotlib PDF embeds the time it was written (the value of `/CreationDate` in its Info
dictionary), in the local timezone: `D:YYYYMMDDhhmmssZ` under UTC, six bytes longer
(`D:YYYYMMDDhhmmss-05'00'`) anywhere else. `render.py` sets `SOURCE_DATE_EPOCH=0` unless the
caller has set it, so its renders carry `D:19700101000000Z` and are byte-identical on every
machine in every timezone; a panel script run on its own does not set it. Everything else the
scripts write is deterministic. `compare_pdf.py` computes two hashes of every file.
`sha256_dateless` is the SHA-256 of the bytes after the value of every `/CreationDate` and
`/ModDate`, in either form and of any length, is replaced by zeros of the same length: equal
hashes mean that every other byte agrees, fonts, coordinates and text included.
`sha256_drawing` is the SHA-256 of a canonical serialisation of every page as pypdf parses it
(the page dictionary without `/Parent`, indirect references resolved, dictionary keys sorted,
streams decoded): it depends on what is drawn and not on the Info dictionary, the object
numbers, the compression or the byte offsets. The second hash settles the case the first
cannot: a date stamp of another length moves the cross-reference offsets after the Info
dictionary, so a render made without `SOURCE_DATE_EPOCH` in a non-UTC timezone differs from
the reference in `sha256_dateless` although nothing drawn differs. Against a directory of
reference PDFs the report says `BYTE-IDENTICAL`, `IDENTICAL (dates aside; N date value(s)
differ)`, `IDENTICAL (drawing); metadata differs: ...` (the Info keys whose values differ, dates
as written, and the sizes) or `DIFFERENT: ...` (sizes, page sizes and whether the extracted text
differs); against a manifest the same three verdicts come from its hashes, with the byte size,
the page count and the page size in points. The exit status is 0 only when every expected file
is present and identical in drawing. `--names a.pdf,b.pdf` restricts either form to a subset.

`reference_manifest.json` holds these values for the eleven reference PDFs: renders of the
research `figure2/`, `figure3/` and `figure4/` scripts at commit `835b750` from the benchmark
records at `2c66219`, written by `render.py` on 2026-11-14 with `SOURCE_DATE_EPOCH=0`. Its
`source` says so symbolically (never a path) and records that they match the research
`Submission/figures` working tree of 2026-11-12 in everything but the date stamp. To regenerate
it after the scripts or the records change, render the eleven files and write the manifest from
them with a label that says what they are:

```bash
python figures/render.py --arms /path/to/arms --out /path/to/out
python figures/compare_pdf.py --write-manifest figures/reference_manifest.json --reference /path/to/out \
    --source-label "renders of the research figure scripts at commit <sha> from the records at <sha>, SOURCE_DATE_EPOCH=0"
```

On 2026-11-14, in the environment of `requirements.txt` (matplotlib 3.10.9, numpy 2.2.6,
Pillow 12.2.0, pypdf 6.19.0), `render.py` wrote the eleven files from the `arms/` directory of
the data archive byte-identically under `TZ=UTC` and `TZ=EST5EDT`, and `compare_pdf.py` reported
every file `IDENTICAL (dates aside, manifest)` against the manifest (11/11 identical in drawing,
exit status 0). Byte identity dates aside is expected only with these versions: other versions
render the same figures, but glyph subsetting and PDF object layout differ between matplotlib
builds, and pypdf 7 writes the two composite sheets differently (the `figures` extra of
`pyproject.toml` keeps pypdf below 7 for that reason).

## Reproducing the figures from scratch: run, then score, then create

Every panel has three scripts, run in this order from the `figures/` directory: `_run.py` calls
the models and writes one raw record per pipeline run (or one trajectory per starting model and
rule), `_score.py` adds the held-out labels by MCMC and the independent readers' verdicts in side
files, and `_create.py` counts and draws. The per-figure READMEs document each step, its
options, what it writes and what it costs: [`figure2/README.md`](figure2/README.md) (the census
of Figure 2: ten arms, 618 runs), [`figure3/README.md`](figure3/README.md) (the two drift arms
and the real-data census arm of Figure 3) and [`figure4/README.md`](figure4/README.md) (the five
cost arms, four of them Figure 2's). Every script prints its full usage with `--help`, and every
run script lists what it would do with `--dry-run` without calling anything. The scripts are run
as files (`python figure2/figure2a_run.py`), not with `python -I`: each puts its own directory
on `sys.path` and reaches `figure2/`, `figure3/` and `bench/` through `figures/`.

```bash
pip install -e '.[providers,figures]'    # the package, the OpenAI and Google clients, the drawing libraries
cd figures
python figure2/figure2a_run.py    --out-root runs/2026-11 --dry-run                          # the 618 runs of Figure 2, nothing called
python figure2/figure2a_run.py    --out-root runs/2026-11 --arm correct-baseline_haiku-4.5   # one arm, 72 runs (proposer key)
python figure2/figure2a_score.py  --out-root runs/2026-11 --arm correct-baseline_haiku-4.5   # labels (MCMC), then readers (keys)
python figure2/figure2a_create.py runs/2026-11/correct-baseline_haiku-4.5 --out out/figure2a_rerun.pdf
python figure3/figure3a_run.py    --out-root runs/2026-11 --dry-run                          # the 32 drift trajectories of Fig. 3a and 3b
python figure3/figure3e_run.py    --out-root runs/2026-11 --dry-run                          # the 28 real-data trajectories of Fig. 3c
python figure3/figure3c_run.py    --out-root runs/2026-11 --dry-run                          # the 21 census runs of Fig. 3d
python figure4/figure4_run.py     --out-root runs/2026-11 --dry-run                          # the 295 census runs of the cost figure
```

Keys: the run step needs the API key of the proposer family it runs (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GOOGLE_API_KEY`), the score step the keys of the arm's independent readers,
and the real-data arms the Semantic Scholar key of the live literature search (`S2_API_KEY`).
Export them or put them in `figures/.env`, which the engines load (an exported variable wins;
`.env` is ignored by git). The labels need no key: they are MCMC on the stored programs.
`--out-root` is any directory outside the committed records (`figures/runs/` is ignored by git);
a run whose record exists is skipped and a partial drift trajectory is resumed, so an
interrupted pass continues where it stopped.

Cost: Figure 2 is 618 records, about $300-400 of proposer and judge calls and 25-40 hours of CPU
with four arms in parallel shells, plus about $70 for the reader passes; Figure 3 is 192 synthetic
drift steps (about $70 and 16 hours), 168 real-data drift steps (about $85 and 20 hours) and the
21 census runs of its real-data arm (about $17 and 5 hours); the cost figure adds the 7 runs of
the default-budget real-data arm to four arms Figure 2 already ran. A re-run reproduces the
protocol, not the bytes: model replies, verdict texts, timings and the cost blocks differ between
runs, and the counts the create scripts print will differ from the paper's within the spread the
figures show. The committed records are the paper's; nothing a re-run writes replaces them.

## Data

No record ships with this repository. The panels read `<arm>/*.json` under the data root, one
JSON record per pipeline run, with the drift trajectories under `<arm>/drift/`. The records are
published as an asset of the GitHub release v0.3.0, `compass_benchmark_records_v0.3.0.zip`
(https://github.com/M-Lab-Repo/compass/releases/download/v0.3.0/compass_benchmark_records_v0.3.0.zip).
Its `arms/` directory holds 18 arm directories and 13 `<arm>_summary.json` files (738 JSON
files, about 57 MB), the benchmark records of the research repository at commit `2c66219`; the
figures read 13 of the directories. Beside `arms/` the archive holds `datasets/` (the seven
vendored textbook datasets of the real-data arms, with the `README.md` that documents their
sources), its own `README.md` and `CHECKSUMS.sha256` (SHA-256 of every file; `sha256sum -c
CHECKSUMS.sha256` from inside the unpacked directory).

Three environment variables point the code at the unpacked archive and at a cache:

- `COMPASS_ARMS_DIR`, the `arms/` directory: the data root of `render.py` (chosen, in this
  order, by its `--arms DIR` flag, this variable, or the default `../bench/arms` relative to
  this repository, the layout of the research checkout where `compass-git/` sits inside
  `code/`), and the directory the per-figure READMEs' commands pass to the create scripts as
  `$COMPASS_ARMS_DIR/<arm>`. The driver stops before drawing when a required arm directory is
  missing or holds no record, and names the missing paths.
- `COMPASS_DATA_DIR`, the `datasets/` directory: where `bench/realdata.py` reads the seven
  textbook datasets for the run and score steps of the real-data arms (default: the research
  layout, `../tutorials/demos/data` relative to this repository).
- `COMPASS_CACHE_DIR`, the MCMC trace cache the referee, the labels and the arbiter and control
  fits share (default `figures/.test_cache`, ignored by git). It is not part of the archive; the
  fits regenerate it.

A re-run writes under `--out-root/<arm>/` in the side-file layout (records, `scores/`,
`readers/`, `drift/`, `drift/scores/`); the committed arms keep the labels and reader verdicts
inline in each record, and the create scripts read both forms the same way. The committed
records keep the older arm names (`e5_anthropic`, `e1_anthropic`, ...); the protocol tables
`figure2/figure2_protocols.py` and `figure3/figure3_protocols.py` record, for every arm name of the
run scripts, the committed directory it corresponds to, and the `--help` of every run and score
script lists the pair.

## Provenance

```bash
python tools/sync_figures.py --research /path/to/code --check
python tools/check_provenance.py --research /path/to/code --manifest figures/provenance.json
```

`tools/sync_figures.py` copies the 43 files of the research `figure2/`, `figure3/`, `figure4/`
and `bench/` directories into `figures/` (39 scripts and modules and the four `__init__.py`
files); `--check` writes nothing and exits 1 if any file differs from the result. The copies are
verbatim modulo the import rewrite: the research scripts' sibling imports and their imports of
`bench`, `figure2`, `figure3` and `figure4` stay as they are, because `figures/` mirrors the
research layout, and only the imports of the pipeline itself (`llm`, `graph`, `judge`,
`agents.*`, `state`) are rewritten to the installed `compass` package (the manifest's
`import_map`). The exceptions are a few line windows that drop an eighth research dataset this
repository does not distribute (its entries in `bench/realdata.py`, `REAL_ORDER` and
`DATASET_NAME` of `figure3/figure3_common.py`, the arm on it in the protocol tables and the
usage lines that named it) and the two roots that honour `COMPASS_CACHE_DIR` and
`COMPASS_DATA_DIR`; each window is recorded in `provenance.json` with its line number, the
copied text and the sha256 of the research lines. `tools/check_provenance.py` compares every
top-level definition and statement of each copy with the research file, enforces the import
block, fails on any difference the manifest does not allow and on a listed research source that
changed after the pinned commit, and exits 0 only when nothing else differs; its sweep also
fails on a Python file under `figures/` that is neither in `files` nor in `not_covered`
(`render.py` and `compare_pdf.py` are listed there as new code without a research source), so
the directory cannot grow unverified code unnoticed. `provenance.json` pins `research_commit`
`8e04643`, the research HEAD the copies were generated from, with `research_sources_dirty` empty
(the sources were committed and clean) and the sha256 of every source and copy.

## Tests

```bash
python -m pytest tests/figures -q
```

`tests/figures/conftest.py` puts `figures/` on `sys.path`, so the tests import the figure code
the way `render.py` does (`from figure2 import figure2_common`, `import render`). Most tests
build their own records under a temporary directory with the fixtures of
`tests/figures/figure_records.py`, so the suite passes in a clean clone once the `figures` extra
is installed; `test_render.py` renders every file from such synthetic records and checks the
byte-identical render across timezones, and `test_compare_pdf.py` exercises the two hashes and
the verdicts. Two groups of tests look outside the repository and skip when what they look for
is absent: the tests of `test_figure2.py`, `test_figure3.py` and `test_figure4.py` that read the
committed arms at `render.arms_dir()` (`COMPASS_ARMS_DIR` when it is set, else `../bench/arms`),
and `test_provenance.py`, which runs the two provenance commands above when the research
checkout sits next to this repository. The overlay check of the Figure 2 sheet in
`test_figure2.py` needs the optional `pymupdf` and skips without it. `ruff check figures tools`
lints the directory (configuration in `pyproject.toml`), and the import-hygiene check

```bash
python -I -c "import sys; sys.path.insert(0, 'figures'); import render, compare_pdf; \
  print(sorted(m for m in sys.modules if m.split('.')[0] in {'langchain','jax','numpyro','compass','bench'}))"
```

must print an empty list: nothing on the drawing path imports the agent pipeline, the benchmark
modules, LangChain, JAX or NumPyro.
