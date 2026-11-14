"""Pytest configuration of the figure tests (``tests/figures/``).

Puts the ``figures/`` directory on ``sys.path`` so the tests import the figure code the way
``figures/render.py`` does (``from figure2 import figure2_common``, ``import render``). Nothing
here reads the environment or a ``.env`` file: the figure code needs no API key, and most tests
build their own records under ``tmp_path`` (``figure_records.py`` in this directory).

Two groups of tests look outside this repository, and skip when what they look for is absent,
so the suite passes in a clean clone:

- the tests of ``test_figure2.py``, ``test_figure3.py`` and ``test_figure4.py`` that render the
  committed benchmark arms read them at ``render.arms_dir()``: ``COMPASS_ARMS_DIR`` when it is
  set, else ``<repo>/../bench/arms`` (the research layout, where ``compass-git/`` sits inside
  ``code/``). Point ``COMPASS_ARMS_DIR`` at the ``arms/`` directory of the data archive to run
  them elsewhere.
- ``test_provenance.py`` checks the committed copies against the research tree when it sits
  next to this repository.

``test_render.py`` clears ``COMPASS_ARMS_DIR`` per test, so a value exported in the shell never
reaches the driver under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"
if str(FIGURES_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURES_DIR))
