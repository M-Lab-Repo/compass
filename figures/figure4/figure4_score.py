"""Figure 4, What a verdict costs: score the arms behind the panel (hidden-fold labels; readers only on request).

    python figure4/figure4_score.py --out-root runs/2026-11 [--arm ARM ...] [--readers LIST] [--force]

Options
  --out-root DIR   the root the run script wrote to; side files go to DIR/<arm>/scores/ (and readers/)
  --arm ARM        score only this arm; repeat the flag for several; default: every arm of the panel
  --readers LIST   independent readers to add, provider:model pairs separated by commas (default: none;
                   Figure 4 draws no reader verdict, Figure 2's score scripts add the arms' readers)
  --force          recompute what is already stored (default: skip it, so a pass resumes)
  --help           this text plus the panel's arms, one line each

Values
  ARM      one of the names in ARMS below (figure4_run.py says what each means).
  LIST     e.g. anthropic:claude-opus-4-5-20251101,openai:gpt-5 (needs those providers' keys).
  DIR      the --out-root that the run script wrote to.

The panel itself needs no label: it reads the cost block of the raw records. This step exists
because the panel's footnote reports the MCMC seconds of hidden-fold scoring, which the score step
measures and stores in the scores/ side files, and because the four synthetic arms are shared with
Figure 2. Scoring is figure2/figure2_score.py, MCMC only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from figure2 import figure2_protocols as protocols  # noqa: E402
from figure2 import figure2_score as engine  # noqa: E402

PANEL = "4"
ARMS = ("misspecified-baseline_sonnet-4.6", "misspecified-baseline_haiku-4.5", "misspecified-baseline_gpt-5",
        "misspecified-baseline_gemini-3.8-flash", "real-data-default-budget_sonnet-4.6")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, epilog=protocols.arms_help(ARMS),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True, metavar="DIR")
    ap.add_argument("--arm", action="append", choices=ARMS, metavar="ARM")
    ap.add_argument("--readers", default="", metavar="LIST")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    readers = engine.parse_readers(args.readers)
    return {arm: engine.score_arm(Path(args.out_root) / arm, readers=readers, force=args.force)
            for arm in (args.arm or ARMS)}


if __name__ == "__main__":
    main()
