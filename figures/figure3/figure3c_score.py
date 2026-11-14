"""Figure 3c, Outcome of the accepted revision (real datasets): score the arms behind the panel.

    python figure3/figure3c_score.py --out-root runs/2026-11 [--arm ARM ...] [--readers LIST] [--no-readers] [--force]

Options
  --out-root DIR   the root the run script wrote to; side files go to DIR/<arm>/scores/ and DIR/<arm>/readers/
  --arm ARM        score only this arm; default: the arm of the panel
  --readers LIST   the independent readers; default: the arm's own readers from the protocol table
  --no-readers     labels only (MCMC on the stored programs; needs no key)
  --force          recompute labels and verdicts that are already stored (default: skip them, so a pass resumes)
  --help           this text plus the panel's arms, one line each

Values
  ARM      real-data_sonnet-4.6 (figure3c_run.py says what it is).
  LIST     provider:model pairs separated by commas, e.g. openai:gpt-5. Without --readers the arm gets its
           GPT-5 reader (the matrix's "Independent model" column).
  DIR      the --out-root that the run script wrote to.

Labels are the hidden-fold Δ of every candidate against the textbook model (real data has no fresh
draw); the matrix colours cells by them. Scoring is figure2/figure2_score.py, MCMC only; readers
need that provider's key and a reader whose key is missing is skipped with a note.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import figure3_protocols as protocols
except ImportError:
    from figure3 import figure3_protocols as protocols
from figure2 import figure2_protocols as census  # noqa: E402
from figure2 import figure2_score as engine  # noqa: E402

PANEL = "c"
ARMS = ("real-data_sonnet-4.6",)
assert ARMS == protocols.MATRIX_ARMS


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, epilog=census.arms_help(ARMS),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True, metavar="DIR")
    ap.add_argument("--arm", action="append", choices=ARMS, metavar="ARM")
    ap.add_argument("--readers", metavar="LIST")
    ap.add_argument("--no-readers", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    out = {}
    for arm in args.arm or ARMS:
        readers = () if args.no_readers else (engine.parse_readers(args.readers) if args.readers
                                              else census.readers_of(arm))
        out[arm] = engine.score_arm(Path(args.out_root) / arm, readers=readers, force=args.force)
    return out


if __name__ == "__main__":
    main()
