"""Figure 2b, Who accepts them: score the arms behind the panel (labels, then independent readers).

    python figure2/figure2b_score.py --out-root runs/2026-11 [--arm ARM ...] [--readers LIST] [--no-readers] [--force]

Options
  --out-root DIR   the root the run script wrote to; side files go to DIR/<arm>/scores/ and DIR/<arm>/readers/
  --arm ARM        score only this arm; repeat the flag for several; default: every arm of the panel
  --readers LIST   the independent readers; default: the arm's own readers from the protocol table
  --no-readers     labels only (MCMC on the stored programs; needs no key)
  --force          recompute labels and verdicts that are already stored (default: skip them, so a pass resumes)
  --help           this text plus the panel's arms, one line each

Values
  ARM      one of the names in ARMS below, <baseline>_<proposer> (the run script of this panel
           says what each part means).
  LIST     provider:model pairs separated by commas. Providers: anthropic, openai, google. The models
           used in the paper: claude-sonnet-4-6, claude-haiku-4-5-20251001, claude-opus-4-5-20251101,
           gpt-5, gemini-3.8-flash. Without --readers each arm gets its own readers, listed below.
  DIR      the --out-root that the run script wrote to.

Arms: all ten census arms, shared with panels a and d.
Labels need no key; readers need that provider's key, and a reader whose key is missing is
skipped with a note.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import figure2_protocols as protocols
    import figure2_score as engine
except ImportError:
    from figure2 import figure2_protocols as protocols
    from figure2 import figure2_score as engine

PANEL = "b"
ARMS = (
    "correct-baseline_sonnet-4.6", "misspecified-baseline_sonnet-4.6", "real-data_sonnet-4.6",
    "correct-baseline_haiku-4.5", "misspecified-baseline_haiku-4.5",
    "correct-baseline_gpt-5", "misspecified-baseline_gpt-5", "real-data_gpt-5",
    "correct-baseline_gemini-3.8-flash", "misspecified-baseline_gemini-3.8-flash",
)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, epilog=protocols.arms_help(ARMS),
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
                                              else protocols.readers_of(arm))
        out[arm] = engine.score_arm(Path(args.out_root) / arm, readers=readers, force=args.force)
    return out


if __name__ == "__main__":
    main()
