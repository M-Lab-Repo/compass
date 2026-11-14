"""Figure 3e (SI), Deployed model over successive rounds on real data: score the trajectories behind the panel.

    python figure3/figure3e_score.py --out-root runs/2026-11 [--arm ARM] [--force]

Options
  --out-root DIR   the root the run script wrote to; side files go to DIR/<arm>/drift/scores/
  --arm ARM        score only this arm; default: every arm of the panel (here there is one)
  --force          recompute scores that are already stored (default: skip them, so a pass resumes)
  --help           this text plus the panel's arms, one line each

Values
  ARM      drift_real-data_sonnet-4.6 (figure3e_run.py says what the name means).
  DIR      the --out-root that the run script wrote to.

Real data has no fresh draw, so every round's deployed model is scored against the starting model on
the hidden fold only; the round's candidate is scored against the model it was proposed for. MCMC
only, no API key; the double budget of the arm is read from each trajectory's flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import figure3_protocols as protocols
    import figure3_score as engine
except ImportError:
    from figure3 import figure3_protocols as protocols
    from figure3 import figure3_score as engine

PANEL = "e"
ARMS = ("drift_real-data_sonnet-4.6",)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, epilog=protocols.arms_help(ARMS),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True, metavar="DIR")
    ap.add_argument("--arm", action="append", choices=ARMS, metavar="ARM")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    return {arm: engine.score_arm(Path(args.out_root) / arm / "drift", force=args.force)
            for arm in (args.arm or ARMS)}


if __name__ == "__main__":
    main()
