"""Figure 3a, Deployed model over successive rounds (drift): run the trajectories behind the panel.

    python figure3/figure3a_run.py --out-root runs/2026-11 [--arm ARM] [--only PREFIX] [--dry-run]

Options
  --out-root DIR   where the records go: DIR/<arm>/drift/<type>_seed<s>_<rule>.json (required; the
                   committed bench/arms/ is refused, so a re-run never overwrites the paper's data)
  --arm ARM        run only this arm; default: every arm of the panel (here there is one)
  --only PREFIX    run only the trajectories whose file name starts with PREFIX
  --dry-run        print the trajectories that would run and call nothing
  --help           this text plus the panel's arms, one line each

Values
  ARM      one of the names in ARMS below, of the form drift_<baseline>_<proposer>:
             drift_correct-baseline   every trajectory starts from the correct (generating) model, so any
                                      accepted revision is unnecessary by construction
           and <proposer> the model that revises (sonnet-4.6). The paper's committed trajectories use
           the older short name e2_anthropic (figure3_protocols.LEGACY_DIR).
  PREFIX   the start of a trajectory file name, <type>_seed0_<rule>, with <type> one of overdispersion,
           zero_inflation, heavy_tails, binomial_overdispersion, positive_skew, heteroscedasticity,
           missing_nonlinearity, missing_hierarchy and <rule> one of execution, self, control, referee.
           Examples: --only heavy_tails (the type's 4 trajectories), --only heavy_tails_seed0_referee (one).
  DIR      any directory outside bench/arms, e.g. runs/2026-11.

Arms: the synthetic drift arm, shared with panel b. 8 problems x 4 rules = 32 trajectories of 6
rounds, one proposer step each (about 5 minutes and $0.35 at Sonnet prices per round, so about
$70 and 16 hours in one shell; run several --only types in parallel shells). A trajectory file
with all its rounds is skipped and a partial one is resumed, so an interrupted pass continues.
Needs the proposer's API key.
"""

from __future__ import annotations

import argparse

try:
    import figure3_protocols as protocols
    import figure3_run as engine
except ImportError:
    from figure3 import figure3_protocols as protocols
    from figure3 import figure3_run as engine

PANEL = "a"
ARMS = ("drift_correct-baseline_sonnet-4.6",)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, epilog=protocols.arms_help(ARMS),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True, metavar="DIR")
    ap.add_argument("--arm", action="append", choices=ARMS, metavar="ARM")
    ap.add_argument("--only", metavar="PREFIX")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    return {arm: engine.run_arm(arm, args.out_root, only=args.only, dry_run=args.dry_run)
            for arm in (args.arm or ARMS)}


if __name__ == "__main__":
    main()
