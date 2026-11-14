"""Figure 3e (SI), Deployed model over successive rounds on real data: run the trajectories behind the panel.

    python figure3/figure3e_run.py --out-root runs/2026-11 [--arm ARM] [--only PREFIX] [--dry-run]

Options
  --out-root DIR   where the records go: DIR/<arm>/drift/<dataset>_seed0_<rule>.json (required; the
                   committed bench/arms/ is refused)
  --arm ARM        run only this arm; default: every arm of the panel (here there is one)
  --only PREFIX    run only the trajectories whose file name starts with PREFIX
  --dry-run        print the trajectories that would run and call nothing
  --help           this text plus the panel's arms, one line each

Values
  ARM      drift_real-data_sonnet-4.6: every trajectory starts from the textbook model of a real dataset
           (which is misspecified, so a fix exists), with live literature search and double MCMC budget;
           the paper's committed trajectories are bench/arms/e2_anthropic_real/drift.
  PREFIX   <dataset>_seed0_<rule>, with <dataset> one of roaches, ucbadmit, baseball, seeds, radon, kidiq,
           golf and <rule> one of execution, self, control, referee. Example: --only roaches (4 trajectories).
  DIR      any directory outside bench/arms, e.g. runs/2026-11.

Arms: the real-data drift arm. 7 datasets x 4 rules = 28 trajectories of 6 rounds, about 7 minutes
and $0.50 per round (about $85 and 20 hours in one shell). A complete trajectory is skipped, a
partial one resumed. Needs the proposer's API key and the Semantic Scholar key for the live search.
"""

from __future__ import annotations

import argparse

try:
    import figure3_protocols as protocols
    import figure3_run as engine
except ImportError:
    from figure3 import figure3_protocols as protocols
    from figure3 import figure3_run as engine

PANEL = "e"
ARMS = ("drift_real-data_sonnet-4.6",)


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
