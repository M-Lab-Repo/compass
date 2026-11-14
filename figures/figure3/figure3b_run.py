"""Figure 3b, Why the deployed model drifts: run the trajectories behind the panel.

    python figure3/figure3b_run.py --out-root runs/2026-11 [--arm ARM] [--only PREFIX] [--dry-run]

Options
  --out-root DIR   where the records go: DIR/<arm>/drift/<type>_seed<s>_<rule>.json (required; the
                   committed bench/arms/ is refused)
  --arm ARM        run only this arm; default: every arm of the panel (here there is one)
  --only PREFIX    run only the trajectories whose file name starts with PREFIX
  --dry-run        print the trajectories that would run and call nothing
  --help           this text plus the panel's arms, one line each

Values
  ARM      drift_correct-baseline_sonnet-4.6, the same arm as panel a (figure3a_run.py spells out the name).
  PREFIX   <type>_seed0_<rule>, with <type> one of overdispersion, zero_inflation, heavy_tails,
           binomial_overdispersion, positive_skew, heteroscedasticity, missing_nonlinearity,
           missing_hierarchy and <rule> one of execution, self, control, referee.
  DIR      any directory outside bench/arms, e.g. runs/2026-11.

Arms: the synthetic drift arm, shared with panel a; a trajectory that exists under --out-root is
skipped, so running this after figure3a_run.py repeats nothing. Panel b reads the raw records
alone (which rounds changed the model, how many latent sites it has), so it has no score step:
figure3b_create.py follows this script directly.
"""

from __future__ import annotations

import argparse

try:
    import figure3_protocols as protocols
    import figure3_run as engine
except ImportError:
    from figure3 import figure3_protocols as protocols
    from figure3 import figure3_run as engine

PANEL = "b"
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
