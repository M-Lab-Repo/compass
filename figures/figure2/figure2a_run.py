"""Figure 2a, What the agent proposed (census): run the experiment arms behind the panel.

    python figure2/figure2a_run.py --out-root runs/2026-11 [--arm ARM ...] [--only PREFIX] [--dry-run]

Options
  --out-root DIR   where the records go: DIR/<arm>/<type>_seed<s>_r<k>.json (required; the
                   committed bench/arms/ is refused, so a re-run never overwrites the paper's data)
  --arm ARM        run only this arm; repeat the flag for several; default: every arm of the panel
  --only PREFIX    run only the instances whose file name starts with PREFIX
  --dry-run        print the runs that would happen and call nothing
  --help           this text plus the panel's arms, one line each

Values
  ARM      one of the names in ARMS below, of the form <baseline>_<proposer>:
             correct-baseline        the run starts from the correct model; nothing is wrong
             misspecified-baseline   the run starts from a misspecified model; a fix exists
             real-data               seven textbook datasets, three hidden splits, double MCMC budget
           and <proposer> one of sonnet-4.6, haiku-4.5, gpt-5, gemini-3.8-flash (the model that
           revises the model). The paper's committed records use older short names for the same
           arms (e5 = correct, e1 = misspecified; see figure2_protocols.LEGACY_DIR).
  PREFIX   the start of a record file name. Synthetic arms: <type>_seed<s>_r<k> with <type> one of
           overdispersion, zero_inflation, heavy_tails, binomial_overdispersion, positive_skew,
           heteroscedasticity, missing_nonlinearity, missing_hierarchy; <s> 0, 1 or 2; <k> 1, 2 or 3.
           Real-data arms: <dataset>_seed0[_s<split>]_r1 with <dataset> one of roaches, ucbadmit,
           baseball, seeds, radon, kidiq, golf and <split> 1 or 2 (split 0 has no suffix).
           Examples: --only heavy_tails (one type, 9 runs), --only heavy_tails_seed0 (3 runs),
           --only heavy_tails_seed0_r1 (1 run), --only roaches (3 runs, one per split).
  DIR      any directory outside bench/arms, e.g. runs/2026-11; records land in DIR/<arm>/.

Arms: all ten census arms, shared with panels b and d.
A synthetic arm is 72 runs (8 types x 3 seeds x 3 repeats), a real-data arm 21 (7 datasets x 3
splits). A run whose record already exists under --out-root is skipped, so running this after
another panel's script repeats nothing. Needs the proposer's API key.
"""

from __future__ import annotations

import argparse

try:
    import figure2_protocols as protocols
    import figure2_run as engine
except ImportError:
    from figure2 import figure2_protocols as protocols
    from figure2 import figure2_run as engine

PANEL = "a"
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
    ap.add_argument("--only", metavar="PREFIX")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    return {arm: engine.run_arm(arm, args.out_root, only=args.only, dry_run=args.dry_run)
            for arm in (args.arm or ARMS)}


if __name__ == "__main__":
    main()
