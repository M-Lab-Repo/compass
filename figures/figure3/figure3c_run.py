"""Figure 3c, Outcome of the accepted revision (real datasets): run the arms behind the panel.

    python figure3/figure3c_run.py --out-root runs/2026-11 [--arm ARM ...] [--only PREFIX] [--dry-run]

Options
  --out-root DIR   where the records go: DIR/<arm>/<dataset>_seed0[_s<split>]_r1.json (required; the
                   committed bench/arms/ is refused)
  --arm ARM        run only this arm; default: the arm of the panel
  --only PREFIX    run only the runs whose file name starts with PREFIX
  --dry-run        print the runs that would happen and call nothing
  --help           this text plus the panel's arms, one line each

Values
  ARM      the one name in ARMS below:
             real-data_sonnet-4.6        seven textbook datasets, three hidden splits, double MCMC budget
                                         (the same arm Figure 2 draws; committed as e1_anthropic_realdata_x2)
  PREFIX   <dataset>_seed0[_s<split>]_r1 with <dataset> one of roaches, ucbadmit, baseball, seeds, radon,
           kidiq, golf, and <split> 1 or 2 (split 0 has no suffix).
           Examples: --only roaches (3 runs), --only golf_seed0_s1 (1 run).
  DIR      any directory outside bench/arms, e.g. runs/2026-11.

This is a census arm: each run is the Figure 2 protocol (the agent revises the textbook model,
the live rules judge every candidate, template controls are fitted and judged), run by
figure2/figure2_run.py. 21 runs, about 15 minutes and $0.80 each. A run whose record exists is skipped.
Needs the proposer's API key and, for the live literature search, the Semantic Scholar key.
"""

from __future__ import annotations

import argparse

try:
    import figure3_protocols as protocols
except ImportError:
    from figure3 import figure3_protocols as protocols
from figure2 import figure2_protocols as census  # noqa: E402
from figure2 import figure2_run as engine  # noqa: E402

PANEL = "c"
ARMS = ("real-data_sonnet-4.6",)
assert ARMS == protocols.MATRIX_ARMS


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, epilog=census.arms_help(ARMS),
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
