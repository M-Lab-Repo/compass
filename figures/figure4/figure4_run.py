"""Figure 4, What a verdict costs: run the arms behind the panel.

    python figure4/figure4_run.py --out-root runs/2026-11 [--arm ARM ...] [--only PREFIX] [--dry-run]

Options
  --out-root DIR   where the records go: DIR/<arm>/<type>_seed<s>_r<k>.json (required; the committed
                   bench/arms/ is refused, so a re-run never overwrites the paper's data)
  --arm ARM        run only this arm; repeat the flag for several; default: every arm of the panel
  --only PREFIX    run only the runs whose file name starts with PREFIX
  --dry-run        print the runs that would happen and call nothing
  --help           this text plus the panel's arms, one line each

Values
  ARM      one of the names in ARMS below:
             misspecified-baseline_<proposer>      the run starts from a misspecified model; a fix exists
                                                   (Figure 2's arms; <proposer> one of sonnet-4.6, haiku-4.5,
                                                   gpt-5, gemini-3.8-flash)
             real-data-default-budget_sonnet-4.6   seven textbook datasets, one hidden split, the default MCMC
                                                   budget (committed as e1_anthropic_realdata)
  PREFIX   synthetic arms: <type>_seed<s>_r<k> with <type> one of overdispersion, zero_inflation, heavy_tails,
           binomial_overdispersion, positive_skew, heteroscedasticity, missing_nonlinearity, missing_hierarchy,
           <s> 0, 1 or 2 and <k> 1, 2 or 3; the real-data arm: <dataset>_seed0_r1 with <dataset> one of roaches,
           ucbadmit, baseball, seeds, radon, kidiq, golf. Examples: --only heavy_tails (9 runs), --only roaches (1).
  DIR      any directory outside bench/arms, e.g. runs/2026-11.

Every record written by figure2/figure2_run.py carries a ``cost`` block (tokens and seconds per
stage and per rule, MCMC seconds per stage), which is all this panel reads. The four synthetic arms
are Figure 2's, so a record that exists under --out-root is skipped. 4 x 72 + 7 = 295 runs, about
$0.10 to $0.70 each by proposer (figure4/README.md). Needs each proposer's API key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from figure2 import figure2_protocols as protocols  # noqa: E402
from figure2 import figure2_run as engine  # noqa: E402

PANEL = "4"
ARMS = ("misspecified-baseline_sonnet-4.6", "misspecified-baseline_haiku-4.5", "misspecified-baseline_gpt-5",
        "misspecified-baseline_gemini-3.8-flash", "real-data-default-budget_sonnet-4.6")


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
