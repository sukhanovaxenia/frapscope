#!/usr/bin/env python3
"""
run_frap.py — the single controller.

Reads frap.config.CONDITIONS, runs the shared pipeline, writes all figures and
CSVs under OUTPUT_BASE. To add a protein or change a data source, edit
frap/config.py only — no analysis code changes.

    python3 run_frap.py
    python3 run_frap.py --out /path/to/results
    python3 run_frap.py --plot-mode gapped
    python3 run_frap.py --plot-mode full        # legacy figures
"""

import argparse
from pathlib import Path

from frap.config import CONDITIONS
from frap import run
from frap.viz import PLOT_MODES

DEFAULT_OUT = "/mnt/user-data/outputs/frap_results"


def main():
    ap = argparse.ArgumentParser(description="FRAP analysis (extraction-agnostic).")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output base directory")
    ap.add_argument("--bleach-time", type=float, default=0.0,
                    help="nominal bleach time on the input axis; only used by "
                         "--plot-mode full (the bleach is otherwise detected)")
    ap.add_argument("--n-pre", type=int, default=3,
                    help="earliest frames averaged as the pre-bleach plateau")
    ap.add_argument("--n-plat", type=int, default=3, help="last frames averaged as plateau")
    ap.add_argument("--min-coverage", type=float, default=0.75,
                    help="drop grid points where fewer than this fraction of "
                         "replicates contribute, and set the quantile used to "
                         "define each condition's post-bleach window (default 0.75)")
    ap.add_argument("--stats-control", default=None,
                    help="condition to test every other condition against "
                         "(e.g. eS28). Restricts the Holm family to k-1 planned "
                         "contrasts instead of all pairs")
    ap.add_argument("--stats-exclude", nargs="*", default=[],
                    help="conditions to leave out of statistical testing "
                         "entirely, e.g. a control pooled across non-identical "
                         "acquisition settings")
    ap.add_argument("--no-harmonise", action="store_true",
                    help="build the comparison on native windows. The overlay "
                         "is then not cross-comparable; use only for legacy figures")
    ap.add_argument("--harmonise-dt", type=float, default=None,
                    help="also resample onto a shared frame interval, in "
                         "seconds. Set to the COARSEST interval in the panel "
                         "and only when half-times are to be compared")
    ap.add_argument("--no-renormalise", action="store_true",
                    help="do not rescale each replicate to its own pre-bleach "
                         "plateau before plotting")
    ap.add_argument("--plot-mode", choices=PLOT_MODES, default="postbleach",
                    help="postbleach: recovery only, re-zeroed per replicate (default); "
                         "gapped: pre-bleach markers + blank bleach interval; "
                         "full: legacy continuous line")
    args = ap.parse_args()

    print("=" * 60)
    print("FRAP analysis — extraction-agnostic library")
    print(f"Conditions: {', '.join(c.display + ' [' + c.loader + ']' for c in CONDITIONS)}")
    print(f"Plot mode : {args.plot_mode}")
    print(f"Output: {args.out}")
    print("=" * 60)
    run(CONDITIONS, Path(args.out), args.bleach_time, args.n_pre, args.n_plat,
        args.plot_mode, not args.no_renormalise, args.min_coverage,
        harmonise_comparison=not args.no_harmonise,
        harmonise_dt=args.harmonise_dt,
        stats_control=args.stats_control,
        stats_exclude=tuple(args.stats_exclude))


if __name__ == "__main__":
    main()