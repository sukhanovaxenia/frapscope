"""Command-line entry point — the single controller.

Reads a condition set from a config file, runs the shared pipeline, and writes
every figure and CSV under the output directory. To add a protein or change a
data source, edit the config file; no analysis code changes.

Formerly ``run_frap.py`` at the repository root, invoked as
``python3 run_frap.py``. It is now reached through the ``frapscope`` console
script declared in pyproject.toml, so it runs from any directory rather than
only from the one containing it. The shebang and ``__main__`` guard are gone:
neither is used by a console entry point, and keeping them invited the older
invocation, which imports a second copy of the package under a different name.

    frapscope --config examples/config_ribosomal.py
    frapscope --config my_study.py --out results/
    frapscope --config my_study.py --plot-mode gapped
    frapscope --config my_study.py --plot-mode full   # legacy figures
"""

import argparse
from pathlib import Path

from frapscope.config import ConfigError, load_conditions
from frapscope import run
from frapscope.viz import PLOT_MODES

# Relative by default: an absolute path here pointed into one machine's
# home directory and silently wrote nothing useful anywhere else.
DEFAULT_OUT = "frap_results"


def main():
    ap = argparse.ArgumentParser(description="FRAP analysis (extraction-agnostic).")
    ap.add_argument("--config", required=True,
                    help="Python file defining CONDITIONS "
                         "(see examples/config_ribosomal.py)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output base directory")
    ap.add_argument("--bleach-time", type=float, default=0.0,
                    help="nominal bleach time on the input axis; only used by "
                         "--plot-mode full (the bleach is otherwise detected)")
    ap.add_argument("--n-pre", type=int, default=3,
                    help="earliest frames averaged as the pre-bleach plateau")
    ap.add_argument("--n-plat", type=int, default=3,
                    help="fallback only: frames averaged as the plateau when the "
                         "observation window is degenerate (zero span, or no "
                         "frame inside the fractional window). The plateau is "
                         "otherwise the mean over the final 20 %% of the observation "
                         "window (core._plateau_level, plat_frac); changing this "
                         "will not affect a normally sampled condition")
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
    try:
        conditions = load_conditions(args.config)
    except ConfigError as exc:
        raise SystemExit(f"configuration error: {exc}")
    print(f"Conditions: {', '.join(c.display + ' [' + c.loader + ']' for c in conditions)}")
    print(f"Plot mode : {args.plot_mode}")
    print(f"Output: {args.out}")
    print("=" * 60)
    run(conditions, Path(args.out), args.bleach_time, args.n_pre, args.n_plat,
        args.plot_mode, not args.no_renormalise, args.min_coverage,
        harmonise_comparison=not args.no_harmonise,
        harmonise_dt=args.harmonise_dt,
        stats_control=args.stats_control,
        stats_exclude=tuple(args.stats_exclude))