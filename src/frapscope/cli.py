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
import csv
from pathlib import Path

from frapscope import run
from frapscope.config import ConfigError, load_conditions
from frapscope.viz import PLOT_MODES

# Relative by default: an absolute path here pointed into one machine's
# home directory and silently wrote nothing useful anywhere else.
DEFAULT_OUT = "frap_results"


def _number(text):
    try:
        return float(text)
    except ValueError:
        return None


def _parse_covariate(tokens):
    """NAME=VALUE pairs, a two-column CSV, or any mixture of the two.

    A CSV keeps the covariate with the data it came from and under version
    control; the inline form is for one-off checks. A header row is detected by
    its second field not parsing as a number, so `condition,regions` is skipped
    and `eS2,5` is not.
    """
    if not tokens:
        return None
    cov = {}
    for token in tokens:
        if "=" in token:
            name, value = token.split("=", 1)
            number = _number(value)
            if number is None:
                raise SystemExit(f"--covariate: {value!r} is not a number")
            cov[name.strip()] = number
            continue
        path = Path(token)
        if not path.exists():
            raise SystemExit(f"--covariate: no such file {token!r} (and no '=' "
                             "in it, so it is not a NAME=VALUE pair)")
        with open(path, newline="") as fh:
            rows = [r for r in csv.reader(fh)
                    if len(r) >= 2 and not r[0].lstrip().startswith("#")]
        if rows and _number(rows[0][1]) is None:
            rows = rows[1:]
        for row in rows:
            number = _number(row[1])
            if number is None:
                raise SystemExit(f"--covariate: {path}: {row[1]!r} is not a number")
            cov[row[0].strip()] = number
    if not cov:
        raise SystemExit("--covariate: nothing parsed")
    return cov


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
    ap.add_argument("--stats-control", nargs="*", default=None,
                    help="condition(s) to test every other condition against "
                         "(e.g. eS28, or 'GFP eS28'). Restricts the Holm family "
                         "to k-1 planned contrasts instead of all pairs. Naming "
                         "more than one runs each as its own family, written to "
                         "mobile_fraction_stats_vs_<control>.csv: a "
                         "free-fluorophore control and a within-class reference "
                         "answer different questions and must not share a "
                         "correction")
    ap.add_argument("--covariate", nargs="*", default=None, metavar="NAME=VALUE|CSV",
                    help="per-condition numbers measured outside this pipeline, "
                         "correlated with --covariate-metric across the panel: "
                         "either NAME=VALUE pairs or a two-column CSV "
                         "(condition,value). Spearman with an exact permutation "
                         "p-value, the smallest p the design admits, and a "
                         "leave-one-out check on every condition")
    ap.add_argument("--covariate-label", default="covariate",
                    help="what the covariate is, for axis labels and the report")
    ap.add_argument("--covariate-metric", action="append", default=None,
                    choices=["mobile_fraction", "bleach_depth"],
                    help="metric to correlate (repeatable; default mobile_fraction)")
    ap.add_argument("--covariate-estimator", default="mean",
                    choices=["mean", "median"],
                    help="how a condition's replicates are reduced to the one "
                         "number that enters the correlation")
    ap.add_argument("--covariate-reference", default=None,
                    help="condition drawn on the covariate figure as a "
                         "horizontal band and left out of the correlation, for "
                         "a control that has no covariate value of its own")
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
        raise SystemExit(f"configuration error: {exc}") from None
    print(f"Conditions: {', '.join(c.display + ' [' + c.loader + ']' for c in conditions)}")
    print(f"Plot mode : {args.plot_mode}")
    print(f"Output: {args.out}")
    print("=" * 60)
    run(conditions, Path(args.out), args.bleach_time, args.n_pre, args.n_plat,
        args.plot_mode, not args.no_renormalise, args.min_coverage,
        harmonise_comparison=not args.no_harmonise,
        harmonise_dt=args.harmonise_dt,
        stats_control=args.stats_control,
        stats_exclude=tuple(args.stats_exclude),
        covariate=_parse_covariate(args.covariate),
        covariate_label=args.covariate_label,
        covariate_metrics=tuple(args.covariate_metric or ["mobile_fraction"]),
        covariate_reference=args.covariate_reference,
        covariate_estimator=args.covariate_estimator)
