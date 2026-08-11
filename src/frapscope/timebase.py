"""
frap.timebase — common post-bleach time base across conditions.

Motivation
----------
Per-condition figures are internally consistent: every replicate is aligned to
its own bleach frame, so ``t = 0`` means the same thing within a condition. A
*cross-condition* overlay is a different claim. It asserts that two curves at
the same abscissa are comparable observations, which is only true if the
conditions share a post-bleach window and a sampling density. In this dataset
they do not: the acquisitions were configured per protein, so the post-bleach
window spans 21.1-53.5 s and the frame interval spans 1.15-6.00 s.

Two derived quantities react differently to that heterogeneity:

* **Mobile fraction** is a ratio of levels. Provided each condition has run for
  several half-times it is close to window-invariant, and the invariance can be
  demonstrated rather than assumed (:func:`invariance_report`).
* **Half-time** is a rate. It is biased low when the recovery is sampled
  coarsely relative to its own t-half, because the unresolved fast phase is
  absorbed into the first retained point. A 5-fold spread in frame interval is
  therefore not comparable, and :func:`sampling_bias_report` quantifies it.

This module operates on the ``all_data`` contract (list of per-replicate dicts
with ``time``, ``intensity``, ``intensity_raw``, ``provenance``), *before*
``plot_frap_summary`` and ``compare_conditions``, so no figure code changes.

Usage
-----
    from frap import build_all_data
    from frap.config import CONDITIONS
    from frap.timebase import (acquisition_report, harmonise,
                               invariance_report, sampling_bias_report)

    datasets = {c.display: build_all_data(c) for c in CONDITIONS}

    print(acquisition_report(datasets))        # what was actually acquired
    print(invariance_report(datasets))         # does truncation change the ranking?
    print(sampling_bias_report(datasets))      # is t-half comparable at all?

    shared, W = harmonise(datasets)            # common window, native sampling
    # -> feed `shared` to plot_frap_summary / compare_conditions as usual
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                    # optional: use the package's own fitter
    from .core import fit_frap_curve, align_to_bleach          # type: ignore
except Exception:                                              # pragma: no cover
    fit_frap_curve = None                                      # type: ignore
    align_to_bleach = None                                     # type: ignore


__all__ = [
    "acquisition_report", "common_window", "condition_window",
    "truncate", "resample_to_grid",
    "harmonise", "invariance_report", "sampling_bias_report",
]

AllData = List[dict]
Datasets = Dict[str, AllData]


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _aligned(all_data: AllData, crop: bool = True) -> AllData:
    """Replicates on a post-bleach axis with t = 0 at the bleach frame.

    Delegates to ``core.align_to_bleach`` when the package is importable so the
    bleach-detection rule stays in one place; otherwise falls back to the same
    rule used there (the trace minimum), which keeps this module usable
    standalone for auditing exported CSVs.
    """
    if align_to_bleach is not None:
        return align_to_bleach(all_data, crop=crop)
    out = []
    for d in all_data:
        t = np.asarray(d["time"], float)
        y = np.asarray(d["intensity"], float)
        k = int(np.nanargmin(y))
        sl = slice(k, None) if crop else slice(None)
        e = copy.deepcopy(d)
        e["time"] = t[sl] - t[k]
        e["intensity"] = y[sl]
        if "intensity_raw" in d:
            e["intensity_raw"] = np.asarray(d["intensity_raw"], float)[sl]
        out.append(e)
    return out


def _point_metrics(t: np.ndarray, y: np.ndarray,
                   n_plat: int = 3, f_pre: float = 1.0) -> dict:
    """Point-based mobile fraction and plateau, no model assumed.

    ``f_pre`` is the normalised pre-bleach level (1.0 after double
    normalisation). Using the observed terminal points rather than a fitted
    asymptote is what makes the estimate window-honest: it reports what was
    seen, not what an exponential extrapolates to beyond the last frame.
    """
    y = np.asarray(y, float)
    f_min = float(y[0])
    f_plat = float(np.nanmean(y[-min(n_plat, len(y)):]))
    denom = f_pre - f_min
    mf = 100.0 * (f_plat - f_min) / denom if denom > 0 else np.nan
    return {"f_min": f_min, "f_plat": f_plat, "mobile_fraction": mf,
            "bleach_depth": 100.0 * (f_pre - f_min) / f_pre if f_pre else np.nan}


def _t_half(t: np.ndarray, y: np.ndarray) -> float:
    """Half-time from a single-exponential fit; NaN if it will not converge."""
    if fit_frap_curve is not None:
        try:
            r = fit_frap_curve(np.asarray(t, float), np.asarray(y, float))
            if r is None:
                return float("nan")
            if isinstance(r, dict):
                return float(r.get("t_half", np.nan))
            return float(r)
        except Exception:
            return float("nan")
    from scipy.optimize import curve_fit                      # local import
    t = np.asarray(t, float); y = np.asarray(y, float)
    if len(t) < 4:
        return float("nan")

    def model(x, i0, imax, tau):
        return imax - (imax - i0) * np.exp(-x / tau)

    try:
        p, _ = curve_fit(model, t, y, p0=[y[0], y[-1], max(t[-1] / 5, 0.5)],
                         bounds=([-0.5, -0.5, 0.05], [2.0, 2.0, 5 * t[-1] + 10]),
                         maxfev=40000)
        return float(p[2] * np.log(2))
    except Exception:
        return float("nan")


def _mean_curve(all_data: AllData, n_grid: int = 200,
                min_coverage: float = 0.75
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coverage-masked mean of the aligned replicates on a shared grid.

    The grid ends at the coverage quantile, not at the shortest replicate.
    Using the minimum makes a condition's own "native" curve hostage to its
    worst replicate: one EGFP acquisition covering 1.6 s truncated the whole
    condition's mean to 1.6 s, so its native and truncated mobile fractions
    came out identical and its rank moved between the two orderings. The
    invariance verdict then reported CHANGED for a reason that had nothing to
    do with truncation. This is the same quantile :func:`condition_window`
    uses, so the two agree by construction.

    Replicates ending before the grid does contribute NaN rather than a
    flat-extrapolated endpoint, and the mean is taken NaN-aware.
    """
    view = _aligned(all_data)
    t_end = condition_window(all_data, min_coverage)
    if t_end <= 0:
        t_end = max(float(d["time"][-1]) for d in view)
    grid = np.linspace(0.0, t_end, n_grid)
    rows = []
    for d in view:
        t = np.asarray(d["time"], float)
        y = np.interp(grid, t, np.asarray(d["intensity"], float))
        rows.append(np.where(grid > t[-1], np.nan, y))
    stack = np.vstack(rows)
    with np.errstate(invalid="ignore"):
        return grid, np.nanmean(stack, axis=0), stack


def _fmt(rows: Sequence[Sequence], header: Sequence[str]) -> str:
    cols = list(zip(*([header] + [[str(x) for x in r] for r in rows])))
    w = [max(len(str(x)) for x in c) for c in cols]
    line = "  ".join(str(h).ljust(wi) for h, wi in zip(header, w))
    out = [line, "-" * len(line)]
    for r in rows:
        out.append("  ".join(str(x).ljust(wi) for x, wi in zip(r, w)))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# 1. What was actually acquired
# --------------------------------------------------------------------------- #
def acquisition_report(datasets: Datasets) -> str:
    """Per-condition post-bleach window, frame interval and window/t-half.

    ``window / t-half`` is the number of half-times observed. Below roughly 3
    the terminal level is still climbing and any plateau is an extrapolation;
    above 5 the point-based mobile fraction is safe. Print this before writing
    any cross-condition sentence.
    """
    rows = []
    for name, ad in datasets.items():
        view = _aligned(ad)
        wins = [float(d["time"][-1]) for d in view]
        dts = np.concatenate([np.diff(np.asarray(d["time"], float)) for d in view])
        grid, mean, _ = _mean_curve(ad)
        th = _t_half(grid, mean)
        rows.append([name, len(view),
                     f"{np.min(wins):.1f}-{np.max(wins):.1f}",
                     f"{np.median(dts):.2f}",
                     f"{np.median([len(d['time']) for d in view]):.0f}",
                     f"{th:.2f}",
                     f"{np.median(wins) / th:.1f}" if th == th and th > 0 else "n/a"])
    return "ACQUISITION\n" + _fmt(
        rows, ["condition", "n", "window (s)", "dt (s)", "frames",
               "t-half (s)", "windows/t-half"])


# --------------------------------------------------------------------------- #
# 2. Harmonisation
# --------------------------------------------------------------------------- #
def condition_window(all_data: AllData, min_coverage: float = 0.75) -> float:
    """Post-bleach duration that at least ``min_coverage`` of replicates cover.

    Taking the *shortest* replicate would be correct only if every replicate
    were a valid observation. It is not robust: a single replicate whose
    detected bleach frame sits near the end of its trace contributes a
    near-zero post-bleach duration and, through :func:`common_window`, drives
    the shared window for the entire panel to zero. That is a real failure mode
    in this dataset (one EGFP replicate covers a single post-bleach frame), and
    a quantile is the right estimator: it keeps the window that the bulk of the
    replicates support and lets the pathological minority be dropped by
    coverage trimming downstream, where the drop is reported.
    """
    w = sorted(float(d["time"][-1]) for d in _aligned(all_data))
    if not w:
        return 0.0
    k = min(int(np.floor((1.0 - min_coverage) * len(w))), len(w) - 1)
    return float(w[k])


def common_window(datasets: Datasets, rule: str = "min",
                  min_coverage: float = 0.75) -> float:
    """Longest post-bleach interval every condition actually covers.

    ``rule="min"`` takes the shortest per-condition window (every plotted point
    is then backed by every condition). ``rule="median"`` is more permissive and
    should only be used together with an explicit coverage annotation.
    """
    wins = [condition_window(ad, min_coverage) for ad in datasets.values()]
    return float(np.min(wins)) if rule == "min" else float(np.median(wins))


def truncate(all_data: AllData, window: float, min_points: int = 3,
             name: str = "?") -> AllData:
    """Crop each aligned replicate to ``[0, window]``.

    Replicates left with fewer than ``min_points`` samples are dropped, because
    three points is the minimum a single-exponential fit can consume. Drops are
    reported rather than silent: a replicate disappearing here means its
    detected bleach frame was late relative to its own acquisition, which is a
    data-quality signal, not routine housekeeping.
    """
    out, dropped = [], 0
    for d in _aligned(all_data, crop=False):
        t = np.asarray(d["time"], float)
        m = t <= window + 1e-9
        # Pre-bleach frames (t < 0) are retained. They are not plotted in
        # postbleach mode, but prebleach_qc and renormalise_to_prebleach both
        # read them, and a view stripped of them reports a NaN baseline for
        # every condition — which would bury the two conditions whose baseline
        # genuinely is contaminated. Only post-bleach samples count towards
        # min_points, since those are what a recovery fit consumes.
        if int(np.sum(m & (t >= 0))) < min_points:
            dropped += 1
            continue
        e = copy.deepcopy(d)
        e["time"] = t[m]
        e["intensity"] = np.asarray(d["intensity"], float)[m]
        if "intensity_raw" in d:
            e["intensity_raw"] = np.asarray(d["intensity_raw"], float)[m]
        out.append(e)
    if dropped:
        print(f"  [{name}] {dropped} replicate(s) dropped: fewer than "
              f"{min_points} samples inside the {window:.2f} s window")
    return out


def resample_to_grid(all_data: AllData, dt: float,
                     window: Optional[float] = None,
                     min_points: int = 3) -> AllData:
    """Re-sample replicates onto a shared ``dt`` grid.

    Use only to *equalise* sampling downward, i.e. with ``dt`` set to the
    coarsest condition's interval. Interpolating a coarse trace onto a fine grid
    manufactures timepoints that were never acquired and makes the fast phase
    look resolved when it was not.
    """
    out = []
    for d in _aligned(all_data, crop=False):
        t = np.asarray(d["time"], float)
        pre = t[t < 0]
        post = t[t >= 0]
        end = post[-1] if window is None else min(post[-1], window)
        g = np.concatenate([pre, np.arange(0.0, end + 1e-9, dt)])
        if len(g) - len(pre) < min_points:
            continue
        e = copy.deepcopy(d)
        e["time"] = g
        e["intensity"] = np.interp(g, t, np.asarray(d["intensity"], float))
        if "intensity_raw" in d:
            e["intensity_raw"] = np.interp(
                g, t, np.asarray(d["intensity_raw"], float))
        e["provenance"] = f"{d.get('provenance', '?')}+resampled@{dt:g}s"
        out.append(e)
    return out


def harmonise(datasets: Datasets, window: Optional[float] = None,
              dt: Optional[float] = None, rule: str = "min",
              min_coverage: float = 0.75) -> Tuple[Datasets, float]:
    """Put every condition on a common post-bleach window (and optionally grid).

    Returns ``(datasets, window)`` containing only conditions that retain at
    least one usable replicate. A condition emptied by truncation is excluded
    with a message rather than passed on as an empty list, which downstream
    would surface as an opaque "need at least one array to concatenate" from
    the interpolation step.
    """
    W = common_window(datasets, rule, min_coverage) if window is None \
        else float(window)

    # Attribute the window so the constraint is visible rather than inferred.
    per = {k: condition_window(v, min_coverage) for k, v in datasets.items()}
    binding = min(per, key=per.get)
    print(f"  common window W = {W:.2f} s, set by {binding} "
          f"(per-condition: " +
          ", ".join(f"{k} {v:.1f}" for k, v in sorted(per.items(),
                                                      key=lambda kv: kv[1])) + ")")

    if dt is not None:
        coarsest = max(
            float(np.median(np.concatenate(
                [np.diff(np.asarray(d["time"], float)) for d in _aligned(v)])))
            for v in datasets.values())
        if dt < coarsest - 1e-9:
            raise ValueError(
                f"harmonise_dt = {dt:g} s is finer than the coarsest frame "
                f"interval in the panel ({coarsest:.2f} s). Resampling a coarse "
                "trace onto a finer grid manufactures timepoints that were "
                "never acquired; set dt to the coarsest interval or larger.")
        n_grid = int(np.floor(W / dt)) + 1
        if n_grid < 3:
            raise ValueError(
                f"a {dt:g} s grid yields only {n_grid} point(s) inside the "
                f"{W:.2f} s common window, below the 3 a recovery fit needs. "
                "Either widen the window (drop the condition that binds it) or "
                "compare mobile fractions at native sampling instead of "
                "half-times.")

    if W <= 0:
        raise ValueError(
            "common post-bleach window is zero. At least one condition has no "
            "replicate with a usable recovery phase — check bleach detection "
            "(core.detect_bleach_index) and the loader's time origin before "
            "comparing conditions.")

    out = {}
    for k, v in datasets.items():
        w = truncate(v, W, name=k) if dt is None else resample_to_grid(v, dt, W)
        if not w:
            print(f"  [{k}] excluded from the comparison: no replicate "
                  f"survives the {W:.2f} s window")
            continue
        out[k] = w
    if len(out) < 2:
        raise ValueError(
            f"only {len(out)} condition(s) survive harmonisation at "
            f"W = {W:.2f} s; a cross-condition comparison is not defined.")
    return out, W


# --------------------------------------------------------------------------- #
# 3. Audits a reviewer will ask for
# --------------------------------------------------------------------------- #
def _measured_f_pre(all_data: AllData, n_pre: int = 3) -> float:
    """Mean normalised pre-bleach plateau across replicates, defaulting to 1.0."""
    vals = []
    for d in all_data:
        y = np.asarray(d["intensity"], float)
        t = np.asarray(d["time"], float)
        pre = y[t < 0] if np.any(t < 0) else y[:0]
        if pre.size:
            vals.append(float(np.nanmean(pre[:min(n_pre, pre.size)])))
    if not vals:
        return 1.0
    v = float(np.nanmean(vals))
    return v if np.isfinite(v) and v > 0 else 1.0


def invariance_report(datasets: Datasets, window: Optional[float] = None
                      ) -> str:
    """Mobile fraction on the native window vs the common window.

    The comparative claim in the manuscript is an ordering, not an absolute
    level. If the ordering survives truncation to the shortest shared window,
    the unequal acquisition durations are a presentational issue rather than a
    confound, and the report is the evidence for saying so.
    """
    W = common_window(datasets) if window is None else float(window)
    rows, native, trunc = [], [], []
    for name, ad in datasets.items():
        g, m, _ = _mean_curve(ad)
        # Use each condition's own measured pre-bleach plateau, not 1.0. Double
        # normalisation should put it at unity, but two conditions here sit at
        # 0.894 and 1.418, and forcing the denominator to 1.0 rescales their
        # mobile fractions by exactly that factor -- overstating GFP and
        # understating eL27 relative to the per-replicate metric in the main
        # table. The mean curve carries no pre-bleach frames, so the value is
        # taken from the replicates before alignment.
        f_pre = _measured_f_pre(ad)
        a = _point_metrics(g, m, f_pre=f_pre)["mobile_fraction"]
        msk = g <= W + 1e-9
        b = _point_metrics(g[msk], m[msk], f_pre=f_pre)["mobile_fraction"]
        native.append(a); trunc.append(b)
        rows.append([name, f"{g[-1]:.1f}", f"{a:.1f}", f"{b:.1f}", f"{b - a:+.1f}"])
    order_a = [n for _, n in sorted(zip(native, datasets), reverse=True)]
    order_b = [n for _, n in sorted(zip(trunc, datasets), reverse=True)]
    verdict = ("PRESERVED" if order_a == order_b else "CHANGED  <-- do not overlay")
    return (f"WINDOW INVARIANCE (common window = {W:.2f} s)\n"
            + _fmt(rows, ["condition", "native (s)", "MF native %",
                          "MF common %", "delta"])
            + f"\nranking native: {' > '.join(order_a)}"
            + f"\nranking common: {' > '.join(order_b)}"
            + f"\nordering: {verdict}")


def sampling_bias_report(datasets: Datasets) -> str:
    """How much of each half-time is an artefact of the frame interval.

    Every condition is re-fitted after being down-sampled to the coarsest
    interval in the panel. A ratio far from 1 means the reported half-time is
    set by the acquisition, not by the assembly, and half-times must not be
    compared across conditions without this correction.
    """
    dts = {}
    for name, ad in datasets.items():
        view = _aligned(ad)
        dts[name] = float(np.median(np.concatenate(
            [np.diff(np.asarray(d["time"], float)) for d in view])))
    coarse = max(dts.values())
    rows = []
    for name, ad in datasets.items():
        g, m, _ = _mean_curve(ad)
        native = _t_half(g, m)
        gg = np.arange(0.0, g[-1] + 1e-9, coarse)
        down = _t_half(gg, np.interp(gg, g, m)) if len(gg) >= 4 else float("nan")
        ratio = down / native if native == native and native > 0 else float("nan")
        rows.append([name, f"{dts[name]:.2f}", f"{native:.2f}",
                     f"{down:.2f}", f"{ratio:.2f}"])
    spread = max(dts.values()) / min(dts.values())
    flag = ("" if spread < 1.5 else
            f"\nWARNING: frame intervals differ {spread:.1f}-fold. "
            "Report t-half per condition only, or re-fit all conditions on the "
            f"{coarse:.2f} s grid before any cross-condition statement.")
    return (f"SAMPLING BIAS (coarsest interval = {coarse:.2f} s)\n"
            + _fmt(rows, ["condition", "dt (s)", "t-half native",
                          "t-half down-sampled", "ratio"]) + flag)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # tiny self-test on synthetic replicates
    rng = np.random.default_rng(0)

    def synth(n, dt, window, mf, th):
        out = []
        for _ in range(n):
            t = np.arange(0, window + 1e-9, dt)
            tau = th / np.log(2) * rng.uniform(.8, 1.2)
            f0 = 0.08
            y = f0 + (mf / 100) * (1 - f0) * (1 - np.exp(-t / tau))
            out.append({"file": "s", "time": t,
                        "intensity": y + rng.normal(0, .004, t.size),
                        "intensity_raw": 30 * y, "provenance": "synthetic"})
        return out

    ds = {"eS2": synth(8, 1.26, 21.1, 3.5, 3.7),
          "eL27": synth(6, 1.15, 21.6, 7.1, 4.7),
          "eL42": synth(10, 6.00, 53.5, 12.3, 9.5),
          "eS28": synth(12, 2.19, 39.1, 20.6, 5.0)}
    print(acquisition_report(ds), "\n")
    print(invariance_report(ds), "\n")
    print(sampling_bias_report(ds), "\n")
    shared, W = harmonise(ds)
    print("harmonised window:", round(W, 2), "s;",
          {k: len(v) for k, v in shared.items()})