"""
frap.core — extraction-agnostic analysis of FRAP recovery.

Everything here operates on the `all_data` contract (see frap.loaders): a list
of per-replicate dicts with keys `time` (s, time-since-bleach; pre-bleach
negative), `intensity` (full-scale double-normalized, pre-bleach -> 1.0), and
`intensity_raw` (mean ROI fluorescence in detector a.u.).

Biological grounding
--------------------
The single-exponential recovery model and the mobile-fraction decomposition
follow the standard FRAP framework (Axelrod et al., Biophys J 1976; Phair &
Misteli, Nature 2000), with the caveats on over-interpreting a single rate
constant articulated by Sprague & McNally (Trends Cell Biol 2005): when recovery
is dominated by binding/exchange in a near-immobile assembly rather than free
diffusion, the fitted tau reports an effective exchange rate, not a diffusion
coefficient. The mobile fraction is the biologically load-bearing quantity for
distinguishing a dynamic, liquid-like condensate from a stable, solid- or
amyloid-like assembly (cf. Taylor et al., Biophys J 2019).

Bleach detection
----------------
The nominal time origin is not trusted. The bleach frame is instead detected as
the minimum of each replicate's own trace, which is what it is by construction:
the moment after photobleaching and before any recovery. This makes every
downstream quantity invariant to how a loader chose to place t = 0 — necessary
because the plot digitiser splits at the midpoint of the assumed axis range and
can displace the origin by several seconds in either direction.
"""

import numpy as np


def exponential_recovery(t, y0, ymax, tau):
    """I(t) = ymax - (ymax - y0) * exp(-t/tau). t is time since bleach (>=0)."""
    return ymax - (ymax - y0) * np.exp(-t / tau)


# --------------------------------------------------------------------------- #
# Bleach detection and post-bleach alignment
# --------------------------------------------------------------------------- #

def detect_bleach_index(intensity, min_post=1, min_drop_frac=0.5):
    """Index of the bleach frame: the largest drop below the running maximum.

    A bleach is defined by its *shape* — a sharp fall from the signal level
    established by the preceding frames — not by being the numerically smallest
    sample. Using the global minimum with a "leave N frames behind it" guard
    fails in both directions:

      * with no guard, the global minimum of a noisy trace can land on the final
        frame, which is not a bleach (nothing recovers after it) and collapses
        the replicate to a zero-length observation window;
      * with a guard of N frames, a trace whose post-bleach segment is SHORTER
        than N — which is exactly what down-sampling to a coarse common grid
        produces — has its real bleach excluded from the search, and the
        detector returns a frame inside the pre-bleach plateau. Every downstream
        quantity then inverts: f_min is taken at the pre-bleach level, the
        denominator (f_pre - f_min) collapses toward zero, and the mobile
        fraction diverges to large negative values.

    Scoring the drop against ``np.maximum.accumulate`` avoids both, because it
    does not depend on how many frames follow the bleach. ``min_post`` now only
    excludes the final frame, and ``min_drop_frac`` requires the winning drop to
    be at least that fraction of the trace's full range — without it, a shallow
    dip in an already-cropped post-bleach trace (which has no bleach in it at
    all) would be mistaken for one. When no drop qualifies, the minimum is
    returned, which is the correct answer for a trace that starts at the bleach.
    """
    y = np.asarray(intensity, float)
    if y.size == 0 or np.all(np.isnan(y)):
        return 0
    limit = max(1, y.size - min_post)
    cand = y[:limit]
    rng = np.nanmax(y) - np.nanmin(y)

    run_max = np.maximum.accumulate(np.where(np.isnan(cand), -np.inf, cand))
    drop = run_max - cand
    if np.all(~np.isfinite(drop)):
        return int(np.nanargmin(cand))
    i = int(np.nanargmax(np.where(np.isfinite(drop), drop, -np.inf)))
    if np.isfinite(rng) and rng > 0 and drop[i] >= min_drop_frac * rng:
        return i
    return int(np.nanargmin(cand))


def align_to_bleach(all_data, crop=False):
    """Re-zero every replicate's clock on its own detected bleach frame.

    With ``crop=False`` the pre-bleach frames are retained at negative times,
    which is what the gapped plotting style needs. With ``crop=True`` only the
    recovery phase is returned, starting at t = 0.

    The originals are not modified. Each returned entry records
    ``t_bleach_original`` (the time, on the input axis, at which the bleach was
    found) and ``n_prebleach`` so that the shift stays auditable.
    """
    out = []
    for d in all_data:
        t = np.asarray(d["time"], float)
        y = np.asarray(d["intensity"], float)
        i = detect_bleach_index(y)
        e = dict(d)
        sl = slice(i, None) if crop else slice(None)
        e["time"] = t[sl] - t[i]
        e["intensity"] = y[sl]
        if "intensity_raw" in d and d["intensity_raw"] is not None:
            e["intensity_raw"] = np.asarray(d["intensity_raw"], float)[sl]
        e["t_bleach_original"] = float(t[i])
        e["n_prebleach"] = int(i)
        out.append(e)
    return out


def prebleach_plateau(all_data, n_pre=3):
    """Normalised pre-bleach plateau of each replicate, from its EARLIEST frames.

    Double normalisation fixes the pre-bleach level at 1.0 by construction, so
    this is a quality-control quantity, not a measurement: a plateau that
    departs from unity means the window a loader used to compute its pre-bleach
    mean did not contain only pre-bleach frames.

    The earliest frames are used deliberately. Averaging over everything before
    the bleach returns 1.0 whatever happened, because that is the window the
    normalisation divided by — which is precisely why a contaminated baseline
    can pass unnoticed.
    """
    vals = []
    for d in all_data:
        y = np.asarray(d["intensity"], float)
        i = detect_bleach_index(y)
        pre = y[:i]
        if pre.size == 0:
            vals.append(np.nan)
            continue
        vals.append(float(np.nanmean(pre[: min(n_pre, pre.size)])))
    return np.array(vals, float)


def prebleach_qc(all_data, n_pre=3, tol=0.10):
    """Check that the normalised pre-bleach plateau is unity within ``tol``."""
    vals = prebleach_plateau(all_data, n_pre)
    mean = float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else np.nan
    return dict(per_rep=vals, mean=mean, tol=tol,
                passed=bool(np.isfinite(mean) and abs(mean - 1.0) <= tol))


def renormalise_to_prebleach(all_data, n_pre=3):
    """Rescale each replicate so its own measured pre-bleach plateau is 1.0.

    Double normalisation is supposed to guarantee this, but a loader whose
    pre-bleach window was contaminated leaves the whole trace scaled by a
    constant (2.34x for eS28, 1.41x for GFP in the digitised route). The mobile
    fraction is a ratio of differences and survives that untouched, but the
    plotted curve does not: on a shared axis an inflated condition appears to
    recover further than it did, and the "pre-bleach = 1.0" reference line
    stops meaning anything for it.

    Rescaling here makes the y-axis honest and the conditions mutually
    comparable. It cannot change any derived parameter, by construction.
    """
    out = []
    for d in all_data:
        y = np.asarray(d["intensity"], float)
        i = detect_bleach_index(y)
        pre = y[:i]
        f = float(np.nanmean(pre[: min(n_pre, pre.size)])) if pre.size else 1.0
        e = dict(d)
        e["intensity"] = y / f if np.isfinite(f) and f > 0 else y
        e["prebleach_scale"] = f
        out.append(e)
    return out


def _longest_true_run(mask):
    """(start, stop) of the longest contiguous True run in a boolean array."""
    best = (0, 0); start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if start is not None and len(mask) - start > best[1] - best[0]:
        best = (start, len(mask))
    return best


def trim_to_coverage(common, arr, min_fraction=0.5, min_n=3):
    """Restrict the grid to where enough replicates actually contribute.

    Masked interpolation leaves NaN outside each replicate's sampled range, so
    at the extremes of a grid spanning the union of all ranges the mean can be
    computed from a single replicate — which is why the tail of an averaged
    curve jumps to that replicate's value and its SEM collapses to zero. Grid
    points below the coverage threshold are dropped rather than plotted.

    Returns (common, arr, n, dropped) restricted to the longest contiguous run
    meeting the threshold.
    """
    n = np.sum(~np.isnan(arr), axis=0)
    n_rep = arr.shape[0]
    need = max(min(min_n, n_rep), int(np.ceil(min_fraction * n_rep)))
    ok = n >= need
    if not ok.any():
        return common, arr, n, 0
    a, b = _longest_true_run(ok)
    return common[a:b], arr[:, a:b], n[a:b], int(len(common) - (b - a))


# --------------------------------------------------------------------------- #
# Fitting and resampling
# --------------------------------------------------------------------------- #

def fit_frap_curve(time, intensity, bleach_time=None, r2_min=0.0):
    """Fit single-exponential recovery to the post-bleach phase.

    The post-bleach phase is located by bleach detection rather than by the
    nominal ``bleach_time``; passing an explicit ``bleach_time`` overrides that.
    Fitting from a displaced origin is what drives the initial-intensity
    parameter negative — physically impossible, and the signature of a wrong
    time zero rather than of slow recovery.

    Returns None if fewer than 3 post-bleach points, if the fit fails, or if the
    coefficient of determination falls below ``r2_min``.
    """
    time = np.asarray(time, float)
    intensity = np.asarray(intensity, float)
    if bleach_time is None:
        i = detect_bleach_index(intensity)
    else:
        i = int(np.argmax(time >= bleach_time))
    t_post = time[i:] - time[i]
    y_post = intensity[i:]
    pre = intensity[:i]
    pre_bleach = float(np.nanmean(pre[: min(3, pre.size)])) if pre.size else 1.0
    if len(t_post) < 3:
        return None

    from scipy.optimize import curve_fit
    try:
        popt, _ = curve_fit(exponential_recovery, t_post, y_post,
                            p0=[y_post[0], y_post[-1], 10.0],
                            bounds=([-0.2, 0, 0.1], [1.5, 1.5, 300]), maxfev=10000)
    except Exception as e:
        print(f"  fit failed: {e}")
        return None

    y0, ymax, tau = popt
    resid = y_post - exponential_recovery(t_post, *popt)
    ss_tot = float(np.sum((y_post - np.mean(y_post)) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
    if np.isfinite(r2) and r2 < r2_min:
        return None
    mf = (ymax - y0) / (pre_bleach - y0) * 100 if pre_bleach > y0 else 0.0
    return dict(y0=y0, plateau=ymax, tau=tau, t_half=tau * np.log(2),
                mobile_fraction=min(mf, 100.0), pre_bleach=pre_bleach, r2=r2,
                tau_at_bound=bool(tau > 299.0))


def interpolate_to_common_timepoints(all_data, n_points=100, key="intensity",
                                     mask_outside=True):
    """Resample every replicate onto one time grid so means/SEM are well-defined.

    Grid points outside a replicate's own sampled range are returned as NaN
    rather than clamped to its endpoint. ``np.interp`` extrapolates flat, so a
    replicate that ends early would otherwise contribute its final value as a
    horizontal line across the rest of the grid, biasing the mean and shrinking
    the SEM at exactly the timepoints where the fewest replicates contribute.
    """
    all_times = np.concatenate([np.asarray(d["time"], float) for d in all_data])
    common = np.linspace(np.nanmin(all_times), np.nanmax(all_times), n_points)
    rows = []
    for d in all_data:
        t = np.asarray(d["time"], float)
        y = np.interp(common, t, np.asarray(d[key], float))
        if mask_outside:
            y = np.where((common < t.min()) | (common > t.max()), np.nan, y)
        rows.append(y)
    return common, np.array(rows)


def median_iqr(arr):
    """NaN-aware median, bootstrap-free CI half-width, IQR and replicate count.

    Returned in the same (centre, spread_for_band, spread_wide, n) shape as
    :func:`mean_sem` so callers are interchangeable.

    When to prefer this over the mean: FRAP replicates in condensate work are
    routinely drawn from a heterogeneous population -- expression level,
    condensate size and age, focal plane and ROI placement all vary between
    cells -- so the replicate distribution is skewed rather than Gaussian and
    the sample is small. Under those conditions the mean is a poor location
    estimator: with n = 5 a single trace at 8x the others moves the mean by
    ~150 % while the median does not move at all. The IQR band also states what
    the middle half of cells did, which is the biologically meaningful claim,
    whereas a SEM band around a mean pulled by one cell states nothing.

    The caveat is that robustness is not a substitute for exclusion. A trace
    that recovers above its own pre-bleach level is not an extreme member of a
    biological distribution, it is an artefact, and it should be removed by
    drop_supraceiling rather than absorbed by a robust estimator -- otherwise
    the figure looks clean while the artefact remains unexplained.
    """
    n = np.sum(~np.isnan(arr), axis=0)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(arr, axis=0)
        q1 = np.nanpercentile(arr, 25, axis=0)
        q3 = np.nanpercentile(arr, 75, axis=0)
    iqr = q3 - q1
    # distribution-free standard error of the median, valid for n >= 5
    se = 1.2533 * (iqr / 1.349) / np.sqrt(np.maximum(n, 1))
    return med, se, iqr, n


def mean_sem(arr):
    """NaN-aware mean, SEM, SD and per-timepoint replicate count."""
    n = np.sum(~np.isnan(arr), axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
    sem = sd / np.sqrt(np.maximum(n, 1))
    return mean, sem, sd, n


def gap_mask(all_data, common):
    """Boolean mask of grid points lying inside the bleach interval.

    There are no samples between the last pre-bleach frame and the first
    recovery frame, so any curve drawn across that span is interpolation, not
    data. Masking it makes the plotted line break where the measurement did.
    Expects ``all_data`` already aligned by :func:`align_to_bleach`.
    """
    last_pre = []
    for d in all_data:
        t = np.asarray(d["time"], float)
        neg = t[t < 0]
        if neg.size:
            last_pre.append(neg.max())
    if not last_pre:
        return np.zeros_like(common, dtype=bool)
    return (common > max(last_pre)) & (common < 0)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _plateau_level(t_post, y_post, n_plat=3, plat_frac=0.2):
    """Plateau as the mean over the final ``plat_frac`` of the observation window.

    A fixed frame count is not a fixed amount of recovery when frame intervals
    differ between conditions. In this panel the interval spans 5.2-fold, so the
    last three frames cover 2.5 s of recovery for eS2 and 12 s for eL42; on the
    truncated common window that pulls eL42's plateau down into its own rising
    phase and understates its mobile fraction by roughly a third. Defining the
    plateau by elapsed time makes the estimator sampling-invariant, which is the
    same property the mobile fraction already has with respect to normalisation
    scale.

    ``n_plat`` is used only when the window itself is degenerate. The floor is
    otherwise a single sample: if a condition's frame interval is coarse enough
    that only one frame falls in the plateau region, that one frame is the
    plateau, and padding the average with earlier frames reintroduces exactly
    the bias this function exists to remove. The cost is a noisier plateau for
    coarsely sampled conditions, which is the honest trade — the noise is
    visible in the per-replicate spread, whereas the bias was not.
    """
    t_post = np.asarray(t_post, float)
    y_post = np.asarray(y_post, float)
    if y_post.size == 0:
        return np.nan
    span = float(t_post[-1] - t_post[0])
    if span <= 0:
        return float(np.nanmean(y_post[-n_plat:]))
    cut = t_post[-1] - plat_frac * span
    sel = y_post[t_post >= cut]
    if sel.size == 0:
        sel = y_post[-min(n_plat, y_post.size):]
    return float(np.nanmean(sel))


def drop_supraceiling(all_data, n_plat=3, n_pre=3, plat_frac=0.2,
                     max_mf=100.0, name="?", verbose=True):
    """Remove replicates whose terminal intensity exceeds their pre-bleach level.

    Applied once, before plotting, QC and metrics, so that a trace judged
    physically impossible is impossible everywhere. Filtering inside
    replicate_metrics alone leaves the offending replicate in the mean curve,
    the SEM band and the pre-bleach QC while removing it from the reported
    mobile fraction -- a state in which the figure and the table describe
    different samples.

    The ceiling follows from the physics: photobleaching destroys fluorophore,
    so a bleached ROI exchanging with a finite unbleached pool cannot exceed the
    signal it started from (Axelrod et al., Biophys J 1976). Traces that do are
    reporting axial drift, a reference channel bleaching faster than the ROI, or
    a mislocated ROI, none of which are exchange.
    """
    keep, dropped = [], []
    for d in all_data:
        y = np.asarray(d["intensity"], float)
        i = detect_bleach_index(y)
        pre, post = y[:i], y[i:]
        if pre.size == 0 or post.size == 0:
            keep.append(d)
            continue
        f_pre = float(np.nanmean(pre[: min(n_pre, pre.size)]))
        f_min = float(post[0])
        f_plat = _plateau_level(np.asarray(d["time"], float)[i:], post,
                                n_plat=n_plat, plat_frac=plat_frac)
        if (f_pre - f_min) <= 0:
            keep.append(d)
            continue
        v = (f_plat - f_min) / (f_pre - f_min) * 100
        (dropped if v > max_mf else keep).append(
            (d.get("file", "?"), v) if v > max_mf else d)
    if dropped and verbose:
        print(f"  ! [{name}] {len(dropped)} replicate(s) excluded everywhere: "
              f"terminal intensity exceeds the pre-bleach reference")
        for nm, v in dropped:
            print(f"      {nm}: apparent MF = {v:.0f} %")
    return keep, dropped


def replicate_metrics(all_data, n_plat=3, n_pre=3, plat_frac=0.2,
                      max_mf=100.0):
    """Per-replicate bleach depth and mobile fraction, computed POINT-WISE on
    each replicate's real (non-interpolated) samples.

    All three terms are taken relative to the detected bleach frame:
      f_pre  : mean of the earliest ``n_pre`` frames (the true plateau)
      f_min  : the bleach frame itself
      f_plat : mean of the final ``n_plat`` frames

    Because the mobile fraction is a ratio of differences drawn from the same
    trace, defining f_pre this way makes it invariant to the overall scale of
    the normalisation. That matters: a loader whose pre-bleach window was
    contaminated inflates the whole curve by a constant factor, which cancels
    here but does not cancel if f_pre is forced to 1.0 by convention.
    """
    mf, bd, pres, mins, plats, excluded = [], [], [], [], [], []
    for d in all_data:
        y = np.asarray(d["intensity"], float)
        i = detect_bleach_index(y)
        pre, post = y[:i], y[i:]
        if pre.size == 0 or post.size == 0:
            continue
        f_pre = float(np.nanmean(pre[: min(n_pre, pre.size)]))
        f_min = float(post[0])
        f_plat = _plateau_level(np.asarray(d["time"], float)[i:], post,
                                n_plat=n_plat, plat_frac=plat_frac)
        pres.append(f_pre); mins.append(f_min); plats.append(f_plat)
        if (f_pre - f_min) > 0:
            v = (f_plat - f_min) / (f_pre - f_min) * 100
            # A mobile fraction above 100 % means the ROI ended brighter than
            # its own pre-bleach reference. No passive exchange process can do
            # that: the bleached molecules are destroyed, so recovery is bounded
            # by the pre-bleach level. Values above the ceiling therefore mark an
            # acquisition artefact rather than fast exchange -- most often axial
            # drift bringing a brighter plane into the ROI, or over-correction by
            # a reference channel that photobleaches faster than the ROI does
            # (the top-percentile proxy in load_lif is the brightest, hence
            # fastest-bleaching, pixels). Such replicates are excluded rather
            # than clamped, because clamping to 100 % would silently retain a
            # corrupted trace at the extreme of the distribution and inflate both
            # the mean and its variance.
            if v > max_mf:
                excluded.append((d.get("file", "?"), v))
                continue
            mf.append(v)
            bd.append((f_pre - f_min) / f_pre * 100)
    if excluded:
        print(f"  ! {len(excluded)} replicate(s) excluded: terminal intensity "
              f"exceeds the pre-bleach reference (physically impossible)")
        for nm, v in excluded:
            print(f"      {nm}: apparent MF = {v:.0f} %")
    return dict(mf_per_rep=np.array(mf), bd_per_rep=np.array(bd),
                n_excluded=len(excluded), excluded=excluded,
                f_pre=float(np.mean(pres)) if pres else np.nan,
                f_min=float(np.mean(mins)) if mins else np.nan,
                f_plat=float(np.mean(plats)) if plats else np.nan)


def compute_metrics(summary, n_pre=3, n_plat=3):
    """Condition-level bleach depth and mobile/immobile fraction.

    Prefers the point-based per-replicate metrics that plot_frap_summary stores
    (via replicate_metrics); the reported mobile fraction and bleach depth are
    the means of the per-replicate values, so the bar heights match the swarm
    points. Falls back to a mean-curve estimate only for summaries lacking
    replicate data.
    """
    if "mf_per_rep" in summary and "f_pre" in summary:
        f_pre, f_min, f_plat = summary["f_pre"], summary["f_min"], summary["f_plat"]
        mf = float(np.mean(summary["mf_per_rep"])) if len(summary["mf_per_rep"]) else 0.0
        bd = float(np.mean(summary["bd_per_rep"])) if len(summary["bd_per_rep"]) else 0.0
        return dict(f_pre=f_pre, f_min=f_min, f_plat=f_plat,
                    bleach_depth_=bd, mobile_fraction_=mf, immobile_fraction_=100 - mf,
                    mf_per_rep=summary["mf_per_rep"], bd_per_rep=summary["bd_per_rep"])
    t, y = summary["common_time"], summary["mean"]
    i = detect_bleach_index(y)
    f_pre = float(np.nanmean(y[: min(n_pre, max(i, 1))]))
    f_min = float(y[i]); f_plat = float(np.nanmean(y[-n_plat:]))
    bd = (f_pre - f_min) / f_pre * 100 if f_pre > 0 else 0.0
    mf = (f_plat - f_min) / (f_pre - f_min) * 100 if (f_pre - f_min) > 0 else 0.0
    return dict(f_pre=f_pre, f_min=f_min, f_plat=f_plat,
                bleach_depth_=bd, mobile_fraction_=mf, immobile_fraction_=100 - mf)