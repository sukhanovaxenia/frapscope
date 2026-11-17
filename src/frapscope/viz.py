"""
frapscope.viz — visualization applied uniformly to any extraction route.

Two normalized representations and one raw track:
  * normalized recovery + parameter boxplots  (kinetic readout; cross-comparable)
  * raw mean-intensity                          (QC + concentration; within-protein only)
  * cross-condition comparison                  (overlay, metric bars, swarm, table)

Plotting modes (``plot_mode``)
------------------------------
``postbleach`` (default)
    Recovery phase only, each replicate re-zeroed on its own detected bleach
    frame. This is the convention used in the FRAP literature and it is the
    honest one here: there are no samples between the last pre-bleach frame and
    the first recovery frame, so any line drawn across that span is
    interpolation. Re-zeroing per replicate also aligns the recovery phases
    before averaging, which a shared nominal origin does not.
``gapped``
    Pre-bleach frames retained as discrete markers at negative time, the bleach
    interval left blank, recovery drawn from zero. Use when baseline stability
    is worth showing.
``full``
    Legacy behaviour: everything joined by a continuous line. Retained for
    reproduction of earlier figures only; the bleach interval it draws is not
    data.

The comparison emits a provenance warning when conditions are extracted by
different loaders, and a baseline warning when a condition's normalised
pre-bleach plateau departs from unity, because both inject method-dependent
bias into an otherwise biological contrast.
"""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import (
    align_to_bleach,
    compute_metrics,
    drop_supraceiling,
    fit_frap_curve,
    gap_mask,
    interpolate_to_common_timepoints,
    mean_sem,
    prebleach_qc,
    renormalise_to_prebleach,
    replicate_metrics,
    trim_to_coverage,
)

COMPARISON_COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E8505B',
                     '#9C27B0', '#00BCD4', '#795548', '#607D8B',
                     '#3F51B5', '#CDDC39', '#F06292', '#FFC107']
COMPARISON_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '<', '>', 'p', '*']


def comparison_styles(n):
    """Colour and marker for each of ``n`` conditions, every pair distinct.

    Slicing the palettes to ``n`` silently returned fewer entries than
    conditions once a config listed more of them than the palettes held, and
    the comparison overlay then failed with IndexError on the first condition
    past the end. Cycling both lists in step would not fail but would repeat
    pairs, giving two conditions the same line on the same axes. The marker
    index is therefore advanced by one on each pass through the colours, so a
    colour that recurs always recurs with a different marker.
    """
    nc, nm = len(COMPARISON_COLORS), len(COMPARISON_MARKERS)
    colors = [COMPARISON_COLORS[i % nc] for i in range(n)]
    markers = [COMPARISON_MARKERS[(i + i // nc) % nm] for i in range(n)]
    return colors, markers

PLOT_MODES = ("postbleach", "gapped", "full")


def _view(all_data, plot_mode, renormalise=True, n_pre=3):
    """Replicates prepared for plotting under the requested mode.

    ``renormalise`` rescales each replicate so its own pre-bleach plateau is
    1.0, which is what the y-axis and the reference line claim. It is a no-op
    for a condition whose normalisation was already sound and cannot change any
    derived parameter.
    """
    data = renormalise_to_prebleach(all_data, n_pre=n_pre) if renormalise else all_data
    if plot_mode == "postbleach":
        return align_to_bleach(data, crop=True)
    if plot_mode == "gapped":
        return align_to_bleach(data, crop=False)
    return data


def _axis_label(plot_mode):
    return "Time after bleach (s)" if plot_mode == "postbleach" else "Time since bleach (s)"


def _draw_mean(ax, common, mean, sem, mask=None, color="k", label="Mean"):
    """Mean +/- SEM, broken wherever ``mask`` marks an unsampled interval."""
    m = mean.copy()
    if mask is not None and mask.any():
        m = np.where(mask, np.nan, m)
    ax.plot(common, m, "-", lw=2.5, color=color, label=label)
    lo, hi = mean - sem, mean + sem
    if mask is not None and mask.any():
        lo = np.where(mask, np.nan, lo); hi = np.where(mask, np.nan, hi)
    ax.fill_between(common, lo, hi, alpha=0.3, color="gray", label="±SEM")


def plot_frap_summary(all_data, name, output_dir, bleach_time=0.0,
                      n_pre=3, n_plat=3, plot_mode="postbleach",
                      renormalise=True, min_coverage=0.5):
    # Safety net only -- run() filters once before the audits so that every
    # downstream consumer sees the same replicate set. Quiet here to avoid
    # reporting the same exclusion twice.
    all_data, _dropped = drop_supraceiling(all_data, n_plat=n_plat,
                                           n_pre=n_pre, name=name,
                                           verbose=False)
    if not all_data:
        print(f"  x {name}: every replicate exceeded the pre-bleach ceiling")
        return None
    if plot_mode not in PLOT_MODES:
        raise ValueError(f"plot_mode must be one of {PLOT_MODES}, got {plot_mode!r}")
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    # --- baseline QC on the ORIGINAL traces, before any cropping ------------ #
    qc = prebleach_qc(all_data, n_pre=n_pre)
    if not qc["passed"]:
        print(f"  ⚠ BASELINE WARNING [{name}]: normalised pre-bleach plateau = "
              f"{qc['mean']:.3f}, expected 1.000 ± {qc['tol']:.2f}.")
        print("    The pre-bleach window used by the loader was contaminated by the "
              "bleach or the recovery; derived intensities are scaled by that factor.")

    view = _view(all_data, plot_mode, renormalise, n_pre)
    common, allint = interpolate_to_common_timepoints(view, key="intensity")
    common, allint, ncov, dropped = trim_to_coverage(common, allint, min_coverage)
    if dropped:
        need = max(min(3, len(view)), int(np.ceil(min_coverage * len(view))))
        print(f"  [{name}] trimmed {dropped} grid point(s) supported by fewer "
              f"than {need}/{len(view)} replicates")
    mean, sem, std, npt = mean_sem(allint)
    mask = gap_mask(view, common) if plot_mode == "gapped" else None

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.get_cmap("turbo")(np.linspace(0, 1, len(view)))
    for i, d in enumerate(view):
        t, y = np.asarray(d["time"], float), np.asarray(d["intensity"], float)
        if plot_mode == "gapped":
            pre, post = t < 0, t >= 0
            ax.plot(t[pre], y[pre], "o", alpha=0.35, ms=4, color=colors[i])
            ax.plot(t[post], y[post], "o-", alpha=0.35, ms=4, color=colors[i],
                    label=f"Rep {i+1}")
        else:
            ax.plot(t, y, "o-", alpha=0.3, ms=4, color=colors[i], label=f"Rep {i+1}")
    _draw_mean(ax, common, mean, sem, mask)

    if plot_mode == "postbleach":
        ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.6)
        ax.set_xlim(left=0)
    else:
        ax.axvline(0.0, color="red", ls="--", alpha=0.5, label="Bleach")
        if plot_mode == "gapped" and mask is not None and mask.any():
            ax.axvspan(common[mask].min(), 0, color="#BBBBBB", alpha=0.25, lw=0)
    ax.set_xlabel(_axis_label(plot_mode)); ax.set_ylabel("Normalized intensity")
    ax.set_title(f"FRAP Recovery: {name} (n={len(view)})")
    ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(output_dir / f"{name}_frap_curves.png", dpi=150); plt.close(fig)

    # --- fits: always on the original traces; the fitter finds the bleach --- #
    fits = [p for p in (fit_frap_curve(d["time"], d["intensity"],
                                     bleach_time=0.0 if d.get("bleach_at_t0") else None)
                      for d in all_data) if p]
    if len(fits) >= 2:
        th = [p["t_half"] for p in fits]; mb = [p["mobile_fraction"] for p in fits]
        pl = [p["plateau"] for p in fits]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, data, ylab, tfmt in zip(axes, [th, mb, pl],
                ['t½ (s)', 'Mobile Fraction (%)', 'Plateau (norm.)'],
                ['t½ = {:.2f} ± {:.2f} s', 'MF = {:.1f} ± {:.1f} %', 'Plateau = {:.2f} ± {:.2f}']):
            ax.boxplot(data, widths=0.6)
            ax.scatter(np.ones(len(data)), data, alpha=0.6, s=50, c='steelblue')
            ax.set_ylabel(ylab); ax.set_title(tfmt.format(np.mean(data), np.std(data))); ax.set_xticks([])
        fig.suptitle(f'{name}: FRAP Parameters (n={len(fits)})')
        fig.tight_layout(); fig.savefig(output_dir / f'{name}_frap_boxplots.png', dpi=150); plt.close(fig)
        n_bound = sum(p["tau_at_bound"] for p in fits)
        if n_bound:
            print(f"  ⚠ [{name}] {n_bound}/{len(fits)} fits hit the tau upper bound; "
                  f"t½ is unidentifiable for those replicates.")

    with open(output_dir / f'{name}_averaged_data.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['Time_s', 'Mean', 'SEM', 'SD', 'n'])
        for i in range(len(common)):
            w.writerow([f'{common[i]:.3f}', f'{mean[i]:.4f}', f'{sem[i]:.4f}',
                        f'{std[i]:.4f}', int(npt[i])])

    if fits:
        print(f"  [{name}] n={len(all_data)}  t½={np.mean([p['t_half'] for p in fits]):.1f}s  "
              f"MF={np.mean([p['mobile_fraction'] for p in fits]):.1f}%")
    prov = sorted({d.get('provenance', '?') for d in all_data})
    summary = dict(common_time=common, mean=mean, sem=sem, std=std, n=len(all_data),
                   fit_params=fits, all_intensities=allint, provenance=prov,
                   plot_mode=plot_mode, prebleach_plateau=qc["mean"],
                   prebleach_ok=qc["passed"], renormalised=bool(renormalise),
                   n_per_timepoint=npt,
                   t_bleach_shift=[d.get("t_bleach_original", 0.0) for d in view])
    # metrics on the ORIGINAL traces (they need the pre-bleach frames)
    met = replicate_metrics(all_data, n_plat=n_plat, n_pre=n_pre)
    summary.update(met)
    # The mobile fraction is the mean over RETAINED replicates, so the n printed
    # beside it must be that number. Reporting n = len(all_data) next to a mean
    # computed from fewer traces is the kind of mismatch that survives into a
    # manuscript unnoticed; n_loaded is kept separately for the figure legends,
    # which do plot every trace.
    summary["n_loaded"] = len(all_data)
    summary["n"] = int(len(met.get("mf_per_rep", []))) or len(all_data)
    return summary


def plot_intensity_raw(all_data, name, output_dir, bleach_time=0.0,
                       plot_mode="postbleach", min_coverage=0.5):
    """Raw mean-intensity (a.u.). QC + within-protein heterogeneity only;
    absolute a.u. is not comparable across proteins/sessions."""
    # Must apply the SAME filter as plot_frap_summary. When only the summary
    # filtered, the two functions built their grids from different replicate
    # sets, coverage trimming cut them to different lengths, and the comparison
    # figure raised "x and y must have same first dimension" on
    # common_time (99) against mean_raw (56).
    all_data, _ = drop_supraceiling(all_data, name=name, verbose=False)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    view = _view(all_data, plot_mode, renormalise=False)
    common, allraw = interpolate_to_common_timepoints(view, key='intensity_raw')
    common, allraw, _, _ = trim_to_coverage(common, allraw, min_coverage)
    mean, sem, std, npt = mean_sem(allraw)
    mask = gap_mask(view, common) if plot_mode == "gapped" else None

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.get_cmap('turbo')(np.linspace(0, 1, len(view)))
    for i, d in enumerate(view):
        t, y = np.asarray(d["time"], float), np.asarray(d["intensity_raw"], float)
        if plot_mode == "gapped":
            pre, post = t < 0, t >= 0
            ax.plot(t[pre], y[pre], "o", alpha=0.35, ms=4, color=colors[i])
            ax.plot(t[post], y[post], "o-", alpha=0.35, ms=4, color=colors[i], label=f'Rep {i+1}')
        else:
            ax.plot(t, y, 'o-', alpha=0.3, ms=4, color=colors[i], label=f'Rep {i+1}')
    _draw_mean(ax, common, mean, sem, mask)
    if plot_mode == "postbleach":
        ax.set_xlim(left=0)
    else:
        ax.axvline(0.0, color='red', ls='--', alpha=0.5, label='Bleach')
    ax.set_xlabel(_axis_label(plot_mode)); ax.set_ylabel('Mean Intensity (a.u.)')
    ax.set_title(f'FRAP Raw Intensity: {name} (n={len(view)})')
    ax.legend(fontsize=8, loc='best'); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(output_dir / f'{name}_frap_intensity_raw.png', dpi=150); plt.close(fig)

    with open(output_dir / f'{name}_raw_intensity.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['Time_s', 'Mean_Intensity_au', 'SEM', 'SD', 'n'])
        for i in range(len(common)):
            w.writerow([f'{common[i]:.3f}', f'{mean[i]:.3f}', f'{sem[i]:.3f}',
                        f'{std[i]:.3f}', int(npt[i])])
    return dict(common_time=common, mean_raw=mean, sem_raw=sem, all_raw=allraw)


def compare_conditions(summaries, output_dir, bleach_time=0.0, n_pre=3, n_plat=3,
                       plot_mode="postbleach"):
    """Cross-condition figures. Curves are taken from the per-condition
    summaries, so they inherit the renormalisation and coverage trimming
    applied there and are directly comparable on one axis."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    names = list(summaries.keys()); n = len(names)
    colors, markers = comparison_styles(n)

    provs = {nm: tuple(summaries[nm].get('provenance', ['?'])) for nm in names}
    distinct = {p for tup in provs.values() for p in tup}
    if len(distinct) > 1:
        print("  ⚠ PROVENANCE WARNING: conditions were extracted by different methods:")
        for nm in names:
            print(f"      {nm}: {provs[nm]}")
        print("    Cross-condition quantitation may be confounded by method-dependent bias.")
        print("    Re-extract all conditions by the same route (preferably .lif) before"
              " drawing quantitative conclusions.")

    bad = [nm for nm in names if summaries[nm].get('prebleach_ok') is False]
    if bad:
        print("  ⚠ BASELINE WARNING: normalised pre-bleach plateau is not unity for:")
        for nm in bad:
            print(f"      {nm}: {summaries[nm].get('prebleach_plateau', float('nan')):.3f}")
        print("    Those conditions' intensities are scaled by that factor; the mobile")
        print("    fraction is scale-invariant and survives, absolute levels do not.")

    metrics = {nm: compute_metrics(s, n_pre, n_plat) for nm, s in summaries.items()}

    # normalized overlay
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, nm in enumerate(names):
        s = summaries[nm]
        ax.plot(s['common_time'], s['mean'], f'{markers[i]}-', color=colors[i], lw=2, ms=4,
                markevery=max(1, len(s['common_time']) // 16),
                label=f"{nm} (n={s['n']})")
        ax.fill_between(s['common_time'], s['mean'] - s['sem'], s['mean'] + s['sem'],
                        alpha=0.15, color=colors[i])
    ax.axhline(1.0, color='gray', ls=':', alpha=0.4)
    if plot_mode == "postbleach":
        ax.set_xlim(left=0)
        ax.text(0.01, 1.015, 'pre-bleach level', transform=ax.get_yaxis_transform(),
                fontsize=8, color='gray')
    else:
        ax.axvline(bleach_time, color='gray', ls=':', alpha=0.5, label='Bleach')
    ax.set_xlabel(_axis_label(plot_mode)); ax.set_ylabel('Normalized intensity')
    ax.set_title('FRAP: Recovery Comparison'); ax.legend(fontsize=10); ax.grid(alpha=0.2)
    fig.tight_layout()
    for ext in ['png', 'svg']:
        fig.savefig(output_dir / f'comparison_overlay.{ext}', dpi=300)
    plt.close(fig)


    # raw overlay (QC only)
    if all('mean_raw' in summaries[nm] for nm in names):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, nm in enumerate(names):
            s = summaries[nm]
            ax.plot(s['common_time'], s['mean_raw'], f'{markers[i]}-', color=colors[i], lw=2, ms=4,
                    markevery=max(1, len(s['common_time']) // 16), label=f"{nm} (n={s['n']})")
            ax.fill_between(s['common_time'], s['mean_raw'] - s['sem_raw'],
                            s['mean_raw'] + s['sem_raw'], alpha=0.15, color=colors[i])
        if plot_mode == "postbleach":
            ax.set_xlim(left=0)
        else:
            ax.axvline(bleach_time, color='gray', ls=':', alpha=0.5, label='Bleach')
        ax.set_xlabel(_axis_label(plot_mode)); ax.set_ylabel('Mean Intensity (a.u.)')
        ax.set_title('FRAP: Raw Intensity (QC only — not for cross-condition quantitation)')
        ax.legend(fontsize=10); ax.grid(alpha=0.2)
        fig.tight_layout()
        for ext in ['png', 'svg']:
            fig.savefig(output_dir / f'comparison_raw_intensity.{ext}', dpi=300)
        plt.close(fig)

    # metric bars
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, field, title, ylim in [
        (axes[0], 'bleach_depth_', 'Bleach Depth (%)', (0, 110)),
        (axes[1], 'mobile_fraction_', 'Mobile Fraction (%)', None),
        (axes[2], 'immobile_fraction_', 'Immobile Fraction (%)', (0, 115))]:
        vals = [metrics[nm][field] for nm in names]
        key = 'bd_per_rep' if field == 'bleach_depth_' else 'mf_per_rep'
        errs = [float(np.std(metrics[nm].get(key, [0]))) for nm in names]
        bars = ax.bar(names, vals, color=colors, edgecolor='white', lw=1.5, width=0.6,
                      yerr=errs, capsize=4)
        ax.set_ylabel(title); ax.set_title(title.split('(')[0].strip())
        ax.set_ylim(*ylim) if ylim else ax.set_ylim(0, max(vals) * 2.0 + 5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.1f}', ha='center',
                    fontsize=10, fontweight='bold')
        ax.grid(alpha=0.2, axis='y'); ax.tick_params(axis='x', rotation=15)
    fig.suptitle('FRAP: Quantitative Parameters', y=1.02)
    fig.tight_layout()
    for ext in ['png', 'svg']:
        fig.savefig(output_dir / f'comparison_metrics.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # per-replicate mobile-fraction swarm
    if all('mf_per_rep' in metrics[nm] for nm in names):
        fig, ax = plt.subplots(figsize=(8, 5))
        rng = np.random.default_rng(0)
        for i, nm in enumerate(names):
            rv = np.asarray(metrics[nm]['mf_per_rep'], float)
            ax.scatter(np.full_like(rv, i) + rng.normal(0, 0.06, len(rv)), rv,
                       color=colors[i], s=50, alpha=0.6, edgecolors='white', lw=0.5, zorder=3)
            ax.errorbar(i, rv.mean(), yerr=rv.std(), fmt='_', color='black', ms=15, lw=2, capsize=6, zorder=4)
        ax.axhline(0, color='#999999', lw=0.8, ls=':')
        ax.set_xticks(range(n)); ax.set_xticklabels(names)
        ax.set_ylabel('Mobile Fraction (%)'); ax.set_title('Mobile Fraction per Replicate')
        ax.grid(alpha=0.2, axis='y')
        fig.tight_layout()
        for ext in ['png', 'svg']:
            fig.savefig(output_dir / f'comparison_mf_swarm.{ext}', dpi=300)
        plt.close(fig)

    # summary table
    rows = []
    print("\n" + "=" * 96)
    print(f"{'Condition':<12}{'prov':>16}{'n':>4}{'preQC':>8}{'F_pre':>7}{'I0':>7}"
          f"{'Plat':>7}{'Bleach%':>9}{'MF%':>8}{'IF%':>7}")
    print("-" * 96)
    for nm in names:
        m, s = metrics[nm], summaries[nm]
        flag = 'ok' if s.get('prebleach_ok', True) else f"{s.get('prebleach_plateau', float('nan')):.2f}!"
        print(f"{nm:<12}{','.join(provs[nm]):>16}{s['n']:>4}{flag:>8}{m['f_pre']:>7.2f}"
              f"{m['f_min']:>7.2f}{m['f_plat']:>7.2f}{m['bleach_depth_']:>8.1f}%"
              f"{m['mobile_fraction_']:>7.1f}%{m['immobile_fraction_']:>6.0f}%")
        rows.append(dict(condition=nm, provenance='|'.join(provs[nm]), n=s['n'],
                         n_loaded=s.get('n_loaded', s['n']),
                         n_excluded=s.get('n_excluded', 0),
                         prebleach_plateau=s.get('prebleach_plateau'),
                         prebleach_ok=s.get('prebleach_ok'),
                         renormalised=s.get('renormalised'),
                         plot_mode=s.get('plot_mode'),
                         f_pre=m['f_pre'], I0=m['f_min'], plateau=m['f_plat'],
                         bleach_depth_pct=m['bleach_depth_'],
                         mobile_fraction_pct=m['mobile_fraction_'],
                         mobile_fraction_sd=float(np.std(m.get('mf_per_rep', [0]))),
                         immobile_fraction_pct=m['immobile_fraction_']))
    print("=" * 96)
    with open(output_dir / 'comparison_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    return metrics


# ------------------------------------------------------ covariate scatter ---

#: Offsets tried, in order, when a label has to go somewhere that is free.
_LABEL_CANDIDATES = ((13, -4), (13, 7), (-13, -4), (-13, 7), (13, -17),
                     (-13, -17), (0, 13), (0, -24), (22, 13), (-22, -24))


def _jitter_ties(values, width=0.24):
    """Spread conditions sharing an x value, so their labels can be told apart.

    A covariate that counts something -- predicted regions, passages, days --
    puts several conditions on the same abscissa, where markers and error bars
    overlap exactly and the panel reads as fewer points than it has. The jitter
    is cosmetic and never enters the statistics, which are computed on the
    untouched values in :func:`frapscope.stats.correlate_with_covariate`.
    """
    v = np.asarray(values, float)
    out = v.copy()
    for value in np.unique(v):
        idx = np.flatnonzero(v == value)
        if idx.size > 1:
            out[idx] = value + np.linspace(-width, width, idx.size)
    return out


def _place_labels(ax, items, avoid, fontsize=10):
    """Annotate each point at the first candidate offset that collides with nothing.

    Hand-tuned offsets are the usual way to do this and they are wrong the
    moment a condition is added, a value changes, or the panel is re-run on
    another dataset -- the figure still renders, so the overlap is found by
    eye, late, or not at all. Candidates are tried in a fixed order and the
    first clear one wins, which makes the placement deterministic.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer=renderer)
    occupied = list(avoid)
    placed = []
    for x, y, text, colour in items:
        chosen, fallback = None, None
        for dx, dy in _LABEL_CANDIDATES:
            ann = ax.annotate(text, (x, y), textcoords="offset points",
                              xytext=(dx, dy),
                              ha="left" if dx >= 0 else "right",
                              fontsize=fontsize, color=colour, zorder=6)
            bb = ann.get_window_extent(renderer=renderer).expanded(1.06, 1.25)
            if fallback is None:
                fallback = (ann, bb)
            inside = frame.contains(bb.x0, bb.y0) and frame.contains(bb.x1, bb.y1)
            if inside and not any(bb.overlaps(o) for o in occupied):
                chosen = (ann, bb)
                break
            if chosen is None and (ann, bb) != fallback:
                ann.remove()
        if chosen is None:
            chosen = fallback
        elif fallback[0] is not chosen[0]:
            # compare the artists, not the tuples: a label accepted at the
            # first candidate has `chosen` and `fallback` as two equal tuples
            # holding one Text, and `is not` on the tuples removed it
            fallback[0].remove()
        occupied.append(chosen[1])
        placed.append(chosen[0])
    return placed


def plot_covariate_scatter(summaries, covariate, output_dir,
                           metric="mobile_fraction", estimator="mean",
                           reference=None, exclude=(), label="covariate",
                           result=None, xlabel=None, ylabel=None, title=None,
                           stem=None):
    """One point per condition: an external covariate against a FRAP metric.

    The figure carries what the test carries, so the two cannot drift apart:
    rho, the exact permutation p-value and the smallest p the design admits are
    written above the axes, and a condition that one replicate can move further
    than the panel spans is drawn twice -- filled where it is reported, open
    where it would sit without that replicate, joined by a dotted line. That
    second marker is the one piece of this plot that is not decoration: it
    shows, without a caption, how much of the panel's ordering rests on a
    single cell.

    ``reference`` names a condition drawn as a horizontal band rather than a
    point. It is excluded from the correlation by
    :func:`frapscope.stats.correlate_with_covariate`.
    """
    from .stats import COVARIATE_METRICS, correlate_with_covariate

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, metric_label, unit = COVARIATE_METRICS[metric]
    key = COVARIATE_METRICS[metric][0]

    if result is None:
        result = correlate_with_covariate(
            summaries, covariate, metric=metric, estimator=estimator,
            exclude=exclude, reference=reference, label=label)
    rows = result["rows"]
    if len(rows) < 2:
        print("  [covariate] fewer than two points to plot")
        return None

    fragile = {r["condition"] for r in rows if r.get("fragile")}

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    # reference condition as a band, not a point
    ref_level = None
    if reference and reference in summaries:
        rv = np.asarray(summaries[reference].get(key, []), float)
        rv = rv[np.isfinite(rv)]
        if rv.size:
            m, sd = float(rv.mean()), float(rv.std(ddof=1)) if rv.size > 1 else 0.0
            ax.axhspan(m - sd, m + sd, color="#C5CAE9", alpha=0.55, zorder=0)
            ax.axhline(m, color="#3F51B5", ls="--", lw=1.3, zorder=1,
                       label=f"{reference}: {m:.1f} ± {sd:.1f} {unit} (n={rv.size})")
            ref_level = m

    xs = _jitter_ties([r["covariate"] for r in rows])
    ys = [r[estimator] for r in rows]
    colors, _ = comparison_styles(len(rows))

    for (r, x, y, c) in zip(rows, xs, ys, colors):
        ax.errorbar(x, y, yerr=r["sd"], fmt="o", ms=9, color=c, ecolor=c,
                    elinewidth=1.3, capsize=4, alpha=0.9, zorder=3)
    first_alt = True
    for (r, x, y, c) in zip(rows, xs, ys, colors):
        if r["condition"] not in fragile:
            continue
        ax.plot([x, x], [y, r["loo_value"]], color=c, ls=":", lw=1.2, zorder=2)
        ax.plot([x], [r["loo_value"]], marker="o", ms=9, mfc="white", mec=c,
                mew=1.8, zorder=4,
                label="without its most influential replicate"
                if first_alt else None)
        first_alt = False

    ax.set_xlabel(xlabel or f"{label}")
    ax.set_ylabel(ylabel or f"{metric_label} ({unit})")
    vals = [r["covariate"] for r in rows]
    if all(float(v).is_integer() for v in vals):
        lo, hi = int(min(vals)), int(max(vals))
        ax.set_xticks(range(lo, hi + 1))
    ax.margins(x=0.18, y=0.12)
    ax.grid(alpha=0.2, axis="y")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=9, loc="best", framealpha=0.85)

    ax.set_title(title or f"{metric_label} vs {label}", pad=36)
    sub = (f"Spearman rho = {result['rho']:+.2f}, exact p = {result['p']:.3f} "
           f"(n = {result['n']}); smallest attainable p = {result['p_floor']:.3f}")
    rob = result.get("robust")
    if rob:
        sub += (f"\nwithout the most influential replicate of "
                f"{', '.join(rob['conditions'])}: rho = {rob['rho']:+.2f}, "
                f"p = {rob['p']:.3f}")
    ax.text(0.5, 1.022, sub, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="#555555")

    fig.tight_layout()

    # labels last, once the axes have their final position
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    avoid = []
    for r, x, y in zip(rows, xs, ys):
        for yy in (y, r["loo_value"] if r["condition"] in fragile else y):
            px, py = ax.transData.transform((x, yy))
            avoid.append(matplotlib.transforms.Bbox.from_bounds(
                px - 13, py - 13, 26, 26))
    if ref_level is not None:
        # the reference line runs the width of the axes; a label crossing it is
        # legible but reads as if it belonged to the line
        frame = ax.get_window_extent(renderer=renderer)
        py = ax.transData.transform((0.0, ref_level))[1]
        avoid.append(matplotlib.transforms.Bbox.from_bounds(
            frame.x0, py - 7, frame.width, 14))
    leg = ax.get_legend()
    if leg is not None:
        avoid.append(leg.get_window_extent(renderer=renderer))
    _place_labels(ax, [(x, y, f"{r['condition']} (n={r['n']})", c)
                       for r, x, y, c in zip(rows, xs, ys, colors)], avoid)

    stem = stem or f"covariate_{metric}"
    for ext in ["png", "svg"]:
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return result
