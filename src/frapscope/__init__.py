"""
frap — a FRAP analysis library factored on the `all_data` contract.

Architecture
------------
    loaders (swappable extraction) -> all_data -> timebase -> core + viz

Extraction route is pluggable (load_lif / load_roi_csv / load_image_digitized);
fitting, normalisation, metrics and figures are identical regardless of source,
so a quantitative conclusion never depends on how the fluorescence was obtained.
Provenance is carried per replicate and the comparison warns on mixes.

`timebase` sits between loading and analysis. Per-condition figures are produced
on each protein's native window, because that is the honest description of what
was acquired for that protein. The cross-condition comparison is produced on the
harmonised window, because an overlay asserts that two curves at the same
abscissa are comparable observations — which is only true after truncation to a
window every condition covers.

Public API
----------
    from frap import build_all_data, run
    from frap.loaders import LOADER_REGISTRY
    from frap.config import Condition, CONDITIONS
"""

from pathlib import Path

from .loaders import LOADER_REGISTRY
from .viz import plot_frap_summary, plot_intensity_raw, compare_conditions
from .core import drop_supraceiling
from .stats import compare_mobile_fractions
from .timebase import (acquisition_report, invariance_report,
                       sampling_bias_report, harmonise)

__all__ = ["build_all_data", "run"]


def build_all_data(condition):
    """Dispatch a Condition to its loader and return the all_data contract."""
    loader = LOADER_REGISTRY[condition.loader]
    return loader(condition.source, **condition.loader_kwargs)


def _load_all(conditions):
    """Load every condition once, reporting failures instead of raising.

    Loading is the expensive step for the .lif route, so it happens exactly
    once here; both the audits and the figures consume the same in-memory
    replicates. Loading twice would also risk the two copies diverging if a
    loader is stochastic in any way.
    """
    datasets, provenance = {}, {}
    for c in conditions:
        if c.loader not in LOADER_REGISTRY:
            print(f"  x {c.display}: unknown loader {c.loader!r}")
            continue
        try:
            ad = build_all_data(c)
        except Exception as e:
            print(f"  x {c.display}: loader failed - {e}")
            continue
        if not ad:
            print(f"  x {c.display}: loader returned no replicates")
            continue
        datasets[c.display] = ad
        provenance[c.display] = c.loader
    return datasets, provenance


def run(conditions, out_base, bleach_time=0.0, n_pre=3, n_plat=3,
        plot_mode="postbleach", renormalise=True, min_coverage=0.75,
        harmonise_comparison=True, harmonise_dt=None,
        stats_control=None, stats_exclude=()):
    """Full pipeline: load -> audit -> per-condition figures -> comparison.

    Parameters
    ----------
    harmonise_comparison : bool
        Truncate every condition to the longest post-bleach window all of them
        cover before building the comparison. Leave True unless the conditions
        were acquired with a matched protocol.
    harmonise_dt : float or None
        Also resample onto a shared frame interval. Set this only when
        half-times are to be compared across conditions, and set it to the
        coarsest interval in the panel; resampling a coarse trace onto a fine
        grid manufactures timepoints that were never acquired.
    min_coverage : float
        Fraction of a condition's replicates that must contribute to a grid
        point for it to be plotted. 0.75 keeps the tail of a mean curve from
        being drawn by a minority of replicates with an artificially collapsed
        SEM.
    """
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    # ---- load once -------------------------------------------------------
    datasets, provenance = _load_all(conditions)
    if not datasets:
        print("no conditions loaded")
        return {}

    # ---- exclude physically impossible traces BEFORE anything reads them --
    # The audits are computed from the same replicates as the figures, so the
    # filter has to precede them. Applying it only inside plot_frap_summary left
    # invariance_report averaging traces that recover above their own pre-bleach
    # level: eL27 reported 48.7 % there against 5.1 % in the metrics table.
    exclusions = {}
    for _name in list(datasets):
        n_before = len(datasets[_name])
        kept, dropped = drop_supraceiling(datasets[_name], name=_name)
        # Filtering upstream means plot_frap_summary now receives already-clean
        # data and would report n_loaded == n, n_excluded == 0. The counts have
        # to be declared in the Methods, so they are carried forward here.
        exclusions[_name] = dict(n_loaded=n_before, n_excluded=len(dropped),
                                 excluded=[nm for nm, _ in dropped])
        if not kept:
            print(f"  x {_name}: every replicate exceeded the pre-bleach ceiling")
            datasets.pop(_name)
            continue
        datasets[_name] = kept

    # ---- audits: run before anything is plotted or written --------------
    print(acquisition_report(datasets), "\n")
    if len(datasets) >= 2:
        print(invariance_report(datasets), "\n")
        print(sampling_bias_report(datasets), "\n")

    def _figures(data_map, root, tag):
        """Per-condition summary + raw QC track for one view of the data."""
        out = {}
        for name, ad in data_map.items():
            if not ad:
                print(f"  x {name}: no replicates in the {tag} view - skipped")
                continue
            d = root / f"{name}_frap_viz"
            s = plot_frap_summary(ad, name, d, bleach_time,
                                  n_pre, n_plat, plot_mode,
                                  renormalise, min_coverage)
            raw = plot_intensity_raw(ad, name, d, bleach_time, plot_mode,
                                     min_coverage)
            s.update({k: raw[k] for k in ("mean_raw", "sem_raw", "all_raw")})
            # NB: viz already sets s["provenance"] to the sorted set of
            # per-replicate provenances. Record the declared loader under a
            # separate key; overwriting it with a bare string makes the
            # comparison iterate the string character-by-character.
            s["loader"] = provenance.get(name, "?")
            s["view"] = tag
            if name in exclusions:
                s["n_loaded"] = exclusions[name]["n_loaded"]
                s["n_excluded"] = exclusions[name]["n_excluded"]
                s["excluded_replicates"] = exclusions[name]["excluded"]
            out[name] = s
        return out

    # ---- per-condition figures on the native window ---------------------
    summaries = _figures(datasets, out_base, "native")

    # ---- comparison on the harmonised window ----------------------------
    if len(summaries) >= 2:
        cmp_summaries = summaries
        if harmonise_comparison:
            try:
                shared, W = harmonise(datasets, dt=harmonise_dt,
                                      min_coverage=min_coverage)
            except ValueError as e:
                print(f"  ! harmonisation failed: {e}")
                print("  ! falling back to native windows; the overlay is NOT "
                      "cross-comparable and must not be published as one")
                shared, W = None, None
            if shared:
                print(f"comparison built on a common post-bleach window of "
                      f"{W:.2f} s" + (f", resampled to {harmonise_dt:g} s"
                                      if harmonise_dt else " at native sampling"))
                cmp_root = out_base / "harmonised"
                cmp_summaries = _figures(shared, cmp_root,
                                         f"harmonised@{W:.2f}s")
                for _n, _s in cmp_summaries.items():
                    _s["common_window_s"] = W

        if len(cmp_summaries) >= 2:
            compare_conditions(cmp_summaries, out_base / "comparison",
                               bleach_time, n_pre, n_plat, plot_mode)
            compare_mobile_fractions(cmp_summaries, out_base / "comparison",
                                     control=stats_control,
                                     exclude=stats_exclude)
        summaries = {"native": summaries, "comparison": cmp_summaries}

    return summaries