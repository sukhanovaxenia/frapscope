"""
frapscope.stats — cross-condition comparison of mobile fractions.

Until now the p-values quoted in the manuscript were computed outside the
pipeline, which means they were not reproducible from the same command that
produced the figures and could drift out of step with them after any change to
exclusion or extraction. This module closes that gap: it consumes the same
per-replicate mobile fractions that the summary table reports and writes both a
console table and a CSV.

Test choice
-----------
Two tests are reported for every contrast, deliberately.

Mann-Whitney U is the primary test. Replicate counts here are 3-10 and the
replicate distributions are skewed rather than Gaussian, because FRAP samples a
heterogeneous cell population: expression level, condensate size and age, focal
plane and ROI placement all vary between cells. A rank test makes no
distributional assumption and is not moved by a single extreme cell.

Welch's t-test is reported alongside because it uses the magnitudes rather than
only the ranks, and because at n = 3 the exact Mann-Whitney null distribution
cannot produce a p-value below 0.1 whatever the separation -- with three
observations against six there are only 84 distinguishable orderings. A contrast
involving eL27 that is significant by Welch and non-significant by
Mann-Whitney is reporting that limit, not an absence of effect, and the two
tests together make that visible instead of hiding it behind one number.

Hedges' g accompanies both, since with small n the effect size carries more
information than the p-value: a large g with a non-significant p identifies a
contrast that is under-powered rather than null, which is the actionable
distinction when deciding whether to acquire more replicates.

Multiplicity
------------
Holm's step-down correction is applied across the planned contrasts. It controls
the family-wise error rate under arbitrary dependence, is uniformly more
powerful than Bonferroni, and needs no assumption about the correlation between
tests -- appropriate here because the contrasts share conditions and are
therefore not independent.

Correlation with an external covariate
--------------------------------------
``correlate_with_covariate`` answers a different question from the contrasts
above: not whether two conditions differ, but whether a FRAP metric tracks some
number measured outside the pipeline across the whole panel. The unit of
analysis is then the condition rather than the replicate, so k is four to eight
and every property of small-k rank statistics applies at once -- a p-value with
a floor that ties can push above 0.05, a coefficient that can change sign with
the choice of estimator, and an asymptotic p-value that is simply wrong.
``spearman_exact`` and the warnings in ``_report_covariate`` exist so that none
of those has to be remembered at the point of use.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from itertools import combinations
from pathlib import Path

import numpy as np

__all__ = ["compare_mobile_fractions", "planned_contrasts", "holm", "bh",
           "spearman_exact", "correlate_with_covariate",
           "COVARIATE_METRICS"]


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj


def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Bias-corrected standardised mean difference."""
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                 / (na + nb - 2))
    if sp == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)))


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    from scipy import stats as st
    r = st.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    return float(r.statistic), float(df), float(r.pvalue)


def _mwu(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    from scipy import stats as st
    r = st.mannwhitneyu(a, b, alternative="two-sided")
    return float(r.statistic), float(r.pvalue)


def bh(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), order preserved.

    Controls the expected proportion of false discoveries among the rejected
    hypotheses, not the probability of any false rejection. That is the right
    guarantee for a screen -- many conditions times many metrics, where the
    question is which differences are worth following up -- and the wrong one
    for the handful of planned contrasts a paper's conclusion rests on, which
    keep Holm. Non-finite p-values are passed through as NaN and do not count
    towards the family size: an untestable cell is not a test.
    """
    p = np.asarray(pvals, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0.0, 1.0)
    q[ok] = out
    return q


def _min_p(na: int, nb: int) -> float:
    """Smallest two-sided p the exact Mann-Whitney null can return at this n."""
    from math import comb
    total = comb(na + nb, na)
    return min(1.0, 2.0 / total)


def planned_contrasts(names: Iterable[str], control: str | None = None,
                      exclude: Iterable[str] = ()) -> list[tuple[str, str]]:
    """The contrast family to correct over.

    Holm's correction divides the available significance among the tests in the
    family, so the family has to be the set of comparisons the study was
    designed to make -- not every pair the data happen to permit. Testing all
    pairs of five conditions is ten tests, and in this panel that multiplicity
    alone moves eS28 vs eL27 from p = 0.048 to p = 0.095: the contrast does not
    change, only how many other questions were asked alongside it. Pre-specify
    the family and the correction stops charging for questions the manuscript
    never poses.

    With ``control`` given, returns each remaining condition against it -- the
    natural family when the claim is "each candidate differs from the
    non-aggregating control". ``exclude`` removes conditions that are not
    testable at all, such as one pooled across non-identical acquisition
    settings; those belong in the Results as descriptive observations, and
    including them costs power on the contrasts that carry the argument.
    """
    ns = [n for n in names if n not in set(exclude)]
    if control is None:
        return list(combinations(ns, 2))
    if control not in ns:
        raise ValueError(f"control {control!r} not among testable conditions {ns}")
    return [(n, control) for n in ns if n != control]


def compare_mobile_fractions(summaries: dict[str, dict],
                             out_dir: Path | None = None,
                             contrasts: Iterable[tuple[str, str]] | None = None,
                             control: str | None = None,
                             exclude: Iterable[str] = (),
                             qc_only: bool = False,
                             out_name: str = "mobile_fraction_stats.csv"
                             ) -> list[dict]:
    """Pairwise tests on per-replicate mobile fractions.

    ``summaries`` is the dict returned by :func:`frap.run`; each value must
    carry ``mf_per_rep``. ``contrasts`` defaults to every pair.

    ``qc_only`` defaults to False, and deliberately. A pre-bleach plateau away
    from unity means the loader's normalisation reference disagrees with the
    earliest pre-bleach frames, so the whole trace is scaled by a constant. The
    mobile fraction is a ratio of differences drawn from that same trace and the
    constant cancels exactly, which is why a condition can fail baseline QC and
    still contribute a valid mobile fraction. What such a condition cannot
    contribute is an absolute level -- plateau and bleach depth are not
    comparable across a scale mismatch. Excluding it from testing altogether
    would therefore discard usable evidence; the failure is flagged instead so
    the manuscript can state it.

    Set ``qc_only=True`` only when the comparison is over absolute levels rather
    than mobile fractions. Conditions unsuitable for a different reason -- for
    instance a control pooled across non-identical acquisition settings -- should
    be handled by passing an explicit ``contrasts`` list, since that is a
    judgement about the experiment rather than about the baseline.
    """
    usable = {}
    for name, s in summaries.items():
        mf = np.asarray(s.get("mf_per_rep", []), float)
        mf = mf[np.isfinite(mf)]
        if mf.size < 3:
            print(f"  [stats] {name}: fewer than 3 usable replicates, skipped")
            continue
        if not s.get("prebleach_ok", True):
            msg = (f"  [stats] {name}: pre-bleach plateau "
                   f"{s.get('prebleach_plateau', float('nan')):.3f} != 1.0")
            if qc_only:
                print(msg + " -- excluded (qc_only=True)")
                continue
            print(msg + " -- mobile fraction is scale-invariant and retained; "
                        "do not quote its plateau or bleach depth")
        usable[name] = mf
    if len(usable) < 2:
        print("  [stats] fewer than two testable conditions")
        return []

    if contrasts is not None:
        pairs = list(contrasts)
    else:
        pairs = planned_contrasts(usable, control=control, exclude=exclude)
    pairs = [(a, b) for a, b in pairs if a in usable and b in usable]
    if exclude:
        print(f"  [stats] excluded from testing (reported descriptively): "
              f"{', '.join(exclude)}")
    print(f"  [stats] correcting over {len(pairs)} planned contrast(s)")

    rows = []
    for a, b in pairs:
        x, y = usable[a], usable[b]
        t, df, p_t = _welch(x, y)
        u, p_u = _mwu(x, y)
        rows.append(dict(
            group_a=a, group_b=b, n_a=int(x.size), n_b=int(y.size),
            mean_a=float(x.mean()), mean_b=float(y.mean()),
            median_a=float(np.median(x)), median_b=float(np.median(y)),
            welch_t=t, welch_df=df, welch_p=p_t,
            mwu_U=u, mwu_p=p_u, mwu_p_floor=_min_p(x.size, y.size),
            hedges_g=_hedges_g(x, y)))

    for key, src in (("welch_p_holm", "welch_p"), ("mwu_p_holm", "mwu_p")):
        for r, q in zip(rows, holm([r[src] for r in rows])):
            r[key] = float(q)

    def star(q):
        return "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "ns"

    print("\nMOBILE FRACTION: pairwise comparison "
          f"(Holm-corrected over {len(rows)} contrasts)")
    hdr = (f"{'contrast':18s}{'n':>7s}{'g':>7s}{'MWU p':>9s}{'p_holm':>8s}"
           f"{'Welch p':>9s}{'p_holm':>8s}  verdict")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        note = ""
        if r["mwu_p_holm"] >= .05 and r["welch_p_holm"] < .05:
            note = "  under-powered (MWU floor "
            note += f"{r['mwu_p_floor']:.2f})"
        print(f"{r['group_a']+' vs '+r['group_b']:18s}"
              f"{str(r['n_a'])+'/'+str(r['n_b']):>7s}{r['hedges_g']:7.2f}"
              f"{r['mwu_p']:9.4f}{r['mwu_p_holm']:8.4f}"
              f"{r['welch_p']:9.5f}{r['welch_p_holm']:8.4f}  "
              f"{star(max(r['mwu_p_holm'], r['welch_p_holm']))}{note}")

    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / out_name
        with open(fp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"  [stats] written to {fp}")
    return rows


# ------------------------------------------------------- rank correlation ---

def spearman_exact(x, y, max_exact=9, n_mc=200_000, seed=0):
    """Spearman rho, its permutation p-value, and the smallest attainable p.

    With k points the null distribution has k! equally likely orderings, so the
    p-value has a floor: 2/24 = 0.083 for k = 4 without ties. A rho of 1.0 at
    k = 4 is therefore not significant at 0.05 however it is computed, and the
    floor is reported so that this reads as a property of the design rather
    than of the data. Ties in either variable raise the floor further, because
    permutations that only exchange tied values are not distinguishable.
    Exact enumeration up to ``max_exact`` points, Monte Carlo beyond.

    Lives here rather than in :mod:`frapscope.crossrun`, where it was first
    written, because two callers now need it and ``crossrun`` already imports
    from this module.
    """
    import itertools

    from scipy import stats as st

    x, y = np.asarray(x, float), np.asarray(y, float)
    k = x.size
    if k < 3:
        return np.nan, np.nan, np.nan
    rx, ry = st.rankdata(x), st.rankdata(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return np.nan, np.nan, np.nan

    def corr(perm_y):
        a = rx - rx.mean()
        b = perm_y - perm_y.mean(axis=-1, keepdims=True)
        return (b @ a) / np.sqrt((a @ a) * np.sum(b * b, axis=-1))

    rho = float(corr(ry[None, :])[0])
    if k <= max_exact:
        perms = np.array(list(itertools.permutations(range(k))))
    else:
        rng = np.random.default_rng(seed)
        perms = np.array([rng.permutation(k) for _ in range(n_mc)])
    null = np.abs(corr(ry[perms]))
    eps = 1e-12
    p = float(np.mean(null >= abs(rho) - eps))
    floor = float(np.mean(null >= null.max() - eps))
    return rho, p, floor


# ------------------------------------------------- covariate correlation ---

#: Per-replicate metrics a condition summary carries, and how to label them.
COVARIATE_METRICS = {
    "mobile_fraction": ("mf_per_rep", "Mobile fraction", "%"),
    "bleach_depth": ("bd_per_rep", "Bleach depth", "%"),
}


def _leave_one_out(values, estimator):
    """How far one replicate can move a condition, and where it would land.

    A condition entering a correlation is one number standing for n cells, and
    with n in the low teens a single cell can decide where that number sits --
    in FRAP most often a cell whose bleach failed, whose mobile fraction is
    then a ratio with a near-zero denominator and can reach hundreds of percent
    in either direction. The leave-one-out shift measures that directly instead
    of inferring it from a standard deviation, which an extreme value inflates
    along with the mean it is supposed to qualify.
    """
    v = np.asarray(values, float)
    if v.size < 3:
        return float("nan"), float("nan"), -1
    f = np.mean if estimator == "mean" else np.median
    base = float(f(v))
    shifted = np.array([float(f(np.delete(v, i))) for i in range(v.size)])
    i = int(np.argmax(np.abs(shifted - base)))
    return float(abs(shifted[i] - base)), float(shifted[i]), i


def _condition_rows(summaries, covariate, key, exclude, estimator):
    """One row per condition that has both a covariate value and replicates."""
    rows, missing = [], []
    for name, s in summaries.items():
        if name in exclude:
            continue
        if name not in covariate:
            missing.append(name)
            continue
        v = np.asarray(s.get(key, []), float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            print(f"  [covariate] {name}: no usable replicates, skipped")
            continue
        influence, loo, idx = _leave_one_out(v, estimator)
        rows.append(dict(
            condition=name, covariate=float(covariate[name]), n=int(v.size),
            mean=float(v.mean()), median=float(np.median(v)),
            sd=float(v.std(ddof=1)) if v.size > 1 else float("nan"),
            sem=float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1
            else float("nan"),
            influence=influence, loo_value=loo, loo_replicate=idx,
            fragile=False))
    extra = [n for n in covariate if n not in summaries]
    return rows, missing, extra


def correlate_with_covariate(summaries: dict[str, dict],
                             covariate: dict[str, float],
                             metric: str = "mobile_fraction",
                             estimator: str = "mean",
                             exclude: Iterable[str] = (),
                             reference: str | None = None,
                             out_dir: Path | None = None,
                             max_exact: int = 9,
                             fragile_frac: float = 0.5,
                             label: str = "covariate") -> dict:
    """Rank correlation between a per-condition covariate and a FRAP metric.

    ``covariate`` maps display name to a number measured outside this pipeline
    -- predicted aggregation-prone regions, expression level, construct length,
    days post-transfection. The function deliberately knows nothing about what
    the number means: a FRAP package that imported a sequence-analysis package
    to test one hypothesis would acquire that dependency for every user.

    The unit of analysis is the condition, not the replicate, so k is the
    number of conditions -- four to eight in a normal panel. Four consequences
    follow, and all four are reported rather than left to the reader.

    **The p-value has a floor.** At k conditions there are k! orderings, and
    ties in the covariate collapse some of them, so the smallest attainable
    two-sided p may sit above 0.05. When it does, no arrangement of the data
    could have produced a significant result, and calling rho non-significant
    describes the design rather than the data. ``p_floor`` carries this.

    **One replicate can decide a condition's place.** Each condition's
    leave-one-out influence is computed, and a condition that a single
    replicate moves by more than ``fragile_frac`` of the spread across the
    whole panel is marked ``fragile``. The correlation is then recomputed with
    every fragile condition replaced by its leave-one-out value and returned
    under ``robust``. A conclusion that holds only in one of the two is not
    yet a conclusion; one that holds in both can be stated without the caveat.

    **The estimator matters.** Mean and median can order the conditions
    differently. Both are computed and the alternative is returned under
    ``alternate``; a warning is raised only when the two disagree about the
    sign of rho or about significance, since small reorderings among close
    conditions are expected and not informative.

    **Spearman's p is not the t-approximation.** ``scipy.stats.spearmanr``
    returns a p-value from a t-distribution that assumes a sample large enough
    for the approximation to hold; at k = 6 it differs from the exact value in
    the second decimal. :func:`spearman_exact` enumerates the permutations.

    ``reference`` names a condition to exclude from the correlation and keep
    for plotting as a horizontal band -- a free-fluorophore control belongs on
    the axes but has no covariate value and is not a point of the regression.
    """
    if metric not in COVARIATE_METRICS:
        raise ValueError(
            f"metric must be one of {sorted(COVARIATE_METRICS)}, got {metric!r}. "
            "t_half is not carried on condition summaries; take it from "
            "core.replicate_table, which fits each replicate separately.")
    if estimator not in ("mean", "median"):
        raise ValueError(f"estimator must be 'mean' or 'median', got {estimator!r}")

    key, metric_label, unit = COVARIATE_METRICS[metric]
    skip = set(exclude) | ({reference} if reference else set())
    rows, missing, extra = _condition_rows(summaries, covariate, key, skip,
                                           estimator)

    if missing:
        print(f"  [covariate] no {label} value for: {', '.join(sorted(missing))}"
              " -- left out of the correlation")
    if extra:
        print(f"  [covariate] {label} given for conditions not in this run: "
              f"{', '.join(sorted(extra))}")

    out: dict = dict(label=label, metric=metric, metric_label=metric_label,
                     unit=unit, estimator=estimator, reference=reference,
                     n=len(rows), rows=rows, rho=float("nan"),
                     p=float("nan"), p_floor=float("nan"),
                     underpowered=False, alternate=None, robust=None,
                     estimators_disagree=False, spread=float("nan"))
    if len(rows) < 3:
        print(f"  [covariate] fewer than three conditions carry both a {label} "
              "value and replicates; no correlation computed")
        return out

    x = [r["covariate"] for r in rows]
    fits = {}
    for est in ("mean", "median"):
        y = [r[est] for r in rows]
        rho, p, floor = spearman_exact(x, y, max_exact=max_exact)
        fits[est] = dict(estimator=est, rho=rho, p=p, p_floor=floor)

    other = "median" if estimator == "mean" else "mean"
    out.update(fits[estimator])
    out["alternate"] = fits[other]
    out["underpowered"] = bool(out["p_floor"] > 0.05)

    a, b = out["rho"], fits[other]["rho"]
    out["estimators_disagree"] = bool(
        (np.sign(a) != np.sign(b) and min(abs(a), abs(b)) > 0.05)
        or ((out["p"] < 0.05) != (fits[other]["p"] < 0.05)))

    # fragility: one replicate against the spread of the whole panel
    levels = [r[estimator] for r in rows]
    out["spread"] = float(max(levels) - min(levels))
    for r in rows:
        r["fragile"] = bool(np.isfinite(r["influence"])
                            and r["influence"] > fragile_frac * out["spread"])
    if any(r["fragile"] for r in rows):
        y_rob = [r["loo_value"] if r["fragile"] else r[estimator] for r in rows]
        rho_r, p_r, floor_r = spearman_exact(x, y_rob, max_exact=max_exact)
        out["robust"] = dict(estimator=f"{estimator}, fragile conditions "
                                       "without their most influential replicate",
                             rho=rho_r, p=p_r, p_floor=floor_r,
                             conditions=[r["condition"] for r in rows
                                         if r["fragile"]])

    _report_covariate(out)
    if out_dir is not None:
        _write_covariate(out, Path(out_dir))
    return out


def _report_covariate(res: dict) -> None:
    """Console table, then only the warnings this design actually earns."""
    label, est = res["label"], res["estimator"]
    print(f"\n{res['metric_label'].upper()} vs {label.upper()} "
          f"(Spearman, exact permutation p, {res['n']} conditions)")
    hdr = (f"{'condition':<14}{label[:10]:>11}{'mean':>9}{'median':>9}"
           f"{'sd':>8}{'n':>5}{'1-rep':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(res["rows"], key=lambda r: -r["covariate"]):
        mark = " !" if r["fragile"] else ""
        print(f"{r['condition']:<14}{r['covariate']:>11.3g}{r['mean']:>9.2f}"
              f"{r['median']:>9.2f}{r['sd']:>8.2f}{r['n']:>5d}"
              f"{r['influence']:>7.1f}{mark}")
    print(f"\n  rho = {res['rho']:+.3f}   p = {res['p']:.4f}   "
          f"(smallest attainable p at this design: {res['p_floor']:.4f})")

    alt = res["alternate"]
    if alt is not None:
        print(f"  on {alt['estimator']}s instead: rho = {alt['rho']:+.3f}, "
              f"p = {alt['p']:.4f}")
    rob = res["robust"]
    if rob is not None:
        print(f"  dropping the most influential replicate of "
              f"{', '.join(rob['conditions'])}: rho = {rob['rho']:+.3f}, "
              f"p = {rob['p']:.4f}")

    if res["underpowered"]:
        print(f"  ! no result can reach 0.05 here: {res['n']} conditions and "
              f"the ties in {label} put the floor at {res['p_floor']:.3f}. "
              "Report rho descriptively; the design, not the data, decided "
              "that it would not be significant.")
    for r in res["rows"]:
        if r["fragile"]:
            print(f"  ! {r['condition']}: one replicate of {r['n']} moves its "
                  f"{est} by {r['influence']:.1f} {res['unit']}, against a "
                  f"spread of {res['spread']:.1f} across the whole panel. "
                  f"Check that replicate's bleach depth: a cell that did not "
                  f"bleach divides by nearly zero.")
    if res["estimators_disagree"]:
        print("  ! mean and median do not agree on the sign or the "
              "significance of this correlation. Neither should be quoted "
              "without the other.")
    distinct = len({r["covariate"] for r in res["rows"]})
    if distinct < 3:
        print(f"  ! {label} takes only {distinct} distinct value(s) across the "
              "panel; a rank correlation over that is a two-group comparison "
              "and is better reported as one.")


def _write_covariate(res: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"covariate_{res['metric']}"
    fp = out_dir / f"{stem}_values.csv"
    with open(fp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res["rows"][0]))
        w.writeheader()
        w.writerows(res["rows"])
    alt, rob = res["alternate"] or {}, res["robust"] or {}
    test = dict(label=res["label"], metric=res["metric"],
                estimator=res["estimator"], n=res["n"], rho=res["rho"],
                p=res["p"], p_floor=res["p_floor"],
                underpowered=res["underpowered"], spread=res["spread"],
                alt_estimator=alt.get("estimator"), alt_rho=alt.get("rho"),
                alt_p=alt.get("p"),
                estimators_disagree=res["estimators_disagree"],
                robust_rho=rob.get("rho"), robust_p=rob.get("p"),
                fragile="|".join(rob.get("conditions", [])),
                reference=res["reference"])
    fq = out_dir / f"{stem}_test.csv"
    with open(fq, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(test))
        w.writeheader()
        w.writerow(test)
    print(f"  [covariate] written to {fp} and {fq}")
