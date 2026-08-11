"""
frap.stats — cross-condition comparison of mobile fractions.

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
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["compare_mobile_fractions", "planned_contrasts", "holm"]


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


def _welch(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    from scipy import stats as st
    r = st.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    return float(r.statistic), float(df), float(r.pvalue)


def _mwu(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    from scipy import stats as st
    r = st.mannwhitneyu(a, b, alternative="two-sided")
    return float(r.statistic), float(r.pvalue)


def _min_p(na: int, nb: int) -> float:
    """Smallest two-sided p the exact Mann-Whitney null can return at this n."""
    from math import comb
    total = comb(na + nb, na)
    return min(1.0, 2.0 / total)


def planned_contrasts(names: Iterable[str], control: Optional[str] = None,
                      exclude: Iterable[str] = ()) -> List[Tuple[str, str]]:
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


def compare_mobile_fractions(summaries: Dict[str, dict],
                             out_dir: Optional[Path] = None,
                             contrasts: Optional[Iterable[Tuple[str, str]]] = None,
                             control: Optional[str] = None,
                             exclude: Iterable[str] = (),
                             qc_only: bool = False) -> List[dict]:
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
        fp = out_dir / "mobile_fraction_stats.csv"
        with open(fp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"  [stats] written to {fp}")
    return rows