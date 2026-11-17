"""Compare two runs of the same FRAP panel: does the second reproduce the first?

Two runs -- two sessions, two extraction routes, or both -- are each described
by a config file, re-extracted from their raw data and reduced to per-replicate
metrics with exactly the steps ``frapscope`` uses. Conditions are paired by
display name, or through an explicit mapping where the names differ.

Two questions are asked, and they need different statistics.

**Has a condition shifted between runs?** Its replicates in run A and run B are
two independent samples, so a two-sample test applies: the exact
Kolmogorov-Smirnov test, whose statistic D is the largest vertical gap between
the two empirical distribution functions (0 = identical, 1 = no overlap). It
is a distance between distributions, not a correlation coefficient. The size
and direction of the shift are reported as the difference of medians and the
rank-biserial r (+1: every B replicate above every A replicate). One test per
condition and metric; the family is corrected with Benjamini-Hochberg.

**Is the pattern across conditions reproduced?** Here each condition is one
point (its median in A against its median in B), so the sample size is the
number of conditions, typically four or five. Correlation is computed
accordingly: Spearman's rho with an exact permutation p-value and the smallest
p that k conditions can produce, Lin's concordance correlation coefficient --
which, unlike a correlation, is 1 only when B equals A, not merely when B rises
with A -- and an ordinary least-squares line with a confidence interval for its
slope. Uncertainty from finite replicates enters through a bootstrap that
resamples replicates within each condition. One test per metric, corrected
with Benjamini-Hochberg.

The mobile fraction depends on how long recovery was observed. By default each
pair is truncated to the shorter of its two observation windows
(``--window matched``); bleach depth does not depend on the window at all.

    frapscope-compare --config-a old.py --config-b new.py \\
        --label-a "Aug 2026 (.lif)" --label-b "Nov 2026 (LAS X CSV)" \\
        --map eL42=eL36 --exclude GFP --out crossrun/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats as st

from .config import ConfigError, load_conditions
from .core import drop_supraceiling, replicate_table
from .stats import _min_p, bh, spearman_exact
from .timebase import condition_window, truncate

METRICS = {
    "mobile_fraction": ("Mobile fraction", "%"),
    "bleach_depth": ("Bleach depth", "%"),
    "t_half": ("t½", "s"),
}
COLOR_A, COLOR_B = "#607D8B", "#E8505B"


# --------------------------------------------------------------- loading ---

def load_run(config, n_pre=3, n_plat=3):
    """Datasets of one run after the same ceiling filter ``frapscope`` applies."""
    from . import _load_all

    datasets, provenance = _load_all(load_conditions(config))
    out = {}
    for name, ad in datasets.items():
        kept, _ = drop_supraceiling(ad, n_plat=n_plat, n_pre=n_pre, name=name)
        if kept:
            out[name] = kept
    return out, provenance


def pair_conditions(names_a, names_b, mapping=None, exclude=()):
    """Pairs (a, b) in run-A order; names left without a partner on each side."""
    mapping = dict(mapping or {})
    exclude = set(exclude)
    pairs, used_b = [], set()
    for a in names_a:
        b = mapping.get(a, a)
        if a in exclude or b in exclude:
            continue
        if b in names_b:
            pairs.append((a, b))
            used_b.add(b)
    only_a = [a for a in names_a if a not in {p[0] for p in pairs} and a not in exclude]
    only_b = [b for b in names_b if b not in used_b and b not in exclude]
    return pairs, only_a, only_b


def pair_label(a, b):
    return a if a == b else f"{a} / {b}"


# ------------------------------------------------------------ statistics ---

def rank_biserial(a, b):
    """+1 when every value of ``b`` exceeds every value of ``a``, -1 the reverse."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = st.mannwhitneyu(b, a, alternative="two-sided").statistic
    return float(2.0 * u / (a.size * b.size) - 1.0)


def ks_exact(a, b):
    """Two-sample KS statistic D and exact two-sided p, NaN if untestable."""
    a = np.asarray(a, float)[np.isfinite(a)]
    b = np.asarray(b, float)[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan, np.nan
    r = st.ks_2samp(a, b, alternative="two-sided", method="exact")
    return float(r.statistic), float(r.pvalue)


def lin_ccc(x, y):
    """Lin's concordance correlation coefficient: agreement with the identity."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2:
        return np.nan
    sxy = np.mean((x - x.mean()) * (y - y.mean()))
    den = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    return float(2 * sxy / den) if den > 0 else np.nan


def ols(x, y):
    """Slope, intercept, R^2 and the 95 % CI of the slope (t, k - 2 df)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    k = x.size
    if k < 3 or np.ptp(x) == 0:
        return dict(slope=np.nan, intercept=np.nan, r2=np.nan,
                    slope_lo=np.nan, slope_hi=np.nan)
    r = st.linregress(x, y)
    h = st.t.ppf(0.975, k - 2) * r.stderr
    return dict(slope=float(r.slope), intercept=float(r.intercept),
                r2=float(r.rvalue ** 2), slope_lo=float(r.slope - h),
                slope_hi=float(r.slope + h))


def bootstrap_agreement(vals_a, vals_b, n_boot=2000, seed=0):
    """Percentile 95 % CIs of Spearman rho and CCC over replicate resampling."""
    rng = np.random.default_rng(seed)
    rhos, cccs = [], []
    for _ in range(n_boot):
        xa = [np.median(rng.choice(v, v.size)) for v in vals_a]
        xb = [np.median(rng.choice(v, v.size)) for v in vals_b]
        if np.ptp(xa) > 0 and np.ptp(xb) > 0:
            rhos.append(st.spearmanr(xa, xb).statistic)
        cccs.append(lin_ccc(xa, xb))

    def ci(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) \
            if v.size else (np.nan, np.nan)

    return ci(rhos), ci(cccs)


# ------------------------------------------------------------ comparison ---

def compare_runs(runs_a, runs_b, pairs, window="matched", min_coverage=0.75,
                 n_pre=3, n_plat=3, n_boot=2000, seed=0, metrics=METRICS):
    """Per-replicate tables, per-condition tests and cross-condition agreement."""
    reps, pair_rows = [], []
    for a, b in pairs:
        da, db = runs_a[a], runs_b[b]
        wa, wb = condition_window(da, min_coverage), condition_window(db, min_coverage)
        w = min(wa, wb) if window == "matched" else np.nan
        if window == "matched":
            da = truncate(da, w, name=f"A:{a}")
            db = truncate(db, w, name=f"B:{b}")
        ta = replicate_table(da, n_plat=n_plat, n_pre=n_pre)
        tb = replicate_table(db, n_plat=n_plat, n_pre=n_pre)
        label = pair_label(a, b)
        for run, cond, rows in (("A", a, ta), ("B", b, tb)):
            for r in rows:
                reps.append(dict(pair=label, run=run, condition=cond, **r))
        pair_rows.append(dict(
            pair=label, condition_a=a, condition_b=b, n_a=len(ta), n_b=len(tb),
            window_a_s=wa, window_b_s=wb, window_used_s=w,
            provenance_a="|".join(sorted({r["provenance"] for r in ta})),
            provenance_b="|".join(sorted({r["provenance"] for r in tb}))))

    labels = [p["pair"] for p in pair_rows]

    def values(label, run, metric):
        v = np.array([r[metric] for r in reps if r["pair"] == label and r["run"] == run], float)
        return v[np.isfinite(v)]

    dist = []
    for label in labels:
        for m in metrics:
            a, b = values(label, "A", m), values(label, "B", m)
            d, p = ks_exact(a, b)
            dist.append(dict(
                pair=label, metric=m, n_a=a.size, n_b=b.size,
                median_a=float(np.median(a)) if a.size else np.nan,
                median_b=float(np.median(b)) if b.size else np.nan,
                delta_median=(float(np.median(b) - np.median(a))
                              if a.size and b.size else np.nan),
                rank_biserial=rank_biserial(a, b) if a.size and b.size else np.nan,
                ks_D=d, ks_p=p,
                p_floor=_min_p(a.size, b.size) if a.size >= 2 and b.size >= 2 else np.nan))
    q = bh([r["ks_p"] for r in dist])
    for r, qq in zip(dist, q, strict=True):
        r["ks_q_bh"] = float(qq)

    cross = []
    for m in metrics:
        keep = [lab for lab in labels if values(lab, "A", m).size and values(lab, "B", m).size]
        va = [values(lab, "A", m) for lab in keep]
        vb = [values(lab, "B", m) for lab in keep]
        x = np.array([np.median(v) for v in va])
        y = np.array([np.median(v) for v in vb])
        rho, p, floor = spearman_exact(x, y)
        (rlo, rhi), (clo, chi) = (bootstrap_agreement(va, vb, n_boot, seed)
                                  if len(keep) >= 3 else ((np.nan,) * 2,) * 2)
        cross.append(dict(
            metric=m, k=len(keep), conditions="|".join(keep),
            spearman_rho=rho, spearman_p=p, spearman_p_floor=floor,
            rho_boot_lo=rlo, rho_boot_hi=rhi,
            pearson_r=(float(np.corrcoef(x, y)[0, 1])
                       if len(keep) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan),
            ccc=lin_ccc(x, y) if len(keep) >= 2 else np.nan,
            ccc_boot_lo=clo, ccc_boot_hi=chi, **ols(x, y)))
    q = bh([r["spearman_p"] for r in cross])
    for r, qq in zip(cross, q, strict=True):
        r["spearman_q_bh"] = float(qq)
    return dict(replicates=reps, pairs=pair_rows, distribution=dist, cross=cross)


# --------------------------------------------------------------- figures ---

def _stars(q):
    if not np.isfinite(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def _fq(q):
    return "< 0.001" if np.isfinite(q) and q < 0.001 else f"= {q:.3f}"


def _save(fig, out, stem):
    fig.savefig(out / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")


def plot_shift_heatmap(res, out, label_a, label_b, metrics=METRICS):
    """Rows = conditions, columns = metrics; colour = rank-biserial shift A -> B."""
    import matplotlib.pyplot as plt
    labels = [p["pair"] for p in res["pairs"]]
    ms = list(metrics)
    cell = {(r["pair"], r["metric"]): r for r in res["distribution"]}
    M = np.array([[cell[(lab, m)]["rank_biserial"] for m in ms] for lab in labels], float)
    fig, ax = plt.subplots(figsize=(1.9 * len(ms) + 2.2, 0.62 * len(labels) + 1.6))
    im = ax.imshow(np.ma.masked_invalid(M), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i, lab in enumerate(labels):
        for j, m in enumerate(ms):
            r = cell[(lab, m)]
            if not np.isfinite(r["rank_biserial"]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="0.5")
                continue
            unit = metrics[m][1]
            txt = (f"Δ {r['delta_median']:+.1f} {unit}\n"
                   f"q {_fq(r['ks_q_bh'])}{_stars(r['ks_q_bh'])}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if abs(r["rank_biserial"]) > 0.6 else "black")
    ax.set_xticks(range(len(ms)), [metrics[m][0] for m in ms])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(f"Shift from run A to run B\nA = {label_a};  B = {label_b}\n"
                 "Δ = difference of medians (B − A); q = exact KS, Benjamini-Hochberg",
                 fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("rank-biserial r  (+: higher in B)")
    _save(fig, out, "crossrun_shift_heatmap")
    plt.close(fig)


def plot_pattern_heatmap(res, out, label_a, label_b, metrics=METRICS):
    """Condition medians, z-scored within each run: is the ranking preserved?"""
    import matplotlib.pyplot as plt
    labels = [p["pair"] for p in res["pairs"]]
    ms = list(metrics)
    cell = {(r["pair"], r["metric"]): r for r in res["distribution"]}
    cols, raw = [], []
    for m in ms:
        for key in ("median_a", "median_b"):
            v = np.array([cell[(lab, m)][key] for lab in labels], float)
            raw.append(v)
            sd = np.nanstd(v)
            cols.append((v - np.nanmean(v)) / sd if sd > 0 else v * np.nan)
    Z, R = np.array(cols).T, np.array(raw).T
    fig, ax = plt.subplots(figsize=(1.25 * Z.shape[1] + 2.2, 0.55 * len(labels) + 1.9))
    im = ax.imshow(np.ma.masked_invalid(Z), cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if np.isfinite(R[i, j]):
                ax.text(j, i, f"{R[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if np.isfinite(Z[i, j]) and abs(Z[i, j]) > 1.4 else "black")
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="0.5")
    for j in range(2, Z.shape[1], 2):
        ax.axvline(j - 0.5, color="white", lw=3)
    ax.set_xticks(range(Z.shape[1]), [f"{metrics[m][0]}\n{run}" for m in ms for run in ("A", "B")],
                  fontsize=8)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(f"Condition medians, z-scored within each run\nA = {label_a};  B = {label_b}",
                 fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("z within run (numbers: medians)")
    _save(fig, out, "crossrun_pattern_heatmap")
    plt.close(fig)


def plot_regression(res, out, label_a, label_b, metrics=METRICS):
    """Median in B against median in A, one panel per metric."""
    import matplotlib.pyplot as plt
    ms = list(metrics)
    reps = res["replicates"]
    fig, axes = plt.subplots(1, len(ms), figsize=(4.6 * len(ms), 5.8), squeeze=False)
    cross = {r["metric"]: r for r in res["cross"]}
    for ax, m in zip(axes[0], ms, strict=True):
        name, unit = metrics[m]
        c = cross[m]
        keep = c["conditions"].split("|") if c["conditions"] else []
        xs, ys = [], []
        for lab in keep:
            a = np.array([r[m] for r in reps if r["pair"] == lab and r["run"] == "A"], float)
            b = np.array([r[m] for r in reps if r["pair"] == lab and r["run"] == "B"], float)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            x, y = np.median(a), np.median(b)
            xe = [[x - np.percentile(a, 25)], [np.percentile(a, 75) - x]]
            ye = [[y - np.percentile(b, 25)], [np.percentile(b, 75) - y]]
            ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", color="0.2", ms=6,
                        ecolor="0.6", elinewidth=1, capsize=2)
            ax.annotate(lab, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
            xs.append(x)
            ys.append(y)
        if xs:
            lo = min(min(xs), min(ys))
            hi = max(max(xs), max(ys))
            pad = 0.15 * (hi - lo or 1)
            g = np.linspace(lo - pad, hi + pad, 50)
            ax.plot(g, g, "--", color="0.7", lw=1, label="identity")
            if np.isfinite(c["slope"]):
                xa, k = np.array(xs), len(xs)
                yhat = c["intercept"] + c["slope"] * g
                resid = np.array(ys) - (c["intercept"] + c["slope"] * xa)
                s = np.sqrt(np.sum(resid ** 2) / (k - 2)) if k > 2 else np.nan
                se = s * np.sqrt(1 / k + (g - xa.mean()) ** 2 / np.sum((xa - xa.mean()) ** 2))
                tq = st.t.ppf(0.975, k - 2)
                ax.plot(g, yhat, color=COLOR_B, lw=1.5, label="OLS")
                ax.fill_between(g, yhat - tq * se, yhat + tq * se, color=COLOR_B, alpha=0.12)
            ax.set_xlim(g[0], g[-1])
            ax.set_ylim(g[0], g[-1])
        if c["k"] >= 3:
            txt = (f"k = {c['k']} conditions\n"
                   f"Spearman ρ = {c['spearman_rho']:.2f} "
                   f"[{c['rho_boot_lo']:.2f}, {c['rho_boot_hi']:.2f}]\n"
                   f"exact p = {c['spearman_p']:.3f} (smallest possible "
                   f"{c['spearman_p_floor']:.3f})\nq {_fq(c['spearman_q_bh'])}\n"
                   f"Lin's CCC = {c['ccc']:.2f} [{c['ccc_boot_lo']:.2f}, {c['ccc_boot_hi']:.2f}]\n"
                   f"OLS slope = {c['slope']:.2f} [{c['slope_lo']:.2f}, {c['slope_hi']:.2f}]")
        else:
            txt = (f"k = {c['k']} conditions with values in both runs:\n"
                   "too few for agreement statistics")
        ax.text(0.0, -0.17, txt, transform=ax.transAxes, va="top", fontsize=8,
                family="monospace")
        ax.set_xlabel(f"{label_a}: {name} ({unit})")
        ax.set_ylabel(f"{label_b}: {name} ({unit})")
        ax.grid(alpha=0.3)
    fig.suptitle("Agreement across conditions: medians ± IQR, identity (dashed), "
                 "OLS with 95 % band; brackets: bootstrap 95 % CI", fontsize=10)
    fig.subplots_adjust(bottom=0.38, wspace=0.32, top=0.9)
    _save(fig, out, "crossrun_regression")
    plt.close(fig)


def plot_distributions(res, out, label_a, label_b, metrics=METRICS):
    """Every replicate, A against B, per condition and metric."""
    import matplotlib.pyplot as plt
    labels = [p["pair"] for p in res["pairs"]]
    ms = list(metrics)
    cell = {(r["pair"], r["metric"]): r for r in res["distribution"]}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(ms), len(labels), squeeze=False,
                             figsize=(2.1 * len(labels) + 0.8, 2.5 * len(ms)))
    for i, m in enumerate(ms):
        for j, lab in enumerate(labels):
            ax = axes[i, j]
            for k, (run, col) in enumerate((("A", COLOR_A), ("B", COLOR_B))):
                v = np.array([r[m] for r in res["replicates"]
                              if r["pair"] == lab and r["run"] == run], float)
                v = v[np.isfinite(v)]
                ax.scatter(k + rng.uniform(-0.12, 0.12, v.size), v, s=18,
                           color=col, alpha=0.7, edgecolors="white", lw=0.4)
                if v.size:
                    ax.hlines(np.median(v), k - 0.25, k + 0.25, color="k", lw=1.5)
            r = cell[(lab, m)]
            if np.isfinite(r["ks_D"]):
                ax.set_title(f"D = {r['ks_D']:.2f}, q {_fq(r['ks_q_bh'])}{_stars(r['ks_q_bh'])}",
                             fontsize=7.5)
            else:
                ax.set_title("not testable", fontsize=7.5, color="0.4")
            ax.set_xticks([0, 1], ["A", "B"])
            ax.set_xlim(-0.6, 1.6)
            ax.grid(alpha=0.3, axis="y")
            if j == 0:
                ax.set_ylabel(f"{metrics[m][0]} ({metrics[m][1]})")
            if i == 0:
                ax.annotate(lab, (0.5, 1.22), xycoords="axes fraction", ha="center",
                            fontsize=9, fontweight="bold")
    note = ("t½ needs ≥ 5 post-bleach points inside the window; "
            "not testable = fewer than 2 replicates with a value in a run")
    fig.suptitle(f"A = {label_a};  B = {label_b}.  Exact KS, Benjamini-Hochberg q\n{note}",
                 fontsize=9, y=1.03)
    fig.tight_layout()
    _save(fig, out, "crossrun_distributions")
    plt.close(fig)


# ------------------------------------------------------------- reporting ---

def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def report(res, label_a, label_b):
    lines = [f"A = {label_a}", f"B = {label_b}", ""]
    for p in res["pairs"]:
        w = (f"{p['window_used_s']:.1f} s" if np.isfinite(p["window_used_s"])
             else f"native ({p['window_a_s']:.0f} / {p['window_b_s']:.0f} s)")
        lines.append(f"  {p['pair']:14s} n {p['n_a']:>2}/{p['n_b']:<2}  window {w:22s} "
                     f"{p['provenance_a']} -> {p['provenance_b']}")
    mixed = [p["pair"] for p in res["pairs"] if p["provenance_a"] != p["provenance_b"]]
    if mixed:
        lines += ["", "  ! extraction route differs between runs for: " + ", ".join(mixed),
                  "    run and route are confounded there; a shift cannot be assigned to either."]
    lines += ["", f"{'condition':14s} {'metric':16s} {'med A':>7s} {'med B':>7s} {'delta':>7s} "
              f"{'r_rb':>6s} {'D':>5s} {'p':>8s} {'q_BH':>7s}"]
    for r in res["distribution"]:
        lines.append(f"{r['pair']:14s} {r['metric']:16s} {r['median_a']:7.1f} {r['median_b']:7.1f} "
                     f"{r['delta_median']:+7.1f} {r['rank_biserial']:+6.2f} {r['ks_D']:5.2f} "
                     f"{r['ks_p']:8.4f} {r['ks_q_bh']:7.4f}")
    lines += ["", f"{'metric':16s} {'k':>2s} {'rho':>6s} {'p':>6s} {'floor':>6s} {'q_BH':>6s} "
              f"{'CCC':>6s} {'slope':>6s}"]
    for c in res["cross"]:
        lines.append(f"{c['metric']:16s} {c['k']:2d} {c['spearman_rho']:+6.2f} {c['spearman_p']:6.3f} "
                     f"{c['spearman_p_floor']:6.3f} {c['spearman_q_bh']:6.3f} {c['ccc']:+6.2f} "
                     f"{c['slope']:+6.2f}")
    small = [c for c in res["cross"] if c["k"] and np.isfinite(c["spearman_p_floor"])
             and c["spearman_p_floor"] > 0.05]
    if small:
        lines += ["", f"  ! with k = {small[0]['k']} conditions no Spearman test can reach "
                  f"p < {small[0]['spearman_p_floor']:.3f}; read rho and CCC as descriptive."]
    return "\n".join(lines)


# ------------------------------------------------------------------- CLI ---

def _parse_map(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--map expects A=B, got {it!r}")
        a, b = it.split("=", 1)
        out[a.strip()] = b.strip()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="frapscope-compare",
        description="Compare two FRAP runs condition by condition.")
    ap.add_argument("--config-a", required=True, help="config of the reference run (A)")
    ap.add_argument("--config-b", required=True, help="config of the run compared with it (B)")
    ap.add_argument("--label-a", default="A", help="name of run A in figures")
    ap.add_argument("--label-b", default="B", help="name of run B in figures")
    ap.add_argument("--map", nargs="*", default=[], metavar="A=B",
                    help="pair conditions whose display names differ, e.g. eL42=eL36")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="conditions to leave out (name in either run)")
    ap.add_argument("--window", choices=("matched", "native"), default="matched",
                    help="truncate each pair to its shorter observation window "
                         "(default) or keep both native windows")
    ap.add_argument("--min-coverage", type=float, default=0.75,
                    help="fraction of replicates that must cover a condition's window")
    ap.add_argument("--n-pre", type=int, default=3)
    ap.add_argument("--n-plat", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="crossrun_results", help="output directory")
    args = ap.parse_args(argv)

    try:
        print(f"== run A: {args.config_a}")
        runs_a, _ = load_run(args.config_a, args.n_pre, args.n_plat)
        print(f"== run B: {args.config_b}")
        runs_b, _ = load_run(args.config_b, args.n_pre, args.n_plat)
    except ConfigError as exc:
        raise SystemExit(f"configuration error: {exc}") from None

    pairs, only_a, only_b = pair_conditions(list(runs_a), list(runs_b),
                                            _parse_map(args.map), args.exclude)
    if only_a or only_b:
        print(f"  no counterpart: A {only_a or '-'}; B {only_b or '-'}")
    if not pairs:
        raise SystemExit("no condition is present in both runs; use --map")

    res = compare_runs(runs_a, runs_b, pairs, window=args.window,
                       min_coverage=args.min_coverage, n_pre=args.n_pre,
                       n_plat=args.n_plat, n_boot=args.n_boot, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "crossrun_replicates.csv", res["replicates"])
    _write_csv(out / "crossrun_pairs.csv", res["pairs"])
    _write_csv(out / "crossrun_distribution_tests.csv", res["distribution"])
    _write_csv(out / "crossrun_cross_condition.csv", res["cross"])

    import matplotlib
    matplotlib.use("Agg")
    plot_shift_heatmap(res, out, args.label_a, args.label_b)
    plot_pattern_heatmap(res, out, args.label_a, args.label_b)
    plot_regression(res, out, args.label_a, args.label_b)
    plot_distributions(res, out, args.label_a, args.label_b)

    text = report(res, args.label_a, args.label_b)
    (out / "crossrun_report.txt").write_text(text + "\n")
    print("\n" + text + f"\n\nwritten to {out}/")
    return 0
