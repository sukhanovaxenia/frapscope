"""Comparison of two runs: statistics, pairing, windows and the CLI."""

import numpy as np
import pytest

from frapscope.core import replicate_metrics, replicate_table
from frapscope.crossrun import (
    compare_runs,
    lin_ccc,
    main,
    pair_conditions,
    spearman_exact,
)
from frapscope.stats import bh


def trace(mf, depth=0.7, tau=15.0, dt=6.0, n_post=20, noise=0.0, rng=None, name="r"):
    """Single-normalised record with the bleach recorded at t = 0."""
    t_post = np.arange(n_post) * dt
    f_min = 1 - depth
    y_post = f_min + (1 - f_min) * mf * (1 - np.exp(-t_post / tau))
    if noise and rng is not None:
        y_post = y_post + rng.normal(0, noise, n_post)
    t = np.concatenate([[-24.0, -21.4], t_post])
    y = np.concatenate([[1.0, 1.0], y_post])
    return dict(file=name, time=t, intensity=y, intensity_raw=100 * y,
                provenance="synthetic", bleach_at_t0=True)


def run(spec, seed=0, n=10, **kw):
    rng = np.random.default_rng(seed)
    return {c: [trace(mf + rng.normal(0, 0.02), depth=d, noise=0.005, rng=rng,
                      name=f"{c}_{i}", **kw) for i in range(n)]
            for c, (mf, d) in spec.items()}


# ---------------------------------------------------------------- stats ---

def test_bh_against_hand_computation():
    q = bh([0.01, 0.04, 0.03, 0.005, np.nan])
    assert q[:4] == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert np.isnan(q[4])


def test_bh_is_monotone_and_bounded():
    p = np.random.default_rng(1).uniform(size=50)
    q = bh(p)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)
    assert np.all((q >= p - 1e-12) & (q <= 1))


@pytest.mark.parametrize("k,floor", [(4, 2 / 24), (5, 2 / 120)])
def test_spearman_exact_floor(k, floor):
    rho, p, f = spearman_exact(np.arange(k), np.arange(k) ** 2)
    assert rho == pytest.approx(1.0)
    assert p == pytest.approx(floor)
    assert f == pytest.approx(floor)


def test_ccc_measures_agreement_not_association():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert lin_ccc(x, x) == pytest.approx(1.0)
    assert np.corrcoef(x, x + 2)[0, 1] == pytest.approx(1.0)
    assert lin_ccc(x, x + 2) < 0.5


# -------------------------------------------------------------- pairing ---

def test_pairing_with_map_and_exclusion():
    pairs, only_a, only_b = pair_conditions(
        ["eS2", "eL42", "eS28", "GFP"], ["eS2", "eL36", "eS28", "eS12", "GFP"],
        mapping={"eL42": "eL36"}, exclude=["GFP"])
    assert pairs == [("eS2", "eS2"), ("eL42", "eL36"), ("eS28", "eS28")]
    assert only_a == [] and only_b == ["eS12"]


# -------------------------------------------------------------- metrics ---

def test_replicate_table_agrees_with_replicate_metrics():
    ad = run({"x": (0.3, 0.7)}, n=8)["x"]
    t = replicate_table(ad)
    m = replicate_metrics(ad)
    assert [r["mobile_fraction"] for r in t] == pytest.approx(list(m["mf_per_rep"]))
    assert [r["bleach_depth"] for r in t] == pytest.approx(list(m["bd_per_rep"]))


def test_t_half_needs_enough_points():
    short = trace(0.3, n_post=3)
    assert np.isnan(replicate_table([short])[0]["t_half"])
    assert np.isfinite(replicate_table([trace(0.3)])[0]["t_half"])


# ----------------------------------------------------------- comparison ---

def test_detects_a_shift_and_leaves_a_stable_condition_alone():
    a = run({"stable": (0.30, 0.7), "moved": (0.10, 0.9)}, seed=0)
    b = run({"stable": (0.30, 0.7), "moved": (0.40, 0.6)}, seed=1)
    res = compare_runs(a, b, [("stable", "stable"), ("moved", "moved")], n_boot=50)
    d = {(r["pair"], r["metric"]): r for r in res["distribution"]}
    assert d[("moved", "mobile_fraction")]["ks_q_bh"] < 0.01
    assert d[("moved", "mobile_fraction")]["rank_biserial"] == pytest.approx(1.0)
    assert d[("moved", "bleach_depth")]["delta_median"] < -20
    assert d[("stable", "mobile_fraction")]["ks_q_bh"] > 0.05


def test_matched_window_truncates_to_the_shorter_run():
    a = run({"x": (0.3, 0.7)}, n_post=4)          # 18 s observed
    b = run({"x": (0.3, 0.7)}, n_post=20)         # 114 s observed
    res = compare_runs(a, b, [("x", "x")], n_boot=10)
    assert res["pairs"][0]["window_used_s"] == pytest.approx(18.0)
    assert max(r["window_s"] for r in res["replicates"]) == pytest.approx(18.0)
    native = compare_runs(a, b, [("x", "x")], window="native", n_boot=10)
    assert max(r["window_s"] for r in native["replicates"]) == pytest.approx(114.0)


def test_cli_writes_tables_and_figures(tmp_path):
    def export(path, mf, depth):
        t = [0.0, 2.6] + [5.7 + 2.6 * i for i in range(8)] + [27.0 + 6 * i for i in range(20)]
        y = [100.0, 100.0] + [None] * 8 + [
            100 * ((1 - depth) + depth * mf * (1 - np.exp(-6 * i / 15))) for i in range(20)]
        rows = ["Channel 1", "Axis [s];ROI1 []"] + [
            f"{ti};{'' if yi is None else yi}".replace(".", ",") for ti, yi in zip(t, y, strict=True)]
        path.write_text("\n".join(rows))

    for run_name, shift in (("a", 0.0), ("b", 0.1)):
        for cond, (mf, depth) in {"P": (0.2, 0.8), "Q": (0.3, 0.6), "R": (0.1, 0.9)}.items():
            d = tmp_path / run_name / cond
            d.mkdir(parents=True)
            for i in range(5):
                export(d / f"{i}.csv", mf + shift + 0.01 * i, depth)
        (tmp_path / run_name / "cfg.py").write_text(
            "from frapscope.config import Condition\n"
            "CONDITIONS = [Condition(display=c, loader='leica_csv', source=c)"
            " for c in ('P', 'Q', 'R')]\n")
    out = tmp_path / "out"
    main(["--config-a", str(tmp_path / "a" / "cfg.py"), "--config-b", str(tmp_path / "b" / "cfg.py"),
          "--out", str(out), "--n-boot", "20"])
    for f in ("crossrun_replicates.csv", "crossrun_distribution_tests.csv",
              "crossrun_cross_condition.csv", "crossrun_report.txt",
              "crossrun_shift_heatmap.png", "crossrun_pattern_heatmap.png",
              "crossrun_regression.png", "crossrun_distributions.png"):
        assert (out / f).exists(), f
