"""Correlating a FRAP metric with a number measured outside the pipeline."""

import csv
import itertools

import matplotlib
import numpy as np
import pytest
from scipy import stats as st

matplotlib.use("Agg")

from frapscope.cli import _parse_covariate
from frapscope.stats import (
    _leave_one_out,
    correlate_with_covariate,
    spearman_exact,
)
from frapscope.viz import _jitter_ties, _place_labels, plot_covariate_scatter

# The ribosomal panel as acquired in the 131126 series: covariate is the number
# of consensus APR regions, values are the per-condition mobile fractions.
PANEL = {"eS2": (5, 19.86, 7.70, 10), "eL27": (3, 18.83, 11.41, 12),
         "eS12": (3, 19.68, 12.42, 10), "eS28": (0, 11.24, 9.96, 12),
         "eL36": (0, 24.41, 11.94, 14)}


def reps(mean, sd, n, seed=0):
    """n values with exactly this mean and sample sd, so tests are exact."""
    v = np.random.default_rng(seed).normal(0, 1, n)
    v = (v - v.mean()) / v.std(ddof=1)
    return mean + sd * v


def panel(with_failed_bleach=True):
    s = {k: {"mf_per_rep": reps(m, sd, n, seed=i)}
         for i, (k, (_, m, sd, n)) in enumerate(PANEL.items())}
    s["GFP"] = {"mf_per_rep": reps(24.85, 4.75, 10, seed=9)}
    tail = [-138.0] if with_failed_bleach else [26.2]
    s["eS13"] = {"mf_per_rep": np.append(reps(26.2, 10.0, 11, seed=7), tail)}
    return s


def covariate():
    return dict({k: v[0] for k, v in PANEL.items()}, eS13=3)


# ------------------------------------------------------- exact Spearman ---

def test_exact_p_differs_from_the_t_approximation():
    """scipy's p is asymptotic; at k = 6 the difference shows in the result.

    This is the error the function exists to prevent, so it is pinned: the
    permutation p is 528/720, which no t-distribution returns.
    """
    x = [5, 3, 3, 3, 0, 0]
    y = [19.86, 18.83, 19.68, 12.56, 24.41, 11.24]
    rho, p, floor = spearman_exact(x, y)
    assert rho == pytest.approx(st.spearmanr(x, y).statistic)
    assert p == pytest.approx(528 / 720)
    assert p != pytest.approx(st.spearmanr(x, y).pvalue, abs=1e-3)
    assert floor == pytest.approx(24 / 720)


def test_floor_is_the_smallest_p_any_arrangement_could_give():
    x = [5, 3, 3, 3, 0, 0]
    best = min(spearman_exact(x, list(perm))[1]
               for perm in itertools.permutations(range(6)))
    assert spearman_exact(x, [6, 5, 4, 3, 2, 1])[2] == pytest.approx(best)


def test_ties_raise_the_floor():
    """Without ties k = 6 reaches 2/720; the panel's 5/3/3/3/0/0 reaches 0.033."""
    untied = spearman_exact([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])[2]
    tied = spearman_exact([5, 3, 3, 3, 0, 0], [1, 2, 3, 4, 5, 6])[2]
    assert untied == pytest.approx(2 / 720)
    assert tied > untied


def test_four_conditions_cannot_be_significant():
    rho, p, floor = spearman_exact([1, 2, 3, 4], [1, 2, 3, 4])
    assert rho == pytest.approx(1.0)
    assert floor == pytest.approx(2 / 24) and floor > 0.05
    assert p >= floor


# -------------------------------------------------------- leave-one-out ---

def test_leave_one_out_finds_the_failed_bleach():
    v = np.append(reps(26.2, 10.0, 11, seed=7), -138.0)
    influence, loo, idx = _leave_one_out(v, "mean")
    assert idx == 11
    assert loo == pytest.approx(26.2, abs=1e-6)
    assert influence == pytest.approx(abs(v.mean() - 26.2), abs=1e-6)


def test_the_median_is_not_moved_by_one_extreme_replicate():
    v = np.append(reps(26.2, 10.0, 11, seed=7), -138.0)
    assert _leave_one_out(v, "median")[0] < _leave_one_out(v, "mean")[0]


# ------------------------------------------------------- the correlation ---

def test_panel_reproduces_the_hand_computation():
    res = correlate_with_covariate(panel(), covariate(), reference="GFP")
    assert res["n"] == 6
    assert res["rho"] == pytest.approx(0.185, abs=0.002)
    assert res["p"] == pytest.approx(0.7333, abs=0.002)
    assert res["p_floor"] == pytest.approx(0.0333, abs=0.002)


def test_the_reference_is_kept_out_of_the_correlation():
    res = correlate_with_covariate(panel(), covariate(), reference="GFP")
    assert "GFP" not in {r["condition"] for r in res["rows"]}
    # and with no covariate value it would be dropped anyway, without warning
    res2 = correlate_with_covariate(panel(), covariate())
    assert res2["n"] == res["n"]


def test_the_failed_bleach_is_flagged_and_the_result_recomputed_without_it():
    res = correlate_with_covariate(panel(), covariate(), reference="GFP")
    fragile = [r["condition"] for r in res["rows"] if r["fragile"]]
    assert fragile == ["eS13"]
    assert res["robust"]["rho"] == pytest.approx(0.216, abs=0.002)
    assert res["robust"]["p"] == pytest.approx(0.6667, abs=0.002)
    # the conclusion is the same either way, which is the point of reporting both
    assert res["p"] > 0.05 and res["robust"]["p"] > 0.05


def test_a_clean_panel_flags_nothing():
    res = correlate_with_covariate(panel(with_failed_bleach=False),
                                   covariate(), reference="GFP")
    assert not any(r["fragile"] for r in res["rows"])
    assert res["robust"] is None


def test_underpowered_is_set_when_no_arrangement_could_reach_005():
    small = {k: {"mf_per_rep": reps(m, sd, n)} for k, (_, m, sd, n)
             in list(PANEL.items())[:4]}
    res = correlate_with_covariate(small, {k: v[0] for k, v in PANEL.items()})
    assert res["n"] == 4
    assert res["underpowered"] and res["p_floor"] > 0.05


def test_conditions_without_a_covariate_value_are_dropped_not_guessed():
    cov = covariate()
    cov.pop("eL36")
    res = correlate_with_covariate(panel(), cov, reference="GFP")
    assert res["n"] == 5
    assert "eL36" not in {r["condition"] for r in res["rows"]}


def test_unknown_metric_names_the_alternative():
    with pytest.raises(ValueError, match="replicate_table"):
        correlate_with_covariate(panel(), covariate(), metric="t_half")


def test_csv_outputs_carry_the_test_as_well_as_the_values(tmp_path):
    correlate_with_covariate(panel(), covariate(), reference="GFP",
                             out_dir=tmp_path)
    values = list(csv.DictReader(
        open(tmp_path / "covariate_mobile_fraction_values.csv")))
    test = list(csv.DictReader(
        open(tmp_path / "covariate_mobile_fraction_test.csv")))
    assert len(values) == 6 and len(test) == 1
    assert test[0]["fragile"] == "eS13"
    assert float(test[0]["p_floor"]) == pytest.approx(0.0333, abs=0.002)


# -------------------------------------------------------------- plotting ---

def test_every_condition_keeps_its_label(tmp_path):
    """A label accepted at the first candidate offset used to be removed again."""
    res = correlate_with_covariate(panel(), covariate(), reference="GFP")
    fig_dir = tmp_path / "fig"
    plot_covariate_scatter(panel(), covariate(), fig_dir, reference="GFP",
                           result=res)
    assert (fig_dir / "covariate_mobile_fraction.png").exists()
    assert (fig_dir / "covariate_mobile_fraction.svg").exists()


def test_labels_are_placed_without_overlapping(tmp_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.set_xlim(-1, 6)
    ax.set_ylim(0, 30)
    items = [(x, y, f"cond{i} (n=12)", "k")
             for i, (x, y) in enumerate([(0, 10), (0.2, 11), (3, 20),
                                         (3.2, 21), (5, 15)])]
    for x, y, _, _ in items:
        ax.plot([x], [y], "o")
    fig.tight_layout()
    placed = _place_labels(ax, items, [])
    assert len(placed) == len(items)
    assert all(a in ax.texts for a in placed)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [a.get_window_extent(renderer=r) for a in placed]
    for a, b in itertools.combinations(boxes, 2):
        assert not a.overlaps(b)
    plt.close(fig)


def test_jitter_moves_ties_only_and_keeps_the_mean():
    out = _jitter_ties([0, 0, 3, 3, 3, 5])
    assert out[5] == 5
    assert len(set(out[:2])) == 2 and len(set(out[2:5])) == 3
    assert np.mean(out[2:5]) == pytest.approx(3.0)


# ------------------------------------------------------------------ cli ---

def test_covariate_parses_pairs_and_csv(tmp_path):
    fp = tmp_path / "regions.csv"
    fp.write_text("condition,regions\neS2,5\n# a comment line\neL27,3\n")
    cov = _parse_covariate([str(fp), "eS28=0"])
    assert cov == {"eS2": 5.0, "eL27": 3.0, "eS28": 0.0}


def test_covariate_rejects_a_path_that_is_not_there():
    with pytest.raises(SystemExit, match="no such file"):
        _parse_covariate(["nowhere.csv"])
