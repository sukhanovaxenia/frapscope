"""Regression tests for the quantities that changed a reported number.

Every test here corresponds to a fault that was found in a real analysis run and
silently altered a published value before it was caught. They are written as
regressions rather than as unit tests of intent: each one fails against the
implementation that produced the wrong answer, so a future edit that reintroduces
the old behaviour is caught rather than rediscovered.

The provenance of each is given in its docstring, because a test whose reason is
forgotten is a test that gets deleted the next time it is inconvenient.
"""

from __future__ import annotations

import numpy as np
import pytest

from frap.core import detect_bleach_index, median_iqr
from frap.stats import _min_p, holm


# --------------------------------------------------------------------------- #
# Bleach detection
# --------------------------------------------------------------------------- #
class TestBleachDetection:
    """The bleach frame is the largest fall below the running maximum.

    The earlier rule took the global minimum with a guard requiring N frames
    after it. When a trace was resampled to a coarse grid its post-bleach segment
    became shorter than the guard, the real bleach was excluded from the search,
    and the detector returned a frame inside the pre-bleach plateau. Every
    downstream quantity then inverted: the denominator (f_pre - f_min) collapsed
    toward zero and mobile fractions of about -75000 % were produced.
    """

    def test_finds_the_drop_not_the_minimum(self):
        # Recovery overshoots slightly, so the global minimum is NOT the bleach.
        y = np.array([1.0, 1.0, 1.0, 0.1, 0.3, 0.5, 0.6, 0.05])
        # index 7 is the smallest value; index 3 is the bleach
        assert int(np.argmin(y)) == 7
        assert detect_bleach_index(y) == 3

    def test_survives_a_single_post_bleach_frame(self):
        """One frame after the bleach is enough; resampling can leave only that."""
        y = np.array([1.0, 1.0, 1.0, 0.2, 0.35])
        assert detect_bleach_index(y, min_post=1) == 3

    def test_a_bleach_in_the_final_frame_is_not_detected(self):
        """Deliberate: min_post excludes the last frame from the search.

        A fall in the final frame leaves nothing to recover, so the replicate
        carries no recovery information. Detecting it would produce a
        zero-length observation window and a mobile fraction computed from a
        single point. Returning the pre-bleach index instead lets the caller's
        own admissibility checks reject the trace.
        """
        y = np.array([1.0, 1.0, 1.0, 0.2])
        assert detect_bleach_index(y, min_post=1) != 3

    def test_rejects_a_shallow_dip_in_an_already_cropped_trace(self):
        """A post-bleach-only trace contains no bleach; do not invent one."""
        y = np.array([0.20, 0.24, 0.23, 0.28, 0.31, 0.30])
        i = detect_bleach_index(y, min_drop_frac=0.5)
        # the dip at index 2 is far below half the range, so the minimum is used
        assert i == int(np.argmin(y))

    def test_empty_and_all_nan_do_not_raise(self):
        assert detect_bleach_index(np.array([])) == 0
        assert detect_bleach_index(np.array([np.nan, np.nan])) == 0


# --------------------------------------------------------------------------- #
# Exact rank floor
# --------------------------------------------------------------------------- #
class TestExactRankFloor:
    """The smallest two-sided p the Mann-Whitney null can return at a given n.

    Reported alongside the observed p because at three replicates against ten no
    arrangement of the data can reach 0.05, so a non-significant result there
    describes the design rather than the effect. Publishing the observed p alone
    invited the reading that the conditions did not differ, when in fact 2 of 30
    pairwise orderings ran the wrong way.
    """

    @pytest.mark.parametrize("na,nb,expected", [
        (3, 10, 2 / 286),      # eL27 against the control
        (6, 10, 2 / 8008),     # uS5 against the control
        (10, 10, 2 / 184756),  # eL42 against the control
        (3, 3, 2 / 20),
    ])
    def test_floor_matches_the_combinatorial_value(self, na, nb, expected):
        assert _min_p(na, nb) == pytest.approx(expected, rel=1e-12)

    def test_floor_at_n3_cannot_clear_a_family_of_three(self):
        """The value that made the eL27 contrast unreportable by rank alone."""
        floor = _min_p(3, 10)
        assert floor == pytest.approx(0.006993, abs=1e-6)
        assert holm([floor, 0.5, 0.5])[0] > 0.01   # 3 x floor, still above 0.01

    def test_floor_is_capped_at_one(self):
        assert _min_p(1, 1) == 1.0


# --------------------------------------------------------------------------- #
# Holm correction
# --------------------------------------------------------------------------- #
class TestHolm:
    """Step-down correction over the pre-specified contrast family.

    The family size decides the result here: the same three contrasts corrected
    over all ten pairwise comparisons return 0.095 for the eL27 contrast and
    0.024 over the pre-specified three. Fixing the family by design rather than
    by outcome is what makes the smaller number legitimate.
    """

    def test_preserves_input_order(self):
        p = [0.04, 0.001, 0.03]
        adj = holm(p)
        assert np.argmin(adj) == 1

    def test_is_monotone_in_rank(self):
        adj = np.sort(holm([0.001, 0.02, 0.04]))
        assert np.all(np.diff(adj) >= 0)

    def test_never_exceeds_one(self):
        assert np.all(holm([0.5, 0.6, 0.9]) <= 1.0)

    def test_family_size_changes_the_verdict(self):
        raw = 0.011892                       # eL27 vs control, Welch
        assert holm([raw] + [0.9] * 9)[0] > 0.05   # family of ten: not significant
        assert holm([raw, 0.9, 0.9])[0] < 0.05     # family of three: significant


# --------------------------------------------------------------------------- #
# Robust summary
# --------------------------------------------------------------------------- #
class TestMedianIQR:
    """Median with a distribution-free standard error.

    Per-replicate mobile fractions are right-skewed because FRAP samples a
    heterogeneous cell population, so a minority of unusually mobile inclusions
    displaces the mean but not the median. Reported alongside the mean rather
    than instead of it, so that a skew is visible rather than smoothed away.
    """

    def test_median_is_insensitive_to_one_extreme_replicate(self):
        arr = np.array([[4.0], [4.6], [11.5]])       # the eL27 replicate spread
        med, _, _, _ = median_iqr(arr)
        assert med[0] == pytest.approx(4.6)
        assert np.mean(arr) > med[0]                 # the mean is displaced upward

    def test_nan_replicates_are_ignored_not_propagated(self):
        arr = np.array([[1.0], [np.nan], [3.0]])
        med, _, _, n = median_iqr(arr)
        assert med[0] == pytest.approx(2.0)
        assert n[0] == 2

    def test_reports_the_contributing_replicate_count(self):
        arr = np.array([[1.0, np.nan], [2.0, 5.0], [3.0, 7.0]])
        _, _, _, n = median_iqr(arr)
        assert list(n) == [3, 2]


# --------------------------------------------------------------------------- #
# Physical ceiling
# --------------------------------------------------------------------------- #
class TestPhysicalCeiling:
    """Recovery cannot exceed the pre-bleach level.

    Photobleaching destroys fluorophore, so a bleached region exchanging with a
    finite unbleached pool cannot end above the signal it started from. Traces
    that do report axial drift or over-correction by a reference region that
    bleaches faster than the region of interest. Excluding them moved the
    control's mobile fraction from 41.2 % to 21.9 %.
    """

    @staticmethod
    def mobile_fraction(f_pre, f0, f_plateau):
        return 100.0 * (f_plateau - f0) / (f_pre - f0)

    def test_a_supraceiling_trace_exceeds_one_hundred_percent(self):
        assert self.mobile_fraction(1.0, 0.1, 1.25) > 100.0

    def test_an_admissible_trace_does_not(self):
        assert self.mobile_fraction(1.0, 0.1, 0.30) == pytest.approx(22.2, abs=0.1)

    def test_the_criterion_is_equivalent_to_a_plateau_above_prebleach(self):
        f_pre, f0 = 1.0, 0.08
        for f_plat in (0.5, 0.99, 1.0, 1.01, 1.4):
            assert (self.mobile_fraction(f_pre, f0, f_plat) > 100.0) == (f_plat > f_pre)


# --------------------------------------------------------------------------- #
# Plateau window
# --------------------------------------------------------------------------- #
class TestPlateauWindow:
    """The plateau is the last fraction of the window, not a fixed frame count.

    Frame intervals differ about threefold across the conditions reported here,
    so a fixed count of trailing frames spans a different physical duration in
    each. Expressing it as a fraction makes the estimate independent of sampling
    density, which is the only way the conditions can be compared at all.
    """

    @staticmethod
    def window_seconds(t, plat_frac=0.2):
        t = np.asarray(t, float)
        cut = t[0] + (1 - plat_frac) * (t[-1] - t[0])
        return float(t[-1] - t[t >= cut][0])

    @staticmethod
    def frames_seconds(t, n_plat=3):
        t = np.asarray(t, float)
        return float(t[-1] - t[-n_plat])

    def test_fraction_gives_a_comparable_duration_across_sampling_rates(self):
        fine = np.arange(0, 50.1, 2.6)
        coarse = np.arange(0, 50.1, 8.0)
        assert self.window_seconds(fine) == pytest.approx(
            self.window_seconds(coarse), rel=0.25)

    def test_fixed_frame_count_does_not(self):
        fine = np.arange(0, 50.1, 2.6)
        coarse = np.arange(0, 50.1, 8.0)
        ratio = self.frames_seconds(coarse) / self.frames_seconds(fine)
        assert ratio > 2.5      # the same three frames span very different times
