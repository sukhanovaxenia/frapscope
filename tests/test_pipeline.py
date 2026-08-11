"""End-to-end tests on synthetic data with known ground truth.

The unit tests in ``test_regressions.py`` check functions in isolation. These
check that the stages agree with one another, which is the property that was
absent when the pipeline returned a mobile fraction of -75000 % from components
that were each individually correct.

Every expected value is computed by ``make_fixtures.expected_mobile_fraction``
from the same specification the trace was built from, so nothing here asserts
against a number a human transcribed.
"""

from __future__ import annotations

import numpy as np
import pytest

from frapscope.core import drop_supraceiling, replicate_metrics
from frapscope.loaders import load_roi_csv
from frapscope.stats import compare_mobile_fractions
from tests.fixtures.make_fixtures import expected_mobile_fraction, time_vector

#: Everything in this module writes fixture CSVs and reads them back through a
#: loader. Marked so `pytest -m "not integration"` gives the sub-second loop that
#: is worth running on every save, while the full suite runs before a commit.
pytestmark = pytest.mark.integration


def _metrics(spec, directory):
    data = load_roi_csv(directory, time_s=time_vector(spec), n_pre=spec.n_pre)
    kept, _ = drop_supraceiling(data, name=spec.name)
    return data, kept, replicate_metrics(kept, n_pre=spec.n_pre) if kept else None


class TestRoundTrip:
    """A mobile fraction put in is the mobile fraction that comes out."""

    @pytest.mark.parametrize("name", ["arrested", "partial", "control",
                                      "heterogeneous"])
    def test_recovers_the_injected_mobile_fraction(self, name, fixture_dirs, specs):
        spec = specs[name]
        _, kept, m = _metrics(spec, fixture_dirs[name])
        recovered = float(np.nanmean(m["mf_per_rep"]))
        assert recovered == pytest.approx(expected_mobile_fraction(spec), abs=0.5)

    def test_every_replicate_is_retained_when_none_is_impossible(
            self, fixture_dirs, specs):
        for name in ("arrested", "partial", "control", "heterogeneous"):
            data, kept, _ = _metrics(specs[name], fixture_dirs[name])
            assert len(kept) == len(data) == specs[name].n_rep

    def test_bleach_depth_matches_the_specification(self, fixture_dirs, specs):
        spec = specs["control"]
        _, _, m = _metrics(spec, fixture_dirs["control"])
        assert float(np.mean(m["bd_per_rep"])) == pytest.approx(
            100 * spec.bleach_depth, abs=1.0)


class TestSupraCeilingExclusion:
    """Replicates recovering above their own pre-bleach level are removed.

    Not clamped. Clamping to 100 % retains a corrupted trace at the extreme of
    the distribution and inflates both the mean and its variance, which is how
    the control's mobile fraction reached 41.2 % before this was added.
    """

    def test_all_impossible_replicates_are_excluded(self, fixture_dirs, specs):
        data, kept, _ = _metrics(specs["supraceiling"], fixture_dirs["supraceiling"])
        assert len(data) == specs["supraceiling"].n_rep
        assert kept == []

    def test_exclusion_happens_before_metrics_not_inside_them(
            self, fixture_dirs, specs):
        """drop_supraceiling must be the gate, so every consumer sees one set."""
        spec = specs["supraceiling"]
        data = load_roi_csv(fixture_dirs["supraceiling"],
                            time_s=time_vector(spec), n_pre=spec.n_pre)
        m_all = replicate_metrics(data, n_pre=spec.n_pre)
        # replicate_metrics also refuses them, so the two agree rather than
        # one silently admitting what the other rejects
        assert m_all["n_excluded"] == spec.n_rep
        assert len(m_all["mf_per_rep"]) == 0


class TestSamplingInvariance:
    """Two conditions sampled at different intervals stay comparable.

    ``arrested`` is sampled at 2.6 s and ``partial`` at 6.0 s. A plateau taken
    as a fixed frame count spans 5.2 s in one and 12 s in the other, which pulls
    the coarsely sampled condition into its own rising phase. The fractional
    window removes that dependence, and this test fails if it is reverted.
    """

    def test_ordering_is_preserved_across_sampling_rates(
            self, fixture_dirs, specs):
        vals = {}
        for name in ("arrested", "partial", "control"):
            _, _, m = _metrics(specs[name], fixture_dirs[name])
            vals[name] = float(np.nanmean(m["mf_per_rep"]))
        assert vals["arrested"] < vals["partial"] < vals["control"]

    def test_the_coarse_condition_is_not_understated(self, fixture_dirs, specs):
        spec = specs["partial"]
        _, _, m = _metrics(spec, fixture_dirs["partial"])
        recovered = float(np.nanmean(m["mf_per_rep"]))
        expected = expected_mobile_fraction(spec)
        # a fixed three-frame plateau understates this by roughly a third
        assert recovered > 0.9 * expected


class TestContrasts:
    """The statistics run over the values the pipeline actually produced."""

    def test_planned_contrasts_against_a_named_control(self, fixture_dirs, specs):
        groups = {}
        for name in ("arrested", "partial", "control", "heterogeneous"):
            _, _, m = _metrics(specs[name], fixture_dirs[name])
            groups[name] = np.asarray(m["mf_per_rep"], float)

        summaries = {k: {"mf_per_rep": v} for k, v in groups.items()}
        rows = compare_mobile_fractions(summaries, control="control")
        assert len(rows) == 3                      # k-1 planned contrasts
        assert all(r["group_b"] == "control" for r in rows)

        arrested = next(r for r in rows if r["group_a"] == "arrested")
        assert arrested["hedges_g"] < 0            # less mobile than the control
        assert arrested["mwu_U"] == 0              # complete rank separation

    def test_the_heterogeneous_condition_reports_its_rank_floor(
            self, fixture_dirs, specs):
        """At n = 3 the exact null cannot reach significance whatever the data."""
        groups = {}
        for name in ("heterogeneous", "control"):
            _, _, m = _metrics(specs[name], fixture_dirs[name])
            groups[name] = np.asarray(m["mf_per_rep"], float)
        summaries = {k: {"mf_per_rep": v} for k, v in groups.items()}
        row = compare_mobile_fractions(summaries, control="control")[0]
        assert row["n_a"] == 3
        assert row["mwu_p_floor"] > 0.005
        # and the observed p cannot beat it
        assert row["mwu_p"] >= row["mwu_p_floor"]