# frapscope

FRAP recovery analysis for live-cell imaging: extraction from Leica `.lif`
archives, double normalisation, mobile-fraction estimation, and statistics over
a pre-specified contrast family.

Written for a study of nuclear inclusions formed by human ribosomal proteins,
but nothing in the package is specific to that panel: conditions, file routes and
the control are declared in configuration.

---

## What it computes

Given a set of conditions, each a directory of `.lif` archives, the package
produces per-condition recovery curves, mobile fractions, and the statistical
comparison against a named control.

The mobile fraction is point-based, from the measured plateau rather than a
fitted asymptote:

```
MF = (mean I over the last 20 % of the window − I₀) / (mean pre-bleach I − I₀)
```

so it does not depend on the quality of a kinetic fit. The plateau window is a
fraction of the observation window rather than a fixed number of frames, because
frame intervals differ several-fold between conditions and a fixed count spans a
different physical duration in each.

## Install

```bash
git clone https://github.com/sukhanovaxenia/frapscope
cd frapscope
pip install -e ".[dev]"
```

## Use

```bash
frapscope --out results/ --stats-control eS28 --stats-exclude GFP --no-harmonise
```

| flag | effect |
|---|---|
| `--out` | output directory |
| `--stats-control` | condition every other is compared against |
| `--stats-exclude` | conditions held out of the statistical comparison |
| `--no-harmonise` | report on native time windows rather than a common one |
| `--harmonise-dt` | resample to a fixed frame interval |
| `--min-coverage` | fraction of replicates a common window must span (default 0.75) |

Conditions are declared in `config.py` as `Condition` objects: an identifier, a
path, a loader route (`lif`, `roi_csv` or `image_digitized`) and a display label.

## Five decisions that change the numbers

These are documented here rather than buried in the source because each was a
fault in a real analysis run that altered a reported value before it was caught.
Each has a regression test in `tests/test_regressions.py`.

**Bleach frame by largest drop, not by minimum.** The bleach is defined by its
shape — a sharp fall from the level the preceding frames establish — not by being
the numerically smallest sample. A minimum-based rule with a "leave N frames
behind it" guard fails when resampling leaves fewer than N frames after the
bleach: the real bleach is excluded from the search, the detector returns a frame
inside the pre-bleach plateau, and the mobile fraction diverges. Scoring against
`np.maximum.accumulate` does not depend on how many frames follow.

**Replicates above the physical ceiling are excluded.** Photobleaching destroys
fluorophore, so a bleached region exchanging with a finite unbleached pool cannot
end above the signal it started from. A trace that does reports axial drift or
over-correction by a reference region bleaching faster than the region of
interest. Exclusion happens once, before plotting, QC and testing, so a trace
judged impossible is impossible everywhere. In the study this package was written
for, this moved the control's mobile fraction from 41.2 % to 21.9 %.

**The plateau is a time fraction.** See above.

**Native time windows, with an invariance check.** Acquisition windows and frame
intervals were not synchronised across conditions and no valid common window
existed: one condition needed 42.3 s for three half-times while another's
shortest replicate capped the window at 13.0 s. Values are reported on native
windows and the ordering is verified to survive truncation to a common window,
rather than forcing a harmonisation that discards most of the data.

**Family size is fixed by design, not by outcome.** The contrast family is each
candidate against the control, declared through `--stats-control`. The same
three contrasts corrected over all ten pairwise comparisons return an adjusted
*p* of 0.095 where the pre-specified three return 0.024. Fixing the family before
looking is what makes the smaller number legitimate.

## Both tests are reported, and so is the rank floor

Mann-Whitney is reported because replicate counts are small and mobile fractions
are right-skewed. Welch is reported alongside because at the smallest counts the
exact rank null is coarse: with three observations against ten there are 286
distinguishable arrangements, so no configuration of the data can return a
two-sided *p* below 0.007, or below 0.021 after correction over three contrasts.

`stats.py` therefore reports `_min_p(na, nb)` next to the observed *p*. A contrast
significant by Welch and marginal by Mann-Whitney at these counts is reporting the
resolution of the rank statistic, not an absence of effect — and the *U* statistic
is given with the number of possible orderings so the reader can tell the two
apart. `U = 0` means every replicate of one condition lies below every replicate
of the other, whatever the *p*-value says.

## Data

Raw `.lif` archives are not in this repository and are excluded by `.gitignore`.
They are large and they are the primary record of an unpublished experiment.

The tests run on synthetic fixtures and require no microscopy data. To reproduce
the published values from derived data, see `docs/reproducing.md`.

## Tests

```bash
pytest                         # all 35
pytest -m "not integration"    # the 23 that need no fixture data
```

Thirty-five tests in two layers, in two files, kept apart deliberately. `test_regressions.py` checks functions in
isolation — each corresponds to a fault that reached a reported number, and each
fails against the implementation that produced the wrong answer. `test_pipeline.py`
checks that the stages agree with one another, by putting a known mobile fraction
into a synthetic replicate and asserting the pipeline returns it:

```
condition        expected  recovered
arrested            2.00%      1.95%
partial            12.00%     11.93%
control            22.00%     21.96%
heterogeneous       5.67%       5.42%
supraceiling       90.00%   all replicates excluded
```

That second layer is the one that would have caught the -75000 % mobile fraction:
every component was individually correct when the pipeline as a whole was wrong.
Fixtures are generated by `tests/fixtures/make_fixtures.py`, never committed, and
require no microscopy data.

The split is by what a failure means, not by tidiness. A `test_regressions.py`
failure says a function's behaviour changed; a `test_pipeline.py` failure says the
stages stopped agreeing with each other. Knowing which before reading the
traceback is worth a second file. They also differ in cost — 0.7 s and no disk
against 1.3 s and 31 CSVs written — which is why the integration layer carries a
marker and the fast loop can skip it.

## Citation

See `CITATION.cff`.

## License

See `LICENSE`.