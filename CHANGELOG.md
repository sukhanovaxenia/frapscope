# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

A release that changes a number a previous version reported opens with a
**Results affected** section.

## [Unreleased]

### Added

- `stats.correlate_with_covariate` and `viz.plot_covariate_scatter`
  (`--covariate`): rank correlation between a per-condition number measured
  outside the pipeline — predicted aggregation-prone regions, expression
  level, days post-transfection — and a FRAP metric. Exact permutation
  Spearman ρ, the smallest p the design admits, a leave-one-out check that
  names any condition one replicate can move further than the panel spans and
  recomputes the correlation without it, and the same test on medians.
  `--covariate-reference` keeps a control on the figure and out of the
  correlation. Labels are placed by collision search rather than by hand.
- `--stats-control` accepts several controls, each corrected as its own Holm
  family and written to `mobile_fraction_stats_vs_<control>.csv`.
- `examples/covariate_ribosomal.csv`: consensus APR counts for the ribosomal
  panel.
- `frapscope-compare` (`crossrun.py`): compares two runs of a panel from their
  configs. Per condition and metric, an exact two-sample KS test with median
  shift and rank-biserial r; across conditions, exact permutation Spearman ρ
  with its attainable floor, Lin's CCC, OLS slope with CI, and bootstrap CIs
  over replicates. Benjamini-Hochberg within each family. Matched observation
  windows by default. Shift and pattern heatmaps, regression panels,
  per-replicate distributions, CSV tables and a text report.
- `stats.bh`: Benjamini-Hochberg q-values.
- `core.replicate_table`: per-replicate mobile fraction, bleach depth and t½,
  with the same definitions and exclusions as `replicate_metrics`.
- `leica_csv` loader for LAS X Quantify ROI exports, for acquisitions whose
  `.lif` was not kept. Reads both regional dialects, takes timestamps from the
  file, and places the bleach at the frames LAS X leaves unmeasured. Double
  normalisation when reference and background ROIs are exported, single
  otherwise, with distinct provenance so mixed comparisons are flagged.
- `core.bleach_index(record)`: uses the loader's bleach when the record carries
  `bleach_at_t0`, and `detect_bleach_index` otherwise.

### Changed

- `stats.spearman_exact` moved from `crossrun` to `stats`, where both callers
  can reach it. `crossrun.spearman_exact` still resolves to it.
- `stats.compare_mobile_fractions` takes `out_name`.

### Fixed

- The comparison overlay failed with `IndexError` for more than eight
  conditions: the colour and marker palettes held eight and were sliced to
  the condition count. Palettes now hold twelve, and `viz.comparison_styles`
  cycles beyond that without repeating a colour-marker pair.
- `load_lif` accepted only a file, while the configuration documents a folder
  of archives and the example config passes one; a folder failed as "loader
  failed". A folder now loads every `.lif` in it, with replicate names prefixed
  by the archive stem when there are several.
- On single-normalised traces of slowly exchanging cells, acquisition
  photobleaching moves the post-bleach minimum to the end of the trace, and
  re-detection placed the bleach there (frame 20 instead of 2 in one replicate).
  Records with a recorded bleach are no longer re-detected. Records from the
  `.lif`, `roi_csv` and `image_digitized` routes are unaffected.

## [0.1.0] — 2026-08-11

First public release. The package existed as analysis scripts before this;
the history below records the corrections made while the analysis was in
progress, because each altered a value that had already been written down.

### Results affected

| Quantity | Before | After | Cause |
|---|---|---|---|
| Control mobile fraction | 41.2 % | 21.9 % | replicates recovering above their own pre-bleach level were included |
| Mobile fraction, resampled traces | down to −75 000 % | correct | bleach frame taken as the global minimum with a frame-count guard |
| Contrast p-values | corrected over 10 pairs | corrected over the pre-specified 3 | family size was not fixed by design |

### Added

- Extraction directly from Leica `.lif` archives, replacing digitisation of
  exported TIFF images by colour thresholding.
- `stats.py`: Welch and Mann-Whitney reported together with Hedges' *g*, Holm
  correction over a pre-specified family, and `_min_p`, the smallest two-sided
  *p* the exact rank null can return at the given replicate counts.
- `timebase.py`: acquisition, invariance and sampling-bias reports; native-window
  reporting with a verified ordering under truncation to a common window.
- `median_iqr` alongside mean and SEM, since per-replicate mobile fractions are
  right-skewed.
- Regression tests for every quantity listed under "Results affected".

### Changed

- Bleach frame identified as the largest fall below the running maximum rather
  than as the global minimum with a frame-count guard.
- Plateau window expressed as a fraction of the observation window rather than a
  fixed number of trailing frames, so the estimate is independent of sampling
  density.
- Supra-ceiling exclusion applied once in `run()` before plotting, QC and
  testing, so every consumer sees the same set of replicates.

### Known limitations

- The loaders are not exercised in CI, because the microscopy data are not
  distributable. CI asserts only that the entry points exist and import.
- `load_image_digitized` is retained for provenance and should not be used for
  new analysis; it recovers data points from rendered figures by colour
  thresholding, with an estimated uncertainty of ±1–3 % per axis.
