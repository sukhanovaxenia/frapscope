# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

A release that changes a number a previous version reported opens with a
**Results affected** section.

## [Unreleased]

### Added

- `leica_csv` loader for LAS X Quantify ROI exports, for acquisitions whose
  `.lif` was not kept. Reads both regional dialects, takes timestamps from the
  file, and places the bleach at the frames LAS X leaves unmeasured. Double
  normalisation when reference and background ROIs are exported, single
  otherwise, with distinct provenance so mixed comparisons are flagged.
- `core.bleach_index(record)`: uses the loader's bleach when the record carries
  `bleach_at_t0`, and `detect_bleach_index` otherwise.

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
