# Reproducing the published values

The raw `.lif` archives are not distributable, so the published numbers cannot be
regenerated from primary data outside the originating lab. This document sets out
what *can* be reproduced, and from what.

## Three levels

| From | Reproduces | Needs |
|---|---|---|
| `.lif` archives | everything | the raw data, not in this repository |
| per-replicate CSVs | summary statistics, contrasts, all figures | `data/derived/` (deposited with the manuscript) |
| synthetic fixtures | the numerical behaviour of every correction | nothing; runs in CI |

## From derived data

Deposit one CSV per condition with the columns below, and the statistics and
figures follow without the archives:

```
condition,replicate,time_s,intensity_norm,intensity_raw,provenance
```

`intensity_norm` is already double-normalised, so the extraction step is skipped
and everything downstream — bleach detection, plateau, mobile fraction,
exclusion, contrasts — runs unchanged. This is the level at which an independent
reader can check the reported values.

Load these with the `roi_csv` route rather than `lif`:

```python
Condition(display="uS5", source="data/derived/uS5.csv", loader="roi_csv")
```

## What the fixtures cover

`tests/fixtures/make_fixtures.py` builds traces that exercise each correction
rather than imitating a measurement: a trace whose global minimum is not the
bleach, one with a single post-bleach frame, one recovering above its own
pre-bleach level, and one sampled coarsely enough to separate a fractional
plateau window from a fixed-frame one. These are what CI runs.

## What cannot be reproduced from anything deposited

The region of interest was placed by hand on each inclusion. Two analysts
choosing regions on the same archives will not obtain identical traces, and the
per-replicate spread reported in the manuscript includes that variation. The
derived CSVs fix the choice that was made; they do not make it reproducible.

Report this rather than leaving it implied. A reader who reruns the pipeline on
the derived data and obtains the published numbers to the last digit has checked
the analysis, not the measurement.

## The exact invocation behind the published values

The controller was a script at the repository root, `run_frap.py` (see `src/backlog/run_frap.py`), before the
package was laid out. It is now `frapscope.cli`, reached through the `frapscope`
console entry point. The historical command is recorded here because the paper
cites values produced by it, and a reader following this repository needs to be
able to map one onto the other.

Published values were produced by:

```bash
python3 run_frap.py --out ../../FRAP/frap_submission/ --no-harmonise \
        --stats-control eS28 --stats-exclude GFP
```

The equivalent under the packaged entry point is:

```bash
frapscope --config examples/config_ribosomal.py --out results/ --no-harmonise \
          --stats-control eS28 --stats-exclude GFP
```

The only difference is `--config`. `run_frap.py` imported a module-level `CONDITIONS` list, so the condition set was fixed at import and the same command
meant something different on a machine with a different `config.py`. Naming the
config file makes the run self-describing: the command and the file together
determine the output, with nothing supplied by the environment.