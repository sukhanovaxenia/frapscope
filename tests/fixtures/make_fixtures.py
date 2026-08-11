"""Generate synthetic FRAP replicates with known ground truth.

Why these exist
---------------
The unit tests exercise individual functions on inline arrays. What they cannot
reach is the path that produced every wrong number in this project's history:
loader, normalisation, bleach detection, plateau, mobile fraction, contrast.
Each stage was correct in isolation at the moment the pipeline as a whole
returned -75000 %.

These fixtures close that gap without microscopy data. Each replicate is built
from a mobile fraction chosen in advance, written in the ImageJ Multi-Measure
column layout that ``load_roi_csv`` reads, and recovered through the real
pipeline. A test then asserts that what comes out is what went in — a stronger
claim than any unit test here makes, because it can only hold if every stage
agrees with every other about what the numbers mean.

What is deliberately not modelled
---------------------------------
Photon shot noise, bleaching of the reference region, focal drift and
inhomogeneous illumination are all absent; the traces are smooth. Fixtures that
imitate real data invite the reading that passing on them means the pipeline
works on real data, which it does not. They test arithmetic and contracts, not
measurement. Realism would buy nothing here and would cost the ability to state
an exact expected value.

Usage
-----
    python -m tests.fixtures.make_fixtures --out tests/fixtures/data

Deterministic: the same seed gives byte-identical files, so a diff in the
fixtures means a change in this file and nothing else.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Reference and background are constant, so the double normalisation in
#: ``load_roi_csv`` reduces to a scale factor and the expected mobile fraction
#: is exact rather than approximate. That is the point of a fixture.
REF_LEVEL = 1000.0
BG_LEVEL = 100.0
PRE_LEVEL = 800.0


@dataclass(frozen=True)
class Spec:
    """One synthetic condition."""

    name: str
    n_rep: int
    dt: float                 # frame interval, s
    n_pre: int                # pre-bleach frames
    n_post: int               # post-bleach frames
    bleach_depth: float       # fraction of signal removed
    mobile: float             # ground-truth mobile fraction, 0-1
    tau: float                # recovery time constant, s
    jitter: float = 0.0       # per-replicate spread in `mobile`, absolute
    overshoot: float = 0.0    # added to the plateau; >0 breaks the ceiling


#: Chosen to span the situations that broke the pipeline, not to imitate a panel.
SPECS = (
    # Arrested, sampled finely.
    Spec("arrested", n_rep=6, dt=2.6, n_pre=3, n_post=10,
         bleach_depth=0.92, mobile=0.02, tau=6.0),
    # Partially mobile, sampled 2.3x more coarsely. The pair exists so that a
    # fixed-frame plateau and a fractional one give different answers.
    Spec("partial", n_rep=10, dt=6.0, n_pre=3, n_post=10,
         bleach_depth=0.92, mobile=0.12, tau=9.8),
    # Control, bleached less deeply, with replicate spread.
    Spec("control", n_rep=10, dt=8.0, n_pre=3, n_post=10,
         bleach_depth=0.55, mobile=0.22, tau=10.6, jitter=0.10),
    # Two clustered replicates and one far above: the shape that cannot be told
    # from a single atypical cell at this n.
    Spec("heterogeneous", n_rep=3, dt=2.6, n_pre=3, n_post=10,
         bleach_depth=0.90, mobile=0.05, tau=7.0, jitter=0.07),
    # Recovers above its own pre-bleach level. Physically impossible; must be
    # excluded before any averaging.
    Spec("supraceiling", n_rep=2, dt=8.0, n_pre=3, n_post=10,
         bleach_depth=0.55, mobile=0.90, tau=10.6, overshoot=0.30),
)


def _mobile_for(spec: Spec, k: int) -> float:
    """Mobile fraction of replicate k. Deterministic offsets, not draws.

    The k-th replicate of a spec is the same trace on every machine and in every
    run, so a fixture diff can only come from an edit to this file.
    """
    if not spec.jitter or spec.n_rep == 1:
        return spec.mobile
    offsets = np.linspace(-1.0, 1.0, spec.n_rep)
    return float(np.clip(spec.mobile + spec.jitter * offsets[k], 0.0, 1.5))


def replicate(spec: Spec, k: int):
    """One replicate as (time_s, frap, bg, ref, mobile) in raw detector units."""
    mobile = _mobile_for(spec, k)
    n = spec.n_pre + spec.n_post
    t = np.arange(n, dtype=float) * spec.dt

    f0 = PRE_LEVEL * (1.0 - spec.bleach_depth)
    plateau = f0 + mobile * (PRE_LEVEL - f0) + spec.overshoot * PRE_LEVEL
    t_post = np.arange(spec.n_post, dtype=float) * spec.dt
    post = plateau - (plateau - f0) * np.exp(-t_post / spec.tau)

    frap = np.concatenate([np.full(spec.n_pre, PRE_LEVEL), post]) + BG_LEVEL
    return t, frap, np.full(n, BG_LEVEL), np.full(n, REF_LEVEL), mobile


def expected_mobile_fraction(spec: Spec) -> float:
    """Ground-truth mobile fraction of a condition, as a percentage.

    Computed here rather than written down, so a test asserts against a value
    this module derived and not one someone copied into an assertion.
    """
    return 100.0 * float(np.mean([_mobile_for(spec, k) for k in range(spec.n_rep)]))


def time_vector(spec: Spec) -> list[float]:
    """Acquisition times, as ``load_roi_csv`` expects in ``time_s``.

    Plain Python floats, not numpy scalars. ``list(np.arange(...))`` yields
    ``np.float64`` objects whose repr is ``np.float64(2.6)``, which is valid in
    memory and invalid the moment anyone writes the vector into a config file,
    a JSON record or a log. The conversion costs nothing and removes a failure
    that only appears at the point of serialisation.
    """
    n = spec.n_pre + spec.n_post
    return [float(v) for v in np.arange(n, dtype=float) * spec.dt]


def write(out_dir: Path, specs=SPECS) -> dict[str, Path]:
    """Write one directory of replicate CSVs per spec. Returns name -> directory."""
    written: dict[str, Path] = {}
    for spec in specs:
        d = Path(out_dir) / spec.name
        d.mkdir(parents=True, exist_ok=True)
        for k in range(spec.n_rep):
            t, frap, bg, ref, _ = replicate(spec, k)
            with (d / f"{spec.name}_{k + 1:02d}.csv").open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow([" ", "Mean1", "Mean2", "Mean3"])
                for i in range(len(t)):
                    w.writerow([i + 1, f"{frap[i]:.4f}",
                                f"{bg[i]:.4f}", f"{ref[i]:.4f}"])
        written[spec.name] = d
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=Path("tests/fixtures/data"), type=Path,
                    help="directory to write the fixture set into")
    args = ap.parse_args()

    written = write(args.out)
    total = sum(len(list(d.glob("*.csv"))) for d in written.values())
    print(f"wrote {total} replicates across {len(written)} conditions into {args.out}")
    for spec in SPECS:
        note = "   (supra-ceiling: must be excluded)" if spec.overshoot else ""
        print(f"  {spec.name:15s} n={spec.n_rep:2d}  dt={spec.dt:4.1f} s  "
              f"expected MF = {expected_mobile_fraction(spec):6.2f} %{note}")


if __name__ == "__main__":
    main()