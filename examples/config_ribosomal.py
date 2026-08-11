"""Condition set for the ribosomal-protein study.

Set DATA to the directory holding the .lif archives. Everything else follows.
The path is resolved against this file, not the working directory, so the config
and its data move together and the same command means the same thing wherever it
is run from.

    frapscope --config examples/config_ribosomal.py --out results/ \
              --stats-control eS28 --stats-exclude GFP --no-harmonise

That is the invocation behind the published values; see docs/reproducing.md.

The archives are not distributed with this repository. See README, "Data".
"""

from frapscope.config import Condition

#: Directory holding the per-protein subfolders of .lif archives.
DATA = "../../FRAP"


CONDITIONS = [
    # Display names are the ones the manuscript uses. Note uS5, not eS2: RPS2 is
    # conserved in all three domains of life and therefore takes the u prefix.
    # There is no eS2, although earlier drafts of the manuscript used it.
    Condition(key="RPS_2", display="uS5", loader="lif",
              source=f"{DATA}/RPS_2"),

    Condition(key="RPL_27", display="eL27", loader="lif",
              source=f"{DATA}/RPL_27"),

    # The folder is named RPL_36 and the protein is eL42. This is not a typo in
    # either direction: the constructs were cloned from RPL36A (UniProt P83881,
    # 106 aa, eL42) and the acquisition folder was named for RPL36 (Q9Y3U8,
    # 105 aa, eL36), a different protein of the same family. The two differ by
    # one residue in length, which is why the mismatch survived for months. The
    # data here are eL42's. Do not rename the folder to match the display name
    # without checking the archives; do not rename the display to match the
    # folder at all.
    Condition(key="RPL_36", display="eL42", loader="lif",
              source=f"{DATA}/RPL_36"),

    Condition(key="RPS_28", display="eS28", loader="lif",
              source=f"{DATA}/RPS_28"),

    # Free EGFP is retained so that --stats-exclude GFP excludes something. Its
    # replicates were acquired while acquisition settings were still being
    # optimised and are therefore not technical replicates of one another, so it
    # is a qualitative mobility reference and must not enter the statistics.
    # Excluding a condition that is absent from the config looks identical, on
    # the command line, to excluding one that is present.
    Condition(key="EGFP", display="GFP", loader="image_digitized",
              source=f"{DATA}/EGFP"),
]

# ---------------------------------------------------------------------------
# Notes on loader_kwargs
#
# None are set here, deliberately. load_lif already defaults to
# roi_radius_px=6, and restating a default on some conditions but not others
# reads as a difference in how they were measured. Set a kwarg only where the
# value genuinely differs, and then set it on every condition so the asymmetry
# is visible rather than implied.
#
# For a protein with only ImageJ ROI tables, use the roi_csv route and supply
# real per-frame seconds — frame indices distort the recovery rate, because the
# Leica pre-bleach to recovery interval is not uniform:
#
#   Condition(key="RPL_27", display="eL27", loader="roi_csv",
#             source=f"{DATA}/RPL_27_roi_csvs",
#             loader_kwargs=dict(time_s=[0, 2.6, 27.36, 33.36, 39.35, 45.37,
#                                        51.36, 57.37, 63.36, 69.37, 75.36,
#                                        81.37])),
#
# Mixing loader routes across conditions is allowed and triggers a provenance
# warning in the comparison, because the extraction method is a source of
# between-condition bias that should not be silent. GFP above is the one case
# in this panel, and it is excluded from the statistics for that reason.
# ---------------------------------------------------------------------------