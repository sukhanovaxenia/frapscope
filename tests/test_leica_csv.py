"""LAS X Quantify CSV loader and the bleach position it records.

The exports are synthesised here rather than shipped, in both regional
dialects, so the tests pin the parser to the format and not to one file.
"""

import numpy as np
import pytest

from frapscope import loaders
from frapscope.core import bleach_index, detect_bleach_index, replicate_metrics
from frapscope.loaders import LOADER_REGISTRY, load_leica_csv

# Protocol of the 13.11.2026 acquisitions: 2 pre-bleach frames at 2.6 s,
# 8 bleach-pulse frames that LAS X leaves empty, 20 recovery frames at 6 s.
T_PRE = [0.0, 2.598]
T_BLEACH = [5.703 + 2.6 * i for i in range(8)]
T_POST = [26.988 + 6.0 * i for i in range(20)]
T = np.array(T_PRE + T_BLEACH + T_POST)
N_PRE, N_BLEACH = len(T_PRE), len(T_BLEACH)


def recovery(mf=0.30, depth=0.75, tau=25.0, f0=100.0):
    """Noise-free ROI trace; NaN on the bleach frames."""
    y = np.full(T.size, np.nan)
    y[:N_PRE] = f0
    tp = T[N_PRE + N_BLEACH:] - T[N_PRE + N_BLEACH]
    f_min = f0 * (1 - depth)
    y[N_PRE + N_BLEACH:] = f_min + (f0 - f_min) * mf * (1 - np.exp(-tp / tau))
    return y


def write_export(path, cols, dialect="semicolon", channels=None):
    """Write a LAS X Quantify export. ``cols`` maps ROI name -> values."""
    sep, dec = (";", ",") if dialect == "semicolon" else (",", ".")

    def fmt(v):
        return "" if not np.isfinite(v) else repr(float(v)).replace(".", dec)

    def block(ch, c):
        names = list(c)
        out = [f"Channel {ch}", sep.join(["Axis [s]"] + [f"{n} []" for n in names])]
        for i, t in enumerate(T):
            out.append(sep.join([fmt(t)] + [fmt(c[n][i]) for n in names]))
        return out

    lines = []
    for ch, c in (channels or {1: cols}).items():
        lines += block(ch, c)
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode())
    return path


def mf_of(rec):
    return float(replicate_metrics([rec])["mf_per_rep"][0])


@pytest.mark.parametrize("dialect", ["semicolon", "comma"])
def test_both_dialects_parse_to_the_same_trace(tmp_path, dialect):
    y = recovery()
    rec = load_leica_csv(write_export(tmp_path / "a.csv", {"ROI1": y}, dialect))[0]
    assert rec["intensity_raw"] == pytest.approx(y[np.isfinite(y)])
    assert rec["provenance"] == "leica_csv_single"


def test_bleach_is_read_from_the_gap_and_zeroes_time(tmp_path):
    rec = load_leica_csv(write_export(tmp_path / "a.csv", {"ROI1": recovery()}))[0]
    assert rec["bleach_at_t0"] is True
    assert rec["time"][N_PRE] == 0.0
    assert np.all(rec["time"][:N_PRE] < 0)
    assert rec["time"][:N_PRE] == pytest.approx(np.array(T_PRE) - T_POST[0])
    assert bleach_index(rec) == N_PRE


def test_single_normalisation_recovers_mobile_fraction(tmp_path):
    rec = load_leica_csv(write_export(tmp_path / "a.csv", {"ROI1": recovery(mf=0.30)}))[0]
    assert rec["intensity"][:N_PRE] == pytest.approx(1.0)
    assert mf_of(rec) == pytest.approx(30.0, abs=1.5)    # plateau not fully reached


def test_reference_roi_corrects_acquisition_photobleaching(tmp_path):
    """The reason to export a reference ROI: single normalisation reads
    acquisition bleaching as a lower plateau, double normalisation removes it."""
    decay = np.exp(-T / 150.0)
    roi, ref, bg = recovery(mf=0.30) * decay + 5, 200 * decay + 5, np.full(T.size, 5.0)
    roi[N_PRE:N_PRE + N_BLEACH] = np.nan
    f = write_export(tmp_path / "a.csv", {"ROI1": roi, "ROI2": ref, "ROI3": bg})

    single = load_leica_csv(f, background_roi="ROI3")[0]
    double = load_leica_csv(f, reference_roi="ROI2", background_roi="ROI3")[0]
    assert double["provenance"] == "leica_csv"
    assert mf_of(double) == pytest.approx(mf_of(load_leica_csv(
        write_export(tmp_path / "clean.csv", {"ROI1": recovery(mf=0.30)}))[0]), abs=0.5)
    assert mf_of(single) < mf_of(double) - 10


def test_decaying_trace_keeps_its_recorded_bleach(tmp_path):
    """Immobile ROI under acquisition bleaching: the post-bleach signal keeps
    falling. Re-detection would move the bleach to the end of the trace."""
    y = recovery(mf=0.0, depth=0.70)
    y[N_PRE + N_BLEACH:] *= np.exp(-np.arange(20) / 15.0)
    rec = load_leica_csv(write_export(tmp_path / "a.csv", {"ROI1": y}))[0]

    assert detect_bleach_index(rec["intensity"]) > N_PRE      # the failure mode
    assert bleach_index(rec) == N_PRE                         # the fix
    assert mf_of(rec) < 0                                     # reported, not hidden


def test_files_load_in_natural_order(tmp_path):
    for name in ["12_10", "12", "12_2"]:
        write_export(tmp_path / f"{name}.csv", {"ROI1": recovery()})
    (tmp_path / "12_2.tif").write_bytes(b"not a csv")
    assert [r["file"] for r in load_leica_csv(tmp_path)] == ["12", "12_2", "12_10"]


def test_missing_roi_names_what_is_available(tmp_path):
    f = write_export(tmp_path / "a.csv", {"ROI1": recovery()})
    with pytest.raises(KeyError, match="ROI1"):
        load_leica_csv(f, reference_roi="ROI2")


def test_channel_is_selectable(tmp_path):
    a, b = recovery(f0=100.0), recovery(f0=40.0)
    f = write_export(tmp_path / "a.csv", None, channels={1: {"ROI1": a}, 2: {"ROI1": b}})
    assert load_leica_csv(f, channel=2)[0]["intensity_raw"][0] == pytest.approx(40.0)
    with pytest.raises(ValueError, match="no channel 3"):
        load_leica_csv(f, channel=3)


def test_registered_for_configs():
    assert LOADER_REGISTRY["leica_csv"] is load_leica_csv


def test_lif_accepts_a_directory(tmp_path, monkeypatch):
    calls = []

    def fake(path, roi_radius_px, single_series):
        calls.append(path.name)
        return [dict(file="FRAP 001")]

    monkeypatch.setattr(loaders, "_load_lif_file", fake)
    (tmp_path / "b.lif").touch()
    (tmp_path / "a.lif").touch()
    out = loaders.load_lif(tmp_path)
    assert calls == ["a.lif", "b.lif"]
    assert [r["file"] for r in out] == ["a/FRAP 001", "b/FRAP 001"]

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no .lif"):
        loaders.load_lif(empty)
