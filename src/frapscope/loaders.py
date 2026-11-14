"""
frapscope.loaders — interchangeable extractors, all emitting the `all_data` contract.

The all_data contract (the library's invariant interface):
    all_data : list[dict] with keys
        file          : str   replicate/experiment identifier
        time          : np.ndarray  seconds, time-since-bleach (pre-bleach < 0)
        intensity     : np.ndarray  full-scale double-normalized (pre-bleach ~ 1)
        intensity_raw : np.ndarray  mean ROI fluorescence, detector a.u.
        provenance    : str   which loader produced it (for mix-method warnings)
        centre        : tuple|None  detected bleach-spot pixel (.lif only)
        bleach_at_t0  : bool, optional  the bleach was read from the file, not
                        inferred; t = 0 is the first post-bleach frame and core
                        uses it instead of re-detecting (core.bleach_index)

Four loaders are provided:
    load_lif            preferred; lossless extraction from raw Leica .lif
    load_leica_csv      LAS X Quantify export: per-frame ROI means with real
                        timestamps, for acquisitions whose .lif was not kept
    load_roi_csv        honest manual path: ImageJ/easyFRAP ROI tables + timing
    load_image_digitized  DEPRECATED legacy plot-digitizer (lossy; see warning)

Register additional loaders in LOADER_REGISTRY; the runner dispatches by name.
"""

import struct
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------- .lif loader

def _read_lif_xml(path):
    with open(path, "rb") as f:
        struct.unpack("<i", f.read(4)); struct.unpack("<I", f.read(4)); f.read(1)
        n = struct.unpack("<I", f.read(4))[0]
        return f.read(n * 2).decode("utf-16-le", errors="replace")

def _ln(t): return t.split('}')[-1]

def _find_element(root, name):
    for el in root.iter():
        if _ln(el.tag) == "Element" and el.get("Name") == name:
            return el
    return None

def _timestamps(el):
    for d in el.iter():
        if _ln(d.tag) == "TimeStampList":
            ch = [c for c in d if _ln(c.tag) == "TimeStamp"]
            if ch:
                return np.array([(int(c.get("HighInteger")) << 32) | int(c.get("LowInteger"))
                                 for c in ch], dtype=np.int64)
            if d.text and d.text.strip():
                return np.array([int(x, 16) for x in d.text.split()], dtype=np.int64)
    return None

def _phase(n):
    return "pre" if "Pre" in n else "bleach" if "Bleach" in n else "pb" if "Pb" in n else None

def _expkey(n):
    return n.split("/")[0] if "/" in n else n

def _frame(img, t):
    return np.asarray(img.get_frame(z=0, t=t, c=0)).astype(float)

def _detect_bleach_centre(pre_img, pb_img):
    pre = _frame(pre_img, pre_img.dims.t - 1)
    pb0 = _frame(pb_img, 0)
    diff = ndimage.gaussian_filter(pre, 2) - ndimage.gaussian_filter(pb0, 2)
    diff[diff < 0] = 0
    cy, cx = np.unravel_index(diff.argmax(), diff.shape)
    return int(cx), int(cy)

def _series_traces(img, disc, ref_pct=99.0):
    """ROI, whole-cell reference and background traces for one series.

    The reference proxy is the mean of the brightest pixels, used to correct for
    acquisition photobleaching. A strict ``>`` against the percentile returns an
    empty selection whenever the top of the histogram is flat — a saturated or
    near-uniform frame — and ``np.mean([])`` is NaN. That NaN then divides
    through the double normalisation and propagates silently into every
    downstream fit, surfacing much later as "array must not contain infs or
    NaNs". Selecting with ``>=`` and falling back to the frame maximum keeps the
    reference finite for every frame.
    """
    roi, ref, bg = [], [], []
    for t in range(img.dims.t):
        fr = _frame(img, t)
        roi.append(fr[disc].mean())
        thr = np.percentile(fr, ref_pct)
        top = fr[fr >= thr]
        ref.append(float(top.mean()) if top.size else float(fr.max()))
        bg.append(np.percentile(fr, 10))                      # background
    return map(np.array, (roi, ref, bg))

def _extract_experiment_lif(root, imgs, exp, roi_radius_px):
    pre, pb = imgs[exp["pre"]], imgs[exp["pb"]]
    cx, cy = _detect_bleach_centre(pre, pb)
    H, W = pre.dims.y, pre.dims.x
    yy, xx = np.mgrid[0:H, 0:W]
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= roi_radius_px ** 2

    p_roi, p_ref, p_bg = _series_traces(pre, disc)
    q_roi, q_ref, q_bg = _series_traces(pb, disc)

    t_pre = _timestamps(_find_element(root, exp["pre"].split("/")[-1]))
    t_pb  = _timestamps(_find_element(root, exp["pb"].split("/")[-1]))
    t0 = min(t_pre.min(), t_pb.min())
    s_pre = (t_pre - t0) * 1e-7
    s_pb  = (t_pb  - t0) * 1e-7

    prc, prefc = p_roi - p_bg, p_ref - p_bg
    qrc, qrefc = q_roi - q_bg, q_ref - q_bg
    roi_pre, ref_pre = prc.mean(), prefc.mean()
    I_pre = (ref_pre / prefc) * (prc / roi_pre)
    I_pb  = (ref_pre / qrefc) * (qrc / roi_pre)

    t_bleach = s_pb[0]
    time = np.concatenate([s_pre - t_bleach, s_pb - t_bleach])
    inten = np.concatenate([I_pre, I_pb])
    inten_raw = np.concatenate([p_roi, q_roi])
    return time, inten, inten_raw, (cx, cy)

def _extract_single_series_lif(root, img, name, roi_radius_px, n_pre_min=2):
    """One time series containing pre-bleach, bleach and recovery together.

    LAS X writes a FRAP experiment either as separate Pre/Bleach/Pb series or,
    when the whole protocol runs in a single acquisition, as one continuous
    series named without a phase token. The Pre/Pb pairing rule cannot see the
    second form, which is why eL27 loaded 3 replicates from a .lif holding 6 and
    eS2 loaded 6 from a .lif holding 8.

    Nothing about the measurement differs: the same ROI, the same reference and
    background channels, the same double normalisation. Only the bleach frame
    has to be found from the data rather than read off the series name. It is
    located as the largest fall of the whole-frame mean below its running
    maximum -- the same criterion core.detect_bleach_index applies per replicate,
    which is the physically correct one because photobleaching is a sharp drop
    from the established signal level rather than merely the smallest sample.

    The bleach centre is then the brightest pixel of the smoothed difference
    between the frames immediately before and after that fall, exactly as in the
    paired route.
    """
    nt = int(img.dims.t)
    if nt < n_pre_min + 3:
        return None

    # Locate the bleach by the largest LOCALISED single-frame drop, not by the
    # whole-frame mean. A bleach ROI is small relative to the field, so it barely
    # moves the frame average, whereas acquisition photobleaching lowers every
    # pixel gradually across the whole series. Scoring the cumulative fall of the
    # frame mean therefore returns the last frame of the series rather than the
    # bleach. The two processes are separable by their spatial and temporal
    # signature: acquisition bleaching is uniform and gradual, a photobleach is
    # focal and completes within one frame interval. Taking the peak of the
    # smoothed frame-to-frame difference image isolates the second.
    sm = [ndimage.gaussian_filter(_frame(img, t), 2) for t in range(nt)]
    step = np.array([float(np.max(sm[i] - sm[i + 1])) for i in range(nt - 1)])
    lo, hi = n_pre_min - 1, nt - 3          # leave pre-bleach ahead, recovery behind
    if hi <= lo:
        return None
    j = int(np.argmax(step[lo:hi])) + lo
    k = j + 1                                # the bleach lands on the NEXT frame
    span = float(np.max(sm[0]) - np.min(sm[0]))
    if span <= 0 or step[j] <= 0.10 * span:
        return None                          # no bleach-like event: not a FRAP series

    d = sm[k - 1] - sm[k]
    d[d < 0] = 0
    cy, cx = np.unravel_index(d.argmax(), d.shape)
    H, W = d.shape
    yy, xx = np.mgrid[0:H, 0:W]
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= roi_radius_px ** 2

    roi, ref, bg = _series_traces(img, disc)
    ts = _timestamps(_find_element(root, name.split("/")[-1]))
    if ts is None or len(ts) != nt:
        return None
    sec = (ts - ts.min()) * 1e-7

    rc, refc = roi - bg, ref - bg
    roi_pre, ref_pre = rc[:k].mean(), refc[:k].mean()
    inten = (ref_pre / refc) * (rc / roi_pre)

    # Validate before returning. In the paired route the bleach frame is given
    # by the series name; here it is inferred, so a wrong k silently reassigns
    # part of the pre-bleach phase to the recovery and inflates the trace by the
    # ratio of the two levels. A bleach depth outside 20-99 % is the signature:
    # too shallow means k landed on a focus fluctuation rather than the bleach,
    # ~100 % means the ROI was placed off the cell.
    f_pre = float(inten[:k].mean()) if k else np.nan
    f_min = float(inten[k])
    f_plat = float(inten[-max(1, (nt - k) // 5):].mean())
    bd = 100.0 * (1.0 - f_min / f_pre) if k else np.nan
    mf = 100.0 * (f_plat - f_min) / (f_pre - f_min) if (f_pre - f_min) > 0 else np.nan
    print(f"  [lif] single-series {name!r}: {nt} frames, bleach at frame {k} "
          f"({k} pre-bleach), bleach depth {bd:.1f} %, apparent MF {mf:.0f} %")

    # Two gates, because the first alone is not sufficient. Empirically every
    # unpaired multi-frame series in this dataset bleaches to 93.6-100.1 %,
    # against 92.1-92.5 % for the paired route, and then "recovers" to 153-172 %
    # of its own pre-bleach level. Both are the signature of a BLEACH-phase
    # series rather than a complete experiment: LAS X writes the triplet as
    # Pre=Series(n-1), Bleach=Series(n), Pb1=Series(n+1), and when the middle
    # member is named without the Bleach keyword it reaches this branch. Its
    # ROI is driven to zero by design, so a bleach depth near 100 % and a
    # supra-ceiling apparent recovery identify it unambiguously.
    if not (20.0 <= bd <= 95.0):
        print(f"  [lif]   -> REJECTED: bleach depth {bd:.1f} % outside 20-95 %; "
              f"this is most likely the Bleach-phase series of a paired "
              f"experiment, not a complete acquisition")
        return None
    if not np.isfinite(mf) or not (0.0 <= mf <= 100.0):
        print(f"  [lif]   -> REJECTED: apparent mobile fraction {mf:.0f} % is "
              f"outside the physical range 0-100 %")
        return None
    return dict(file=name, time=sec - sec[k], intensity=inten,
                intensity_raw=roi, centre=(int(cx), int(cy)),
                provenance="lif_single_series")


def load_lif(source, roi_radius_px=6, single_series=True, **_):
    """Lossless extraction from Leica .lif. Each FRAP experiment -> one replicate.

    ``source`` is a single archive or a directory of them. A directory is the
    form the configuration documents, and before this it reached ``open()`` as
    a path and failed with IsADirectoryError, which the runner reports only as
    "loader failed". With several archives in one directory, replicate names are
    prefixed with the archive stem, because LAS X numbers series from 1 in every
    project and two archives would otherwise both contribute "FRAP 001".
    """
    from pathlib import Path
    src = Path(source)
    if src.is_dir():
        files = sorted(src.glob("*.lif"), key=_natural_key)
        if not files:
            raise FileNotFoundError(f"no .lif archives in {src}")
        out = []
        for fp in files:
            print(f"  [lif] {fp.name}")
            recs = _load_lif_file(fp, roi_radius_px, single_series)
            if len(files) > 1:
                for r in recs:
                    r["file"] = f"{fp.stem}/{r['file']}"
            out.extend(recs)
        return out
    return _load_lif_file(src, roi_radius_px, single_series)


def _load_lif_file(source, roi_radius_px=6, single_series=True):
    """One .lif archive -> list of replicates. See :func:`load_lif`."""
    from readlif.reader import LifFile
    source = str(source)
    root = ET.fromstring(_read_lif_xml(source))
    lif = LifFile(source)
    imgs = {img.info["name"]: img for img in lif.get_iter_image()}
    exps = defaultdict(dict)
    unclassified = []
    for nm in imgs:
        ph = _phase(nm)
        if ph:
            # Several recovery series (Pb1, Pb2, ...) all classify as "pb" and
            # would overwrite one another in this dict; keep the first, which is
            # the one that starts at the bleach, and record the rest.
            exps[_expkey(nm)].setdefault(ph, nm)
            if ph == "pb" and exps[_expkey(nm)]["pb"] != nm:
                exps[_expkey(nm)].setdefault("pb_extra", []).append(nm)
        else:
            unclassified.append(nm)

    complete = {k: v for k, v in exps.items() if {"pre", "pb"} <= set(v)}
    incomplete = {k: sorted(set(v) - {"pb_extra"}) for k, v in exps.items()
                  if k not in complete}
    # Replicate count is a reported number in the manuscript, so the series that
    # did NOT become a replicate must be visible rather than silently dropped.
    print(f"  [lif] {len(imgs)} series -> {len(complete)} replicate(s)")
    if incomplete:
        for k, v in incomplete.items():
            print(f"  [lif] SKIPPED experiment {k!r}: has {v}, needs both "
                  f"'pre' and 'pb'")
    if unclassified:
        # Report the frame count with each name. A series with dims.t == 1 is a
        # snapshot and can never be a replicate; a series with many frames and no
        # phase keyword is a complete single-series FRAP acquisition (pre-bleach,
        # bleach and recovery in one time series) that the Pre/Pb pairing rule
        # cannot see. Those are recoverable replicates, and telling them apart
        # from snapshots is the whole point of printing this.
        print(f"  [lif] {len(unclassified)} series matched no phase keyword "
              f"(Pre/Bleach/Pb):")
        multi = 0
        for nm in unclassified:
            nt = getattr(imgs[nm].dims, "t", 1)
            tag = "snapshot" if nt <= 1 else f"{nt} frames  <-- RECOVERABLE"
            if nt > 1:
                multi += 1
            print(f"           {nm!r}: {tag}")
        if multi and not single_series:
            print(f"  [lif] {multi} unpaired multi-frame series could be loaded "
                  f"as single-series FRAP experiments (pass single_series=True).")
    singles = [nm for nm in unclassified
               if single_series and getattr(imgs[nm].dims, "t", 1) > 1]
    exps = complete
    all_data = []
    for name in sorted(exps):
        time, inten, inten_raw, centre = _extract_experiment_lif(root, imgs, exps[name], roi_radius_px)
        all_data.append(dict(file=name, time=time, intensity=inten,
                             intensity_raw=inten_raw, centre=centre, provenance="lif"))

    recovered = 0
    for nm in sorted(singles):
        try:
            rec = _extract_single_series_lif(root, imgs[nm], nm, roi_radius_px)
        except Exception as e:
            print(f"  [lif] single-series {nm!r} failed: {e}")
            continue
        if rec is None:
            continue
        all_data.append(rec)
        recovered += 1
    if recovered:
        print(f"  [lif] +{recovered} replicate(s) recovered as single-series "
              f"FRAP -> n = {len(all_data)}")
    return all_data


# ------------------------------------------------- LAS X Quantify CSV loader

_TIME_UNITS = {"s": 1.0, "sec": 1.0, "ms": 1e-3, "min": 60.0}


def _natural_key(path):
    """Sort 12, 12_2, 12_10 in that order, not 12, 12_10, 12_2."""
    import re
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", Path(path).name)]


def _read_leica_quantify(path, channel=1):
    """Parse one LAS X Quantify export into (time_s, {roi_name: values}).

    The export is a small text table, preceded by a ``Channel N`` line for each
    channel the ROIs were measured in::

        Channel 1
        Axis [s];ROI1 [];ROI2 []
        0;98,27;61,40
        2,598;95,98;61,22
        5,703;;
        ...

    Two dialects occur, and which one depends on the regional settings of the
    acquisition PC rather than on anything in LAS X: semicolon-separated with a
    decimal comma, and comma-separated with a decimal point. The separator is
    read from the header line and the decimal mark follows from it -- a comma
    cannot be both, so a comma-separated file has a decimal point.

    Empty cells are frames on which the ROI was not measured. In a FRAP protocol
    these are the bleach-pulse frames, and they are returned as NaN rather than
    dropped here so that the caller can see where the bleach was.
    """
    import re
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp1251"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"{Path(path).name}: cannot decode")

    blocks, cur = {}, None
    for line in text.replace("\r", "").split("\n"):
        line = line.strip()
        m = re.match(r"^channel\s*(\d+)", line, re.I)
        if m:
            cur = int(m.group(1))
            blocks[cur] = []
            continue
        if not line:
            continue
        if cur is None:                      # export without a Channel line
            cur = 1
            blocks[cur] = []
        blocks[cur].append(line)
    if channel not in blocks:
        raise ValueError(f"{Path(path).name}: no channel {channel}; "
                         f"found {sorted(blocks)}")
    lines = blocks[channel]
    if not lines:
        raise ValueError(f"{Path(path).name}: channel {channel} is empty")

    header = lines[0]
    sep = ";" if ";" in header else ("\t" if "\t" in header else ",")
    decimal_comma = sep != ","
    names = [re.sub(r"\s*\[.*?\]\s*$", "", h).strip() for h in header.split(sep)]
    unit_m = re.search(r"\[(.*?)\]", header.split(sep)[0])
    unit = (unit_m.group(1).strip().lower() if unit_m else "s") or "s"
    if unit not in _TIME_UNITS:
        raise ValueError(f"{Path(path).name}: first column is {header.split(sep)[0]!r}; "
                         f"expected a time axis in s, ms or min")

    def num(cell):
        cell = cell.strip()
        if not cell:
            return np.nan
        return float(cell.replace(",", ".") if decimal_comma else cell)

    rows = [ln.split(sep) for ln in lines[1:]]
    cols = {nm: np.array([num(r[j]) if j < len(r) else np.nan for r in rows])
            for j, nm in enumerate(names)}
    t = cols.pop(names[0]) * _TIME_UNITS[unit]
    return t, cols


def load_leica_csv(source, frap_roi="ROI1", reference_roi=None,
                   background_roi=None, channel=1, pattern="*.csv", **_):
    """Load LAS X Quantify ROI exports into the all_data contract.

    ``source`` is one exported CSV or a directory of them, one per replicate.
    Unlike ImageJ tables (:func:`load_roi_csv`) these carry the acquisition
    timestamps, so no timing has to be supplied and none is assumed.

    The bleach is placed where the file records it. LAS X leaves the ROI cells
    empty on the bleach-pulse frames, so the first measured frame after that
    gap is the first post-bleach frame and becomes t = 0, the same origin the
    .lif route uses. Only when a file has no gap is the bleach detected from
    the trace, by the criterion :func:`core.detect_bleach_index` applies.

    Normalisation depends on what was exported:

    * ``reference_roi`` and ``background_roi`` given: double normalisation,
      identical to the .lif route. Provenance ``leica_csv``.
    * neither given (one ROI per file): single normalisation to the pre-bleach
      mean. Acquisition photobleaching is then NOT corrected, which lowers the
      apparent plateau and so the mobile fraction, by an amount that grows with
      the number of post-bleach frames. Provenance ``leica_csv_single``, so a
      comparison against double-normalised conditions raises the provenance
      warning instead of passing silently.

    Supplying only one of the two is accepted. A background without a reference
    is subtracted before single normalisation; a reference without a background
    is used uncorrected for offset, which is what LAS X reports when no
    background ROI was drawn.
    """
    from .core import detect_bleach_index

    src = Path(source)
    files = sorted(src.glob(pattern), key=_natural_key) if src.is_dir() else [src]
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} in {src}")
    double = reference_roi is not None
    prov = "leica_csv" if double else "leica_csv_single"
    print(f"  [leica_csv] {len(files)} file(s) in {src.name}: "
          f"{'double' if double else 'single'} normalisation"
          + ("" if double else " (no reference ROI: acquisition photobleaching "
                               "is not corrected)"))

    all_data = []
    for fp in files:
        t, cols = _read_leica_quantify(fp, channel=channel)
        need = [frap_roi] + [r for r in (reference_roi, background_roi) if r]
        missing = [r for r in need if r not in cols]
        if missing:
            raise KeyError(f"{fp.name}: ROI {missing} not in export; "
                           f"available: {sorted(cols)}")
        frap = cols[frap_roi]
        ok = np.isfinite(t) & np.isfinite(frap)
        for r in (reference_roi, background_roi):
            if r:
                ok &= np.isfinite(cols[r])

        # Bleach = first measured frame after the first run of unmeasured ones.
        idx = np.flatnonzero(ok)
        gaps = np.flatnonzero(np.diff(idx) > 1)
        if idx.size == 0:
            print(f"  [leica_csv] {fp.name}: no measured frames, skipped")
            continue
        n_gap = 0
        if gaps.size:
            k = int(gaps[0]) + 1                         # index into idx
            n_gap = int(idx[gaps[0] + 1] - idx[gaps[0]] - 1)
        else:
            k = int(detect_bleach_index(frap[idx]))

        t, frap = t[idx], frap[idx]
        bg = cols[background_roi][idx] if background_roi else np.zeros_like(frap)
        fc = frap - bg
        if k < 1 or frap.size - k < 2:
            print(f"  [leica_csv] {fp.name}: {k} pre-bleach and {frap.size - k} "
                  f"post-bleach frames; needs at least 1 and 2, skipped")
            continue
        roi_pre = float(np.mean(fc[:k]))
        if roi_pre <= 0:
            print(f"  [leica_csv] {fp.name}: pre-bleach ROI mean <= 0 after "
                  f"background subtraction, skipped")
            continue
        if double:
            refc = cols[reference_roi][idx] - bg
            inten = (float(np.mean(refc[:k])) / refc) * (fc / roi_pre)
        else:
            inten = fc / roi_pre

        f_min = float(inten[k])
        tail = inten[k:][-max(1, (inten.size - k) // 5):]
        bd = 100.0 * (1.0 - f_min / float(np.mean(inten[:k])))
        mf = 100.0 * (float(np.mean(tail)) - f_min) / (1.0 - f_min) if f_min < 1 else np.nan
        flag = ""
        if np.isfinite(mf) and mf < 0:
            flag = "  <-- signal FALLS after the bleach"
        elif np.isfinite(mf) and mf > 100:
            flag = "  <-- above pre-bleach (excluded downstream)"
        print(f"  [leica_csv] {fp.stem}: {k} pre, {n_gap} bleach, "
              f"{inten.size - k} post frames; depth {bd:.0f} %, "
              f"apparent MF {mf:.0f} %{flag}")

        all_data.append(dict(file=fp.stem, time=t - t[k], intensity=inten,
                             intensity_raw=frap, centre=None, provenance=prov,
                             bleach_at_t0=n_gap > 0))
    return all_data


# ------------------------------------------------------ ROI-CSV (manual) loader

def load_roi_csv(source, time_s=None, frame_interval=None, n_pre=2,
                 col_frap="Mean1", col_bg="Mean2", col_ref="Mean3", **_):
    """Load ImageJ/easyFRAP ROI Multi-Measure tables into the all_data contract.

    `source` may be a single CSV (one replicate) or a directory of CSVs. Each CSV
    must contain per-frame ROI means with columns for the bleach spot, background,
    and reference ROIs (defaults match ImageJ Multi Measure: Mean1/Mean2/Mean3).

    Timing MUST be supplied because frame indices distort the recovery rate (the
    Leica pre->recovery interval is non-uniform). Provide either `time_s` (an
    explicit per-frame second vector, recommended — pull it from the .lif
    metadata) or `frame_interval` (uniform spacing, only valid if acquisition was
    truly uniform). The bleach is placed at the first frame whose value drops
    below the pre-bleach mean; pre-bleach frames map to negative time."""
    import csv
    from pathlib import Path
    src = Path(source)
    files = sorted(src.glob("*.csv")) if src.is_dir() else [src]
    all_data = []
    for fp in files:
        rows = list(csv.DictReader(open(fp)))
        frap = np.array([float(r[col_frap]) for r in rows])
        bg   = np.array([float(r[col_bg])  for r in rows]) if col_bg in rows[0] else np.zeros_like(frap)
        ref  = np.array([float(r[col_ref]) for r in rows]) if col_ref in rows[0] else np.ones_like(frap)
        nfr = len(frap)
        if time_s is not None:
            t_abs = np.asarray(time_s, float)
            if len(t_abs) != nfr:
                raise ValueError(f"{fp.name}: time_s length {len(t_abs)} != {nfr} frames")
        elif frame_interval is not None:
            t_abs = np.arange(nfr) * float(frame_interval)
        else:
            raise ValueError("Provide time_s (preferred) or frame_interval for ROI CSV timing.")
        # locate bleach: first frame below pre-bleach baseline
        pre_mean = frap[:n_pre].mean()
        drop = np.where(frap < 0.5 * pre_mean)[0]
        bleach_idx = drop[0] if len(drop) else n_pre
        t_bleach = t_abs[bleach_idx]
        # double normalization
        fc, rc = frap - bg, ref - bg
        roi_pre = fc[:n_pre].mean(); ref_pre = rc[:n_pre].mean()
        inten = (ref_pre / rc) * (fc / roi_pre)
        all_data.append(dict(file=fp.stem, time=t_abs - t_bleach, intensity=inten,
                             intensity_raw=frap, centre=None, provenance="roi_csv"))
    return all_data


# ------------------------------------------- DEPRECATED legacy plot-digitizer

def load_image_digitized(source, x_range=(0, 60), y_range=(0, 250), **_):
    """DEPRECATED. Recovers data points from rendered plot TIFs via green-marker
    detection. Retained only for legacy reproduction.

    KNOWN LIMITATIONS (do not use for quantitation):
      * Axis calibration assumes the plot area fills the whole image (no margin),
        biasing intensities by the figure's margins — typically >>3%.
      * Overlapping markers merge into single centroids, silently dropping points.
      * No background/reference channel: cannot do acquisition-bleach correction.
    Prefer load_lif (lossless) or load_roi_csv (honest manual path)."""
    import warnings
    from pathlib import Path

    from PIL import Image
    warnings.warn("load_image_digitized is deprecated and lossy; prefer load_lif/load_roi_csv.",
                  stacklevel=2)
    src = Path(source)
    files = sorted(src.glob("*.tif")) if src.is_dir() else [src]
    all_data = []
    for fp in files:
        arr = np.array(Image.open(fp))
        if arr.ndim < 3:
            continue
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        mask = (g > 150) & (r < 100) & (b < 100)
        lbl, nlab = ndimage.label(mask)
        if nlab == 0:
            continue
        centres = np.array(ndimage.center_of_mass(mask, lbl, range(1, nlab + 1)))
        H, W = arr.shape[:2]                       # NOTE: full-frame assumption (the bug)
        ypx, xpx = centres[:, 0], centres[:, 1]
        x = x_range[0] + xpx / W * (x_range[1] - x_range[0])
        y = y_range[1] - ypx / H * (y_range[1] - y_range[0])
        order = np.argsort(x)
        x, y = x[order], y[order]
        # crude bleach split + double-norm-ish to fit the contract
        bt = x_range[0] + 0.5 * (x_range[1] - x_range[0])
        pre = y[x < bt]
        f_pre = pre.mean() if len(pre) else y.max()
        inten = (y - y.min()) / (f_pre - y.min() + 1e-9)
        all_data.append(dict(file=fp.stem, time=x - bt, intensity=inten,
                             intensity_raw=y, centre=None, provenance="image_digitized"))
    return all_data


LOADER_REGISTRY = {
    "lif": load_lif,
    "leica_csv": load_leica_csv,
    "roi_csv": load_roi_csv,
    "image_digitized": load_image_digitized,
}
