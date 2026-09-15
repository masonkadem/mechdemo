"""export_csv.py -- plain-text export of everything a session produced.

    python export_csv.py                    # every session and recording in data/ -> export/
    python export_csv.py --session 20260915_1430_s01
    python export_csv.py --no-waveforms     # skip the big per-sample files

Why CSV as well as .npz
----------------------
The .npz files are the analysis format and stay authoritative -- they hold the per-site signals
the figures are built from. But a cuff-validation session has to be readable by whatever the lab
already uses (R, Excel, SPSS, a statistician who will not install numpy), and it has to be
readable in five years when this code no longer runs. CSV is the format that survives.

Three kinds of file come out, because they answer different questions:

  *_calib.csv    one row per cuff reading, with the features that reading is paired to.
                 This is the file you fit a model on, and the only one that is irreplaceable --
                 a recording can be repeated, a subject with a cuff already on their arm cannot.
  *_marks.csv    one row per sync mark, in both session-relative and absolute time.
  *_summary.csv  one row per recording: fps, frames, HR, path length, sites accepted.
  *_waveform.csv one row per SAMPLE, one column per site. Large (a 60 s recording at 30 fps over
                 34 sites is ~1.8k rows x 35 columns), so it is opt-out.

Every table carries the derived quantities alongside the raw ones -- MAP beside SBP/DBP, wave
speed and ln(PWV) beside the transit time -- so the person doing the statistics does not have to
re-derive the definitions this rig used and risk using a different ones.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import bp_model as BP

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EXPORT = ROOT / "export"

CALIB_COLS = ["i", "iso", "t_rel_s", "label", "sbp", "dbp", "map",
              "ptt_ms", "spread_ms", "hr", "path_cm", "pwv_ms", "ln_pwv", "usable"]
MARK_COLS = ["i", "iso", "t_rel_s", "t_wall", "label", "note", "recording", "ptt_ms", "hr"]
SUMMARY_COLS = ["file", "subject", "tag", "session",
                # Clock times first after the identifiers: reconciling a recording against a cuff
                # log or a lab notebook starts with "which run was going at 13:30", and that
                # should be readable in the leftmost columns rather than decoded from a float.
                "clock_record", "clock_end", "duration_s", "timezone",
                "fps", "n_frames", "n_points", "n_accepted", "n_accepted_strict",
                "gate_profile", "method", "consensus_hr", "path_cm",
                "t_wall_capture", "t_wall_record", "clock_capture", "n_marks", "n_calib"]
FIT_COLS = ["target", "mode", "n", "intercept", "coef_ln_pwv", "coef_hr",
            "sigma_mmHg", "hr_term", "note"]


def _f(v, key=""):
    """Blank rather than 'nan' or 'None' for a missing cell: every reader treats blank as NA.

    Time columns are formatted as fixed-point, NOT with %g. A unix timestamp under %.6g renders as
    1.78949e+09 -- which throws away the milliseconds that the sync marks exist to record, and
    %g would also start rounding t_rel_s past 1000 s, i.e. 17 minutes into a session. Fixed 3
    decimals keeps milliseconds exact for both, forever.
    """
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        return f"{float(v):.3f}" if key.startswith("t_") else f"{float(v):.6g}"
    return v


def _clock_ms(t):
    """Local time of day to the millisecond, or blank. The human-readable alignment column."""
    if t is None or not np.isfinite(t):
        return ""
    # Rounded to ms before splitting, so a .9996 fraction carries into the next second instead
    # of wrapping to .000 and leaving the seconds field a full second behind. Must stay identical
    # to app_ptt.Worker._iso and to the timestamp in the recording's filename.
    t = round(float(t), 3)
    return (time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))
            + f".{int(round((t % 1) * 1000)):03d}")


def _write(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: _f(r.get(k), k) for k in cols})
    return path


# ==================================================================== session tables
def calib_rows(session):
    """Cuff readings with their paired features, plus what the model derives from them.

    `usable` is the column to filter on: a reading logged when no transit time was on screen is
    still a real observation and is kept, but it cannot enter a fit. Exporting it without the
    flag is how a 12-reading session turns into a 5-point model without anyone noticing.
    """
    out = []
    for i, p in enumerate(getattr(session, "calib", [])):
        ptt, path = p.get("ptt_ms"), p.get("path_cm")
        sbp, dbp = p.get("sbp"), p.get("dbp")
        mapv = (float(dbp) + (float(sbp) - float(dbp)) / 3.0
                if sbp is not None and dbp is not None else None)
        out.append({**p, "i": i, "map": mapv,
                    "pwv_ms": BP.pwv_ms(ptt, path), "ln_pwv": BP.ln_pwv(ptt, path),
                    "usable": BP.features(ptt, p.get("hr"), path) is not None})
    return out


def mark_rows(session):
    return [{**m, "i": i} for i, m in enumerate(getattr(session, "marks", []))]


def fit_rows(session):
    """The calibration actually in force, so an exported estimate can be reproduced exactly."""
    rows = []
    for t, f in session.fits().items():
        rows.append({"target": t, "mode": f.mode, "n": f.n, "intercept": f.intercept,
                     "coef_ln_pwv": f.coef[0] if len(f.coef) else None,
                     "coef_hr": f.coef[1] if len(f.coef) > 1 else None,
                     "sigma_mmHg": f.sigma, "hr_term": session.hr_term, "note": f.note})
    return rows


def export_session(session, outdir=EXPORT):
    """Write the calib, marks and fit tables for one session. Returns the paths written."""
    stem = f"session_{session.name}"
    wrote = [_write(Path(outdir) / f"{stem}_calib.csv", CALIB_COLS, calib_rows(session)),
             _write(Path(outdir) / f"{stem}_marks.csv", MARK_COLS, mark_rows(session))]
    if session.calib:
        wrote.append(_write(Path(outdir) / f"{stem}_fit.csv", FIT_COLS, fit_rows(session)))
    return wrote


# ================================================================== recording tables
def recording_rows(paths=None):
    """One row per completed recording, from the per-recording json summaries."""
    rows = []
    for j in sorted(paths if paths is not None else DATA.glob("rppg_pose_*.json")):
        try:
            d = json.loads(j.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(d, dict) or not (d.get("fps") or d.get("tag")):
            # A stub or half-written json would otherwise contribute a row of blank cells that
            # reads as a failed recording rather than as a file that was never one.
            print(f"[skip] {j.name}: not a completed recording")
            continue
        rows.append({**d, "file": j.stem,
                     "n_marks": len(d.get("marks", [])), "n_calib": len(d.get("calib", []))})
    return rows


def export_waveform(npz_path, outdir=EXPORT):
    """Per-sample signals for one recording, timestamped in absolute local time.

    Three time columns, because alignment needs different things at different moments:

      t_s      seconds from the start of CAPTURE, matching the timebase the sync marks and the
               cuff readings are recorded in. This is the join key.
      t_wall   unix seconds with millisecond resolution -- unambiguous, timezone-free, and what
               anything automated should use.
      clock    local time of day to the millisecond, for reading against a cuff printout or a
               lab notebook by eye.

    t_s previously started at 0, which was wrong by the warm-up: the signals sit on a grid that
    begins at T[0] >= WARMUP_S (3 s of auto-exposure settling is discarded), so every sample was
    labelled about three seconds earlier than it happened. Sync marks are stored in capture time
    and so were misaligned against these waveforms by that same constant -- invisible, because a
    3 s error still looks like a plausible time. The offset is now taken from the saved `t`.

    Column names carry the site's segment AND its nominal arterial distance, because the column
    order alone does not say which trace is proximal.
    """
    with np.load(npz_path, allow_pickle=False) as z:
        sigs = z["sigs"]
        fs = float(z["fs"])
        seg = z["seg"].astype(str) if "seg" in z else np.array([""] * len(sigs))
        dist = z["dist"] if "dist" in z else np.full(len(sigs), np.nan)
        acc = z["accepted"] if "accepted" in z else np.ones(len(sigs), bool)
        quals = z["quals"] if "quals" in z else np.full(len(sigs), np.nan)
        # Prefer the exact uniform grid the signals were built on; fall back to the raw frame
        # times, then to a bare index. Older recordings carry none of these, and they get the
        # old behaviour with `t_s` starting at 0 -- flagged in the sites file rather than guessed.
        tu = z["tu"] if "tu" in z else (z["t"] if "t" in z else None)
        t_cap = float(z["t_wall_capture"]) if "t_wall_capture" in z else float("nan")

    n = sigs.shape[1]
    if tu is not None and len(tu) == n:
        t_rel = np.asarray(tu, float)
    elif tu is not None and len(tu) >= 1:
        t_rel = float(tu[0]) + np.arange(n) / fs
    else:
        t_rel = np.arange(n) / fs

    names = [f"s{i:02d}_{seg[i] if i < len(seg) else ''}_{dist[i]:.0f}cm"
             if i < len(dist) and np.isfinite(dist[i])
             else f"s{i:02d}_{seg[i] if i < len(seg) else ''}"
             for i in range(len(sigs))]
    out = Path(outdir) / f"{Path(npz_path).stem}_waveform.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        wr = csv.writer(fh)
        # A header comment would break strict CSV readers, so provenance goes in the sibling
        # sites table instead -- accepted/quality per column, keyed by the same name.
        wr.writerow(["t_s", "t_wall", "clock"] + names)
        for k in range(n):
            tw = t_cap + t_rel[k]
            wr.writerow([f"{t_rel[k]:.6f}",
                         f"{tw:.3f}" if np.isfinite(tw) else "",
                         _clock_ms(tw)]
                        + [f"{v:.6g}" for v in sigs[:, k]])

    sites = Path(outdir) / f"{Path(npz_path).stem}_sites.csv"
    _write(sites, ["column", "index", "segment", "distance_cm", "accepted", "quality"],
           [{"column": names[i], "index": i,
             "segment": seg[i] if i < len(seg) else "",
             "distance_cm": dist[i] if i < len(dist) else None,
             "accepted": bool(acc[i]) if i < len(acc) else None,
             "quality": quals[i] if i < len(quals) else None}
            for i in range(len(sigs))])
    return [out, sites]


def export_all(outdir=EXPORT, waveforms=True, session=None):
    """Everything on disk -> CSV. Returns the list of paths written."""
    outdir = Path(outdir)
    wrote = []
    sessions = [session] if session is not None else [
        BP.Session.load(p) for p in sorted(DATA.glob("session_*.json"))]
    for s in sessions:
        wrote += export_session(s, outdir)
    rows = recording_rows()
    if rows:
        wrote.append(_write(outdir / "recordings_summary.csv", SUMMARY_COLS, rows))
        # Cuff readings and marks copied into each recording's own clock -- the join key a
        # per-beat analysis needs, and the reason the recording json carries them at all.
        ev = []
        for r in rows:
            for kind in ("marks", "calib"):
                for e in r.get(kind, []):
                    ev.append({**e, "file": r["file"], "kind": kind[:-1] if kind[-1] == "s"
                               else kind, "tag": r.get("tag")})
        if ev:
            wrote.append(_write(outdir / "recordings_events.csv",
                                ["file", "tag", "kind", "t_in_record_s", "iso", "t_rel_s",
                                 "label", "note", "sbp", "dbp", "ptt_ms", "hr"], ev))
    if waveforms:
        for z in sorted(DATA.glob("rppg_pose_*.npz")):
            try:
                wrote += export_waveform(z, outdir)
            except (OSError, KeyError, ValueError) as e:      # a truncated npz must not abort
                print(f"[warn] {z.name}: {e}")
    return wrote


def main():
    ap = argparse.ArgumentParser(description="Export sessions and recordings to CSV.")
    ap.add_argument("--out", default=str(EXPORT))
    ap.add_argument("--session", help="one session name, e.g. 20260915_1430_s01")
    ap.add_argument("--no-waveforms", action="store_true",
                    help="skip the per-sample files, which are much the largest")
    a = ap.parse_args()
    sess = None
    if a.session:
        p = DATA / f"session_{a.session}.json"
        if not p.exists():
            raise SystemExit(f"no such session: {p}")
        sess = BP.Session.load(p)
    wrote = export_all(a.out, waveforms=not a.no_waveforms, session=sess)
    if not wrote:
        print("nothing to export -- data/ has no sessions or recordings yet")
        return
    for p in wrote:
        print(f"{p.stat().st_size:>9,d}  {p}")
    print(f"\n{len(wrote)} file(s) -> {Path(a.out).resolve()}")


if __name__ == "__main__":
    main()
