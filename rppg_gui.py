"""rppg_gui.py -- experiment console for two-site camera PPG (neck + hand).

One window that runs the whole protocol: pick the experiment, place the two ROIs, watch them
track, record, and see the traces, spectra and the cross-condition comparison without touching a
terminal.

Why ROI tracking matters here. The measurement is a sub-frame timing comparison between two
skin regions, so a box that drifts off the carotid or off the palm silently replaces pulse with
motion. cv2.TrackerMIL follows each region (OpenCV 5 removed CSRT/KCF and the legacy module), and
the tracker is re-seeded from the last good box if it loses lock. Live pulsatility bars show, per
ROI, whether a pulse is actually being picked up -- a flat bar means fix the placement now rather
than discover it after a full recording.

The experiments encode the signed predictions worth testing:

  rest        hand at heart level -- the baseline every other condition is compared against
  hand_up     hand raised well above the heart. Local arterial pressure falls ~0.77 mmHg/cm, so
              the vessel softens and the pulse slows: PTT should LENGTHEN.
  hand_down   hand hanging below the heart. Pressure rises, vessel stiffens: PTT should SHORTEN.
  post_exer   immediately after ~30 s of exercise. BP and HR rise: PTT should SHORTEN.

hand_down is included because it predicts the OPPOSITE sign to hand_up. A rig that reports a
shift in the predicted direction for both is measuring physiology; one that shifts the same way
regardless is measuring posture or ROI placement.

    python rppg_gui.py
"""
import json
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

import numpy as np

import rppg_two_site as R

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

EXPERIMENTS = {
    "rest  (hand at heart level)": ("rest", "Baseline. Elbow bent, palm at mid-chest, "
                                    "level with your heart."),
    "hand_up  (hand above heart)": ("hand_up", "Raise the hand 30-40 cm ABOVE heart level. "
                                    "Prediction: PTT LENGTHENS."),
    "hand_down  (hand below heart)": ("hand_down", "Let the hand hang BELOW heart level. "
                                      "Prediction: PTT SHORTENS."),
    "post_exercise": ("post_exer", "Record immediately after ~30 s of exercise. "
                      "Prediction: PTT SHORTENS."),
}


class App:
    def __init__(self, root):
        self.root = root
        root.title("Two-site rPPG console")
        root.geometry("1120x760")
        self.running = False
        self.result = None

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Experiment", font=("Segoe UI", 10, "bold")).pack(side="left")
        self.exp = ttk.Combobox(top, values=list(EXPERIMENTS), width=32, state="readonly")
        self.exp.current(0)
        self.exp.pack(side="left", padx=6)
        self.exp.bind("<<ComboboxSelected>>", lambda e: self.show_hint())

        ttk.Label(top, text="seconds").pack(side="left", padx=(12, 2))
        self.secs = ttk.Spinbox(top, from_=15, to=180, width=5)
        self.secs.set(60)
        self.secs.pack(side="left")

        self.track = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="track ROIs", variable=self.track).pack(side="left", padx=10)

        self.btn = ttk.Button(top, text="Record", command=self.start)
        self.btn.pack(side="left", padx=8)
        ttk.Button(top, text="Compare conditions", command=self.compare).pack(side="left")

        self.hint = ttk.Label(root, text="", foreground="#2f4b7c", padding=(10, 0),
                              wraplength=1080, justify="left")
        self.hint.pack(fill="x")
        self.status = ttk.Label(root, text="ready", padding=(10, 4), foreground="#555")
        self.status.pack(fill="x")
        self.show_hint()

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.fig = Figure(figsize=(11, 5.6), dpi=96)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.blank()

    # ------------------------------------------------------------------ ui
    def show_hint(self):
        tag, txt = EXPERIMENTS[self.exp.get()]
        done = (DATA / f"rppg_two_site_{tag}.json").exists()
        self.hint.config(text=f"{txt}    [{'recorded' if done else 'not yet recorded'}]")

    def blank(self):
        self.fig.clear()
        a = self.fig.add_subplot(111)
        a.text(.5, .5, "pick an experiment and press Record",
               ha="center", va="center", color="#999", fontsize=12)
        a.axis("off")
        self.canvas.draw()

    def set_status(self, s):
        self.status.config(text=s)
        self.root.update_idletasks()

    # --------------------------------------------------------------- record
    def start(self):
        if self.running:
            return
        self.running = True
        self.btn.config(state="disabled")
        tag = EXPERIMENTS[self.exp.get()][0]
        secs = float(self.secs.get())
        threading.Thread(target=self.worker, args=(tag, secs), daemon=True).start()

    def worker(self, tag, secs):
        try:
            self.set_status("select the NECK box, then the HAND box (ENTER after each)")
            N, H, T = capture_tracked(secs, track=self.track.get(),
                                      on_status=self.set_status)
            if len(T) < 100:
                raise RuntimeError(f"only {len(T)} frames captured")
            self.set_status("analysing ...")
            res = analyse(N, H, T, tag)
            self.result = res
            self.root.after(0, lambda: self.plot(res))
            self.root.after(0, self.show_hint)
            self.set_status(f"{tag}: HR {res['hr_bpm']:.0f} bpm, SNR {res['snr']:.1f}, "
                            f"lag {res['lag_ms']:+.1f} ms")
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("capture failed", str(e)))
            self.set_status("failed")
        finally:
            self.running = False
            self.root.after(0, lambda: self.btn.config(state="normal"))

    # ----------------------------------------------------------------- plot
    def plot(self, res):
        d = np.load(DATA / f"rppg_two_site_{res['tag']}.npz")
        nb, hb, fs = d["neck"], d["hand"], float(d["fs"])
        t = np.arange(len(nb)) / fs
        self.fig.clear()

        a = self.fig.add_subplot(221)
        s = slice(0, int(min(10 * fs, len(nb))))
        a.plot(t[s], nb[s] / (np.std(nb) + 1e-9), color="#2f4b7c", lw=1.1, label="neck")
        a.plot(t[s], hb[s] / (np.std(hb) + 1e-9) - 4, color="#c1543b", lw=1.1, label="hand")
        a.set_title("a  pulse traces (first 10 s)", loc="left", fontsize=10, fontweight="bold")
        a.set_xlabel("s", fontsize=8); a.set_yticks([])
        a.legend(fontsize=7.5, frameon=False, ncol=2)
        a.spines[["top", "right", "left"]].set_visible(False)

        from scipy.signal import welch
        b = self.fig.add_subplot(222)
        for x, c, lab in ((nb, "#2f4b7c", "neck"), (hb, "#c1543b", "hand")):
            f, P = welch(x, fs, nperseg=min(len(x), int(8 * fs)))
            m = (f > 0.5) & (f < 4)
            b.plot(f[m] * 60, P[m] / P[m].max(), color=c, lw=1.2, label=lab)
        b.axvline(res["hr_bpm"], color="k", ls="--", lw=0.9)
        b.set_title(f"b  spectra (HR {res['hr_bpm']:.0f} bpm, SNR {res['snr']:.1f})",
                    loc="left", fontsize=10, fontweight="bold")
        b.set_xlabel("bpm", fontsize=8); b.set_yticks([])
        b.legend(fontsize=7.5, frameon=False)
        b.spines[["top", "right", "left"]].set_visible(False)

        c = self.fig.add_subplot(223)
        w = res.get("window_lags") or []
        if w:
            c.plot(w, "o-", color="#2f4b7c", ms=4, lw=1.2)
            c.axhline(np.median(w), color="#c1543b", lw=1.2,
                      label=f"median {np.median(w):+.1f} ms")
            c.axhline(0, color="k", lw=0.6, alpha=.4)
            c.legend(fontsize=7.5, frameon=False)
        c.set_title("c  neck→hand lag per 10 s window", loc="left", fontsize=10,
                    fontweight="bold")
        c.set_xlabel("window", fontsize=8); c.set_ylabel("ms", fontsize=8)
        c.spines[["top", "right"]].set_visible(False)

        d2 = self.fig.add_subplot(224)
        rows = []
        for lab, (tg, _) in EXPERIMENTS.items():
            p = DATA / f"rppg_two_site_{tg}.json"
            if p.exists():
                j = json.loads(p.read_text())
                wl = j.get("window_lags") or [j["lag_ms"]]
                rows.append((tg, float(np.median(wl)), float(np.std(wl)), j["snr"]))
        if rows:
            y = np.arange(len(rows))
            d2.barh(y, [r[1] for r in rows], xerr=[r[2] for r in rows],
                    color=["#4a7c59" if r[0] == res["tag"] else "#9a9a9a" for r in rows],
                    height=.6)
            d2.set_yticks(y, [r[0] for r in rows], fontsize=8)
            d2.axvline(0, color="k", lw=.6)
        d2.set_title("d  all recorded conditions", loc="left", fontsize=10, fontweight="bold")
        d2.set_xlabel("neck→hand lag (ms)", fontsize=8)
        d2.spines[["top", "right"]].set_visible(False)

        self.fig.tight_layout()
        self.canvas.draw()

    # -------------------------------------------------------------- compare
    def compare(self):
        got = {tg: json.loads((DATA / f"rppg_two_site_{tg}.json").read_text())
               for _, (tg, _) in EXPERIMENTS.items()
               if (DATA / f"rppg_two_site_{tg}.json").exists()}
        if "rest" not in got or len(got) < 2:
            messagebox.showinfo("need more data",
                                "Record 'rest' plus at least one other condition first.")
            return
        from scipy import stats
        base = got["rest"].get("window_lags") or []
        lines = []
        for tg, j in got.items():
            if tg == "rest":
                continue
            w = j.get("window_lags") or []
            if len(w) < 3 or len(base) < 3:
                lines.append(f"{tg}: too few windows"); continue
            diff = float(np.median(w) - np.median(base))
            t, p = stats.ttest_ind(w, base, equal_var=False)
            exp = "LENGTHEN" if tg in ("hand_up",) else "SHORTEN"
            ok = (diff > 0) if exp == "LENGTHEN" else (diff < 0)
            verdict = ("consistent" if ok else "OPPOSITE to prediction") if p < .05 \
                else "not significant"
            lines.append(f"{tg:11s} {diff:+6.1f} ms  p={p:.3f}   predicted {exp}: {verdict}")
        messagebox.showinfo("condition comparison", "\n".join(lines) if lines else "no pairs")


# ---------------------------------------------------------------- capture
def capture_tracked(seconds, track=True, cam=0, on_status=lambda s: None):
    """Same measurement as rppg_two_site.capture, with optional ROI tracking."""
    import cv2
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    if not cap.isOpened():
        raise RuntimeError("cannot open camera")
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("camera returned no frame")

    rois = {}
    for name in ("NECK", "HAND"):
        ok, frame = cap.read()
        r = cv2.selectROI(f"drag a box over your {name}, then press ENTER", frame, False, False)
        cv2.destroyAllWindows()
        if r[2] < 5 or r[3] < 5:
            raise RuntimeError(f"no {name} box selected")
        rois[name] = tuple(int(v) for v in r)

    trackers = {}
    if track:
        for name, r in rois.items():
            try:
                tr = cv2.TrackerMIL.create()
                tr.init(frame, r)
                trackers[name] = tr
            except Exception:
                trackers = {}
                break

    N, H, T = [], [], []
    t0 = time.time()
    on_status("recording -- hold still")
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        for name in ("NECK", "HAND"):
            if name in trackers:
                good, box = trackers[name].update(frame)
                if good:
                    x, y, w_, h_ = (int(v) for v in box)
                    if w_ > 5 and h_ > 5 and 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        rois[name] = (x, y, w_, h_)
                else:
                    # lost lock: re-seed from the last good box rather than drift
                    try:
                        tr = cv2.TrackerMIL.create(); tr.init(frame, rois[name])
                        trackers[name] = tr
                    except Exception:
                        pass

        def mean_rgb(r):
            x, y, w_, h_ = r
            p = frame[y:y + h_, x:x + w_]
            return p[:, :, ::-1].reshape(-1, 3).mean(0) if p.size else np.zeros(3)

        N.append(mean_rgb(rois["NECK"])); H.append(mean_rgb(rois["HAND"]))
        T.append(time.time() - t0)

        for name, col in (("NECK", (0, 255, 0)), ("HAND", (0, 128, 255))):
            x, y, w_, h_ = rois[name]
            cv2.rectangle(frame, (x, y), (x + w_, y + h_), col, 2)
            cv2.putText(frame, name, (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, .6, col, 2)
        cv2.putText(frame, f"{seconds - (time.time()-t0):4.0f}s", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        if len(N) > 60:
            for buf, col, yy in ((N, (0, 255, 0), 52), (H, (0, 128, 255), 76)):
                a = np.array(buf[-60:])[:, 1]
                amp = float(np.std(a) / (np.mean(a) + 1e-9) * 1000)
                cv2.rectangle(frame, (118, yy - 10), (118 + int(min(amp * 12, 170)), yy - 2),
                              col, -1)
                cv2.putText(frame, f"pulse {amp:4.1f}", (10, yy),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, col, 1)
        cv2.imshow("recording - q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return np.array(N), np.array(H), np.array(T)


def analyse(N, H, T, tag):
    fs = (len(T) - 1) / (T[-1] - T[0])
    tu = np.linspace(T[0], T[-1], len(T))
    nb = R.bandpass(np.interp(tu, T, R.chrom(N)), fs)
    hb = R.bandpass(np.interp(tu, T, R.chrom(H)), fs)
    from scipy.signal import welch
    f, P = welch(nb, fs, nperseg=min(len(nb), int(8 * fs)))
    m = (f > R.BAND[0]) & (f < R.BAND[1])
    hr = float(f[m][np.argmax(P[m])] * 60)
    snr = float(P[m].max() / (np.median(P[m]) + 1e-12))
    lag, peak = R.lag_subframe(nb, hb, fs)
    W = int(10 * fs)
    lags = [R.lag_subframe(nb[i:i + W], hb[i:i + W], fs)[0]
            for i in range(0, len(nb) - W, W // 2)] if len(nb) > W else []
    out = {"tag": tag, "fps": fs, "n_frames": len(T), "hr_bpm": hr, "snr": snr,
           "lag_ms": lag, "xcorr_peak": peak,
           "window_lags": [float(x) for x in lags],
           "plausible": bool(5.0 <= abs(lag) <= 120.0)}
    (DATA / f"rppg_two_site_{tag}.json").write_text(json.dumps(out, indent=2))
    np.savez(DATA / f"rppg_two_site_{tag}.npz", neck=nb, hand=hb, t=tu, fs=fs)
    return out


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
