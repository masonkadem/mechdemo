"""fig_equation_variants.py -- one figure: which timing feature makes the calibration equation
work best, and how all of them compare to a fine-tuned model.

Every arm is the SAME 2-parameter per-patient fit -- `DBP = a + b * x` -- so the only thing that
changes is x, the timing feature. That isolates the measurement from the model.

    python fig_equation_variants.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
GAP = 50
KS = [1, 2, 3, 5, 10, 20, 30, 50, 100]

# (label, cache file, invert-and-square?, crosses channels?)
FEATURES = [
    ("PTT  ECG→PPG  (2 sensors)",      "_calib_ptt_maxslope.npy", True,  True),
    ("PAT  ECG→ABP  (invasive)",       "_calib_pat_abp.npy",      True,  True),
    ("PTT  ABP→PPG  (invasive)",       "_calib_ptt_true.npy",     True,  True),
    ("PPG rise time  (1 sensor)",      "_calib_ppg_rise.npy",     False, False),
    ("PPG notch delay  (1 sensor)",    "_calib_ppg_notch.npy",    False, False),
    ("PPG pulse width  (1 sensor)",    "_calib_ppg_width.npy",    False, False),
    ("PPG augmentation idx (1 sensor)", "_calib_ppg_ai.npy",      False, False),
]


def fit_predict(A, yc, B, alpha=10.0):
    """Ridge with alpha scaled by features-per-sample, identical across every arm."""
    if len(A) < 2:
        return np.full(len(B), yc.mean())
    mu, sd = A.mean(0), A.std(0)
    keep = sd > 1e-6
    if not keep.any():
        return np.full(len(B), yc.mean())
    A, B, mu, sd = A[:, keep], B[:, keep], mu[keep], sd[keep]
    a = alpha * max(1.0, A.shape[1] / len(A))
    return Ridge(alpha=a).fit((A - mu) / sd, yc).predict((B - mu) / sd)


def main():
    d = np.load(DATA / "vitaldb_full_calfree.npz", mmap_mode="r")
    g, y = np.array(d["gte"]), np.array(d["yte"])[:, 1]
    patients = np.unique(g)
    rows = {p: np.where(g == p)[0] for p in patients}

    def column(v, invsq):
        """One regressor, gaps filled with the patient's own median."""
        z = 1.0 / np.clip(v, 0.02, None) ** 2 if invsq else v.astype(float).copy()
        out = z.copy()
        for i in rows.values():
            ok = np.isfinite(z[i])
            out[i] = np.where(ok, z[i], np.median(z[i][ok]) if ok.any() else 0.0)
        return out[:, None]

    def curve(M):
        c = {}
        for k in KS:
            e = [np.abs(fit_predict(M[i[:k]], y[i[:k]], M[i[k + GAP:]]) - y[i[k + GAP:]]).mean()
                 for i in rows.values() if k + GAP < len(i) - 5]
            c[k] = float(np.median(e))
        return c

    res, meta = {}, {}
    for label, fn, invsq, crosses in FEATURES:
        p = DATA / fn
        if not p.exists():
            print(f"[skip] {label}: {fn} missing"); continue
        v = np.load(p)
        res[label] = curve(column(v, invsq))
        meta[label] = crosses
        safe = label.replace("→", "->")
        print(f"{safe:34s} k=5 {res[label][5]:5.2f}  k=20 {res[label][20]:5.2f}  "
              f"k=100 {res[label][100]:5.2f}   (valid {np.isfinite(v).mean()*100:3.0f}%)")

    # references: no timing feature at all, and the fine-tuned model
    floor = {k: float(np.median([np.abs(y[i[:k]].mean() - y[i[k + GAP:]]).mean()
                                 for i in rows.values() if k + GAP < len(i) - 5])) for k in KS}
    print(f"{'average of k cuff readings':34s} k=5 {floor[5]:5.2f}  k=20 {floor[20]:5.2f}  "
          f"k=100 {floor[100]:5.2f}")
    cc = DATA / "calibration_cost.json"
    model = None
    if cc.exists():
        g50 = {r["k"]: r for r in json.load(open(cc)) if r["gap"] == 50}
        model = {k: g50[k]["head"] for k in KS if k in g50}

    json.dump({"equations": res, "floor": floor, "model": model},
              open(DATA / "equation_variants.json", "w"), indent=2)

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)
    best = min(res, key=lambda n: res[n][20])
    greens = ["#0072B2", "#56B4E9", "#CC79A7"]      # 1-sensor PPG variants
    greys = ["#8C8C8C", "#B0B0B0", "#C8C8C8"]       # cross-channel variants
    gi = si = 0
    for label in res:
        if meta[label]:
            c, ls, lw = greys[si % 3], "--", 1.5; si += 1
        else:
            c, ls, lw = greens[gi % 3], "-", 1.8; gi += 1
        if label == best:
            c, lw = "#009E73", 3.0
        ax.plot(KS, [res[label][k] for k in KS], "o" + ls, color=c, lw=lw, ms=4, label=label)

    ax.plot(KS, [floor[k] for k in KS], "s-", color="#333333", lw=1.6, ms=4,
            label="no timing feature (average of cuff readings)")
    if model:
        mk = [k for k in KS if k in model]
        ax.plot(mk, [model[k] for k in mk], "o-", color="#D55E00", lw=3.0, ms=5,
                label="transformer + fine-tuned linear head")

    ax.axvspan(0.85, 2.6, color="#BBB", alpha=0.16, lw=0)
    ax.annotate("too few\nreadings", (1.5, 7.75), ha="center", fontsize=8, color="#666")
    ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS, fontsize=9)
    ax.minorticks_off()
    ax.set_xlabel("cuff readings used to calibrate this patient (k)")
    ax.set_ylabel("diastolic BP error (mmHg)")
    ax.set_title("Which timing feature makes the 2-parameter equation work?\n"
                 "same fit, same patients — only the feature changes", loc="left", fontsize=11)
    # the two numbers the slide is about
    i20 = KS.index(20)
    ax.annotate("", (20, res[best][20]), (20, model[20] if model else res[best][20]),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.6))
    if model:
        ax.annotate(f"  model is {res[best][20]-model[20]:.2f} mmHg better\n"
                    "  than the best equation",
                    (20.5, (res[best][20] + model[20]) / 2), fontsize=9, va="center")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=1)
    ax.spines[["top", "right"]].set_visible(False)
    out = ROOT / "figures" / "fig_equation_variants.png"
    fig.savefig(out, dpi=200)
    print(f"\nbest equation feature at k=20: {best} ({res[best][20]:.2f} mmHg)")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
