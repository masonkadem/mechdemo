"""calib_finetune_table.py -- the calibration-burden table, but with FINE-TUNING instead of anchors.

`calib_all_models.py` answers "how much does an OFFSET fitted from k cuff readings buy?" -- one
number per patient, the classical anchor. This answers the harder question: given the same k
readings, how much does fitting a whole LINEAR HEAD on the model's representation buy?

Same models, same patients, same k, so the two tables are directly comparable.

  deep nets     penultimate-layer activations (forward hook on the final Linear) -> ridge head
  LightGBM      the 83 audited waveform features, optionally + age/sex/BMI      -> ridge head
  hand-crafted  PTT equation and the PPG-only dicrotic-notch delay, for scale

Every arm is fitted the same way -- ridge with alpha scaled by features-per-sample -- so the
comparison is the information in each representation, not the fitting procedure.

    python calib_finetune_table.py                # full table
    python calib_finetune_table.py --latex        # + LaTeX
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

import mechlib
import ood_benchmark as ob
import lightgbm_arm as gbm
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCHS = ["lenet1d", "inception1d", "xresnet1d50", "xresnet1d101", "transformer"]
KS = [0, 1, 2, 3, 5, 10, 20, 50]
GAP = 50
TARGET = 1                      # DBP


def penultimate(model, X, bs=256):
    """Activations feeding the final Linear, via a forward hook.

    A hook rather than surgery on each architecture: lenet/inception/xresnet/transformer all end
    differently, and rebuilding each one to expose features would risk silently changing the
    network we are supposed to be measuring.
    """
    last = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
    buf = []
    h = last.register_forward_hook(lambda _m, inp, _o: buf.append(inp[0].detach().cpu().numpy()))
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            model(torch.tensor(X[i:i + bs]).permute(0, 2, 1).to(DEVICE))
    h.remove()
    return np.concatenate(buf)


def head_curve(F, y, rows, ks=KS, gap=GAP, alpha=10.0):
    """Fit a per-patient ridge head on the first k segments; score after a gap."""
    out = {}
    for k in ks:
        errs = []
        for i in rows.values():
            if k + gap >= len(i) - 5:
                continue
            c, t = i[:k], i[k + gap:]
            yc, yt = y[c], y[t]
            if k == 0:
                continue                                  # nothing to fit: reported separately
            A, B = F[c], F[t]
            mu, sd = A.mean(0), A.std(0)
            keep = sd > 1e-6
            if not keep.any() or len(A) < 2:
                errs.append(float(np.abs(yc.mean() - yt).mean()))
                continue
            a = alpha * max(1.0, int(keep.sum()) / len(A))
            p = Ridge(alpha=a).fit((A[:, keep] - mu[keep]) / sd[keep], yc).predict(
                (B[:, keep] - mu[keep]) / sd[keep])
            errs.append(float(np.abs(p - yt).mean()))
        out[k] = float(np.median(errs)) if errs else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    d = np.load(DATA / "vitaldb_full_calfree.npz", mmap_mode="r")
    g, Y = np.array(d["gte"]), np.array(d["yte"])
    y = Y[:, TARGET]
    patients = np.unique(g)
    rows = {p: np.where(g == p)[0] for p in patients}
    X = mechlib.normalize(np.array(d["Xte"])[:, :, [ECG, PPG]])
    print(f"[data] {len(patients)} held-out patients x {len(rows[patients[0]])} segments", flush=True)

    res = {}

    # ---- floor: predict the patient's own mean of their k readings ----------
    floor = {}
    for k in KS:
        if k == 0:
            continue
        e = [float(np.abs(y[i[:k]].mean() - y[i[k + GAP:]]).mean())
             for i in rows.values() if k + GAP < len(i) - 5]
        floor[k] = float(np.median(e))
    res["average of k readings"] = {"kind": "floor", "curve": floor, "n_params": 1}
    print(f"{'average of k readings':28s} " +
          " ".join(f"{floor.get(k, float('nan')):6.2f}" for k in KS if k), flush=True)

    # ---- hand-crafted single features --------------------------------------
    for name, path, invsq in (("PTT equation", "_calib_ptt_maxslope.npy", True),
                              ("PPG-only dicrotic notch", "_calib_ppg_notch.npy", False)):
        f = DATA / path
        if not f.exists():
            print(f"[skip] {name}: {path} not found"); continue
        v = np.load(f)
        z = 1.0 / np.clip(v, 0.02, None) ** 2 if invsq else v.copy()
        for i in rows.values():                       # fill gaps with the patient's own median
            m = np.nanmedian(z[i][np.isfinite(z[i])]) if np.isfinite(z[i]).any() else 0.0
            z[i] = np.where(np.isfinite(z[i]), z[i], m)
        c = head_curve(z[:, None], y, rows)
        res[name] = {"kind": "hand", "curve": c, "n_params": 2}
        print(f"{name:28s} " + " ".join(f"{c[k]:6.2f}" for k in KS if k), flush=True)

    # ---- LightGBM feature sets ---------------------------------------------
    fp = DATA / "_feat_full_ALLtrain.pkl"
    if fp.exists():
        full = pickle.load(open(fp, "rb"))
        Fte = full["Fte"]
        keys = [k for k in full["Ftr"]
                if np.isfinite(np.asarray(full["Ftr"][k], float)).mean() > 0.3
                and np.nanstd(np.asarray(full["Ftr"][k], float)) > 1e-9]
        n = len(y)
        M = np.column_stack([np.asarray(Fte.get(k, np.full(n, np.nan)), float)[:n] for k in keys])
        M = gbm._impute(M, gbm.column_medians(M))
        c = head_curve(M, y, rows)
        res[f"waveform features ({len(keys)})"] = {
            "kind": "features", "curve": c, "n_params": M.shape[1] + 1}
        print(f"{f'waveform features ({len(keys)})':28s} "
              + " ".join(f"{c[k]:6.2f}" for k in KS if k), flush=True)

        # Demographics deliberately NOT added as a per-patient arm. Age and sex have exactly
        # zero within-patient variance (BMI ~2e-1 from rounding), so a per-patient head cannot
        # use them -- the standardisation guard drops them and the row would be a duplicate of
        # the one above. They can only help a POPULATION model, which is measured separately.
        demo = np.c_[np.array(d["age_te"]), np.array(d["sex_te"]), np.array(d["bmi_te"])]
        sd_within = max(float(demo[i].std(0).max()) for i in rows.values())
        print(f"{'  [demographics]':28s} within-patient SD {sd_within:.2e} -> cannot enter a "
              f"per-patient head; see the k=0 population row", flush=True)
        res["_demo_note"] = {"within_patient_sd": sd_within}
    else:
        print(f"[skip] feature table {fp.name} not found")

    # ---- deep nets ----------------------------------------------------------
    for mk in ARCHS:
        ck_path = ROOT / "models" / f"{mk}_ecgppg_full.pt"
        if not ck_path.exists():
            print(f"[skip] {mk}: no checkpoint"); continue
        try:
            ck = torch.load(ck_path, map_location=DEVICE, weights_only=False)
            net = ob.build_model(mk, n_ch=2, L=1250)
            net.load_state_dict(ck["state_dict"])
            net.to(DEVICE).eval()
        except Exception as exc:
            print(f"[skip] {mk}: {str(exc)[:60]}"); continue
        F = penultimate(net, X)
        c = head_curve(F, y, rows)
        res[mk] = {"kind": "deep", "curve": c, "n_params": F.shape[1] + 1,
                   "n_weights": int(sum(p.numel() for p in net.parameters()))}
        print(f"{mk + f' ({F.shape[1]}d)':28s} " + " ".join(f"{c[k]:6.2f}" for k in KS if k),
              flush=True)

    # ---- the self-supervised transformer, for continuity with the notebooks --
    mp = ROOT / "models" / "mae_probe.pt"
    if mp.exists():
        from mae_probe import Supervised
        ck = torch.load(mp, map_location=DEVICE, weights_only=False)
        enc = Supervised().to(DEVICE)
        miss, unexp = enc.load_state_dict(
            {k: v for k, v in ck["mae"].items() if k.startswith(("embed.", "pos", "enc."))},
            strict=False)
        assert not unexp and all(k.startswith("head.") for k in miss)
        enc.eval()
        with torch.no_grad():
            F = np.concatenate([enc.represent(torch.tensor(X[i:i + 512]).to(DEVICE)).cpu().numpy()
                                for i in range(0, len(X), 512)])
        c = head_curve(F, y, rows)
        res["MAE transformer (self-sup)"] = {"kind": "deep", "curve": c,
                                             "n_params": F.shape[1] + 1}
        print(f"{'MAE transformer (self-sup)':28s} " + " ".join(f"{c[k]:6.2f}" for k in KS if k),
              flush=True)

    # ---- k=0: where demographics CAN help, since no per-patient fit exists ---
    if fp.exists():
        Ftr = full["Ftr"]
        ntr = len(full["ytr"])
        Mtr = np.column_stack([np.asarray(Ftr.get(k, np.full(ntr, np.nan)), float) for k in keys])
        med = gbm.column_medians(Mtr)
        Mtr = gbm._impute(Mtr, med)
        ytr = full["ytr"][:, TARGET]
        Dtr = np.c_[np.array(d["age_tr"])[:ntr], np.array(d["sex_tr"])[:ntr],
                    np.array(d["bmi_tr"])[:ntr]]

        def pop(A_tr, A_te):
            p = Ridge(alpha=10.0).fit(A_tr, ytr).predict(A_te)
            return float(np.median([np.abs(p[i] - y[i]).mean() for i in rows.values()]))

        k0_w = pop(Mtr, M)
        k0_d = pop(np.c_[Mtr, Dtr], np.c_[M, demo])
        res["_k0"] = {"waveform": k0_w, "waveform+demo": k0_d}
        print(f"\n[k=0, no calibration] population model on waveform features {k0_w:.2f} mmHg; "
              f"+ age/sex/BMI {k0_d:.2f} mmHg ({k0_w - k0_d:+.2f})", flush=True)

    (DATA / "calib_finetune_table.json").write_text(json.dumps(res, indent=2, default=float))

    # ---- final table --------------------------------------------------------
    ks = [k for k in KS if k]
    hdr = f"\n{'model':30s} {'params':>7s} " + " ".join(f"{'k=' + str(k):>7s}" for k in ks)
    print(hdr); print("-" * len(hdr))
    order = ["floor", "hand", "features", "deep"]
    for kind in order:
        for name, v in res.items():
            if name.startswith("_") or v.get("kind") != kind:
                continue
            print(f"{name:30s} {v['n_params']:7d} " +
                  " ".join(f"{v['curve'].get(k, float('nan')):7.2f}" for k in ks))
    print("\nDBP MAE (mmHg), median over patients. Each model's representation gets a per-patient")
    print("ridge head fitted on the first k segments and is scored after a 50-segment gap.")
    print(f"\n[done] data/calib_finetune_table.json")

    if args.latex:
        write_latex(res, ks)


def write_latex(res, ks):
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Calibration burden under \emph{fine-tuning}: diastolic BP mean absolute "
         r"error (mmHg) when a per-patient linear head is fitted on each model's representation "
         r"from $k$ cuff readings, scored on held-out segments after a gap. Median over 144 "
         r"calibration-free test patients.}",
         r"\label{tab:calib-finetune}",
         r"\begin{tabular}{l r " + "r" * len(ks) + "}", r"\toprule",
         r"Model & Params & " + " & ".join(f"$k{{=}}{k}$" for k in ks) + r" \\", r"\midrule"]
    labels = {"floor": r"\textit{No model}", "hand": r"\textit{Hand-crafted}",
              "features": r"\textit{Feature models}", "deep": r"\textit{Deep networks}"}
    for kind in ("floor", "hand", "features", "deep"):
        block = [(n, v) for n, v in res.items()
                 if not n.startswith("_") and v.get("kind") == kind]
        if not block:
            continue
        L.append(f"\\multicolumn{{{len(ks)+2}}}{{l}}{{{labels[kind]}}} \\\\")
        for n, v in block:
            cells = " & ".join(
                "--" if not np.isfinite(v["curve"].get(k, float("nan")))
                else f"{v['curve'][k]:.2f}" for k in ks)
            L.append(f"\\quad {n.replace('_', ' ')} & {v['n_params']:,} & {cells} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = ROOT / "TABLE_calib_finetune.tex"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"[done] {out.name}")


if __name__ == "__main__":
    main()
