# mechdemo — does a blood-pressure model *use* the physics it should?

A small, self-contained demo of **mechanistic faithfulness auditing** for cuffless
blood-pressure (BP) estimation from ECG + PPG. Accuracy, a linear probe, and a causal
audit measure three *different* things — and only the causal audit tells you whether a
model actually uses pulse transit time (PTT), the physiological cue it is supposed to rely on.

> **Governing physiology.** Higher BP → stiffer artery → faster pulse wave → **shorter PTT**
> (`BP = A/PTT^p + B`). A faithful model's predicted BP should therefore go **down** when PTT
> goes **up**. DBP is the most PTT-coupled target (the pulse propagates during diastole).

## The app (start here)

```bash
pip install -r requirements.txt
streamlit run app_faithfulness.py
```

Three tabs:

1. **Synthetic sandbox** — a controlled testbed with a dial `alpha` that sets how much a model
   routes BP through the real PTT pathway. Only the causal **donor-swap** tracks `alpha`;
   accuracy and a linear probe do not. This *validates the audit*.
2. **Real data (VitalDB)** — a small model on real ECG + PPG. A single amplitude-normalized
   window can't fix a subject's absolute BP, so the raw model sits near a predict-the-mean
   baseline; **per-subject calibration** makes it genuinely accurate (MAE ~11 mmHg SBP). Yet the
   causal PTT-shift audit shows it does **not** use transit time. Shown for the `cal_free` and
   `cal_based` official subsets.
3. **Can we make it faithful?** — the capstone. An auxiliary objective that reconstructs the
   measured PTT **4×'s how decodable PTT is** (probe R² 0.12 → 0.52), yet the **donor-swap**
   (patching only that PTT direction into the activations) leaves DBP unmoved. *Decodable is not
   used; faithfulness is a causal property.*

## What's in here

```
app_faithfulness.py        the demo (three tabs); reads precomputed results in data/
mechlib.py                 self-contained toolkit: PTT detection, causal audits
                           (input-shift + subspace donor-swap), linear probe, calibration
precompute_capstone.py     regenerates data/capstone.* from the bundled subset (reproducible)
notebooks/
  investigate_data.ipynb   what's in the data + verify the dataloader (channel identities)
  models_and_reconstruction.ipynb
                           CNN / local self-attention / local cross-attention regressors,
                           convergence curves, calibration, and ABP reconstruction
                           (conv + cross-attention), reconstruct-then-read-off BP
data/
  vitaldb_mini.npz         bundled VitalDB subset, official patient-disjoint splits
                           (float16, ~42 MB): ECG/PPG/ABP waveforms + [SBP, DBP] labels
  realdata_*.json/.npz     precomputed real-data audit results for the app
  capstone.json/.npz       precomputed vanilla-vs-aux results for the app
```

Channel identities are verified from physics (kurtosis + the ABP↔SBP anchor):
**ECG = 0, PPG = 1, ABP = 2**. ECG and PPG arrive amplitude-normalized to [0, 1]; only ABP is
in mmHg (its per-segment peak is the SBP label).

## Reproduce

```bash
python precompute_capstone.py          # regenerates data/capstone.* from data/vitaldb_mini.npz
jupyter lab notebooks/                 # run the notebooks end-to-end
```

`data/realdata_*.json` were precomputed from the full VitalDB `cal_free` / `cal_based` test
subsets (not bundled); everything else regenerates from `data/vitaldb_mini.npz`.

## Key finding & limitation (important)

The synthetic sandbox is the **rigorously validated** part: the law is true by construction and
the donor-swap correctly tracks faithfulness.

On **real data**, we established a prerequisite that turned out to be decisive: **the PTT→BP law
is essentially absent in resting ICU monitoring data.** Across *two* datasets (VitalDB and raw
MIMIC-BP), with physiological PTT detectors, quality filtering, and SBP/DBP/MAP targets, the
within-subject PTT–BP correlation sits at chance (frac-negative ≈ 0.47–0.54). This is not a
preprocessing or channel-alignment artifact (MIMIC preserves the true 298 ms lag and still shows
no law) — single-site PAT is simply a weak BP surrogate without active BP manipulation. The
strong, real relationship in this data is **full pressure-waveform morphology** (ECG+PPG → ABP
reconstruction, corr **0.93**).

Consequently the real-data audit results here should be read as *"applying the method to real
signals where the PTT law is weak,"* not as a clean faithfulness verdict. A crisp real-data
positive control needs **BP-manipulation data** (e.g. the PhysioNet Pulse Transit Time PPG
exercise dataset); that investigation is ongoing.

## Takeaway

A model can be **accurate** and have the mechanism **decodable** in its activations, yet still
be **right for the wrong reason** — not causally using it. Standard evaluation and linear
probes miss this; a causal donor-swap catches it. And a faithfulness audit is only meaningful
when the governing law is actually present in the data — verifying that is step one.
