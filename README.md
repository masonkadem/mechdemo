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
3. **Faithful to what?** — the capstone. An auxiliary objective reconstructs the **ABP pressure
   waveform** from the shared features. This rebuilds the wave at **corr 0.91** and makes PTT
   **~2× more decodable** (probe R² 0.13 → 0.29), yet the **donor-swap** leaves the causal PTT
   response at chance. Running the audit across *candidate cues* (a mechanism profile) then shows
   what the model *does* use: it is causally faithful to **PPG wave-shape / stiffness morphology**
   (frac-correct 0.62) but **not** to arrival time (PAT frac 0.26). Accuracy, decodability, and
   reconstruction fidelity are each independent of faithfulness — and the audit says faithful *to
   what*, not just faithful/unfaithful. A reconstruct-then-read-off model (predict the pressure
   wave, read BP off its peak/trough) is included as the faithful-by-design alternative.

## What's in here

```
app_faithfulness.py        the demo (three tabs); reads precomputed results in data/
mechlib.py                 self-contained toolkit: PTT detection, candidate-cue extraction,
                           causal audits (input-shift + subspace donor-swap), mechanism
                           profile, linear probe, calibration
precompute_recon.py        regenerates data/capstone.* from the bundled subset (reproducible):
                           vanilla vs ABP-waveform-reconstruction models, four-way dissociation,
                           mechanism-faithfulness profile, reconstruct-then-read-off model
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
python precompute_recon.py             # regenerates data/capstone.* from data/vitaldb_mini.npz
jupyter lab notebooks/                 # run the notebooks end-to-end
```

`data/realdata_*.json` were precomputed from the full VitalDB `cal_free` / `cal_based` test
subsets (not bundled); everything else regenerates from `data/vitaldb_mini.npz`.

## Key finding & limitation (important)

The synthetic sandbox is the **rigorously validated** part: the law is true by construction and
the donor-swap correctly tracks faithfulness.

On **real data**, the interval detectable from ECG→PPG is **pulse *arrival* time (PAT)**, not
clean pulse transit time: PAT = pre-ejection period (PEP) + vascular transit, and PEP moves
independently of BP — the same direction as PTT in exercise, the opposite under vasoconstriction
([Payne 2006](https://journals.physiology.org/doi/abs/10.1152/japplphysiol.00980.2011);
[Mukkamala review](https://pmc.ncbi.nlm.nih.gov/articles/PMC9088838/)). In resting ICU data the
PAT→BP law is therefore weak and can *invert*: our faithful-by-construction analytic control
(detect PAT → map to DBP) fits a **positive** slope and fails its own audit. So a model that
"fails the PTT audit" here is not necessarily unfaithful — it is being tested against a law the
modality does not cleanly express.

The law ECG+PPG *does* express is **pulse-wave morphology / arterial stiffness** — augmentation,
reflected-wave timing, and second-derivative stiffness indices read off the PPG *shape*
([El-Hajj & Kyriacou](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9849280/)). The reconstruction
model rebuilds the full pressure wave at **corr 0.91**, and the **mechanism profile** confirms it:
the model is causally faithful to the wave-shape/stiffness cue (frac 0.62) while failing arrival
time (frac 0.26). This turns the audit from a pass/fail test into **mechanism attribution** —
*faithful to what* — and points to reconstruction as a route to faithful-by-design models.

A crisp positive control for the *transit-time* law specifically needs **BP-manipulation data**
(e.g. the PhysioNet Pulse Transit Time PPG exercise dataset); that investigation is ongoing.

## Takeaway

A model can be **accurate**, have the mechanism **decodable** in its activations, and even
**reconstruct** the target waveform faithfully, yet still not causally use a given mechanism.
These four properties are independent; only a causal audit separates them. Generalized across
candidate cues, the same audit answers *which* physiology a model uses — and a faithfulness
verdict is only meaningful once you check the governing law is actually present in the data.
