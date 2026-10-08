# mechdemo — does a blood-pressure model use the physiology it should?

Cuffless blood-pressure models built from ECG and PPG are expected to rely on pulse arrival time
(PAT): a stiffer, higher-pressure artery carries the pulse faster, so a **longer PAT should mean
lower BP**. Accuracy cannot tell whether a model does this. This repository builds an
intervention audit that can — shift the PPG relative to the ECG and measure how each patient's
predicted BP responds — validates it where the answer is known, and applies it to transformers
trained on VitalDB.

Next steps and open questions: **[TODO.md](TODO.md)**.

## Main findings

1. **The audit is validated in simulation.** ECG and PPG with beat shapes fitted to VitalDB, a
   known PTT -> BP law, and a 400 ms PPG channel offset: the detector recovers the injected PTT
   (r = 0.997) and the audit tracks how much a model uses it (r = -1.00).
2. **VitalDB's PPG channel lags the ECG by several hundred ms.** Detectors that search a fixed
   window after each R peak pair the wrong points. Published methods fix it: Pan-Tompkins R peaks,
   PPG feet by intersecting tangent found on the PPG alone, and beat matching by a constant delay
   (cf. Ruffolo et al. 2025).
3. **The law is in the data.** Within patients, longer PAT goes with lower BP in ~87% of patients
   for SBP and ~74% for DBP (quality-filtered).
4. **The audited DBP transformers read it backwards.** Six training runs give the expected sign
   in 11-43% of held-out patients, while sitting at the accuracy of predicting the mean. The
   response is to the ECG-PPG gap itself, not to beat position.

## Figures

| Figure | What it shows | Script |
|---|---|---|
| `figures/fig_audit.png` | Thesis figure: audit validated in simulation (top), applied to VitalDB (bottom) | `fig_audit.py` |
| `figures/fig_beat_templates.png` | Simulated vs VitalDB average heartbeat | `fig_beat_templates.py` |
| `figures/fig_main.png` | Datasets, in- vs out-of-distribution accuracy, probes, calibration burden | `fig_main.py` |

## Reproduce

Python environment: `requirements.txt` (on the Windows workstation, `miniconda3/envs/dl`).

```bash
python fit_beat_templates.py              # fit simulator beat shapes to VitalDB (prints constants)
python synth_waveform_audit.py            # simulation: train 15 transformers, audit them (GPU, ~6 min)
python real_audit_controls.py             # VitalDB: 5 seeds, PPG/ECG/joint shifts, data's own slope
python real_audit_controls.py --target sbp
python pat_bp_quality.py                  # within-patient PAT -> BP law at four quality levels
python pat_bp_quality.py --dataset mimic  # same on MIMIC-BP (expects ../BP)
python fig_audit.py && python fig_beat_templates.py && python fig_main.py
```

## Layout

```
mechlib.py                PAT detection (r_peaks, ppg_feet, pair_beats, compute_ptt), models, probes
audit_subject.py          subject-level shift audit
synth_waveform_audit.py   simulator (make_segment) and the end-to-end simulated audit
real_audit_controls.py    seeds, shift-type specificity, and the data's own slope on VitalDB
pat_bp_quality.py         quality-filtered within-patient PAT -> BP law (VitalDB, MIMIC-BP)
fit_beat_templates.py     fits the simulator's beat shapes to VitalDB
fig_*.py                  figures
ood_benchmark.py, calib_all_models.py, gbm_families.py, physics_audit.py,
run_feature_full.py, fig_panel_e.py
                          produce the data behind fig_main
data/                     result JSONs (tracked); waveform .npz files are local only
notebooks/                tutorials and walkthroughs
ptt/                      webcam / two-site PTT app (separate deliverable)
not_used/                 retired scripts, figures and plans
```

## Data notes

- Waveforms are PulseDB's VitalDB subsets (`data/vitaldb_mini.npz`, `vitaldb_mini_deep.npz`,
  `vitaldb_full_calfree.npz`), 125 Hz, channels **ECG = 0, PPG = 1, ABP = 2**, labels [SBP, DBP].
  They are not in git.
- ECG and the arterial line are aligned in VitalDB; the PPG channel is not. Use PAT for
  within-patient changes only, never as an absolute transit time.
- MIMIC-BP lives in `../BP` (ECG, PPG, ABP, 30 s segments).
