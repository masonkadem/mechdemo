# Results summary

All numbers below are on disk and reproducible from the scripts named. DBP unless stated.
Companion documents: [PLAN.md](PLAN.md) (status and next steps), [IP.md](IP.md) (patentability).

---

## Table 1 — The audit instrument is validated

Three independent validations. Faithful = the audit responds in the direction the governing law
predicts (`BP = A/PTT^p + B`).

| test | textbook model | inverted model | constant | amplitude | source |
|---|---|---|---|---|---|
| analytic controls (slope) | **+0.032** | −0.032 | 0.000 | 0.000 | `audit_subject.py` |
| fraction of segments correct | 93% | 7% | 0% | 0% | `audit_subject.py` |

| synthetic **waveform** models | α=0 | α=0.25 | α=0.5 | α=0.75 | α=1.0 |
|---|---|---|---|---|---|
| audit slope | +0.046 | −0.023 | −0.083 | −0.148 | **−0.185** |

`r(α, slope) = −0.996` (`synth_waveform_audit.py`). The generator injects
dBP/dPTT = −0.22 mmHg/ms; the audit recovers −0.185 at α=1.0.

**Sign convention**: a negative roll *shortens* measured PTT, and shorter PTT means higher BP, so
a faithful model gives a **negative** slope on this sweep. Verified against the generator, not
assumed.

> **The instrument works end-to-end on raw waveforms.** Every null below is therefore about the
> models or the data, not about the audit.

---

## Table 2 — No trained model is faithful

Corrected audit, per-subject slopes, VitalDB test subjects (`run_weekend2.py` stage 1).

| architecture | slope | 95% CI | subjects faithful |
|---|---|---|---|
| lenet1d | −0.0085 | [−0.0494, +0.0346] | 37% |
| inception1d | −0.0033 | [−0.0639, +0.0550] | 47% |
| xresnet1d50 | −0.0076 | [−0.0899, +0.0606] | 41% |
| xresnet1d101 | −0.0049 | [−0.1069, +0.0600] | 45% |
| transformer | −0.0042 | [−0.0387, +0.0282] | 41% |

Every CI spans zero; every faithful-fraction is near the 50% coin flip. Slopes are ~40× smaller
than the α=1.0 synthetic model (−0.185). Seed variance is small (5 lenet1d seeds, slope sd
0.0031), so this is not training noise.

---

## Table 3 — Four independent instruments agree: PAT carries almost no BP information

| instrument | result | source |
|---|---|---|
| causal roll-audit | all CIs span zero (Table 2) | `run_weekend2.py` |
| measurability | PAT recoverable on only **44%** of PulseDB segments | `audit_subject.py` |
| channel synchronization | VitalDB PAT near-constant 240/242 ms — an instrumental offset, not physiology | `vitaldb_raw.py` |
| calibration value | PAT arm is the **worst** family at every anchor count | `calib_families.py` |

PAT-family permutation importance on calibrated error: `pat_foot` **+0.002**, `xcorr_peak`
+0.042, with `pat_peak` and `xcorr_lag` **negative**.

**Physiological reading.** Moens–Korteweg governs the *aortic* pulse. What these datasets measure
is `R-peak → finger`, which is PEP (cardiac, not vascular) plus peripheral transit through
neurally-toned muscular arteries. The law applies to a quantity the sensors do not capture.

---

## Table 4 — Calibration is where the signal actually is

Per-subject offset fitted on *k* randomly chosen anchors, scored on held-out segments
(`calib_families.py`). Random anchors, not first-*k* — see the correction note.

| arm | n feat | k=0 | k=1 | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|---|---|
| all 83 (waveform) | 83 | 6.88 | 6.01 | 5.37 | 5.16 | 4.86 | **4.81** |
| all 83 + demographics | 86 | 7.16 | 5.90 | 5.09 | 4.91 | **4.55** | 4.49 |
| reflection / APG | 14 | 7.66 | 6.84 | 5.90 | 5.71 | 5.54 | 5.33 |
| complexity / fractal | 5 | 7.88 | 6.58 | 5.99 | 5.76 | 5.47 | 5.45 |
| rate / HRV | 9 | 8.05 | 7.03 | 6.54 | 6.18 | 5.97 | 5.85 |
| demographics only | 3 | 8.85 | 6.94 | 6.42 | 6.03 | 5.90 | 5.75 |
| **PAT / arrival time** | 6 | 8.48 | 7.13 | 6.62 | 6.37 | 6.05 | **6.04** |
| subject-mean floor | — | | | | | | 5.59 |

- Demographics **halve** the anchor requirement: waveform needs k=20 to reach 4.81; +demographics
  reaches it at **k=10**.
- Calibrated models **beat the subject-mean floor** (4.81 vs 5.59) — nothing in the cross-dataset
  work ever did.
- Demographics *hurt* at k=0 (7.16 vs 6.88) but help once anchored: they inform the per-subject
  **offset**, not the population mean.

Top permutation importances on calibrated error: **age +1.347**, dw10 +0.409, vpg_min +0.283,
ppg_skew_g +0.257, t_e +0.256, bmi +0.241.

---

## Table 5 — Cross-dataset transfer is at the mean-predictor floor

| set | constant predictor | oracle set mean | our best model |
|---|---|---|---|
| ID (VitalDB) | 9.43 | 9.43 | **8.07** ✓ |
| MIMIC-BP | 11.39 | 10.22 | 10.30 |
| BCG | 7.92 | 7.41 | 7.60 |
| Sensors | 8.02 | 8.23 | 8.10 |
| UCI2 | 8.49 | 8.59 | 8.40 |
| PPG-BP | 10.89 | 8.84 | 8.90 |

On every external set the best model ties a constant predictor. There is no transferable
between-subject signal. Within-subject SBP sd (13.3 mmHg) ≥ between-subject sd (12.3 mmHg), so
the informative axis is within-subject, which is why Table 4 matters and cross-dataset OOD
ranking does not.

---

## Table 6 — Clinical data carries closed-loop treatment confounding

1066 phenylephrine rate step-ups, 35 VitalDB cases (`fig_drug_feedback.py`).

| pre-dose MAP trend | n | MAP before | MAP after | net |
|---|---|---|---|---|
| falling (treated) | 303 | −4.7 | −0.2 | **+1.8** |
| flat | 387 | +0.1 | +2.5 | +0.9 |
| rising | 376 | +4.9 | +3.8 | +0.4 |

Naive per-case effect **+0.85 mmHg** (p=0.092) for a drug that raises BP 15–30 mmHg. Clinicians
dose *because* MAP fell and titrate to target, so treatment is a function of the outcome and
before/after cancels. **Drug state must not be used as a model feature** — it encodes reverse
causation and inverts out of distribution.

Figure: [figures/fig_drug_feedback.png](figures/fig_drug_feedback.png)

---

## What is publishable now

1. **PAT carries negligible BP information in standard cuffless datasets** — four independent
   instruments, with the audit validated end-to-end on synthetic waveforms (Tables 1–3). The
   field has assumed PAT ≈ PTT for two decades.
2. **Closed-loop treatment confounding in clinical waveform datasets** (Table 6) — a
   methodological warning that generalizes well beyond BP.
3. **Calibration, not architecture, is the lever** (Tables 4–5): all architectures are
   indistinguishable, transfer is at the floor, and cheap demographics halve the cuff burden.

## Known limitations

- The estimator recovers injected PTT at only **r=+0.558** on clean synthetic waveforms, so foot
  detection is itself lossy; α=0.50 trained poorly (test r²=−1.884), so the α grid is not a clean
  dose–response and needs a seeded rerun.
- Checkpointed models give negative slopes where fresh retrains give positive ones; something
  beyond seed differs and is unresolved.
- `perturb_aix` is not specific (moves the notch as much as AIx), so reflection-mechanism claims
  are blocked.
- PTT-PPG could not test the governing law: cuff BP is measured at activity boundaries, giving a
  1.6 mmHg across-activity spread against 14.7 between subjects.

---

## The MIMIC arrival-time result: why it is not a mechanism win

`GBM arrival time only` posts the best raw MAE of any variant on MIMIC-BP (11.09 against
13.1-17.4 for the rest), which reads as evidence that restricting a model to the sanctioned
physics buys out-of-distribution robustness. Three checks say otherwise.

**1. A constant beats it.** MIMIC's DBP mean is 58.4 mmHg against 62.9 in training, a shift of
-4.5. Predicting the training mean scores **10.33** on MIMIC, better than the 11.09. The
subject-mean floor is 4.66, so every variant is 2-4x worse than predicting a number.

**2. It wins by flatness, not by tracking.**

| arm | pred sd | MIMIC MAE | within-subject r | slope vs true BP |
|---|---|---|---|---|
| all features | 5.88 | 14.65 | 0.157 | 0.76 |
| no rate shortcut | 5.68 | 14.60 | 0.167 | 0.80 |
| **PAT only** | **4.45** | **11.09** | **0.032** | **0.40** |
| morphology only | 5.21 | 13.82 | 0.149 | 0.87 |

PAT-only has the best MAE and the **worst** correlation with the quantity it is predicting. A
model winning through mechanism would still track BP within subject; this one does not.

**3. On a shift-invariant metric the advantage disappears.** Removing each subject's offset,
which is what calibration does and what a mean shift cannot fake:

| arm | MIMIC-BP | BCG | Sensors |
|---|---|---|---|
| all features | 5.23 | 2.81 | 6.45 |
| no rate shortcut | 5.18 | 2.94 | 6.38 |
| PAT only | **5.64** | **2.21** | 6.70 |

PAT-only is now worst on MIMIC and best on BCG. There is no consistent ordering.

**What this does and does not license.** It does not license "restricting to arrival time
improves OOD generalisation" -- that claim fails all three checks. It does license the weaker and
still useful statement that **raw cross-dataset MAE is not a measure of mechanism**: under
distribution shift it rewards degenerate predictors, so the field's standard OOD tables are
uninformative about whether a model learned physiology. That is a methodological finding, and it
is the one the evidence supports.

PulseDB and the benchmark papers do not report mean-predictor baselines, so model-versus-model
OOD comparison is standard practice in this literature. The point is not that those papers erred
by omission, but that such a comparison cannot carry mechanistic weight.
