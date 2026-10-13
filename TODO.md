# Next steps

Status as of 2026-10-07. Older plans (PLAN.md, NEXT.md) are retired to `not_used/docs/`; their
conclusions about PAT were drawn with the old detector and should not be cited.

## Where things stand

- **The audit works when the answer is known.** On simulated ECG/PPG whose beat shapes are fitted
  to VitalDB, with a 400 ms PPG channel offset, the detector recovers injected PTT (r = 0.997) and
  the audit slope tracks the weight on PTT (r = -1.00), recovering ~90% of the injected slope.
- **VitalDB's PPG channel lags the ECG by several hundred ms.** The old detector (50-350 ms window
  after each R peak) paired T waves and diastolic dips; it matched the arterial-line beat count in
  32% of segments. The new one (Pan-Tompkins R peaks, PPG feet found on the PPG alone, beats paired
  by a constant delay) reaches 92%, and PAT is measurable in 97% of segments.
- **The PAT -> BP law is in the data.** Within patients, longer PAT goes with lower BP in
  ~87% of patients for SBP and ~74% for DBP once segments are quality-filtered
  (`pat_bp_quality.py`). The SBP figure is near the noise ceiling (~95%); DBP stays below it,
  consistent with the pre-ejection period inside PAT hiding more of the DBP relation
  (Payne et al. 2006).
- **The DBP transformer reads arrival time backwards.** All six training runs (original + 5 seeds)
  give the expected sign in only 11-43% of 142 held-out patients, against 74% in the data. The
  response follows the ECG-PPG gap (ECG-later = PPG-earlier; both shifted together ~ 0) and holds
  for 8-16 ms shifts. Accuracy cannot see this: every run sits at MAE 9.1-9.4 vs 9.45 for
  predicting the mean.
- **SBP transformers lean the right way, weakly.** Five seeds give the expected sign in 54-70% of
  patients (data: 79% unfiltered), MAE 13.9-14.4 vs 14.76 for the mean. Less specific than DBP:
  ECG-later matches PPG-earlier in only 3 of 5 runs, and joint shifts give small responses. So
  the direction a model learns depends on the target -- backwards for DBP, weakly right for SBP.
  (`data/real_audit_controls_sbp.json`). For SBP, runs that learn more beyond the mean follow the
  law more (r = 0.97, 5 runs; suggestive). Figure: `figures/fig_targets.png`.
- **On MIMIC-BP (out of distribution) no model beats predicting the VitalDB mean** (DBP 10.6-11.3
  vs 10.6; SBP 15.2-16.2 vs 14.7); all overestimate by 4.5-7.6 mmHg. With each patient's offset
  removed they track within-patient BP no better than a flat line (r ~ 0.1). So whether following
  the law predicts OOD performance cannot be tested on these near-mean models
  (`mimic_ood_runs.py`, `data/mimic_ood_runs.json`). fig_main panel e (r = -0.67, old detector)
  is not reproduced and should be retired.

## 1. Finish the audit chapter

- [x] **SBP transformer** (`--target sbp`): weakly right-way, see above.
- [ ] **Decide how to frame DBP vs SBP.** Why do DBP models learn the gap backwards while SBP
      models learn it weakly forwards? Check whether the DBP sign holds on quality-filtered
      segments, and whether SBP's weaker specificity (joint-shift response) survives more seeds.
- [x] **SBP vs DBP figure:** `fig_targets.py`. Regenerated `data/pat_bp_quality_vitaldb.json`
      (VitalDB PPG lags the arterial line by 476 ms, IQR 461-490).
- [ ] **CalBased (next, highest value).** Models that have seen each patient know the patient's
      baseline, so they should track within-patient change -- where the law lives. Prediction: they
      follow the law far more than calibration-free models. Then re-test law-following vs MIMIC
      within-patient tracking on models that actually track. Data: train on the VitalDB training
      set (`data/vitaldb_full_calfree.npz` train + val = the 1,293 CalBased patients; verify the
      IDs), test on `../BP/dbdata/VitalDB_CalBased_Test_Subset.mat` (1.5 GB, h5py; loader in
      `not_used/calbased_vs_anchor.py`). An earlier CalBased transformer reached DBP 6.4 / SBP 11.5
      MAE (`../BP/dbdata/results/`).
- [ ] **Full-data calibration-free models.** `data/vitaldb_full_calfree.npz` is local (150k train
      segments, 1,100 patients; 12x the mini file). `real_audit_controls.py` needs a training-file
      option. ~20 min per seed on the RTX 5070.
- [ ] **MIMIC-BP.** `python pat_bp_quality.py --dataset mimic` (left running on 2026-10-07; output
      in `data/pat_bp_quality_mimic.json` if it finished, otherwise re-run): measures the ECG -> PPG
      offset against the arterial line and the within-patient law. MIMIC's documentation warns it
      is not built for inter-waveform timing, so check the offset before trusting any PAT there.
      Then audit VitalDB-trained models on MIMIC-BP.
- [ ] **A model that beats the mean.** The audited transformers barely beat predicting the mean,
      so "a model that learned little learned PAT backwards" is the honest claim. Train on
      `data/vitaldb_full_calfree.npz` (3.5 GB) or audit the stronger `ood_benchmark.py` models.
- [ ] **Apply the quality filters to the model audit** (clean beats, arterial line agrees) and
      check the sign result holds on clean segments only.
- [ ] Optional: realign PPG by the estimated offset, retrain, re-audit. Lower priority: simulated
      models trained with a 400 ms offset already give the correct sign.
- [ ] **Update RESULTS.md.** Several rows predate the detector fix ("PAT carries almost no BP
      information", "the law can invert") and are now wrong or untested.

## 2. Bring fig_main up to date

- [ ] Panel **e** (mechanism vs OOD robustness): old detector, old audit, 5 points, and plots
      the audit's size without its sign. Drop: the cleaner re-test (`mimic_ood_runs.py`) does not
      reproduce it.
- [ ] Panel **f** (probes, cardiac period vs PAT): PAT targets came from the old detector.
      Re-run; the message likely holds (the transformer's PAT probe fell 0.25 -> 0.11).
- [ ] Panel **d**: re-run the "PAT only" single tree with the new PAT. It was already the best
      out-of-distribution model (10.96 vs 11.8-15.7 mmHg) and is the natural PTT-model baseline.
- [ ] Panels a, b, g do not depend on PAT timing; keep.
- [ ] Restore the modules that `fig_main`'s data scripts import from `not_used/`:
      `features_ext.py` (gbm_families), `features_full.py` (run_feature_full),
      `pat_estimators.py` (fig_panel_e).

## 3. Next chapter: PTT-based BP models

- [ ] **SBP first, DBP second** -- the law is clearest for SBP with PAT.
- [ ] **Baselines:** BP = a / PAT^p + b per patient (Moens-Korteweg / Bramwell-Hill form) with
      per-patient calibration; PAT-only tree; deep models. In-distribution (VitalDB) and
      out-of-distribution (MIMIC-BP).
- [ ] **The offset:** treat the instrumental ECG-PPG offset as part of the per-patient intercept,
      or estimate it against the arterial line where one exists. Never report absolute PAT from
      VitalDB's PPG.
- [ ] **The pre-ejection period:** PAT = PEP + PTT. Options: heart-rate adjustment, PPG-only
      timing features that need no ECG (notch delay, rise time), and ECG -> arterial-line foot as an
      upper bound on what clean transit time could give.
- [ ] **Report the data's own law** (share of patients with the expected sign, and the noise
      ceiling) as the reference every model is judged against.

## Key numbers

| | Value |
|---|---|
| Detector vs injected PTT (simulation) | r = 0.997, bias -1 ms |
| Audit slope vs weight on PTT (simulation) | r = -1.00; -0.20 recovered vs -0.22 injected |
| R-peak count matches arterial line, old -> new | 32% -> 92% |
| PAT measurable, old -> new | 45% -> 97% |
| Patients with expected sign in the data, SBP / DBP (clean) | 87% / 74% (noise ceiling ~95% / ~89%) |
| DBP transformer runs with expected sign | 11-43% of 142 patients |
| DBP MAE, transformers vs mean predictor | 9.1-9.4 vs 9.45 mmHg |
