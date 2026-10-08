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
  (`data/real_audit_controls_sbp.json`)

## 1. Finish the audit chapter

- [x] **SBP transformer** (`--target sbp`): weakly right-way, see above.
- [ ] **Decide how to frame DBP vs SBP.** Why do DBP models learn the gap backwards while SBP
      models learn it weakly forwards? Check whether the DBP sign holds on quality-filtered
      segments, and whether SBP's weaker specificity (joint-shift response) survives more seeds.
- [ ] **Show SBP in fig_audit** (panel f/h with both targets, or an SBP row in the supplement).
- [ ] **Regenerate `data/pat_bp_quality_vitaldb.json`:** `python pat_bp_quality.py` (the numbers
      above came from the run before the script gained `--dataset`; the JSON was not kept).
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
      the audit's size without its sign. Re-run or drop.
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
