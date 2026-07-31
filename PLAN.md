# Project 3 — status and plan

Living document. Updated 2026-07-31. Newest decisions at the top of each section.

The one-line state of the project: **the cross-dataset OOD framing is dead (everything sits at
the mean-predictor floor), the causal audit is validated and working, and the live hypothesis is
that mechanistic faithfulness predicts WITHIN-subject BP tracking.**

---

## 1. Established results (verified, safe to cite)

| # | result | evidence |
|---|---|---|
| 1 | **All architectures are indistinguishable on ID accuracy.** DBP MAE 8.1–8.8 from a 32-leaf single tree to a 1.8M-param CNN. | `data/gbm_variants_ood.json`, deep runs |
| 2 | **Heart rate dominates arrival time causally.** HR perturbation moves predictions 1.4–2.5× more than the PAT roll. | `data/weekend_results.json` stage 2 |
| 3 | **Probing cannot distinguish faithful from shortcut models.** Probe decodability is flat across models (period 0.7–0.8, PAT ~0.2) while causal slopes differ. | stage 1 + probe battery |
| 4 | **Robustness ≠ faithfulness.** Noise augmentation improved OOD (16.2→15.0) but *weakened* the roll-audit response (−19.9→−9.6). | `data/noise_faithfulness.json` |
| 5 | **The causal audit is validated.** Textbook `BP=−3/PAT+b` → slope +0.032 (93% of segments); inverted → −0.032 (7%); constant and amplitude nulls → exactly 0.000. | `data/audit_subject_controls.json` |
| 6 | **Models carry no transferable between-subject signal.** On all five external sets the best model ties a constant predictor. | mean-predictor baseline (§2) |
| 7 | **Models DO track within-subject BP change.** within-r 0.44–0.63; not a time artifact (partial r 0.631→0.637). Tracking *separates* models that ID accuracy cannot. | `within_subject.py` (first pass) |

---

## 2. Retractions — things we believed and disproved

Kept deliberately, so they are not re-derived or accidentally re-claimed.

- **"XResNet101 uses PAT backwards (+8.6)."** Did not replicate: −15.0 on a larger sample,
  −18.3 ± 2.1 bootstrapped. Withdrawn.
- **"r = −0.71 between audit slope and OOD error."** Partly an artifact of the above; fell to
  −0.57 and is underpowered at n=5 architectures. Withdrawn pending the per-subject version.
- **"LightGBM demo gets MIMIC 9.8."** Imputation artifact — MIMIC has no demographics, all-NaN
  columns were filled with 0.0 and misrouted the trees. Honest waveform-only number is 13.3.
- **"Novel APG timing features."** `T_a…T_e` already appear in the published BP-Benchmark
  feature set. Relabelled "reproduces on a held-out split".
- **"Reflection-only GBM beats the deep nets on MIMIC (10.3)."** The oracle within-set mean gets
  10.22 — the model was doing nothing beyond centring. Withdrawn.
- **"The audit fails its positive control."** My error, twice over: a sign-convention mistake (on
  a negative-only sweep a faithful model gives a POSITIVE `dBP/d(shift)`) plus NaN-imputation in
  `audit_controls.py` pinning the median slope at zero. **The audit works** — see result 5.
- **"VitalDB has 0 vasoactive-drug cases."** My query bug: channels are named by pump
  abbreviation (`Orchestra/PHEN_RATE`), not drug name. Real count is 2,905.

---

## 3. Known bugs / open technical debt

- `mechlib.causal_ptt_audit` still **imputes** non-finite PAT and sweeps the **positive arm**,
  where the tangent-foot estimator saturates (+48 ms → +26 ms measured) and ~1% of segments
  beat-slip (up to 225 ms). Both bias slope magnitude **toward zero**, so published PAT
  dependence should get *stronger* once fixed. `audit_subject.py` has the corrected method.
- PAT is measurable on only **44%** of segments. Root cause is PulseDB's window-level
  preprocessing, which never needed a reliable foot. A beat-level re-segmentation with an
  explicit foot-quality gate is the highest-value preprocessing change.
- `perturb_aix` is **not specific** — it moves the notch (+0.283) about as much as AIx (+0.290).
  All reflection-mechanism claims are blocked until this is fixed.
- Stage 4 (`deep_seeds`) never finished; seed variance is large (inception1d gave slope −8.0 in
  stage 1 and +18.9 in stage 4 — a sign flip). **No cross-model mechanism claim is safe until
  the seed distribution is known.**

---

## 4. The live hypothesis

> Cuffless BP models carry essentially no transferable *between-subject* signal, but they do
> track *within-subject* BP change — and mechanistic faithfulness to pulse arrival time predicts
> how well they track it.

Why this reframing was forced: within-subject SBP sd (13.3 mmHg) ≥ between-subject sd
(12.3 mmHg); every subject spans >30 mmHg, median 73 mmHg. These are surgical cases, so the BP
swings are drug and fluid events already present in our waveforms. Cross-dataset transfer was
measuring the axis with the least signal.

**What must hold for this to be a finding (in dependency order):**

1. **Baselines** — HR-only, PAT-only, AIx-only, amplitude-null. *If HR-only reaches ~0.6, the
   tracking is another shortcut and the hypothesis changes.* ← gating everything
2. **Power** — pair per-subject audit slope with per-subject tracking (n≈144, not n=5), bootstrap CI.
3. **Corrected slopes** — re-run all five nets through the validated audit.
4. **Seed distribution** — finish stage 4; report mechanism per architecture with error bars.

---

## 5. Next steps

**Now (running):** `within_subject.py` — steps 1 and 2 above.

**Then, in order:**
- Re-run the five deep nets through the corrected audit (`audit_subject.py` method).
- Finish `deep_seeds` for seed error bars.
- CalBased protocol run. Within-subject tracking is essentially what calibrated deployment
  measures, so this connects directly rather than being a side quest.
- Beat-level re-segmentation with a foot-quality gate (lifts PAT validity above 44%).

**Clinical validation arm (native VitalDB — scoped, not started).**
Use `api.vitaldb.net` directly, **not** PulseDB: PulseDB anonymized VitalDB case IDs into
sequential `p000001` labels, so joining it to the clinical tables would require matching on
age/sex/BMI — a quasi-identifier linkage attack on a deliberately de-identified release.
VitalDB's own API serves waveforms with `caseid` intact, so the join is native. Open access, no DUA.

Available (`data/vitaldb_scope.json`):

| | n |
|---|---|
| ART + PLETH + ECG_II together | 3,458 / 6,388 |
| trio + ≥1 vasoactive agent | 2,905 |
| phenylephrine / norepinephrine / epinephrine | 103 / 61 / 63 |
| preop hypertension | 1,162 (34%) |
| labs (hb / alb / cr) | ~93% |

Two designs, chosen to match the clustering in the data:
- **Between-subject stiffness contrast** — hypertensive vs normotensive (n=1,162). Largest
  usable subgroup, needs no drug timing. Probably the first analysis.
- **Within-subject causal perturbation** — BP and PAT 2 min before vs 2 min after each
  phenylephrine bolus, patient as their own control. Drug timing is exogenous, so this is a
  genuine causal estimate.

**Analysis-method note.** The data are strongly clustered (within ≈ between variance), so
between- and within-subject effects can have *opposite signs* (Simpson's paradox). Pooled
correlations or pooled SHAP would silently average the two and mislead. Use:
- **linear mixed models** (`BP ~ drug + CO + labs + (1|patient)`) for effects with correct SEs;
- **GAMs with per-patient random effects** for nonlinear dose–response shape;
- **mutual information / distance correlation** for nonmonotonic screening, then partial
  correlation controlling for age/sex/HR;
- **SHAP only as a faithfulness instrument** — does the model's SHAP dependence for PAT carry
  the sign the governing law predicts, and does it agree with the causal roll-audit? SHAP
  explains the *model*, not the physiology. Compute it within-subject.
- FDR correction across agents/labs; use dose (`_RATE` integrated over time), not a binary flag.

**Deferred (deliberately):**
- Foundation-model pretraining then auditing. Large compute, and uninterpretable until the seed
  distribution is known — inception1d already flipped slope sign between runs.
- Cross-attention ECG-PPG with auditable attention lag. Still the most novel architecture idea,
  but it needs the audit and seed baselines settled first.
- Fractal-dimension governing law.

---

## 6. Figures

`fig_main.py` needs rebuilding: corrected slopes/CIs, real leaf counts (32 single-tree, 50,400
SFS-19, 479,790 Optuna — currently a placeholder "~8k"), and the ±16 ms sweep (100% tracking)
rather than ±48 ms.
