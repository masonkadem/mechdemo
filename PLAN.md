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
| 7 | **Models DO track within-subject BP change.** within-r 0.44–0.65; not a time artifact (partial r 0.631→0.637), and not a rate shortcut (HR-only reaches only 0.154). Tracking *separates* models that ID accuracy cannot. | `data/within_subject.json` |
| 8 | **Faithfulness does NOT predict tracking.** Per-subject pairing at n=67 subjects/model: r = +0.07, +0.11, −0.06, −0.07, +0.12, every CI spanning zero. Well-powered null. | `data/within_subject.json` |
| 9 | **All five models are anti-faithful at the median subject.** Median audit slope negative for every architecture; only 36–49% of subjects faithful (faithful = positive on the validated negative-arm sweep). Yet they track BP at 0.44–0.65. | `data/within_subject.json` |
| 10 | **The amplitude null out-tracks every named physiological cue.** amp 0.270 vs PAT 0.190, AIx 0.184, HR 0.154. Within-subject tracking is substantially perfusion/signal-quality driven, not arterial mechanics. | `data/within_subject.json` |

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
- **"Mechanistic faithfulness predicts within-subject BP tracking."** The live hypothesis for
  about two hours. Killed by its own test: per-subject pairing at n=67 subjects per model gives
  r = +0.07 / +0.11 / −0.06 / −0.07 / +0.12, all CIs spanning zero. This is a well-powered null,
  not an underpowered one. Faithfulness and capability are unrelated here, not merely decoupled.
- **"~2000 VitalDB cases have cardiac output."** My substring match caught `Primus/CO2`
  (capnography). Real CO is EV1000/Vigileo, ~325–508 cases.
- **"PEP is 63% of PAT"** and its corrected successor **"33%"**. Both withdrawn. The first came
  from a foot detector firing twice per beat (146 feet for 74 R peaks); the second from data
  whose absolute timing is not trustworthy — see below.
- **"Absolute PEP/PTT is recoverable from VitalDB."** It is not. On case 1: R→ART foot 242 ms,
  R→PPG foot 742 ms, so the implied radial→finger transit is 500 ms against an RR of 842 ms.
  A real radial→finger transit is ~20–50 ms. The values are also near-constant beat to beat
  (240, 240, 238, 242, 244…) where real PAT varies by tens of ms. **The ART and PPG channels
  carry independent device latencies and are not mutually time-synchronized**, and a fixed
  offset is indistinguishable from a real transit time. Within-subject *changes* remain valid
  (a constant offset cancels in differences); absolute intervals do not.

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

## 4. The live hypothesis — PEP as the missing term

The previous hypothesis (faithfulness predicts within-subject tracking) is dead; see §2. What
its autopsy leaves is a sharper question. Every model is **anti-faithful** to PAT (median slope
negative, 36–49% of subjects faithful) while still tracking BP at 0.44–0.65. Two readings:

  (a) the models are wrong and exploit shortcuts; or
  (b) **PAT itself is a bad target, and the models are right to distrust it.**

(b) is physiologically motivated and testable. `PAT = PEP + PTT`. Only PTT carries arterial
stiffness; **PEP (pre-ejection period) is cardiac** — electrical-to-mechanical delay plus
isovolumic contraction — and it is large and time-varying. The entire cuffless-BP literature
uses PAT as a stiffness proxy while PEP contaminates it.

> **Hypothesis.** Pharmacological state gives a measurable handle on PEP. If PEP-corrected PAT
> is what models actually track, that explains the anti-faithfulness, repairs the audit, and
> converts part of the per-subject calibration constant into a measurable covariate.

Why this is novel and why it is ours to do: propofol depresses contractility (lengthens PEP);
ephedrine/epinephrine increase it (shorten PEP). Nobody has tested this at scale because nobody
had PulseDB-scale waveforms joined to infusion records — VitalDB has both, natively.

**Predictions, each falsifiable:**
1. Audit slope becomes **more positive** (more faithful) when conditioned on drug state.
2. Subjects on inotropes show **systematically different** PAT→BP slopes than drug-free subjects.
3. Offset calibration is **partly replaceable** by drug/BIS/SV covariates (measured in stage 3).
4. If (1)–(3) all fail, the models are simply unfaithful and (a) is the answer — also publishable,
   and the audit method still stands.

**Constraint discovered while building the loader (§2, last entry).** VitalDB's ART and PPG
channels are not mutually time-synchronized, so *absolute* PEP/PTT cannot be measured and
predictions must be phrased as **within-subject changes**, where the fixed offset cancels:
"does the arrival interval *shift* after ephedrine?", never "what is PEP?".

**This may be an upstream finding in its own right.** If channel pairs carry an arbitrary fixed
offset, then PulseDB's PAT — and therefore the x-axis of our entire roll-audit — is measured
against an arbitrary zero. That is a candidate cause of the observed anti-faithfulness that has
nothing to do with the models, and it is worth testing directly: estimate the per-dataset
constant offset and re-run the audit relative to it.

---

## 4b. Pharmacodynamic audit — first results (2026-07-31)

**Stage 1 of `run_weekend2` is complete and is a clean null.** All five architectures, validated
audit, per-subject CIs:

| arch | slope | 95% CI | subjects faithful |
|---|---|---|---|
| lenet1d | −0.0085 | [−0.0494, +0.0346] | 37% |
| inception1d | −0.0033 | [−0.0639, +0.0550] | 47% |
| xresnet1d50 | −0.0076 | [−0.0899, +0.0606] | 41% |
| xresnet1d101 | −0.0049 | [−0.1069, +0.0600] | 45% |
| transformer | −0.0042 | [−0.0387, +0.0282] | 41% |

**Every CI spans zero and every faithful-fraction is near the 50% coin flip.** There is no
faithful model in our set and no meaningful ranking among them.

**Seed variance is small, but checkpoints disagree with retrains.** Five fresh lenet1d seeds
give slope +0.0040 to +0.0117 (sd 0.0031) — consistently *positive*, i.e. faithful — while the
stored checkpoint gives −0.0085. Consistent within a training run, opposite between runs. So the
earlier −8.0/+18.9 chaos was the broken audit, not seeds; but something other than seed differs
between checkpoint and retrain (epochs, data subset) and **must be resolved before any slope is
quoted**.

**The phenylephrine audit is confounded by closed-loop treatment.** Per-case MAP change after a
phenylephrine step-up is **+0.85 mmHg, 60% of cases positive, Wilcoxon p=0.092** (30 cases, 466
events) — for a drug that should raise BP 15–30 mmHg. The reason is visible in the timing:

| | MAP |
|---|---|
| case baseline | 73.0 mmHg |
| just **before** drug step | 72.0 mmHg (−1.0 vs baseline) |
| after drug step | 77.1 mmHg (+4.1 vs baseline) |

Clinicians give vasopressors *because* MAP has fallen, and titrate to a target. **Treatment is a
function of the outcome**, so a naive before/after comparison cancels most of the true effect.
This is closed-loop confounding, not a weak drug.

Two consequences:
- Any model that uses drug state as a *feature* is learning "drug on ⇒ patient was hypotensive",
  which is reverse causation and will invert out of distribution. This argues **against** adding
  drugs to the LightGBM arm.
- The pharmacodynamic audit needs a design that survives feedback: dose–response *within* the
  treated window, marginal structural models / inverse-probability weighting, or the
  bolus-onset transient (30–90 s) before the clinician reacts.

En route, one measurement bug worth keeping: sampling the 500 Hz `SNUADC/ART` waveform at 1 Hz
aliases systole/diastole at random (sd 19.9 mmHg). Use `Solar8000/ART_MBP` (sd 7.9), which is
already beat-averaged.

---

## 4c. Dataset triage for the causal arm (2026-07-31)

The phenylephrine result (§4b) means any ICU/OR dataset carries treatment feedback. What we need
instead is an **exogenous** BP perturbation. Assessed:

| dataset | verdict |
|---|---|
| **PTT-PPG** (PhysioNet, 22 subj) | **Best fit.** ECG 500 Hz + 6 PPG channels, sit/walk/run, drug-free. Exercise is exogenous — no feedback loop. |
| **Mendeley `pz2zzr8vhm`** (148 subj) | **Not usable for the causal arm.** Its own docs: BP was collected *only at rest* (`_1` files), not after exercise. So it is 148 subjects × one resting BP = between-subject only — the axis we already showed carries no transferable signal (result 6). Fine as a drug-free sanity set. No public API; needs manual download. |
| **Autonomic Aging** (1,100 subj) | **No PPG.** Header shows 2 channels: ECG + NIBP at 1000 Hz. Cannot support PAT or APG tests. Useful only for BP dynamics. |
| **Graphene tattoo** | URL 404s; not on PhysioNet at the path given. |

**PAT on PTT-PPG is physiologically credible, unlike VitalDB.** Resting record `s4_sit`:
PAT median **126.0 ms, IQR 114–136**, with genuine beat-to-beat variation (literature 100–250 ms).
VitalDB gave a near-constant 240/242 ms that is an instrumental offset, not physiology.

Caveat: walk/run records show inflated PAT sd (140–144 ms) from motion artifact, and `s1_run`
reports HR 83 bpm, which is too low for running — the R-peak detector is dropping beats under
motion. **Motion-state-aware quality gating is required before using the exercise records.**

---

## 5. Next steps

**Running:** `run_weekend2.py` — corrected audit, seed variance (gating), CalBased, tracking pairing.

**Priority now (the PEP arm):**
- Pull ~300 VitalDB cases with ART+PLETH+ECG plus an infusion record. Derive PEP from the
  arterial upstroke (ECG-R → ART foot = PAT_art; PPG foot → PTT), which is what having a real
  arterial line makes possible and PulseDB does not.
- Test whether drug state explains PEP variance, then whether PEP-corrected PAT restores audit
  faithfulness.
- Quantify how much offset calibration drug/BIS/SV covariates buy back.

**Also queued:**
- Chase the amplitude-null result (result 10). It currently undermines the physiological reading
  of within-subject tracking and needs an explanation either way.
- Beat-level re-segmentation with a foot-quality gate (lifts PAT validity above 44%).
- Visual feature→BP mapping: partial-dependence / SHAP panels computed **within-subject**, for
  overall intuition about which features map to BP and how nonlinearly.

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
