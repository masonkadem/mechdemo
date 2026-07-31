# Patentability assessment — what we have, and where the white space is

Written 2026-07-31. Companion to [PLAN.md](PLAN.md). **Not legal advice** — a technical
assessment to decide what merits a provisional filing and a search by a registered agent.

---

## 1. Landscape (what is already claimed)

| area | status | representative |
|---|---|---|
| End-to-end ECG+PPG → BP networks | **Crowded.** Samsung and others. | [US20250090033A1](https://patents.google.com/patent/US20250090033A1/en) |
| Calibration-free BP devices | **Actively filed, unsolved.** | [US20250107717A1](https://patents.google.com/patent/US20250107717A1/en) |
| Self-calibrating / adaptive recalibration | Filed (e.g. 2024 Graphic Era wristband) | — |
| Causal DAG inference over clinical metrics | Filed, but **telemetry//process** metrics, not waveform models | [US20250285766](https://patents.justia.com/patent/20250285766) |
| Closed-loop vasopressor controllers | Well developed — they *are* the controller | [Sensors 26:2180](https://doi.org/10.3390/s26072180) |

**Two observations that matter.** First, a patent-landscape review states that no assignee in the
2012–2025 dataset has published a validated calibration-free solution at scale — the commercial
prize is unclaimed. Second, everything filed is a **predictor** or a **controller**. Nothing
found claims an *instrument that decides whether a physiological predictor is mechanistically
trustworthy*. That asymmetry is where our work sits.

---

## 2. What we actually have (viability triage)

Honest grading. Most of this project's headline claims were retracted (see PLAN.md §2); what
survives is mostly **negative results and instrumentation**, which is a poor basis for a
prediction patent but a good basis for a *diagnostic* one.

| asset | evidence | patentable? |
|---|---|---|
| **A. Closed-loop confounding detector** | 1066 events, 35 cases. Treated group dosed at −4.7 mmHg, returns to −0.2. Naive effect +0.85 mmHg for a 15–30 mmHg drug. `fig_drug_feedback.py` | **Strongest.** Novel, useful, non-obvious, and demonstrably reduces to practice. |
| **B. Validated causal roll-audit + control battery** | Textbook +0.032 / inverted −0.032 / nulls 0.000. `audit_subject.py` | **Yes, as a method.** The *validation protocol* (signed positive/inverted/null controls) is the defensible part, not the roll itself. |
| **C. Channel-desynchronization detector** | VitalDB ART/PPG carry a fixed offset; PAT near-constant 240/242 ms vs PTT-PPG's physiological 126 ms (IQR 114–136) | **Yes, narrow.** A device self-test for "is my sensor pair time-aligned?" is concrete and commercially relevant. |
| **D. Mean-predictor floor / within≈between variance test** | within-subject SBP sd 13.3 vs between 12.3; all OOD sets tie a constant predictor | Weak alone — it is a benchmarking insight, and prior art on "compare against baseline" is broad. Useful as a **claim element**. |
| **E. Faithfulness ≠ accuracy dissociations** | Probe flat while causal slopes differ; noise aug improved OOD 16.2→15.0 but weakened audit −19.9→−9.6 | **Publication, not patent.** A finding about models, not a method with utility. |
| **F. Any specific BP model of ours** | All five architectures null (37–47% faithful, CIs span zero) | **No.** We have no model that outperforms; do not file on this. |

**The blunt summary:** we do not have a better BP predictor and should not pretend to. We have
**measurement instruments that tell you when a BP predictor cannot be trusted.** That is the
patentable core.

---

## 3. Three candidate inventions, most defensible first

### Invention 1 — Detecting and correcting closed-loop treatment confounding in physiological model training data

**Problem.** Clinical training data is generated under a feedback controller: clinicians treat
*because* a vital sign moved. A model trained on it learns "vasopressor present ⇒ patient was
hypotensive" — reverse causation that inverts when deployed where treatment policy differs.
Nothing found claims detection of this in *training-data curation for waveform models*.

**Method (as reduced to practice).**
1. Align physiological signal to intervention events (drug rate step-ups).
2. Stratify events by the **pre-intervention trend** of the target.
3. Compute divergence between strata: convergence-at-dose is the controller signature.
4. Emit a **confounding index**; gate the drug covariate out of the feature set, or reweight
   (IPW) when the index exceeds threshold.

**Why non-obvious.** The naive reading of +0.85 mmHg is "the drug barely works". The insight is
that the null *is* the controller's residual error, and that it is measurable and correctable.

**Commercial hook.** Any company training BP/sepsis/shock models on ICU or OR data has this
defect and no standard test for it. Applies far beyond BP.

### Invention 2 — Mechanistic faithfulness self-test for a deployed physiological estimator

**Method.** At run time, apply a signed input-space intervention (shift PPG relative to ECG),
measure the response slope against the governing law (Moens–Korteweg: shorter arrival ⇒ higher
BP), and **validate the instrument itself** with a built-in control battery (an analytic
textbook-signed model, a sign-inverted model, a constant model, an amplitude-only model). Output
a trust/abstain signal.

**Key differentiator.** Prior art claims *prediction*; this claims *self-validation with
calibrated controls*. Our control battery is what makes it credible — and is exactly what caught
our own false "positive control failed" conclusion.

**Caveat to disclose.** All five of our models fail this test. That does not weaken the claim
(the instrument works — see B) but it means claims must be drafted around the **detector**, never
around "our faithful model".

### Invention 3 — Sensor time-alignment self-test from waveform statistics

**Method.** Estimate per-beat arrival time; flag **abnormally low beat-to-beat variance** and
out-of-range absolute values as evidence of a fixed instrumental offset rather than physiology
(VitalDB 240/242 ms constant vs PTT-PPG 126 ms with genuine variation). Auto-estimate and
subtract the constant, or refuse calibration.

**Why it matters.** A multi-sensor wearable with an uncorrected inter-channel latency produces a
PAT that is mostly instrument. This is a concrete, testable device self-check — the narrowest and
most easily granted of the three.

---

## 4. Recommended path

1. **File a provisional on Invention 1 now.** It is the most novel, the best evidenced, and the
   broadest in application. One year of priority is cheap.
2. **Combine 2 + 3 into a second provisional** as a "trustworthiness monitor for physiological
   estimators", with 3 as a dependent claim.
3. **Do not file on any BP model.** We have no accuracy advantage; a filing would be weak and
   would misrepresent the evidence.
4. **Publication sequencing.** A US provisional preserves rights, but public disclosure starts
   clocks and forfeits most non-US rights. **File before submitting or preprinting.**
5. Get a registered agent to run a proper FTO/novelty search — the searches behind this document
   were keyword-level, not a legal search, and absence of evidence here is not evidence of
   absence.

---

## 5. Honest risks

- **Invention 1's closest prior art is the causal-inference literature** (marginal structural
  models, IPW, confounding-by-indication). Those are decades old. Novelty must rest on the
  *specific application* to waveform-model training-data curation plus the concrete detection
  procedure, not on the general idea of treatment-confounding.
- **Invention 2 rests on a governing law we could not confirm models obey.** The instrument is
  validated; the physiology it tests for is absent in every model we audited. Claims must be
  about detection, not about achieving faithfulness.
- **Our strongest results are negative.** Patents want utility. The utility framing is
  "prevents deployment of an untrustworthy model" — defensible, but it needs to be argued, and it
  is a harder sell than "our model is more accurate".
- **PTT-PPG has only 22 subjects.** Any efficacy claim resting on it is thin; treat it as
  supporting evidence, and strengthen with the exercise transition data once downloaded.
