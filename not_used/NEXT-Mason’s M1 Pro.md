# What to do next

Written after the lambda sweep retracted the PTT-supervision result, which leaves the project
with no constructive machine-learning finding. This document decides what the project is now.

---

## Where the evidence actually stands

**Solid, and unaffected by the retraction:**

| finding | evidence |
|---|---|
| The audit instrument is validated end to end | analytic controls (+0.032 / −0.032 / 0.000) and synthetic waveforms (r = −0.996) |
| No trained model is faithful | six architectures, all CIs spanning zero, 37–47% faithful |
| Arrival time carries little BP information | four independent instruments agree |
| AAMI defeats every model | all 2.3–2.9× worse than predicting each subject's own mean |
| Cross-dataset transfer sits at the mean floor | best model ties a constant predictor on all five external sets |
| Calibration is the real lever | removing the rate shortcut costs 0.02 mmHg; demographics halve the anchor requirement |
| Clinical data carries closed-loop confounding | phenylephrine effect measures +0.85 mmHg for a 15–30 mmHg drug |

**Retracted:** r = −0.71 (audit slope vs OOD); PTT supervision raising faithfulness 66→76%;
faithfulness predicting OOD interventionally (r = −0.089 over 9 runs); the MIMIC arrival-time
"win" (a constant predictor beats it).

---

## The novel finding hiding in the feature reference

Searching the cuffless-BP literature for the specific critique this project has assembled returns
nothing. Recent work notes that "accuracy is strongly influenced by the representativeness of the
test subject's BP distribution" as an observation, but no paper systematically demonstrates it.
Meanwhile the underlying statistical phenomenon — **within-subject and between-subject
associations diverging, and reversing** — is well established in
[fMRI](https://pubmed.ncbi.nlm.nih.gov/27101735/) and
[cognitive neuroscience](https://www.nature.com/articles/s41467-026-71404-0), where it is treated
as a first-order validation pitfall.

**Nobody has applied it to cuffless blood pressure.** Our data shows the effect is large:

| feature | pooled r | within-subject r | retains |
|---|---|---|---|
| `rise` | −0.197 | −0.002 | 1% |
| `rr_mean` | +0.214 | −0.007 | 3% (sign flips) |
| `hr` | −0.214 | −0.016 | 8% |
| `pat_peak` | −0.220 | −0.024 | 11% |
| `pat_foot` | −0.207 | −0.052 | 25% |

**14 of 87 features reverse sign** between pooled and within-subject analysis, and **20 lose more
than half their apparent association**. Critically, the features that collapse hardest are exactly
the ones the field builds on: heart rate, pulse arrival time, and upstroke timing.

This unifies results that currently look like separate negative findings:

- models tie a constant predictor out of distribution → they learned between-subject structure
- AAMI defeats them → AAMI samples the tails where between-subject centring stops working
- PAT "wins" on MIMIC by being flattest → a distribution-shift artifact
- calibration is the only thing that helps → an offset is exactly what removes the between-subject
  component
- within-subject SBP sd (13.3) ≥ between-subject sd (12.3) → the informative axis is the one
  standard evaluation discards

### The deep models do NOT collapse — and that sharpens the claim

Running the same test on the trained networks gives the opposite result:

| model | pooled r | within-subject r | retains |
|---|---|---|---|
| lenet1d | +0.459 | +0.505 | 110% |
| inception1d | +0.435 | **+0.600** | 138% |
| xresnet1d50 | +0.436 | +0.586 | 134% |
| xresnet1d101 | +0.444 | +0.533 | 120% |
| transformer | +0.414 | +0.436 | 105% |

Every network **gains** within subject. So the dissociation is not a property of the task — it is
a property of the *hand-crafted features*. The networks find within-subject structure that
`hr`, `pat_foot`, `rise` and the rest do not carry.

This is a better result than a blanket "the field validates on the wrong axis", and it reframes
several findings at once:

- The deep nets are not merely centring. They track within-subject BP change at r ≈ 0.5–0.6,
  which is real capability the cross-dataset tables hide entirely.
- The feature models' apparent competitiveness is pooled-only. On the axis that matters they are
  carrying far less than their pooled correlations imply.
- It explains why calibration helps so much and why the mechanism arms barely differ: an offset
  removes the between-subject component, after which the features have little left.
- It sits alongside, not against, the faithfulness result. The networks are unfaithful to the
  arrival-time law AND better at within-subject tracking than the physiological features are.
  Whatever they use, it is neither PAT nor the published indices.

**The open question this creates, and the best experiment left: what are the networks using?**
The probe battery and audit both say "not arrival time". A within-subject-specific probe — which
features predict the network's within-subject residual — would answer it, and no existing result
does.

---

## What to do, in order

### 1. Write the paper around the pooled/within dissociation (weeks, no new data)

The framing is no longer "deep models are unfaithful" — that is a negative result about six
models. It is:

> Cuffless BP models are validated on between-subject variation and deployed on within-subject
> variation. These are close to orthogonal in every standard dataset. Reported accuracy therefore
> does not transfer, the physiological features the field trusts are pooled-only artifacts, and
> mechanistic auditing shows models do not use the governing law because the pooled task does not
> require it.

Everything needed is on disk. The audit becomes supporting evidence for *why* the models behave
this way rather than the headline.

**Missing piece worth adding:** report every headline result twice, pooled and within-subject,
including the deep nets. That table is the paper.

### 2. The measurement paper: two-site video PTT (months, new data)

The only route to a PTT free of PEP contamination. Neck-to-hand from one camera excludes the
cardiac component by construction and shares a clock. Keep it simple: three sites, fixed ROIs,
240 fps phone, braced arm. The claim is one number — arrival time rising linearly with anatomical
distance, slope inside 4–12 m/s.

Kill criterion, set in advance: if two independent sessions do not reproduce a plausible PWV,
drop it rather than defend it.

### 3. Things NOT to do

- **More architectures.** Six is enough; they are indistinguishable on every axis measured.
- **More auxiliary-loss variants.** The λ sweep says the knob does not move faithfulness reliably.
- **Chasing PAT further.** Four instruments agree, and the fifth explanation (PEP contamination,
  drug-dependent in surgical data) is now understood.
- **Cross-dataset OOD tables as evidence of mechanism.** Demonstrated to reward degenerate
  predictors.

---

## Immediate housekeeping

- `fig_main` panels e and f still show the retracted r = −0.71 and superseded slopes. Rebuild
  around the pooled/within dissociation or cut them.
- CalBased is a 12k subsample of 51,720. Run in full before submission.
- Every result currently described as "PTT" is PAT = PEP + PTT. The naming is fixed in `mechlib`;
  the write-up needs the same correction throughout.

---

## Quick probes against the arterial ground truth (2026-08-03)

Four short tests using the invasive ABP channel as reference. Together they put a hard ceiling on
what any cuffless method can achieve on this data.

**1. How much does the governing law actually buy?** Fitting a per-subject line from
GROUND-TRUTH arrival time (Q-onset to ABP foot) to DBP:

| | DBP MAE |
|---|---|
| predict the subject own mean | 5.40 mmHg |
| + true arrival time | 5.05 mmHg |
| + true arrival time + true pressure-wave shape | 4.03 mmHg |

**Perfectly measured arrival time buys 0.35 mmHg.** Even reading the shape off the arterial
waveform itself -- a sensor no wearable can have -- buys only 1.37 mmHg. This is the number the
field has been chasing with optical proxies that recover arrival time at r <= 0.21.

**2. The law is real and stronger where BP moves more.** r(PAT, DBP) is -0.201 in subjects with a
small BP range and -0.298 in those with a large one, so the relationship is not an artifact of
subjects who happen to be stable. It is genuine, and small.

**3. The within-subject signal is mostly not slow drift.** Within-subject DBP sd is 7.03 mmHg
while the median beat-to-beat change between consecutive segments is 6.61 mmHg. Nearly all the
within-subject variation is fast, so a model would have to track pressure beat to beat rather
than follow a trend.

**What this means for the story.** The project has been asking why models do not use the
governing law. These tests suggest a blunter answer: **on this data the law is worth about
0.35 mmHg even measured perfectly**, against a 5.40 mmHg floor and 7.03 mmHg of within-subject
variation to explain. Models do not use it because there is very little there to use, and what is
there is only recoverable through an optical fiducial that tracks it at r <= 0.21.

That is a stronger and more useful claim than model unfaithfulness, and it is testable elsewhere:
if the same ceiling appears in a dataset with larger BP excursions, it is a property of cuffless
sensing rather than of surgical data.
