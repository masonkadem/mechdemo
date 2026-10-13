"""build_comps_walkthrough_nb.py -- emit comps_walkthrough.ipynb.

A study notebook for comps and the committee meeting: every step of the project, from the
physiology to the audit, in small runnable blocks, PyTorch-tutorial style. Each part has
by-hand exercises (pen and paper, then a cell that checks the answer without revealing it) and
code exercises (a TODO function with the solution folded underneath). Linear-algebra ideas are
practised where the project uses them: convolution and shifts as matrices, least squares and
the hat matrix, centering and within-patient (fixed-effects) slopes, attention as matrix
products, ridge and the SVD, calibration as a one-column least-squares problem.

Uses the real code (mechlib, real_audit_controls, synth_waveform_audit), the local
data/vitaldb_mini.npz and vitaldb_mini_deep.npz, the saved checkpoints in data/, and the
committed result JSONs. About 5-10 minutes end to end on the RTX 5070; set QUICK = True in the
first code cell for a faster pass on CPU.

    python notebooks/build_comps_walkthrough_nb.py
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "comps_walkthrough.ipynb"
cells = []


def _lines(s):
    return [l + "\n" for l in s.split("\n")[:-1]] + [s.split("\n")[-1]]


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(s.strip())})


def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": _lines(s.strip("\n"))})


def solution(s):
    """A folded answer box under an exercise."""
    md("<details><summary><b>Solution</b> (try it first)</summary>\n\n" + s.strip() + "\n\n</details>")


# =============================================================================================
md(r"""
# Does a blood-pressure model use the physiology it should? A walkthrough

This notebook walks through the whole project in runnable steps, from the physics to the
final audit. Run it top to bottom. Each part has:

- **Concept**: a short explanation and the one equation that matters.
- **✍️ By hand**: work it on paper first, then type your answer into the check cell. The check
  says *correct* or *not quite* without showing the answer.
- **🧪 Code it**: a function with a `TODO`. The solution is folded underneath. Later cells use
  the project's own implementation, so the notebook still runs if you skip an exercise.
- **🗣️ Committee question**: the kind of question you are likely to be asked, with a short
  answer folded underneath.

### The project in four sentences

1. Cuffless BP models read ECG + PPG, and physiology says they *should* use pulse arrival time
   (PAT): a stiffer, higher-pressure artery carries the pulse faster, so **longer PAT → lower BP**.
2. Accuracy cannot tell you whether a model does this. An **intervention audit** can: shift the
   PPG relative to the ECG (a controlled change in PAT) and see which way each patient's
   predicted BP moves.
3. The audit is **validated where the answer is known** (simulated ECG/PPG with an injected law),
   and the **law is checked in the data itself** (within patients, longer PAT goes with lower BP
   in ~74% (DBP) to ~87% (SBP) of patients).
4. Applied to transformers trained on VitalDB, the DBP models **read PAT backwards** (expected
   sign in only 11–43% of patients) while sitting at the accuracy of predicting the mean.

### Roadmap

| Part | Topic | Linear algebra / stats practised |
|---|---|---|
| 1 | Physics: why PAT should predict BP | derivatives, the chain rule, autograd preview |
| 2 | The data as arrays | shapes, broadcasting, z-scores, variance decomposition |
| 3 | Signals: filters, R peaks, PPG feet, beat pairing | convolution = Toeplitz matrix, linearity, SVD/PCA of beats |
| 4 | Least squares, the workhorse | normal equations, hat matrix, centering matrix, Frisch–Waugh–Lovell, Simpson's paradox, sign test |
| 5 | The data's own PAT → BP law | within-patient slopes, noise ceiling |
| 6 | PyTorch from zero | tensors, autograd, `nn.Linear`, gradient descent, conditioning and eigenvalues |
| 7 | The transformer, piece by piece | patch embedding as matmul, attention by hand, multi-head, LayerNorm, parameter counting |
| 8 | Training on VitalDB | MSE vs MAE, regression to the mean, skill over baseline |
| 9 | The intervention audit | shift operators as matrices, rank, controls, specificity |
| 10 | Validating the audit in simulation | dose–response |
| 11 | Probes: decodable is not the same as used | ridge regression, SVD shrinkage |
| 12 | Calibration | one-column least squares, variance ∝ 1/k |
| 13 | Out of distribution (MIMIC-BP) | bias vs within-patient tracking |
| 14 | Results at a glance | |
| 15 | Question bank | |
| 16 | Cheat sheet | |
""")

code(r"""
# ---- setup ---------------------------------------------------------------------------------
import sys, json, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.signal import butter, cheby2, sosfilt, sosfiltfilt, find_peaks

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import mechlib                                   # the project's toolkit
from mechlib import ECG, PPG, ABP                # channel indices: 0, 1, 2

DATA = ROOT / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
QUICK = DEV == "cpu"                             # fewer epochs / smaller sims on CPU
torch.manual_seed(0)
rng = np.random.default_rng(0)
plt.rcParams.update({"figure.dpi": 110, "figure.figsize": (8, 3), "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})
print(f"root: {ROOT}\ndevice: {DEV} | torch {torch.__version__} | QUICK = {QUICK}")


def check(name, yours, answer, tol=1e-2):
    # Compare your answer with the reference without printing the reference.
    if yours is None:
        print(f"[{name}] not attempted: replace None with your answer and re-run this cell")
        return
    try:
        ok = np.allclose(np.asarray(yours, float), np.asarray(answer, float), rtol=tol, atol=tol)
    except (ValueError, TypeError):
        ok = False
    print(f"[{name}] " + ("correct ✓" if ok else "not quite ✗ (check your working; the solution is folded below)"))


def load_json(name):
    p = DATA / name
    return json.loads(p.read_text()) if p.exists() else None
""")

# =============================================================================================
md(r"""
---
# Part 1 — Physics: why PAT should predict BP

**The chain of reasoning** (each link is a question you should be able to answer):

1. **Moens–Korteweg**: the pulse wave travels along an elastic tube at
   $$\text{PWV} = \sqrt{\frac{E\,h}{\rho\,d}}$$
   ($E$ wall stiffness, $h$ wall thickness, $d$ diameter, $\rho$ blood density).
2. **Hughes**: arterial stiffness rises with pressure, $E = E_0\,e^{\gamma P}$.
3. **Transit time** over path length $L$: $\text{PTT} = L / \text{PWV}$.
4. Put together:
   $$\text{PTT} = L\sqrt{\tfrac{\rho d}{E_0 h}}\;e^{-\gamma P/2}
   \;\Rightarrow\; P = \tfrac{2}{\gamma}\ln\!\Big(L\sqrt{\tfrac{\rho d}{E_0 h}}\Big) - \tfrac{2}{\gamma}\ln \text{PTT}.$$
   So **higher pressure → stiffer wall → faster wave → shorter PTT**, and
   $\frac{dP}{d\,\text{PTT}} = -\frac{2}{\gamma\,\text{PTT}} < 0$.
   In practice people fit $BP = a/\text{PTT}^p + b$ or $BP = a\ln \text{PTT} + b$ per patient.
5. **What we actually measure is PAT, not PTT**:
   $$\text{PAT} = \underbrace{\text{PEP}}_{\text{cardiac: R wave → valve opening}} + \underbrace{\text{PTT}}_{\text{vascular}} \;\big(+\ \text{a fixed instrument offset in VitalDB's PPG}\big).$$
   PEP carries no stiffness information and changes with contractility and drugs, so PAT is a
   noisier, diluted version of PTT. The VitalDB PPG offset (~476 ms) is constant, so it cancels in
   *within-patient changes* but makes absolute PAT meaningless.
""")

md(r"""
### ✍️ By hand 1.1 — transit time
The heart-to-finger path is $L = 0.8$ m. PWV rises from 5 m/s to 6 m/s.
What are the PTTs before and after, in ms? Enter `[before, after]`.
""")
code(r"""
my_ptt = None          # e.g. [ ..., ... ]
check("1.1 PTT", my_ptt, [160.0, 133.33])
""")
solution(r"""
PTT = L / PWV: 0.8/5 = 0.160 s = **160 ms**; 0.8/6 = 0.1333 s = **133.3 ms**. A 20% faster wave gives a ~27 ms shorter transit, which is a few samples at 125 Hz (8 ms per sample).
""")

md(r"""
### ✍️ By hand 1.2 — the local slope of the law
Take $DBP = a / \text{PAT}$ with $a = 12000$ mmHg·ms.

(a) DBP at PAT = 200 ms? (b) The slope $d\,DBP / d\,\text{PAT}$ there, in mmHg/ms?
Enter `[dbp, slope]`.
""")
code(r"""
my_12 = None
check("1.2 law", my_12, [60.0, -0.3])
""")
solution(r"""
(a) 12000 / 200 = **60 mmHg**. (b) $\frac{d}{dx}(a/x) = -a/x^2 = -12000/40000 =$ **−0.3 mmHg/ms**.
Every 10 ms of extra arrival time lowers DBP by about 3 mmHg. The data's own median within-patient slope for SBP (Part 5) is in the same range, about −0.24 to −0.27 mmHg/ms.

**Follow-up:** if the measured PAT includes a constant 476 ms offset, $\text{PAT}_m = \text{PAT} + 476$, then $d\,DBP/d\,\text{PAT}_m = d\,DBP/d\,\text{PAT}$ (the derivative of a constant is 0). The **slope survives the offset; the absolute value does not**. That is why the project only uses within-patient changes.
""")

md(r"""
### Check it with autograd (a first look at PyTorch)
PyTorch can differentiate any computation you write. Here it gives the same slope you computed by hand.
""")
code(r"""
# Step 1: a number PyTorch can take derivatives with respect to
pat = torch.tensor(200.0, requires_grad=True)       # PAT = 200 ms
print("pat =", pat)

# Step 2: the law, DBP = 12000 / PAT
dbp = 12000.0 / pat
print("dbp =", dbp)                                  # 12000 / 200 = 60 mmHg

# Step 3: ask PyTorch for the slope d(dbp) / d(pat)
dbp.backward()
print("slope =", pat.grad)                           # -12000 / 200^2 = -0.3 mmHg per ms
""")
md(r"""
The physics chain, one quantity per line: pressure → wall stiffness → wave speed → transit time.
""")
code(r"""
pressure = np.linspace(50, 130, 100)                 # 100 pressures from 50 to 130 mmHg
print("pressure:", pressure[:3].round(1), "...", pressure[-1])

stiffness = 0.08e6 * np.exp(0.015 * pressure)        # Hughes: E = E0 * exp(gamma * P), in Pa
print("stiffness at 50 and 130 mmHg:", stiffness[0].round(), stiffness[-1].round(), "Pa")

speed = np.sqrt(stiffness * 1e-3 / (1060 * 4e-3))    # Moens-Korteweg: PWV = sqrt(E h / (rho d)), m/s
print("wave speed at 50 and 130 mmHg:", speed[0].round(2), speed[-1].round(2), "m/s")

ptt = 1000 * 1.0 / speed                             # time to travel 1 m, in ms
print("transit time at 50 and 130 mmHg:", ptt[0].round(), ptt[-1].round(), "ms")

fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
ax[0].plot(pressure, ptt)
ax[0].set(xlabel="Pressure (mmHg)", ylabel="PTT (ms)", title="Higher pressure, shorter transit")
ax[1].plot(ptt, pressure)
ax[1].set(xlabel="PTT (ms)", ylabel="Pressure (mmHg)", title="Flipped: the law a model should learn")
plt.tight_layout(); plt.show()
""")

md(r"""
<details><summary>🗣️ <b>Committee question:</b> "Why is PAT a worse surrogate than PTT, and why use it anyway?"</summary>

PAT = PEP + PTT. PEP (electrical activation to aortic valve opening) depends on contractility, preload and drugs (propofol lengthens it, inotropes shorten it), not on arterial stiffness, so it adds variance that does not follow BP and can even move against it. We use PAT because ECG + finger PPG is what monitors and wearables record; true PTT needs two arterial sites or an aortic-valve timing signal. The project therefore states the law as the *data's own* within-patient PAT → BP relation (Part 5) and judges models against that, not against an ideal PTT law. It also reports arterial-line PAT (ECG → ABP foot) as a cleaner reference.
</details>
""")

# =============================================================================================
md(r"""
---
# Part 2 — The data as arrays

PulseDB's VitalDB subset: 10 s segments at **125 Hz** → 1250 samples, three channels
**ECG = 0, PPG = 1, ABP = 2**, labels **[SBP, DBP]**, and a patient ID per segment.
Splits are **by patient**: no test patient appears in training.

A dataset is a 3-D array (a rank-3 tensor) `X[n, t, c]`: segment `n`, time `t`, channel `c`.
""")
code(r"""
d = mechlib.load_mini(str(DATA / "vitaldb_mini.npz"))     # a dictionary of arrays

fs = d["fs"]
print("fs =", fs, "Hz  -> one sample =", 1000 / fs, "ms")

print("Xtr:", d["Xtr"].shape, " = (segments, samples, channels)")
print("ytr:", d["ytr"].shape, "    = (segments, [SBP, DBP])")
print("gtr:", d["gtr"].shape, "       = one patient ID per segment")
print("Xva:", d["Xva"].shape)
print("Xte:", d["Xte"].shape)

n_samples = d["Xtr"].shape[1]
print("one segment =", n_samples, "samples =", n_samples / fs, "s")

train_patients = np.unique(d["gtr"])
val_patients = np.unique(d["gva"])
test_patients = np.unique(d["gte"])
print("patients: train", len(train_patients), "| val", len(val_patients), "| test", len(test_patients))

overlap = set(train_patients) & set(test_patients)
print("test patients that also appear in training:", len(overlap))
""")

code(r"""
i = 0                                    # which test segment; change i to browse
segment = d["Xte"][i]                    # one segment, (1250, 3)
print("segment:", segment.shape)

t = np.arange(n_samples) / fs            # time of each sample: 0, 0.008, 0.016, ... s
print("t:", t[:3], "...", t[-1])

sbp, dbp = d["yte"][i]                   # its labels
patient = d["gte"][i]
print("patient", patient, "| SBP", sbp, "| DBP", dbp)

fig, ax = plt.subplots(3, 1, figsize=(9, 4.5), sharex=True)
ax[0].plot(t, segment[:, ECG], lw=0.8); ax[0].set_ylabel("ECG (au)")     # column 0
ax[1].plot(t, segment[:, PPG], lw=0.8); ax[1].set_ylabel("PPG (au)")     # column 1
ax[2].plot(t, segment[:, ABP], lw=0.8); ax[2].set_ylabel("ABP (mmHg)")   # column 2
ax[0].set_title(f"Test segment {i}: patient {patient}, SBP/DBP = {sbp:.0f}/{dbp:.0f} mmHg")
ax[2].set_xlabel("Time (s)"); plt.tight_layout(); plt.show()
""")

md(r"""
### Indexing and broadcasting
- `X[:, :, PPG]` → every segment's PPG, shape `(N, 1250)`.
- `X.mean(axis=1, keepdims=True)` → one mean per segment and channel, shape `(N, 1, C)`.
- `X - X.mean(1, keepdims=True)`: NumPy **broadcasts** the `(N, 1, C)` array along the time
  axis. This is how per-segment normalisation is written without loops.

Every model in the project sees **z-scored** channels: each segment's ECG and PPG are scaled to
mean 0, SD 1 (`mechlib.normalize`). This removes amplitude, so a model cannot use PPG amplitude
(which depends on sensor contact, not BP). Timing and shape are preserved.

### ✍️ By hand 2.1 — z-score
z-score the vector `[2, 4, 4, 4, 5, 5, 7, 9]` (population SD, i.e. divide by n).
""")
code(r"""
my_z = None
check("2.1 z-score", my_z, [-1.5, -0.5, -0.5, -0.5, 0, 0, 1, 2])
""")
solution(r"""
Mean = 40/8 = 5. Deviations: −3, −1, −1, −1, 0, 0, 2, 4. Squares sum to 9+1+1+1+0+0+4+16 = 32, variance 32/8 = 4, SD = 2.
z = **[−1.5, −0.5, −0.5, −0.5, 0, 0, 1, 2]**.
""")

md(r"""
### 🧪 Code it 2.2 — normalise with broadcasting
Write `my_normalize(X)` for an `(N, L, C)` array: subtract each segment's per-channel mean and
divide by its per-channel SD (+1e-8). No loops.
""")
code(r"""
def my_normalize(X):
    # TODO: one line for the mean, one for the SD
    return None

Xs = d["Xte"][:5, :, :2].astype(np.float32)
if my_normalize(Xs) is not None:
    print("matches mechlib.normalize:", np.allclose(my_normalize(Xs), mechlib.normalize(Xs), atol=1e-5))
""")
solution(r"""
```python
def my_normalize(X):
    X = X.astype(np.float32)
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
```
`keepdims=True` keeps the time axis as size 1, so the `(N, 1, C)` statistics broadcast against `(N, L, C)`.
""")

md(r"""
### Between- vs within-patient variance
The **law of total variance**:
$$\operatorname{Var}(y) = \underbrace{\mathbb{E}\big[\operatorname{Var}(y\mid \text{patient})\big]}_{\text{within}} + \underbrace{\operatorname{Var}\big(\mathbb{E}[y\mid\text{patient}]\big)}_{\text{between}}.$$
This split recurs throughout the project. The PAT law is a *within*-patient statement, while
most of a calibration-free model's error is the *between*-patient offset.

### ✍️ By hand 2.3
Patient A has DBP readings [60, 62]; patient B has [80, 82]. Compute total, within and between
variance (population formulas). Enter `[total, within, between]`.
""")
code(r"""
my_var = None
check("2.3 variance split", my_var, [101.0, 1.0, 100.0])
""")
solution(r"""
Grand mean 71. Total: (121 + 81 + 81 + 121)/4 = **101**. Within: each patient has variance 1, mean **1**. Between: patient means 61 and 81, variance of {61, 81} = **100**. Check: 1 + 100 = 101.
""")
code(r"""
y = d["yte"][:, 1]                      # test DBP, one per segment
g = d["gte"]                            # the patient of each segment
print(len(y), "segments from", len(np.unique(g)), "patients")

patient_mean = np.zeros(len(y))         # for each segment: its patient's average DBP
for p in np.unique(g):
    this_patient = g == p               # True for this patient's segments
    patient_mean[this_patient] = y[this_patient].mean()
print("first 5 DBP:          ", y[:5].round(1))
print("their patient's mean: ", patient_mean[:5].round(1))

within = np.mean((y - patient_mean) ** 2)            # spread around each patient's own mean
between = np.mean((patient_mean - y.mean()) ** 2)    # spread of the patient means
total = y.var()
print("total   =", total.round(1))
print("within  =", within.round(1))
print("between =", between.round(1))
print("within + between =", (within + between).round(1))
print("share between patients:", (between / total).round(2))
""")

md(r"""
### The baseline every model must beat: predict the training mean
The MSE-optimal constant is the **mean**; the MAE-optimal constant is the **median**.

### ✍️ By hand 2.4
Labels `y = [60, 60, 60, 100]`. MAE when predicting the mean, and when predicting the median?
Enter `[mae_mean, mae_median]`.
""")
code(r"""
my_24 = None
check("2.4 baselines", my_24, [15.0, 10.0])
""")
solution(r"""
Mean 70: errors 10, 10, 10, 30, MAE **15**. Median 60: errors 0, 0, 0, 40, MAE **10**. Squared error penalises the one big miss most, so the mean moves toward the outlier.
""")
code(r"""
train_dbp = d["ytr"][:, 1]
test_dbp = d["yte"][:, 1]

guess = train_dbp.mean()                       # the same guess for every test segment
print("training mean DBP:", guess.round(1))

errors = np.abs(test_dbp - guess)
print("first 5 errors:", errors[:5].round(1))
print("DBP MAE of predicting the mean:", errors.mean().round(2), "mmHg")

guess_sbp = d["ytr"][:, 0].mean()
errors_sbp = np.abs(d["yte"][:, 0] - guess_sbp)
print("SBP MAE of predicting the mean:", errors_sbp.mean().round(2), "mmHg")
""")

# =============================================================================================
md(r"""
---
# Part 3 — Signals: filters, R peaks, PPG feet and beat pairing

PAT is measured per segment in three steps, all in `mechlib.py`:
1. **R peaks** on the ECG (`r_peaks`, Pan–Tompkins).
2. **PPG feet** on the PPG alone (`ppg_feet`, intersecting tangent).
3. **Pairing** each foot with the R peak that caused it (`pair_beats`, constant delay).

All three depend on filters, so we start with what a filter is in linear-algebra terms.

## 3.1 A filter is a matrix
Discrete convolution $y[n] = \sum_k h[k]\,x[n-k]$ is linear in $x$, so it is a matrix–vector
product $y = Hx$. $H$ is a **Toeplitz** matrix: each row is the flipped kernel, shifted one
column to the right.
""")
code(r"""
x = np.array([1., 2, 3, 4, 5])
kernel = np.array([1/3, 1/3, 1/3])           # average of 3 neighbours
print("x:", x)

H = np.zeros((3, 5))                         # 3 outputs, 5 inputs
H[0, 0:3] = kernel                           # output 0 averages x[0..2]
H[1, 1:4] = kernel                           # output 1 averages x[1..3]
H[2, 2:5] = kernel                           # output 2 averages x[2..4]
print("H =\n", H.round(3))

y = H @ x
print("H @ x:      ", y)                     # [2, 3, 4]
print("np.convolve:", np.convolve(x, kernel, mode="valid"))
""")
md(r"""
### ✍️ By hand 3.1
Apply the kernel `h = [1, -1]` (valid convolution, so $y[n] = x[n] - x[n-1]$) to
`x = [1, 4, 9, 16, 25]`. This is a discrete derivative.
""")
code(r"""
my_31 = None
check("3.1 derivative filter", my_31, [3, 5, 7, 9])

H_diff = np.zeros((4, 5))                    # 4 outputs from 5 inputs
for row in range(4):
    H_diff[row, row] = -1                    # minus the earlier sample
    H_diff[row, row + 1] = 1                 # plus the later sample
print("H for y[n] = x[n] - x[n-1]:\n", H_diff)
""")
solution(r"""
Differences of successive squares: **[3, 5, 7, 9]**. (Second differences would all be 2: the derivative of a parabola is a line.)
""")

md(r"""
## 3.2 Zero-phase filtering: why the timing survives
A causal (one-way) filter **delays** the signal. Everything here is a timing measurement at the
8 ms scale, so the project filters forward *and* backward (`sosfiltfilt`). The two phase delays
cancel and the timing is unchanged. Linearity is checked too: $H(a x_1 + b x_2) = aHx_1 + bHx_2$.
""")
code(r"""
tt = np.arange(0, 4, 1 / fs)                         # 4 s of time
pulse = np.exp(-0.5 * ((tt - 2) / 0.05) ** 2)        # one smooth pulse at t = 2 s
print("pulse peaks at sample", np.argmax(pulse), "=", tt[np.argmax(pulse)], "s")

sos = butter(2, [0.5, 10], btype="band", fs=fs, output="sos")   # a 0.5-10 Hz band-pass filter

one_way = sosfilt(sos, pulse)                        # filter forwards only
two_way = sosfiltfilt(sos, pulse)                    # forwards, then backwards

lag_one = (np.argmax(one_way) - np.argmax(pulse)) * 1000 / fs
lag_two = (np.argmax(two_way) - np.argmax(pulse)) * 1000 / fs
print("one-way filter moves the peak by", lag_one, "ms")
print("two-way filter moves the peak by", lag_two, "ms")

x1 = rng.normal(size=500)                            # linearity check with two random signals
x2 = rng.normal(size=500)
left = sosfiltfilt(sos, 2 * x1 - 3 * x2)
right = 2 * sosfiltfilt(sos, x1) - 3 * sosfiltfilt(sos, x2)
print("filter(2 x1 - 3 x2) == 2 filter(x1) - 3 filter(x2):", np.allclose(left, right))

fig, ax = plt.subplots(figsize=(7, 2.3))
ax.plot(tt, pulse, label="input")
ax.plot(tt, one_way, label="one-way")
ax.plot(tt, two_way, "--", label="zero-phase")
ax.set(xlim=(1.6, 2.6), xlabel="Time (s)"); ax.legend(frameon=False); plt.show()
""")

md(r"""
## 3.3 R peaks: Pan–Tompkins, stage by stage
`mechlib.r_peaks`: 5–15 Hz bandpass (keeps QRS energy, suppresses T waves) → derivative →
square → 150 ms moving average → peaks ≥ 0.3 s apart → refine to the largest |bandpassed ECG|
within 60 ms. Below, each stage is recomputed and checked against the library.
""")
code(r"""
seg = d["Xte"][0].astype(float)           # one test segment, (1250, 3)
ecg = seg[:, ECG]
ppg = seg[:, PPG]
abp = seg[:, ABP]
print("ecg:", ecg.shape)

# 1. band-pass 5-15 Hz: keeps the sharp QRS, removes the slower T wave
band = sosfiltfilt(butter(2, [5, 15], btype="band", fs=fs, output="sos"), ecg)

# 2. slope, 3. squared: large and positive wherever the ECG changes fast
slope = np.gradient(band)
squared = slope ** 2

# 4. moving average over 150 ms
w = int(0.15 * fs)
print("averaging window:", w, "samples")
smooth = np.convolve(squared, np.ones(w) / w, mode="same")

# 5. candidate beats: peaks at least 0.3 s apart and tall enough
peaks, _ = find_peaks(smooth, distance=int(0.3 * fs), height=0.25 * np.percentile(smooth, 99))
print("candidate peaks:", peaks)

# 6. move each to the biggest |band-passed ECG| within 60 ms
h = int(0.06 * fs)
r_mine = []
for p in peaks:
    start = max(p - h, 0)
    window = np.abs(band[start : p + h])
    r_mine.append(start + int(np.argmax(window)))
r_mine = np.array(r_mine)

r_lib = mechlib.r_peaks(ecg, fs)
print("my R peaks:     ", r_mine)
print("mechlib R peaks:", r_lib)
print("same:", np.array_equal(r_mine, r_lib))

rr = np.diff(r_lib) / fs                   # time between beats, s
print("RR intervals:", rr.round(2), "s -> heart rate", round(60 / np.median(rr)), "bpm")

fig, ax = plt.subplots(4, 1, figsize=(9, 5.5), sharex=True)
ax[0].plot(t, ecg, lw=0.8); ax[0].set_ylabel("raw ECG")
ax[0].plot(t[r_lib], ecg[r_lib], "o", ms=4)
ax[1].plot(t, band, lw=0.8); ax[1].set_ylabel("5–15 Hz")
ax[2].plot(t, squared, lw=0.8); ax[2].set_ylabel("slope²")
ax[3].plot(t, smooth, lw=0.8); ax[3].set_ylabel("moving avg")
ax[3].plot(t[peaks], smooth[peaks], "o", ms=4)
ax[3].set_xlabel("Time (s)"); plt.tight_layout(); plt.show()
""")

md(r"""
## 3.4 PPG feet: the intersecting tangent
For each beat: find the steepest upslope point $k$ (peak of the first derivative), draw the
tangent there, and intersect it with the horizontal line through the preceding minimum $p_{\min}$:
$$\text{foot} = k - \frac{p[k] - p_{\min}}{p'[k]}.$$
This is the standard, most reproducible foot definition. It interpolates between samples, so
the foot is a *float* index.

### ✍️ By hand 3.4
The trough before a beat has value 0.1. At the steepest point $k = 60$, $p[k] = 0.5$ and the
slope is $p'[k] = 0.08$ per sample. Where is the foot (sample index), and at what time (ms) at
125 Hz? Enter `[index, ms]`.
""")
code(r"""
my_foot = None
check("3.4 foot", my_foot, [55.0, 440.0])
""")
solution(r"""
foot = 60 − (0.5 − 0.1)/0.08 = 60 − 5 = **55** samples → 55 × 8 ms = **440 ms**.
""")
code(r"""
feet = mechlib.ppg_feet(ppg, fs)                      # one foot per beat, as a sample index
print("feet (samples):", feet.round(1))               # fractional: between two samples
print("feet (ms):     ", (feet * 1000 / fs).round())

pf = sosfiltfilt(cheby2(4, 20, [0.5, 10], btype="band", fs=fs, output="sos"), ppg)   # filtered PPG
dp = np.gradient(pf)                                  # its slope

start = int(feet[1])                                  # the second beat
k0 = start + int(np.argmax(dp[start : start + 40]))   # its steepest point
print("steepest point of beat 2: sample", k0, "| value", pf[k0].round(3), "| slope", dp[k0].round(3))
tangent_x = np.arange(k0 - 25, k0 + 8)
tangent_y = pf[k0] + dp[k0] * (tangent_x - k0)        # straight line through the steepest point

fig, ax = plt.subplots(2, 1, figsize=(9, 3.6), sharex=True)
ax[0].plot(t, pf, lw=0.9)
ax[0].plot(feet / fs, np.interp(feet, np.arange(len(pf)), pf), "v", ms=6)
ax[0].plot(tangent_x / fs, tangent_y, "k--", lw=0.8)
ax[0].set(ylabel="PPG (filtered)", ylim=(pf.min() - 0.3 * np.ptp(pf), pf.max() + 0.1 * np.ptp(pf)))
ax[1].plot(t, dp, lw=0.8); ax[1].set(ylabel="dPPG/dt", xlabel="Time (s)")
plt.tight_layout(); plt.show()
""")

md(r"""
## 3.5 Pairing beats: the bug that changed the project
VitalDB's PPG channel lags the ECG by **several hundred ms**, often more than one RR interval.
An old-style detector that looks for the foot 50–350 ms after *each* R peak then pairs a foot
with the *wrong* (next) R peak. That turned "PAT" into a mix of T waves, diastolic dips and
heart rate.

The fix (`pair_beats`, after Ruffolo et al. 2025): try every R peak up to 1.5 s before each
foot, and pick the **delay that is the same for every beat**. The true pairing is constant
across beats; a wrong pairing drifts with beat-to-beat heart-rate variability.

### ✍️ By hand 3.5
R peaks at **0, 600, 1240, 1840, 2480 ms**; PPG feet at **700, 1300, 1940, 2540 ms**.
(a) For each foot, the delay to the *most recent* R peak. (b) The delay to the R peak *before*
that. (c) The SD (population) of each list. Which pairing is physiological? Enter
`[sd_most_recent, sd_one_before]`.
""")
code(r"""
my_35 = None
check("3.5 pairing spread", my_35, [20.0, 0.0])
""")
solution(r"""
(a) Most recent: 700−600 = 100, 1300−1240 = 60, 1940−1840 = 100, 2540−2480 = 60, i.e. [100, 60, 100, 60], SD **20 ms**.
(b) One before: 700−0, 1300−600, 1940−1240, 2540−1840 = [700, 700, 700, 700], SD **0**.

The constant 700 ms delay is the true one: the foot follows the beat that caused it by a fixed amount. The "most recent R peak" delay looks plausible (60–100 ms) but it is wrong, and it varies with RR. A fixed 50–350 ms window would pick it.
""")
code(r"""
# the toy example (fs = 1000 Hz, so a sample index is a time in ms)
r_toy = np.array([0, 600, 1240, 1840, 2480])          # R peaks
feet_toy = np.array([700., 1300, 1940, 2540])         # PPG feet
fi, ri, dl = mechlib.pair_beats(r_toy, feet_toy, 1000)
print("paired feet:   ", fi)
print("paired R peaks:", ri)
print("delays (s):    ", dl)                          # the same for every beat
""")
code(r"""
# the real segment
fi, ri, dl = mechlib.pair_beats(r_lib, feet, fs)
delays_ms = dl * 1000
print("constant-delay pairing:", len(dl), "beats")
print("  delays (ms):", delays_ms.round())
print("  SD:", np.std(delays_ms).round(1), "ms")

# the old way: lowest PPG point 50-350 ms (6-44 samples) after each R peak
naive = []
for r in r_lib:
    if r + 44 < len(pf):
        window = pf[r + 6 : r + 44]
        naive.append((6 + np.argmin(window)) * 1000 / fs)
naive = np.array(naive)
print("old fixed window:")
print("  delays (ms):", naive.round())
print("  SD:", np.std(naive).round(1), "ms  <- whatever dip happens to sit in the window")

beat_ms = np.median(np.diff(r_lib)) * 1000 / fs
print("one heartbeat lasts", round(beat_ms), "ms; the true delay is", round(np.median(delays_ms)), "ms")

abp_pat = mechlib.segment_ptt(ecg, abp, fs) * 1000    # ECG -> arterial-line foot
print("ECG -> PPG foot:", round(np.median(delays_ms)), "ms")
print("ECG -> ABP foot:", round(abp_pat), "ms")
print("extra lag of the PPG channel:", round(np.median(delays_ms) - abp_pat), "ms (the monitor, not physiology)")
""")
code(r"""
fig, ax = plt.subplots(2, 1, figsize=(9, 3.6), sharex=True)
ax[0].plot(t, ecg, lw=0.8); ax[0].set_ylabel("ECG")
ax[1].plot(t, ppg, lw=0.8); ax[1].set_ylabel("PPG"); ax[1].set_xlabel("Time (s)")
for f_, r_ in zip(fi, ri):                       # dashed line from each R peak to its paired foot
    ax[0].axvline(r_ / fs, color="grey", ls=":", lw=0.8); ax[1].axvline(f_ / fs, color="grey", ls=":", lw=0.8)
    ax[1].annotate("", xy=(f_ / fs, ppg.min()), xytext=(r_ / fs, ppg.min()),
                   arrowprops=dict(arrowstyle="->", color="C3", lw=0.8))
plt.tight_layout(); plt.show()
""")

md(r"""
## 3.6 Linear-algebra drill: the average heartbeat by SVD
Stack many foot-aligned PPG beats as rows of a matrix $B$ (beats × time). The **SVD**
$B_c = U\Sigma V^\top$ of the centred matrix gives principal components (rows of $V^\top$),
the main ways beats differ from the mean beat. The share of variance in component $i$ is
$\sigma_i^2 / \sum_j \sigma_j^2$. (`fit_beat_templates.py` fits the simulator's beat shapes
to this mean beat.)

### ✍️ By hand 3.6
The symmetric matrix $A = \begin{pmatrix}2&1\\1&2\end{pmatrix}$. Its eigenvalues (largest first)?
Hint: $\det(A-\lambda I) = (2-\lambda)^2 - 1$. Enter `[l1, l2]`.
""")
code(r"""
my_eig = None
check("3.6 eigenvalues", my_eig, [3.0, 1.0])
A = np.array([[2., 1], [1, 2]])
""")
solution(r"""
$(2-\lambda)^2 = 1 \Rightarrow \lambda = 3$ or $1$. Eigenvectors $[1,1]/\sqrt2$ (λ=3) and $[1,-1]/\sqrt2$ (λ=1). For a symmetric positive semi-definite matrix such as a covariance $B_c^\top B_c/n$, the eigenvalues are the squared singular values of $B_c$ divided by $n$, which is the PCA ↔ SVD link. Check with `np.linalg.eigh(A)` (returns ascending order).
""")
code(r"""
beats = []
for one_ppg in d["Xte"][:300, :, PPG].astype(float):          # the first 300 test segments
    z = (one_ppg - one_ppg.mean()) / (one_ppg.std() + 1e-8)
    for foot in mechlib.ppg_feet(one_ppg, fs).astype(int):
        if 10 <= foot < len(z) - 90:
            beat = z[foot - 10 : foot + 90]                      # 100 samples: 80 ms before to 720 ms after the foot
            beat = (beat - beat.min()) / (np.ptp(beat) + 1e-8)   # rescale to 0..1
            beats.append(beat)

B = np.array(beats)
print("B:", B.shape, "= (beats, samples)")

mean_beat = B.mean(0)
Bc = B - mean_beat                                             # centre: subtract the mean beat
U, S, Vt = np.linalg.svd(Bc, full_matrices=False)
print("U:", U.shape, " S:", S.shape, " Vt:", Vt.shape)

share = S ** 2 / (S ** 2).sum()                                # variance share of each component
print("variance share of PC1, PC2, PC3:", share[:3].round(3))

cov = Bc.T @ Bc / len(B)
eigenvalues = np.linalg.eigh(cov)[0][::-1]                     # largest first
print("eigenvalues of the covariance:", eigenvalues[:3].round(4))
print("S^2 / n from the SVD:         ", (S[:3] ** 2 / len(B)).round(4))

time_ms = (np.arange(100) - 10) * 1000 / fs
fig, ax = plt.subplots(1, 2, figsize=(9, 2.6))
ax[0].plot(time_ms, mean_beat); ax[0].set(title="Mean beat", xlabel="Time from foot (ms)")
for k in range(3):
    ax[1].plot(time_ms, Vt[k], label=f"PC{k + 1} ({share[k]:.0%})")
ax[1].set(title="Principal components", xlabel="Time from foot (ms)"); ax[1].legend(frameon=False)
plt.tight_layout(); plt.show()
""")

# =============================================================================================
md(r"""
---
# Part 4 — Least squares, the workhorse

Nearly every number in the project is a least-squares fit: within-patient slopes, audit slopes,
linear probes, calibration offsets. So it is worth knowing well.

**Model** $y = X\beta + \varepsilon$. Minimise $\|y - X\beta\|^2$. Setting the gradient
$-2X^\top(y - X\beta)$ to zero gives the **normal equations**
$$X^\top X\,\hat\beta = X^\top y \quad\Rightarrow\quad \hat\beta = (X^\top X)^{-1}X^\top y.$$
Geometrically, $X\hat\beta$ is the **orthogonal projection** of $y$ onto the column space of
$X$, and the residual is perpendicular to every column: $X^\top(y - X\hat\beta) = 0$.

### ✍️ By hand 4.1 — normal equations
Fit $y = b_0 + b_1 x$ to the points (1, 2), (2, 3), (3, 5).
Write $X$ (with a column of ones), compute $X^\top X$ and $X^\top y$, and solve the 2×2
system. Enter `[b0, b1]`.
""")
code(r"""
my_beta = None
check("4.1 normal equations", my_beta, [1 / 3, 1.5])
""")
solution(r"""
$X = \begin{pmatrix}1&1\\1&2\\1&3\end{pmatrix}$, $y = (2,3,5)^\top$.
$X^\top X = \begin{pmatrix}3&6\\6&14\end{pmatrix}$, $X^\top y = (10, 23)^\top$.
$\det = 42 - 36 = 6$, $(X^\top X)^{-1} = \frac16\begin{pmatrix}14&-6\\-6&3\end{pmatrix}$.
$\hat\beta = \frac16(140-138,\; -60+69) = (\mathbf{1/3},\ \mathbf{1.5})$.
Shortcut: $b_1 = S_{xy}/S_{xx} = 3/2$, $b_0 = \bar y - b_1\bar x = 10/3 - 3 = 1/3$.
""")
code(r"""
X = np.array([[1., 1],              # column 1: ones (the intercept)
              [1, 2],               # column 2: x
              [1, 3]])
y = np.array([2., 3, 5])

XtX = X.T @ X
print("X^T X =\n", XtX)             # [[3, 6], [6, 14]]
Xty = X.T @ y
print("X^T y =", Xty)               # [10, 23]

beta = np.linalg.solve(XtX, Xty)    # solve X^T X beta = X^T y
print("beta = [b0, b1] =", beta)

print("np.linalg.lstsq:", np.linalg.lstsq(X, y, rcond=None)[0])
print("np.polyfit:     ", np.polyfit([1, 2, 3], y, 1), " <- note the order: [slope, intercept]")

H = X @ np.linalg.inv(XtX) @ X.T    # the hat matrix: projects y onto the columns of X
print("H =\n", H.round(3))
fitted = H @ y
residual = y - fitted
print("fitted:  ", fitted.round(3))
print("residual:", residual.round(3))

print("H symmetric (H = H^T):", np.allclose(H, H.T))
print("H @ H = H:", np.allclose(H @ H, H))
print("trace(H) = number of parameters:", np.trace(H).round(6))
print("X^T residual = 0:", np.allclose(X.T @ residual, 0))
""")

md(r"""
## 4.2 Centering, within-patient slopes and Frisch–Waugh–Lovell
The **centering matrix** $C = I - \tfrac1n\mathbf{1}\mathbf{1}^\top$ subtracts the mean:
$Cx = x - \bar x$. The slope of a regression *with* an intercept equals the slope of the
centred data *without* one:
$$b_1 = \frac{(Cx)^\top(Cy)}{(Cx)^\top(Cx)}.$$
With several patients, use a block-diagonal $C$ (one block per patient): subtract each
patient's own mean. This is the **within-patient (fixed-effects) slope**. By the
**Frisch–Waugh–Lovell** theorem it equals the slope on $x$ from a regression that includes one
dummy (intercept) column per patient. Either way, *each patient's baseline is removed before the
slope is estimated*.

**Why this matters here:** VitalDB's PPG offset, age, vessel size and the device all shift a
patient's PAT and BP *levels*. Pooling patients mixes those level differences into the slope,
and can reverse its sign (**Simpson's paradox**).
""")
code(r"""
rs = np.random.default_rng(1)
pat_list, dbp_list, patient_list = [], [], []

# three patients: (typical PAT, typical DBP). Patients with LONGER PAT have HIGHER DBP.
for patient, (pat0, dbp0) in enumerate([(600, 60), (650, 72), (700, 84)]):
    pat = pat0 + rs.normal(0, 12, 20)                         # 20 segments around their typical PAT
    dbp = dbp0 - 0.3 * (pat - pat0) + rs.normal(0, 1.5, 20)  # but WITHIN a patient: slope -0.3
    pat_list.append(pat)
    dbp_list.append(dbp)
    patient_list.append(np.full(20, patient))

pat_toy = np.concatenate(pat_list)                  # 60 segments in one array
dbp_toy = np.concatenate(dbp_list)
patient_toy = np.concatenate(patient_list)
print("pat_toy:", pat_toy.shape, "| patients:", np.unique(patient_toy))

pooled = np.polyfit(pat_toy, dbp_toy, 1)[0]         # one line through everyone
print("pooled slope:", pooled.round(2), "mmHg/ms")

for k in range(3):
    this = patient_toy == k
    slope_k = np.polyfit(pat_toy[this], dbp_toy[this], 1)[0]
    print(f"patient {k} slope:", slope_k.round(2), "mmHg/ms")

fig, ax = plt.subplots(figsize=(5, 3))
for k in range(3):
    this = patient_toy == k
    ax.plot(pat_toy[this], dbp_toy[this], "o", ms=3, label=f"patient {k}")
line_x = np.array([pat_toy.min(), pat_toy.max()])
ax.plot(line_x, np.polyval(np.polyfit(pat_toy, dbp_toy, 1), line_x), "k--", label="pooled fit")
ax.set(xlabel="PAT (ms)", ylabel="DBP (mmHg)", title="Simpson's paradox"); ax.legend(frameon=False, fontsize=7); plt.show()
""")
md(r"""
### 🧪 Code it 4.2 — the fixed-effects slope two ways
(a) `fe_slope_demean(x, y, g)`: subtract each patient's mean from x and y, then compute
$\sum x_c y_c / \sum x_c^2$.
(b) `fe_slope_dummies(x, y, g)`: build $X = [\,x,\ \text{one-hot}(g)\,]$ and solve with
`np.linalg.lstsq`; return the coefficient on $x$.
Both should give the same number (FWL). On the toy data it should be close to −0.3.
""")
code(r"""
def fe_slope_demean(x, y, g):
    # TODO
    return None

def fe_slope_dummies(x, y, g):
    # TODO
    return None

print("demeaned:", fe_slope_demean(pat_toy, dbp_toy, patient_toy))
print("dummies: ", fe_slope_dummies(pat_toy, dbp_toy, patient_toy))
""")
solution(r"""
```python
def fe_slope_demean(x, y, g):
    xc = x.astype(float).copy()
    yc = y.astype(float).copy()
    for k in np.unique(g):
        this = g == k
        xc[this] = xc[this] - xc[this].mean()       # remove this patient's average PAT
        yc[this] = yc[this] - yc[this].mean()       # and average BP
    return (xc @ yc) / (xc @ xc)

def fe_slope_dummies(x, y, g):
    patients = np.unique(g)
    D = np.zeros((len(x), len(patients)))           # one column per patient
    for j, k in enumerate(patients):
        D[g == k, j] = 1                            # 1 where the segment is from patient k
    X = np.column_stack([x, D])                     # [PAT, patient 0?, patient 1?, ...]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta[0]                                  # the coefficient on PAT
```
""")

md(r"""
## 4.3 How sure is one patient's slope? Standard errors and the noise ceiling
For simple regression, $\operatorname{SE}(b_1) = \hat\sigma / \sqrt{S_{xx}}$. With few segments or
little PAT variation, a patient's slope is noisy, and a patient whose true slope is negative can
show a positive estimate by chance. If every patient's true slope were $\beta$ and each estimate
were $\mathcal N(\beta, \operatorname{SE}_i^2)$, the share of negative estimates would be
$$\text{noise ceiling} = \frac1N\sum_i \Phi\!\left(\frac{-\beta}{\operatorname{SE}_i}\right).$$
That is the most one could expect to see even if the law held perfectly. `pat_bp_quality.py`
reports it next to every observed share.

### ✍️ By hand 4.3
True slope β = −0.27 mmHg/ms; a patient's SE = 0.18. Probability that the estimated slope is
negative? (Use Φ(1.5) ≈ 0.933.)
""")
code(r"""
my_43 = None
check("4.3 noise ceiling", my_43, stats.norm.cdf(0.27 / 0.18))
""")
solution(r"""
$P(\hat b < 0) = \Phi(-\beta/\operatorname{SE}) = \Phi(0.27/0.18) = \Phi(1.5) \approx$ **0.933**. So about 7% of such patients show the "wrong" sign from noise alone.
""")

md(r"""
## 4.4 The sign test
The audit's headline statistic is a share: of N patients, k have a negative slope. Under the
null "no systematic direction", $k \sim \text{Binomial}(N, 0.5)$.

### ✍️ By hand 4.4
9 of 10 patients have a negative slope. One-sided $P(K \ge 9)$ and two-sided p-value?
Enter `[one_sided, two_sided]`.
""")
code(r"""
my_44 = None
check("4.4 sign test", my_44, [11 / 1024, 22 / 1024], tol=1e-3)
print("scipy binomtest two-sided:", stats.binomtest(9, 10, 0.5).pvalue)
""")
solution(r"""
$P(K\ge9) = \big[\binom{10}{9} + \binom{10}{10}\big]/2^{10} = 11/1024 \approx$ **0.0107**; two-sided ×2 = **0.0215**.
""")

md(r"""
## 4.5 Theil–Sen: a robust slope
`pat_bp_quality.py` also reports the **Theil–Sen** slope: the median of all pairwise slopes
$(y_j - y_i)/(x_j - x_i)$. A few mis-detected beats then cannot drag the estimate.

### ✍️ By hand 4.5
Points (1, 2), (2, 3), (3, 5), (4, 20). List the 6 pairwise slopes and take the median.
""")
code(r"""
my_45 = None
check("4.5 Theil-Sen", my_45, 4.0)

x = np.array([1., 2, 3, 4])
y = np.array([2., 3, 5, 20])                     # the last point is an outlier
print("Theil-Sen slope:", stats.theilslopes(y, x)[0])
print("least-squares slope:", np.polyfit(x, y, 1)[0])
""")
solution(r"""
Slopes: (1,2)–(2,3) 1; (1,2)–(3,5) 1.5; (1,2)–(4,20) 6; (2,3)–(3,5) 2; (2,3)–(4,20) 8.5; (3,5)–(4,20) 15. Sorted: 1, 1.5, 2, 6, 8.5, 15. Median = (2+6)/2 = **4**. OLS gives 5.6. With 4 points both are pulled by the outlier; with many points only the 1/3 of pairs touching the outlier move, and the median ignores them.
""")

# =============================================================================================
md(r"""
---
# Part 5 — The data's own PAT → BP law

Before asking whether a *model* follows the law, ask whether the *data* does. For each patient,
regress BP on measured PAT across their segments (this is the within-patient slope from Part 4)
and count the share of patients with a negative slope.

This uses the larger `vitaldb_mini_deep.npz` (16.7k segments, the training file of the audited
models) and the project's detector. Expect **about a minute** of PAT detection.
""")
code(r"""
deep = mechlib.load_mini(str(DATA / "vitaldb_mini_deep.npz"))
Xd = np.concatenate([deep["Xtr"], deep["Xva"], deep["Xte"]])      # all segments together
Yd = np.concatenate([deep["ytr"], deep["yva"], deep["yte"]])      # their [SBP, DBP]
Gd = np.concatenate([deep["gtr"], deep["gva"], deep["gte"]])      # their patient IDs
del deep
print("Xd:", Xd.shape, "| Yd:", Yd.shape, "| Gd:", Gd.shape)

n_use = 4000 if QUICK else len(Xd)
t0 = time.time()
pat_ppg = mechlib.compute_ptt(Xd[:n_use], fs, ppg_pos=PPG) * 1000    # R peak -> PPG foot, ms (one per segment)
pat_abp = mechlib.compute_ptt(Xd[:n_use], fs, ppg_pos=ABP) * 1000    # R peak -> arterial-line foot, ms
Yd = Yd[:n_use]
Gd = Gd[:n_use]
del Xd
print(n_use, "segments,", len(np.unique(Gd)), "patients, took", round(time.time() - t0), "s")

print("first 5 PPG PATs (ms):", pat_ppg[:5].round())
print("first 5 ABP PATs (ms):", pat_abp[:5].round())
print("PAT measurable: PPG", np.isfinite(pat_ppg).mean().round(2), "| ABP", np.isfinite(pat_abp).mean().round(2))

both = np.isfinite(pat_ppg) & np.isfinite(pat_abp)
print("median R -> PPG foot:", np.median(pat_ppg[both]).round(), "ms")
print("median R -> ABP foot:", np.median(pat_abp[both]).round(), "ms")
print("PPG minus ABP:", np.median(pat_ppg[both] - pat_abp[both]).round(), "ms  <- the monitor's lag, not physiology")
""")
md(r"""
### 🧪 Code it 5.1 — per-patient slopes
Write `patient_slopes(pat, y, g, min_seg=5, min_range=5)`: for each patient with ≥ `min_seg`
finite-PAT segments and a PAT range ≥ `min_range` ms, fit `np.polyfit(pat, y, 1)` and keep the
slope. Return an array of slopes.
""")
code(r"""
def patient_slopes(pat, y, g, min_seg=5, min_range=5):
    # TODO
    return None
""")
solution(r"""
```python
def patient_slopes(pat, y, g, min_seg=5, min_range=5):
    out = []
    for p in np.unique(g):
        m = (g == p) & np.isfinite(pat)
        if m.sum() >= min_seg and np.ptp(pat[m]) >= min_range:
            out.append(np.polyfit(pat[m], y[m], 1)[0])
    return np.array(out)
```
""")
code(r"""
def _patient_slopes_ref(pat, y, g, min_seg=5, min_range=5):
    out = []
    for p in np.unique(g):
        m = (g == p) & np.isfinite(pat)
        if m.sum() >= min_seg and np.ptp(pat[m]) >= min_range:
            out.append(np.polyfit(pat[m], y[m], 1)[0])
    return np.array(out)

# use your patient_slopes if you wrote it, otherwise the reference version
slope_fn = patient_slopes
if patient_slopes(pat_ppg[:50], Yd[:50, 1], Gd[:50]) is None:
    slope_fn = _patient_slopes_ref

law = {}
for source, pat in [("PPG", pat_ppg), ("ABP", pat_abp)]:
    for column, bp_name in [(0, "SBP"), (1, "DBP")]:
        slopes = slope_fn(pat, Yd[:, column], Gd)        # one slope per patient
        n_negative = int((slopes < 0).sum())
        share = n_negative / len(slopes)
        p_value = stats.binomtest(n_negative, len(slopes)).pvalue
        law[(source, bp_name)] = slopes
        print(f"{source} PAT vs {bp_name}: {len(slopes)} patients, {share:.0%} negative, "
              f"p = {p_value:.1e}, median slope {np.median(slopes):+.3f} mmHg/ms")

q = load_json("pat_bp_quality_vitaldb.json")
print("\nquality-filtered (pat_bp_quality.py), share negative with PPG PAT:")
for level in ["L0 all", "L1 clean beats", "L2 ABP agrees", "L3 BP varies"]:
    sbp_row = q[level]["PPG_SBP"]
    dbp_row = q[level]["PPG_DBP"]
    print(f"  {level:15s} SBP {sbp_row['frac_negative']:.0%} (ceiling {sbp_row['noise_ceiling']:.0%})   "
          f"DBP {dbp_row['frac_negative']:.0%} (ceiling {dbp_row['noise_ceiling']:.0%})")
""")
code(r"""
# subtract each patient's own mean PAT and mean DBP (the centering matrix from Part 4)
pat_c = np.full(len(pat_ppg), np.nan)
dbp_c = np.full(len(pat_ppg), np.nan)
for p in np.unique(Gd):
    this = (Gd == p) & np.isfinite(pat_ppg)
    pat_c[this] = pat_ppg[this] - pat_ppg[this].mean()
    dbp_c[this] = Yd[this, 1] - Yd[this, 1].mean()
print("first 5 centred PAT (ms):", pat_c[:5].round(1))
print("first 5 centred DBP:     ", dbp_c[:5].round(1))

keep = np.abs(pat_c) < 60                      # drop extreme changes (NaN rows are dropped too)
fit = np.polyfit(pat_c[keep], dbp_c[keep], 1)
print("pooled within-patient slope:", fit[0].round(3), "mmHg/ms")

fig, ax = plt.subplots(1, 2, figsize=(9, 2.8))
ax[0].hist(np.clip(law[("PPG", "SBP")], -2, 2), bins=60, alpha=0.6, color="C3", label="SBP")
ax[0].hist(np.clip(law[("PPG", "DBP")], -2, 2), bins=60, alpha=0.6, color="C0", label="DBP")
ax[0].axvline(0, color="k", lw=0.8)
ax[0].set(xlabel="Within-patient slope (mmHg/ms)", ylabel="Patients"); ax[0].legend(frameon=False)
ax[1].hexbin(pat_c[keep], dbp_c[keep], gridsize=40, mincnt=1, cmap="Blues")
ax[1].plot([-60, 60], np.polyval(fit, [-60, 60]), "k--")
ax[1].set(xlabel="PAT minus patient mean (ms)", ylabel="DBP minus patient mean (mmHg)")
plt.tight_layout(); plt.show()
""")
md(r"""
**Reading it.** Most patients show the textbook sign (≈79% SBP, ≈74% DBP before filtering;
≈87% SBP after the quality filters, close to the ~95% noise ceiling). DBP stays well below its
ceiling, which is consistent with PEP hiding more of the DBP relation. **This share is the
reference every model is judged against.**

<details><summary>🗣️ <b>Committee question:</b> "Aren't your quality filters cherry-picking the law?"</summary>

The filters were fixed in advance and judge only signal quality and BP variability (beat agreement, beat-to-beat delay SD, R-peak count matching the arterial line, PPG landmark moving with the arterial line, BP range). None of them looks at the PAT–BP relation. Every level is reported, including the unfiltered L0, and the result is already strongly significant at L0.
</details>
""")

# =============================================================================================
md(r"""
---
# Part 6 — PyTorch from zero

## 6.1 Tensors
A tensor is an n-dimensional array that can live on the GPU and remembers how it was computed
(for gradients). The rules for shapes and broadcasting are NumPy's. Matrix multiplication `@`
works on the **last two** dimensions and broadcasts over the rest. That is how a whole batch of
token sequences goes through one weight matrix.
""")
code(r"""
a = torch.arange(6.).reshape(2, 3)          # 0..5 arranged as 2 rows x 3 columns
print("a =\n", a)
print("shape", a.shape, "| dtype", a.dtype, "| device", a.device)

b = torch.ones(3, 2)
print("b =\n", b)
print("a @ b =\n", a @ b)                   # (2, 3) @ (3, 2) -> (2, 2)

batch = torch.randn(32, 50, 64)             # (segments, tokens, features): what the transformer sees
W = torch.randn(64, 16)
out = batch @ W                             # the same W is applied to every token of every segment
print("(32, 50, 64) @ (64, 16) ->", tuple(out.shape))

a_gpu = a.to(DEV)
print("a is now on:", a_gpu.device)
a_back = a_gpu.cpu().numpy()
print("back to numpy:\n", a_back)
""")
md(r"""
### ✍️ By hand 6.1 — shapes
What is the shape of `Q @ K.transpose(-2, -1)` if `Q` and `K` are both `(8, 4, 50, 16)` (batch,
heads, tokens, head-dim)? Enter the 4 numbers as a list.
""")
code(r"""
my_61 = None
check("6.1 shapes", my_61, [8, 4, 50, 50])
""")
solution(r"""
`K.transpose(-2,-1)` is `(8, 4, 16, 50)`. `(50×16)@(16×50) = 50×50` for each of the 8×4 batch-head pairs: **(8, 4, 50, 50)**, one attention score for every pair of tokens.
""")

md(r"""
## 6.2 Autograd: derivatives of matrix expressions
For the mean-squared error $\mathcal L(w) = \tfrac1n\|Xw - y\|^2$, the gradient is
$$\nabla_w \mathcal L = \tfrac{2}{n}X^\top(Xw - y).$$

### ✍️ By hand 6.2
$\mathcal L(w) = (wx - y)^2$ with $x = 2$, $y = 3$, at $w = 1$. What is $d\mathcal L/dw$?
""")
code(r"""
my_62 = None
check("6.2 gradient", my_62, -4.0)

# the matrix version: 20 data points, 3 weights
Xg = torch.randn(20, 3)
yg = torch.randn(20)
w = torch.randn(3, requires_grad=True)

loss = ((Xg @ w - yg) ** 2).mean()          # mean squared error
loss.backward()                             # fills w.grad
print("autograd gradient:", w.grad)

by_formula = 2 / 20 * Xg.T @ (Xg @ w.detach() - yg)     # (2/n) X^T (Xw - y)
print("formula gradient: ", by_formula)
print("same:", torch.allclose(w.grad, by_formula, atol=1e-6))
""")
solution(r"""
$d\mathcal L/dw = 2(wx - y)\cdot x = 2(2 - 3)(2) =$ **−4**. Negative, so increasing $w$ lowers the loss; gradient descent moves $w$ up.
""")

md(r"""
## 6.3 `nn.Linear` is $y = xW^\top + b$
The weight is stored as `(out_features, in_features)`.

### ✍️ By hand 6.3
$W = \begin{pmatrix}1&2\\3&4\end{pmatrix}$, $b = (1, 0)$, $x = (1, 1)$. Output?
""")
code(r"""
my_63 = None
check("6.3 linear layer", my_63, [4.0, 7.0])
lin = nn.Linear(2, 2)                          # 2 inputs -> 2 outputs
with torch.no_grad():                          # set the weights by hand
    lin.weight[:] = torch.tensor([[1., 2], [3, 4]])
    lin.bias[:] = torch.tensor([1., 0])
print("weight:\n", lin.weight.data)
print("bias:", lin.bias.data)
x = torch.tensor([1., 1])
print("nn.Linear(x) =", lin(x).detach().numpy())
""")
solution(r"""
$Wx = (1+2,\ 3+4) = (3, 7)$; $+b = $ **(4, 7)**.
""")

md(r"""
## 6.4 Gradient descent, and why we standardise inputs
Fit one patient's DBP = $b_0 + b_1\cdot$PAT by gradient descent and compare it with the
closed-form least-squares answer from Part 4.

For a quadratic loss, gradient descent is stable only if the learning rate satisfies
$\eta < 2/\lambda_{\max}$, where $\lambda_{\max}$ is the largest eigenvalue of the Hessian
$\tfrac{2}{n}X^\top X$. Convergence along the slowest direction goes like
$(1 - \eta\lambda_{\min})^t$. So the **condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$
sets how many steps you need. PAT in raw ms (~650) next to a column of ones gives an enormous
κ. Standardising the input makes κ ≈ 1.

### ✍️ By hand 6.4
The Hessian is $\begin{pmatrix}2&0\\0&8\end{pmatrix}$. What is the largest stable learning rate?
""")
code(r"""
my_64 = None
check("6.4 max learning rate", my_64, 0.25)
""")
solution(r"""
$\lambda_{\max} = 8$, so $\eta < 2/8 =$ **0.25**. Above that, the step along the steep direction overshoots and the loss diverges.
""")
code(r"""
# pick the patient with the most measurable PATs
counts = {}
for p in np.unique(Gd):
    counts[p] = np.isfinite(pat_ppg[Gd == p]).sum()
pid = max(counts, key=counts.get)
this = (Gd == pid) & np.isfinite(pat_ppg)
xp = pat_ppg[this]                     # this patient's PATs (ms)
yp = Yd[this, 1]                       # and DBPs
print("patient", pid, ":", len(xp), "segments")

closed = np.polyfit(xp, yp, 1)         # the exact least-squares answer
print("closed-form slope:", closed[0].round(3), "mmHg/ms")

def fit_gd(x, y, lr, steps=3000):
    X = torch.stack([torch.ones(len(x)), torch.tensor(x, dtype=torch.float32)], 1)   # columns: 1, x
    y = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(2, requires_grad=True)               # [intercept, slope], start at 0
    opt = torch.optim.SGD([w], lr=lr)
    for step in range(steps):
        loss = ((X @ w - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    condition = np.linalg.cond((X.T @ X).numpy())
    return w.detach().numpy(), loss.item(), condition

w_raw, loss_raw, cond_raw = fit_gd(xp, yp, lr=1e-6)      # PAT in raw ms
print("raw ms:       cond(X^T X) =", f"{cond_raw:.1e}", "| slope", w_raw[1].round(3), "| loss", round(loss_raw, 1), "<- far from converged")

x_std = (xp - xp.mean()) / xp.std()                      # PAT standardised: mean 0, SD 1
w_std, loss_std, cond_std = fit_gd(x_std, yp, lr=0.1)
slope_back = w_std[1] / xp.std()                         # convert back to mmHg per ms
print("standardised: cond(X^T X) =", f"{cond_std:.1e}", "| slope", slope_back.round(3), "| loss", round(loss_std, 1))
""")

# =============================================================================================
md(r"""
---
# Part 7 — The transformer, piece by piece

`mechlib.WaveTransformer` (the audited model):
```
x (B, 1250, 2)                                  ECG + PPG, z-scored
 └ embed: Conv1d(2, 64, kernel=25, stride=25)   -> (B, 50, 64)   50 tokens of 200 ms
 └ + pos (1, 50, 64)                            learned positional embedding
 └ 3 x TransformerEncoderLayer(64, 4 heads, FFN 128)
 └ mean over tokens                             -> (B, 64)
 └ head: Linear(64, 1)                          -> (B,)  DBP or SBP in mmHg
```

## 7.1 Patch embedding is one matrix multiply
A `Conv1d` with kernel = stride = 25 cuts the signal into non-overlapping 25-sample patches and
applies the same linear map $W \in \mathbb R^{64\times 50}$ to each flattened patch
(25 samples × 2 channels = 50 numbers).

### ✍️ By hand 7.1
(a) How many tokens does a 1250-sample segment give? (b) How long is each token in ms?
(c) How many parameters does the embedding have (weights + biases)? Enter `[tokens, ms, params]`.
""")
code(r"""
my_71 = None
check("7.1 patch embedding", my_71, [50, 200, 3264])
""")
solution(r"""
(a) 1250/25 = **50** tokens. (b) 25 × 8 ms = **200 ms**. (c) 64 × (2 × 25) + 64 = **3264**.

Note that a 16 ms audit shift is *smaller than a token*. The model can still detect it, because the embedding is linear in the patch contents: shifting the PPG 2 samples changes the numbers inside every patch.
""")
code(r"""
torch.manual_seed(0)
x = torch.randn(2, 1250, 2)                     # (segments, samples, channels)
print("x:", tuple(x.shape))

conv = nn.Conv1d(2, 64, kernel_size=25, stride=25)
out_conv = conv(x.transpose(1, 2))              # Conv1d wants (segments, channels, samples)
print("conv output:", tuple(out_conv.shape))    # (2, 64, 50)
out_conv = out_conv.transpose(1, 2)
print("as tokens:", tuple(out_conv.shape))      # (2, 50, 64): 50 tokens of 64 numbers

# the same thing by hand: cut into patches, then ONE matrix multiply
patches = x.reshape(2, 50, 25, 2)               # 50 patches of 25 samples x 2 channels
print("patches:", tuple(patches.shape))
patches = patches.permute(0, 1, 3, 2)           # (2, 50, 2, 25): channel first, like the conv weight
patches = patches.reshape(2, 50, 50)            # flatten each patch to 2 x 25 = 50 numbers
print("flattened patches:", tuple(patches.shape))

W = conv.weight.reshape(64, 50)                 # (64, 2, 25) -> (64, 50)
print("W:", tuple(W.shape))
out_mm = patches @ W.T + conv.bias              # (2, 50, 50) @ (50, 64) -> (2, 50, 64)
print("matmul output:", tuple(out_mm.shape))
print("same as Conv1d:", torch.allclose(out_conv, out_mm, atol=1e-5))
""")

md(r"""
## 7.2 Self-attention by hand
$$\text{Attention}(Q,K,V) = \operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d}}\Big)V,\qquad Q = XW_Q,\ K = XW_K,\ V = XW_V.$$
Each output token is a **weighted average of all value vectors**. The weights (a row of the
softmax) say how much token *i* "looks at" token *j*. The $\sqrt d$ keeps the dot products
from growing with dimension, which would saturate the softmax.

### ✍️ By hand 7.2
Three tokens, $d = 2$: $X = \begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix}$, and
$W_Q = W_K = W_V = I$. For **token 1** compute (a) the scores $x_1 K^\top/\sqrt2$, (b) the softmax
weights, (c) the output. Use $e^{0.707} \approx 2.028$. Enter the output `[o1, o2]`.
""")
code(r"""
my_72 = None
check("7.2 attention output (token 1)", my_72, [0.802, 0.599])
""")
solution(r"""
(a) $XX^\top = \begin{pmatrix}1&0&1\\0&1&1\\1&1&2\end{pmatrix}$, so row 1 is $(1, 0, 1)/\sqrt2 = (0.707, 0, 0.707)$.
(b) exp: (2.028, 1, 2.028), sum 5.056, weights **(0.401, 0.198, 0.401)**.
(c) $0.401(1,0) + 0.198(0,1) + 0.401(1,1) =$ **(0.802, 0.599)**.
""")
code(r"""
X3 = torch.tensor([[1., 0],                     # token 1
                   [0, 1],                      # token 2
                   [1, 1]])                     # token 3
Q = X3                                          # W_Q = W_K = W_V = identity
K = X3
V = X3

scores = Q @ K.T / 2 ** 0.5                     # every token against every token, divided by sqrt(d)
print("scores:\n", scores.numpy().round(3))

weights = scores.softmax(dim=-1)                # each row now sums to 1
print("weights:\n", weights.numpy().round(3))

out3 = weights @ V                              # weighted average of the value vectors
print("output:\n", out3.numpy().round(3))

def attention(Q, K, V):                         # the same three steps, for any shape
    scores = Q @ K.transpose(-2, -1) / Q.shape[-1] ** 0.5
    weights = scores.softmax(-1)
    return weights @ V, weights

library = F.scaled_dot_product_attention(X3[None], X3[None], X3[None])[0]
print("same as PyTorch's attention:", torch.allclose(out3, library, atol=1e-6))

order = torch.tensor([2, 0, 1])                 # shuffle the tokens: 3, 1, 2
X3_shuffled = X3[order]
out_shuffled, _ = attention(X3_shuffled, X3_shuffled, X3_shuffled)
print("output for shuffled tokens:\n", out_shuffled.numpy().round(3))
print("= the original output, shuffled the same way:", torch.allclose(out_shuffled, out3[order]))
""")
md(r"""
The last line is the key fact: **attention without positional embeddings is
permutation-equivariant**. It cannot tell *when* something happened. Everything this model knows
about timing between tokens, including the R peak → PPG foot delay at the token level, comes
from the learned positional embedding `pos` and from what is inside each 200 ms patch.

## 7.3 Multi-head attention = split, attend, concatenate, project
`nn.MultiheadAttention` stores $W_Q, W_K, W_V$ stacked in `in_proj_weight` (192 × 64 for d = 64).
With 4 heads, each head works in its own 16-dim subspace. Below, the library output is
reproduced with plain matrix products.
""")
code(r"""
d_model = 8                                     # small sizes so you can print everything
n_heads = 2                                     # 2 heads of 8 / 2 = 4 features each
mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

x5 = torch.randn(1, 5, d_model)                 # 1 sequence of 5 tokens
library, _ = mha(x5, x5, x5)
print("library output:", tuple(library.shape))

Wq, Wk, Wv = mha.in_proj_weight.chunk(3)        # the stacked (24, 8) weight -> three (8, 8)
bq, bk, bv = mha.in_proj_bias.chunk(3)
Q = x5 @ Wq.T + bq
K = x5 @ Wk.T + bk
V = x5 @ Wv.T + bv
print("Q, K, V:", tuple(Q.shape))               # (1, 5, 8)

Qh = Q.reshape(1, 5, 2, 4).transpose(1, 2)      # split 8 features into 2 heads of 4
Kh = K.reshape(1, 5, 2, 4).transpose(1, 2)
Vh = V.reshape(1, 5, 2, 4).transpose(1, 2)
print("per head:", tuple(Qh.shape))             # (1, 2 heads, 5 tokens, 4)

heads, _ = attention(Qh, Kh, Vh)                # each head attends on its own
print("heads output:", tuple(heads.shape))

joined = heads.transpose(1, 2).reshape(1, 5, 8) # put the 2 heads side by side again
print("concatenated:", tuple(joined.shape))

mine = joined @ mha.out_proj.weight.T + mha.out_proj.bias   # final projection
print("same as nn.MultiheadAttention:", torch.allclose(library, mine, atol=1e-6))
""")

md(r"""
## 7.4 LayerNorm and the encoder layer
LayerNorm z-scores **each token's feature vector** (then applies a learned scale and shift).
PyTorch's default encoder layer is *post-norm*:
$$h = \text{LN}_1\big(x + \text{SelfAttn}(x)\big),\qquad \text{out} = \text{LN}_2\big(h + W_2\,\text{ReLU}(W_1 h)\big).$$
The residual connections ($x + \dots$) let each layer *add* to the representation rather than
replace it.

### ✍️ By hand 7.4
LayerNorm of $(1, 2, 3)$ with unit scale and zero shift (population variance; ignore ε)?
""")
code(r"""
my_74 = None
check("7.4 layernorm", my_74, [-1.2247, 0.0, 1.2247])
print("nn.LayerNorm:", nn.LayerNorm(3)(torch.tensor([1., 2, 3])).detach().numpy().round(4))

layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, batch_first=True).eval()   # eval: dropout off
with torch.no_grad():
    library = layer(x5)

    attn_out = layer.self_attn(x5, x5, x5, need_weights=False)[0]
    h = layer.norm1(x5 + attn_out)              # add attention to the stream, then LayerNorm

    mlp_out = layer.linear2(F.relu(layer.linear1(h)))   # 8 -> 16 -> ReLU -> 8
    mine = layer.norm2(h + mlp_out)             # add the MLP, then LayerNorm
print("attention out:", tuple(attn_out.shape), "| MLP out:", tuple(mlp_out.shape))
print("hand-built layer == nn.TransformerEncoderLayer:", torch.allclose(library, mine, atol=1e-5))
""")
solution(r"""
Mean 2, variance (1+0+1)/3 = 2/3, SD 0.8165. (1−2)/0.8165 = **−1.2247**, 0, **+1.2247**.
""")

md(r"""
## 7.5 Counting parameters
### ✍️ By hand 7.5
Count the parameters of `WaveTransformer(n_ch=2, dm=64, patch=25, heads=4, depth=3, L=1250)`.
Per encoder layer: attention in-projection $3(64\cdot64 + 64)$, out-projection
$64\cdot64+64$, FFN $64\cdot128+128$ and $128\cdot64+64$, two LayerNorms $2\cdot(64+64)$.
Then add the embedding (7.1), the positional table (50 × 64) and the head (64 + 1).
""")
code(r"""
my_75 = None
check("7.5 parameter count", my_75, 106945, tol=1e-6)
""")
solution(r"""
Per layer: 12,480 + 4,160 + 8,320 + 8,256 + 256 = **33,472**; × 3 = 100,416.
Embedding 3,264 + positions 3,200 + head 65. Total = **106,945**. Small next to 12k training segments × 2,500 numbers, but the audited models still barely beat the mean (Part 8).
""")
code(r"""
net_demo = mechlib.WaveTransformer()
total = 0
for name, p in net_demo.named_parameters():
    if name.startswith("tr.layers.") and not name.startswith("tr.layers.0."):
        total += p.numel()
        continue                                # layers 1 and 2 are copies of layer 0: print only once
    print(f"{name:42s} {str(tuple(p.shape)):12s} {p.numel():6d}")
    total += p.numel()
print("total parameters:", total)

x = torch.randn(4, 1250, 2)                     # 4 segments
y_out, acts = net_demo(x, return_acts=True)
print("output:", tuple(y_out.shape))            # one BP per segment
for i, a in enumerate(acts):
    print("after block", i, ":", tuple(a.shape))   # 0 = embedding, 1-3 = encoder layers
""")

# =============================================================================================
md(r"""
---
# Part 8 — Training on VitalDB

The training loop is four lines repeated: **forward → loss → `backward()` → `step()`**. The
project's recipe (`real_audit_controls.train`): Adam lr 2e-3, LR halved every 10 epochs, 30
epochs, MSE loss, keep the epoch with the best validation MAE. Here a short version trains on
the 4,000-segment mini training set. The audited checkpoints were trained on the 12k-segment
file, so their numbers differ slightly.
""")
code(r"""
two = [ECG, PPG]                                  # the model sees channels 0 and 1 only

Xtr = mechlib.normalize(d["Xtr"][:, :, two])      # z-score each segment's ECG and PPG
ytr = d["ytr"][:, 1]                              # DBP
Xva = mechlib.normalize(d["Xva"][:, :, two])
yva = d["yva"][:, 1]
Xte = mechlib.normalize(d["Xte"][:, :, two])
yte = d["yte"][:, 1]
gte = d["gte"]
print("train", Xtr.shape, "| val", Xva.shape, "| test", Xte.shape)

mean_mae = float(np.abs(yte - ytr.mean()).mean())
print("predict-the-mean test MAE:", round(mean_mae, 2))

def train_model(Xtr, ytr, Xva, yva, depth=3, epochs=12, lr=2e-3, bs=128, seed=0, start_at_mean=True):
    torch.manual_seed(seed)
    net = mechlib.WaveTransformer(n_ch=2, depth=depth, L=Xtr.shape[1]).to(DEV)
    if start_at_mean:                                     # see the note below the next cell
        with torch.no_grad():
            net.head.bias.fill_(float(ytr.mean()))
    opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(epochs // 3, 1), gamma=0.5)

    X_all = torch.tensor(Xtr, device=DEV)
    y_all = torch.tensor(ytr, device=DEV)
    hist = {"train": [], "val": []}
    best_val = np.inf
    best_weights = None

    for epoch in range(epochs):
        net.train()
        order = torch.randperm(len(X_all), device=DEV)     # shuffle the segments
        for start in range(0, len(X_all), bs):
            batch = order[start : start + bs]               # the next 128 segments
            pred = net(X_all[batch])                        # 1. forward: (128,) predictions
            loss = ((pred - y_all[batch]) ** 2).mean()      #    mean squared error
            opt.zero_grad()                                 # 2. clear the old gradients
            loss.backward()                                 # 3. d loss / d every weight
            opt.step()                                      # 4. update the weights
        sched.step()

        train_mae = np.abs(mechlib.predict(net, Xtr, DEV) - ytr).mean()
        val_mae = np.abs(mechlib.predict(net, Xva, DEV) - yva).mean()
        hist["train"].append(train_mae)
        hist["val"].append(val_mae)
        print(f"epoch {epoch}: train MAE {train_mae:.2f}, val MAE {val_mae:.2f}")

        if val_mae < best_val:                              # keep the best epoch
            best_val = val_mae
            best_weights = {k: v.clone() for k, v in net.state_dict().items()}

    net.load_state_dict(best_weights)
    net.eval()
    return net, hist

t0 = time.time()
net, hist = train_model(Xtr, ytr, Xva, yva, epochs=6 if QUICK else 15)
print("trained in", round(time.time() - t0), "s")
""")
code(r"""
def load_ckpt(name):
    ck = torch.load(str(DATA / name), map_location=DEV, weights_only=False)
    model = mechlib.WaveTransformer(**ck["config"]).to(DEV)   # rebuild the same architecture
    model.load_state_dict(ck["state_dict"])                   # put the saved weights in
    model.eval()
    return model

orig = load_ckpt("dbp_transformer.pt")            # the original audited DBP model

p_mine = mechlib.predict(net, Xte, DEV)           # one DBP prediction per test segment
p_orig = mechlib.predict(orig, Xte, DEV)
print("first 5 true DBP:     ", yte[:5].round(1))
print("first 5 your model:   ", p_mine[:5].round(1))
print("first 5 original:     ", p_orig[:5].round(1))

print("test MAE: your model", np.abs(p_mine - yte).mean().round(2),
      "| original", np.abs(p_orig - yte).mean().round(2), "| predict the mean", round(mean_mae, 2))
print("SD of predictions: your model", p_mine.std().round(1), "| original", p_orig.std().round(1),
      "| SD of the truth", yte.std().round(1))

fig, ax = plt.subplots(1, 2, figsize=(9, 2.8))
ax[0].plot(hist["train"], label="train"); ax[0].plot(hist["val"], label="validation")
ax[0].axhline(np.abs(yva - ytr.mean()).mean(), color="grey", ls="--", label="predict the mean")
ax[0].set(xlabel="Epoch", ylabel="MAE (mmHg)"); ax[0].legend(frameon=False)
ax[1].plot(yte, p_orig, ".", ms=2, alpha=0.4); ax[1].plot([30, 110], [30, 110], "k--", lw=0.8)
ax[1].set(xlabel="True DBP (mmHg)", ylabel="Predicted DBP (mmHg)", title="Original model: predictions hug the mean")
plt.tight_layout(); plt.show()
""")
md(r"""
**A training trap worth knowing.** A fresh network outputs ≈ 0, but DBP is ≈ 60 mmHg. The
first gradient steps are then spent moving the output up by 60, and on this small dataset the
model can collapse to a constant (prediction SD ≈ 0). Starting the head's bias at the training
mean (`start_at_mean=True`) removes that useless phase. Try `start_at_mean=False` and compare
the prediction SD. (`synth_waveform_audit.train` does the same job by standardising the
targets instead.) Whenever a model's predictions have a tiny SD, check for this before
interpreting anything else.
""")
md(r"""
### ✍️ By hand 8.1 — skill over the baseline
The original DBP model has test MAE 9.20; predicting the mean gives 9.45. What fraction of the
baseline error does the model remove? (skill = 1 − MAE/MAE_mean)
""")
code(r"""
my_81 = None
check("8.1 skill", my_81, 1 - 9.20 / 9.45)
""")
solution(r"""
1 − 9.20/9.45 = **0.026**, i.e. 2.6%. The model has learned little beyond the mean. That frames the main claim honestly: *a model that learned little learned PAT backwards*. MSE training on data where most variance is between patients (Part 2) and hard to read from a z-scored waveform pulls every prediction toward the mean: the prediction SD is far smaller than the truth's.
""")

# =============================================================================================
md(r"""
---
# Part 9 — The intervention audit

**Idea.** Accuracy cannot reveal *how* a model predicts. So intervene on the input: shift the
PPG earlier by δ (PAT shorter by δ, everything else unchanged) and watch the prediction. A model
that uses PAT the textbook way predicts **higher BP for shorter PAT**, so the slope of predicted
BP vs ΔPAT is **negative**. It is computed per patient (Part 4: within patient is where the law
lives), and the headline number is the **share of patients with a negative slope**, compared
with the data's own share from Part 5.

Design choices (each answers a past failure, see `audit_subject.py`):
- **non-circular shift** (`_shift_channel`, edge hold) rather than `np.roll`, which would wrap
  the end of the PPG round to the start and insert a fake pulse;
- **negative arm only** (−48 to 0 ms): the detector tracks these 1:1;
- **measured ΔPAT must match the nominal shift** within 20 ms, else the segment is dropped
  (beat slip);
- **controls**: textbook and inverted models built from the detector's own PAT must give 100%
  and 0%; ECG-later must match PPG-earlier; shifting both channels together must do nothing.

## 9.1 The shift is a matrix too
### ✍️ By hand 9.1
`x = [1, 2, 3, 4, 5]`. Shift by d = −2 (earlier) with edge hold, and by d = +2 (later).
Enter both as `[[...], [...]]`.
""")
code(r"""
my_91 = None
check("9.1 shifts", my_91, [[3, 4, 5, 5, 5], [1, 1, 1, 2, 3]])
""")
solution(r"""
d = −2: **[3, 4, 5, 5, 5]** (everything moves left; the last value is held). d = +2: **[1, 1, 1, 2, 3]**.
""")
md(r"""
### The shift in code, one line at a time
""")
code(r"""
ppg = np.array([1., 2, 3, 4, 5])        # 5 samples of a PPG
shift = 2                               # move 2 samples earlier
n = len(ppg)                            # 5
print("ppg:", ppg)

earlier = np.zeros(n)
earlier[0 : n - shift] = ppg[shift : n]     # slots 0..2 get samples 2..4
print("after line 1:", earlier)             # [3, 4, 5, 0, 0]
earlier[n - shift : n] = ppg[n - 1]         # slots 3..4 get the last sample
print("after line 2:", earlier)             # [3, 4, 5, 5, 5]

print("mechlib._shift_channel:", mechlib._shift_channel(ppg[None], -2)[0])   # the project's version: d = -2
print("np.roll (wraps round!):", np.roll(ppg, -2))                          # [3, 4, 5, 1, 2]: a fake pulse at the end
""")
md(r"""
The same at the real size: 1250 samples (10 s at 125 Hz), shifted 6 samples (6 × 8 ms = 48 ms).
Only the PPG moves; the ECG stays, so the pulse arrives 48 ms sooner after each heartbeat.
""")
code(r"""
ppg = Xte[0, :, PPG]                    # one real test segment's PPG
shift = 6
n = len(ppg)
print("n =", n, "| n - shift =", n - shift)

earlier = np.zeros(n)
earlier[0 : n - shift] = ppg[shift : n]     # slots 0..1243 get samples 6..1249
earlier[n - shift : n] = ppg[n - 1]         # slots 1244..1249 get the last sample
print("last 8 before:", ppg[-8:].round(3))
print("last 8 after: ", earlier[-8:].round(3))
print("matches mechlib:", np.allclose(earlier, mechlib._shift_channel(ppg[None], -6)[0]))

fig, ax = plt.subplots(2, 1, figsize=(9, 3.2), sharex=True)
ax[0].plot(t, Xte[0, :, ECG], color="k", lw=0.8); ax[0].set_ylabel("ECG (not moved)")
ax[1].plot(t, ppg, lw=1, label="original")
ax[1].plot(t, earlier, lw=1, label="48 ms earlier")
ax[1].set(xlim=(0, 3), xlabel="Time (s)", ylabel="PPG"); ax[1].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.show()
""")
md(r"""
### 🧪 Code it 9.2 — the shift matrix
Build the 5×5 matrix `S` with `S @ x == _shift_channel(x, -2)`. Then compute its rank. Why is it
not invertible, and what does that mean for the signal?
""")
code(r"""
S = None   # TODO: np.zeros((5, 5)) and fill in ones
if S is not None:
    ppg = np.array([1., 2, 3, 4, 5])
    print("S =\n", S)
    print("S @ ppg:", S @ ppg)
    print("rank:", np.linalg.matrix_rank(S))
""")
solution(r"""
```python
S = np.zeros((5, 5))
for i in range(5):
    S[i, min(i + 2, 4)] = 1      # output i copies input i+2, clamped to the last sample
```
Rank **3**: rows 3–5 all copy `x[4]`, and `x[0], x[1]` are never read. Information at the edge is lost, which is why the audit uses small shifts (≤ 48 ms of a 10 s window) and the `edge` hold.
""")

md(r"""
## 9.3 Shift every test segment and re-measure PAT
`real_audit_controls.shifted_inputs` makes all the shifted copies: PPG earlier (−48…0 ms),
ECG later (same PAT change, other channel), both together (PAT unchanged), and small PPG shifts
(−16…0 ms, within natural PAT variation). It re-runs the detector on every copy. Takes ~30–60 s.
""")
code(r"""
import real_audit_controls as rac

t0 = time.time()
base_pat, rows = rac.shifted_inputs(Xte, fs)      # every shifted copy of the test set, with PAT re-measured
print("done in", round(time.time() - t0), "s")

print("base_pat:", base_pat.shape, "= PAT of each unshifted test segment (s)")
print("PAT measurable at baseline:", np.isfinite(base_pat).mean().round(2))
print("kinds of shift:", list(rows.keys()))

first = rows["ppg"][0]                            # PPG 48 ms earlier
print("one shifted copy has:", list(first.keys()))
print("  X:", first["X"].shape, "| PAT change asked for:", first["x"], "ms")

for r in rows["ppg"]:
    change = (r["pat"] - base_pat) * 1000         # measured PAT change, ms
    print(f"asked {r['x']:+6.1f} ms -> measured median {np.nanmedian(change):+6.1f} ms, valid {r['ok'].mean():.0%}")
""")
md(r"""
### ✍️ By hand 9.3 — one patient's audit slope
A patient's mean prediction at nominal ΔPAT = [−48, −32, −16, 0] ms is
[61.0, 60.6, 60.2, 59.8] mmHg. Slope (mmHg/ms)? Textbook sign or backwards?
""")
code(r"""
my_93 = None
check("9.3 audit slope", my_93, np.polyfit([-48, -32, -16, 0], [61.0, 60.6, 60.2, 59.8], 1)[0])
""")
solution(r"""
Every +16 ms of PAT lowers the prediction by 0.4 mmHg: slope = −0.4/16 = **−0.025 mmHg/ms**. Negative, so **textbook**: shorter PAT gives higher BP. (It is ~10× smaller than the data's slope, so the sign carries the claim, not the size.)
""")

md(r"""
## 9.4 Controls first, then the models
""")
code(r"""
med_pat = float(np.nanmedian(base_pat))           # a typical PAT (s)
b0 = float(yte[np.isfinite(base_pat)].mean())     # a typical DBP

def control_predictions(a, mode):
    # a fake "model": BP = a / PAT, using the PAT the detector measured on each shifted copy
    preds = []
    for r in rows[mode]:
        pat = np.where(np.isfinite(r["pat"]), r["pat"], med_pat)
        preds.append(a / pat + b0 - a / med_pat)
    return preds                                  # one prediction array per shift size

def model_predictions(model, mode):
    preds = []
    for r in rows[mode]:
        preds.append(mechlib.predict(model, r["X"], DEV))
    return preds

audits = {}
def run_audit(name, get_predictions):
    result = {}
    for mode in rac.SHIFTS:                       # ppg, ecg, both, ppg_small
        preds = get_predictions(mode)
        result[mode] = rac.subject_slopes(preds, rows[mode], gte)   # one slope per patient
    audits[name] = result
    print(f"{name:24s} PPG earlier {result['ppg']['frac_negative']:4.0%} | ECG later {result['ecg']['frac_negative']:4.0%} | "
          f"small PPG {result['ppg_small']['frac_negative']:4.0%} | both: median {result['both']['median']:+.4f}")

def textbook(mode):
    return control_predictions(3.0, mode)         # longer PAT -> lower BP
def inverted(mode):
    return control_predictions(-3.0, mode)        # the opposite

run_audit("textbook control", textbook)
run_audit("inverted control", inverted)

for name in ["dbp_transformer", "dbp_transformer_seed0", "dbp_transformer_seed1", "dbp_transformer_seed2",
             "dbp_transformer_seed3", "dbp_transformer_seed4"]:
    model = load_ckpt(name + ".pt")
    def this_model(mode):
        return model_predictions(model, mode)
    run_audit(name, this_model)

def your_model(mode):
    return model_predictions(net, mode)
run_audit("your model (Part 8)", your_model)

data_share = (law[("PPG", "DBP")] < 0).mean()
print("\nshare of patients with the textbook sign in the DATA (DBP, Part 5):", round(data_share, 2))
""")
code(r"""
slopes = np.array(audits["dbp_transformer"]["ppg"]["slopes"])    # one per patient
print(len(slopes), "patients; first 5 slopes:", slopes[:5].round(4))
print("share negative (textbook):", np.mean(slopes < 0).round(2))

fig, ax = plt.subplots(figsize=(6, 2.8))
ax.hist(np.clip(slopes, -0.1, 0.1), bins=40, color="C0")
ax.axvline(0, color="k", lw=0.8)
ax.set(xlabel="Audit slope per patient (mmHg/ms)", ylabel="Patients",
       title=f"Original DBP model: {np.mean(slopes < 0):.0%} of patients negative (textbook)")
plt.show()

saved = load_json("real_audit_controls.json")
print("saved run (real_audit_controls.json), share negative:")
for name, result in saved["models"].items():
    print(f"  {name:12s} {result['ppg']['frac_negative']:.2f}")
""")
md(r"""
### Is the repeated last value a problem?
The shift fills the end of the PPG by repeating its last sample (at most 6 of 1250). To check the
model is not reacting to that flat end, make the end flat **without** moving the pulse, and
compare with the real shift.
""")
code(r"""
p0 = mechlib.predict(orig, Xte, DEV)                    # predictions with no change
for shift in [2, 4, 6]:
    X_shift = Xte.copy()
    X_shift[:, :, PPG] = mechlib._shift_channel(Xte[:, :, PPG], -shift)    # the real shift
    X_flat = Xte.copy()
    X_flat[:, -shift:, PPG] = Xte[:, -shift - 1 : -shift, PPG]           # only flatten the end
    move_shift = np.abs(mechlib.predict(orig, X_shift, DEV) - p0).mean()
    move_flat = np.abs(mechlib.predict(orig, X_flat, DEV) - p0).mean()
    print(f"{shift} samples: real shift moves the prediction {move_shift:.3f} mmHg, flat end only {move_flat:.3f} mmHg")
""")
md(r"""
**Reading it.**
- The controls land at 100% and 0%, so the instrument can see both directions.
- PPG-earlier and ECG-later agree, and shifting both does ~nothing: the models respond to the
  **ECG–PPG gap**, i.e. to relative timing, not to where beats sit in the window.
- The DBP transformers give the textbook sign in only **11–43%** of patients, against **~74%**
  in the data: they read arrival time *backwards* while being barely better than the mean.
- The SBP transformers lean the right way, weakly (54–70%; data 79%). Run
  `python real_audit_controls.py --target sbp` to reproduce.

<details><summary>🗣️ <b>Committee question:</b> "Isn't a shifted PPG out of distribution, so the response is meaningless?"</summary>

Two answers. (1) The `ppg_small` mode uses shifts of 8–16 ms, inside the natural within-patient PAT range, and gives the same sign. (2) The shift changes only relative timing: amplitude, shape and heart rate are untouched, the ECG-later version gives the same response, and the both-shifted control is null. So the response is specific to the ECG–PPG gap.
</details>

<details><summary>🗣️ <b>Committee question:</b> "Why would a model learn it backwards?"</summary>

This is open (TODO §1: framing DBP vs SBP). What is established: the response is to the ECG–PPG gap, it holds across 6 runs and at small shifts, and accuracy cannot see it. Candidate explanations to discuss: PAT includes PEP, which can move against DBP (e.g. with drugs that change contractility); the model may use the gap as a proxy for something correlated with it *between* patients, where the sign can differ from within patients (Simpson, Part 4); and a near-mean model's small learned dependencies are not constrained by accuracy. The models are trained on pooled data and never told that the law is a within-patient statement.
</details>
""")

# =============================================================================================
md(r"""
---
# Part 10 — Validating the audit where the answer is known

## 10.0 The question this part answers

Part 9 found that the DBP models read PAT **backwards**. The first thing anyone should ask is:

> *"How do you know that is the model, and not your audit getting it wrong?"*

The audit has several places where it could fail on its own: the detector could pair the
wrong peaks (it did, once: Part 3.5), shifting the PPG could confuse the model, or a network
could use PAT in a way the shift test cannot see. A backwards result is only believable if the
audit is known to give the **right** answer when a model *does* use PAT.

**The idea is the oldest one in measurement: test the instrument on a sample where you already
know the answer.** You check a scale with a known 1 kg weight before you weigh anything else.
Here the "known weight" is a dataset where we decide how much BP depends on PAT.

## 10.1 The recipe: same model, same audit, known truth

| | Real study (Part 9) | Simulation (this part) |
|---|---|---|
| Signals | VitalDB ECG + PPG | synthetic ECG + PPG built to look like VitalDB |
| How BP depends on PAT | **unknown**, that is the question | **set by us** with a dial α |
| Model | WaveTransformer | the same WaveTransformer |
| Audit | shift PPG, re-detect PAT, per-patient slope | the same audit |
| What we check | which sign does the model use? | does the audit read back the dial? |

The only thing that changes is that in the simulation we know the right answer. If the audit
returns it, the audit works, and then Part 9's backwards result has to be about the models.

The dial: every simulated segment gets a PAT and a heart rate, and

$$BP = \alpha\,\underbrace{\big(130 - 0.22\,(\text{PAT} - 190)\big)}_{\text{the textbook law}} + (1-\alpha)\,\underbrace{\big(80 + 0.45\,(\text{HR} - 77)\big)}_{\text{a competing cue}} + \text{patient offset} + \text{noise}.$$

- α = 1: BP comes entirely from PAT, at −0.22 mmHg per ms. A model must use PAT to predict it.
- α = 0: BP comes entirely from heart rate. PAT is useless, so a model should ignore it.
- In between: a mix. A model trained at α should use PAT at a slope of **α × (−0.22)**.

So the test is one plot: **injected slope (x) against audit slope (y)**. A working audit puts
the points on the diagonal; a broken one gives a flat line (it reports ~0 whatever we inject).

### ✍️ By hand 10.1
What slope should the audit recover at α = 0.5? At α = 0?
""")
code(r"""
my_101 = None
check("10.1 injected slopes", my_101, [-0.11, 0.0])
""")
solution(r"""
α × (−0.22): **−0.11** mmHg/ms at α = 0.5, **0** at α = 0 (BP then depends only on HR, so a faithful audit sees no PAT dependence).
""")

md(r"""
## 10.2 The whole idea in ten lines (no waveforms yet)

Before the realistic version, here is the logic with nothing in the way. Each "segment" is just
two numbers, a PAT and a heart rate. The "model" is least squares on those two numbers (Part 4).
The "audit" nudges PAT by −20 ms and measures how much the prediction moves per ms. Run it for
several α and compare with what we injected.
""")
code(r"""
rng_t = np.random.default_rng(1)
n_t = 2000                                         # 2000 segments, each just two numbers
pat_t = rng_t.uniform(120, 260, n_t)               # PAT, ms
hr_t = rng_t.uniform(55, 100, n_t)                 # heart rate, beats per minute
print("first 3 PATs:", pat_t[:3].round(), "| first 3 HRs:", hr_t[:3].round())

def toy_bp(alpha):                                 # the dial, same formula as 10.1
    from_pat = 130 - 0.22 * (pat_t - 190)
    from_hr = 80 + 0.45 * (hr_t - 77)
    noise = rng_t.normal(0, 1.5, n_t)
    return alpha * from_pat + (1 - alpha) * from_hr + noise

def toy_fit(bp):                                   # the "model": least squares on [1, PAT, HR]
    A = np.column_stack([np.ones(n_t), pat_t, hr_t])
    coef = np.linalg.lstsq(A, bp, rcond=None)[0]
    return coef                                    # [intercept, PAT weight, HR weight]

def toy_predict(coef, pat):
    return coef[0] + coef[1] * pat + coef[2] * hr_t

def toy_audit(coef, nudge=-20.0):                  # make PAT 20 ms shorter, see how BP moves
    before = toy_predict(coef, pat_t)
    after = toy_predict(coef, pat_t + nudge)
    return np.mean(after - before) / nudge         # change in BP per ms of PAT

alphas_t = [0, 0.25, 0.5, 0.75, 1.0]
toy_slopes = []
for alpha in alphas_t:
    bp = toy_bp(alpha)
    coef = toy_fit(bp)
    slope = toy_audit(coef)
    toy_slopes.append(slope)
    print(f"alpha {alpha:4.2f}: model weights {coef.round(3)} | injected {alpha * -0.22 + 0.0:+.3f}, audit {slope:+.3f} mmHg/ms")

fig, ax = plt.subplots(figsize=(3.4, 3.0))
ax.plot([a * -0.22 for a in alphas_t], toy_slopes, "o", label="toy audit")
ax.plot([-0.24, 0.02], [-0.24, 0.02], "k--", lw=0.8, label="perfect recovery")
ax.set(xlabel="Injected slope (mmHg/ms)", ylabel="Audit slope (mmHg/ms)"); ax.legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.show()
""")
md(r"""
The audit reads the dial back exactly: every point is on the diagonal. That is no surprise here,
because the model is a straight line and is handed PAT directly, so its PAT coefficient *is* the
injected slope.

**The real test is harder, and that is the point.** The real model never sees a PAT number. It
sees 1,250 samples of raw ECG and PPG and has to find the timing between beats itself. So the
rest of this part does the same experiment with waveforms, step by step, in plain code:

1. draw ECG and PPG signals in which the pulse arrives PAT ms after each heartbeat (10.3);
2. set BP from PAT with the dial (10.4);
3. check the PAT detector can see the PAT we put in (10.5);
4. train the VitalDB transformer on the signals (10.6);
5. run the audit and compare with the dial (10.7).

This is a simplified version of `synth_waveform_audit.py`, which produced the thesis numbers
(it fits beat shapes to VitalDB and adds monitor filtering). The logic is identical.

## 10.3 Draw the signals

Every wave is a **Gaussian bump** at a chosen time. One heartbeat is:

- **ECG**: a tall, narrow R wave at the beat time, and a smaller, wider T wave 0.25 s later
  (the wave the old detector mistook for an R peak);
- **PPG**: the pulse reaches the finger **PAT** ms after the R wave, *plus* a fixed 0.4 s delay
  that VitalDB's PPG channel has (the delay that broke the old detector, Part 3.5). Its shape
  is a systolic bump and a smaller diastolic bump.
""")
code(r"""
FS, L = 125, 1250                          # 125 Hz, 10 s segments, as in VitalDB
time_s = np.arange(L) / FS                 # the time axis, in seconds
OFFSET = 0.4                               # VitalDB's PPG channel lags the ECG by ~0.4 s

def bump(center, width, height):
    # A Gaussian bump: the building block of every wave.
    return height * np.exp(-0.5 * ((time_s - center) / width) ** 2)

def make_segment(hr, pat_ms, rng):
    # 10 s of ECG + PPG at heart rate hr (beats/min) with pulse arrival time pat_ms.
    beats = np.arange(rng.uniform(0, 1), 11, 60 / hr) - 1                    # R-peak times (s)
    ecg, ppg = np.zeros(L), np.zeros(L)
    for b in beats:
        ecg += bump(b, 0.012, 1.0) + bump(b + 0.25, 0.04, 0.35)              # R wave, T wave
        foot = b + pat_ms / 1000 + OFFSET                                     # pulse arrives PAT later
        ppg += bump(foot + 0.13, 0.06, 1.0) + bump(foot + 0.40, 0.08, 0.35)  # systolic, diastolic
    ecg += rng.normal(0, 0.01, L)                                             # a little sensor noise
    ppg += rng.normal(0, 0.01, L)
    return np.stack([ecg, ppg], 1)                                            # (1250, 2), like VitalDB
""")
md(r"""
Two segments with the same heartbeat and noise (same random seed) and only PAT changed. The
PPG moves; the ECG does not. **That gap is the only thing the dial will act on.**
""")
code(r"""
seg_short = make_segment(70, 130, np.random.default_rng(5))
seg_long = make_segment(70, 250, np.random.default_rng(5))        # same seed: only PAT differs
fig, ax = plt.subplots(2, 1, figsize=(9, 3.6), sharex=True)
ax[0].plot(time_s, seg_short[:, 0], color="k", lw=0.8); ax[0].set_ylabel("ECG")
ax[1].plot(time_s, seg_short[:, 1], color="C0", lw=1, label="PAT 130 ms")
ax[1].plot(time_s, seg_long[:, 1], color="C3", lw=1, label="PAT 250 ms")
ax[1].set(xlim=(0, 4), xlabel="Time (s)", ylabel="PPG"); ax[1].legend(frameon=False, fontsize=7, loc="upper right")
plt.tight_layout(); plt.show()
""")
md(r"""
How close is this to the real thing? Left: a simulated segment. Right: a VitalDB test segment
(Part 8). Both are z-scored per channel, as the model sees them.
""")
code(r"""
sim_seg = mechlib.normalize(seg_short[None])[0]
fig, ax = plt.subplots(2, 2, figsize=(10, 3.6), sharex=True)
for col, (x_, name) in enumerate([(sim_seg, "Simulated"), (Xte[0], "VitalDB")]):
    ax[0, col].plot(time_s, x_[:, 0], color="k", lw=0.8); ax[0, col].set_title(name, loc="left")
    ax[1, col].plot(time_s, x_[:, 1], color="C0", lw=0.8); ax[1, col].set_xlabel("Time (s)")
ax[0, 0].set_ylabel("ECG"); ax[1, 0].set_ylabel("PPG"); ax[0, 0].set_xlim(0, 5)
plt.tight_layout(); plt.show()
""")
md(r"""
Simpler (no P wave, no filtering), but the parts that matter for timing are there: sharp R
peaks, T waves, and a PPG pulse that arrives long after its heartbeat.

## 10.4 Turn the dial

Now make a dataset: 40 simulated patients, each segment with a random PAT and heart rate, and
BP from the formula in 10.1. Each patient also gets their own BP offset (people differ), which
no model can read from a waveform. We make three datasets: α = 1, 0.5 and 0.
""")
code(r"""
def make_dataset(alpha, n, n_patients=40, seed=0):
    rng = np.random.default_rng(seed)
    pid = rng.integers(0, n_patients, n)                          # which patient each segment is from
    patient_offset = rng.normal(0, 8, n_patients)                 # each patient's own BP level
    pat = rng.uniform(120, 260, n)                                # ms
    hr = rng.uniform(55, 100, n)                                  # beats/min
    bp = (alpha * (130 - 0.22 * (pat - 190)) + (1 - alpha) * (80 + 0.45 * (hr - 77))
          + patient_offset[pid] + rng.normal(0, 1.5, n))         # the dial
    X = np.zeros((n, L, 2))
    for i in range(n):
        X[i] = make_segment(hr[i], pat[i], rng)                   # draw segment i
    return mechlib.normalize(X), bp.astype(np.float32), pat, pid

n_sim = 1200 if QUICK else 3000
sims = {}
for alpha in [1.0, 0.5, 0.0]:
    sims[alpha] = make_dataset(alpha, n_sim, seed=int(10 * alpha))
    print("alpha", alpha, ": X", sims[alpha][0].shape, "| BP", sims[alpha][1].shape)
print({alpha: d_[0].shape for alpha, d_ in sims.items()})
""")
md(r"""
What the dial does to the data: BP against PAT for each dataset. At α = 1 there is a clear
downward trend (−0.22 mmHg per ms); at α = 0 there is none. The vertical spread is mostly the
patient offsets.
""")
code(r"""
fig, ax = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
for j, alpha in enumerate(sims):
    X_, bp_, pat_, pid_ = sims[alpha]
    ax[j].scatter(pat_, bp_, s=3, alpha=0.3)
    pat_line = np.array([120, 260])
    ax[j].plot(pat_line, alpha * (130 - 0.22 * (pat_line - 190)) + (1 - alpha) * 80, "k--", lw=1)   # the injected law
    ax[j].set(title=f"α = {alpha}: injected {alpha * -0.22 + 0.0:+.2f} mmHg/ms", xlabel="PAT (ms)")
ax[0].set_ylabel("BP (mmHg)")
plt.tight_layout(); plt.show()
""")
md(r"""
## 10.5 Can the detector see the PAT we put in?

The audit relies on `mechlib.compute_ptt` (Part 3) to confirm each shift really changed PAT.
Run it on the α = 1 signals and compare with the PAT we drew. Subtract the 0.4 s channel offset
first: it is the same for every segment, so it cancels in every *change* in PAT.
""")
code(r"""
X1, bp1, pat1, pid1 = sims[1.0]
measured = mechlib.compute_ptt(X1[:600], FS) * 1000 - 1000 * OFFSET
err = measured - pat1[:600]
slip = np.abs(err) > 30                                           # paired with the wrong heartbeat
fig, ax = plt.subplots(figsize=(3.8, 3.4))
ax.scatter(pat1[:600][~slip], measured[~slip], s=4, alpha=0.4, label="segments")
ax.scatter(pat1[:600][slip], np.full(slip.sum(), 108), s=20, marker="v", color="C3",
           label="wrong beat (off the scale)")                    # ~one heartbeat off, drawn at the edge
ax.plot([120, 260], [120, 260], "k--", lw=0.8, label="perfect")
ax.set(xlabel="True PAT (ms)", ylabel="Measured PAT − 400 ms (ms)", ylim=(100, 290))
ax.legend(frameon=False, fontsize=7, loc="upper left")
plt.tight_layout(); plt.show()
print(f"median error {np.nanmedian(err):+.0f} ms, {np.mean(np.abs(err) < 20):.1%} within 20 ms, "
      f"{slip.mean():.1%} paired with the wrong beat")
""")
md(r"""
Almost every point is on the line. The small constant error (~+10 ms) is where the detector
places the "foot" of a smooth bump, and it cancels in any change. A rare segment is paired with
the **wrong heartbeat** (off by one whole beat): that is why the real audit drops any segment
whose measured PAT change does not match the shift.

## 10.6 Train the model

The same `WaveTransformer` as on VitalDB, trained with a plain loop: mini-batches, mean squared
error, Adam. One model per dataset. A minute or so on the GPU.
""")
code(r"""
def train_sim(X, y, epochs, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    mu = y.mean()                                        # train on z-scored BP:
    sd = y.std()                                         # (BP - mu) / sd
    model = mechlib.WaveTransformer().to(DEV)            # same architecture as on VitalDB
    opt = torch.optim.Adam(model.parameters(), lr)
    X_all = torch.tensor(X, dtype=torch.float32, device=DEV)
    y_all = torch.tensor((y - mu) / sd, dtype=torch.float32, device=DEV)
    for epoch in range(epochs):
        order = torch.randperm(len(X), device=DEV)      # shuffle
        for start in range(0, len(X), 128):
            batch = order[start : start + 128]           # 128 segments
            loss = ((model(X_all[batch]) - y_all[batch]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model, mu, sd

def predict_bp(trained, X):
    model, mu, sd = trained
    return mechlib.predict(model, X, DEV) * sd + mu      # back from z-scores to mmHg

ep_sim = 8 if QUICK else 25
n_tr = int(0.8 * n_sim)                                  # first 80% train, last 20% audited
trained = {}
for alpha in sims:
    X_, bp_, pat_, pid_ = sims[alpha]
    t0 = time.time()
    trained[alpha] = train_sim(X_[:n_tr], bp_[:n_tr], ep_sim)
    pred = predict_bp(trained[alpha], X_[n_tr:])
    mae = np.abs(pred - bp_[n_tr:]).mean()
    base = np.abs(bp_[n_tr:] - bp_[:n_tr].mean()).mean()
    print(f"α = {alpha}: test MAE {mae:.1f} mmHg (predicting the mean: {base:.1f}), {time.time() - t0:.0f} s")
""")
md(r"""
## 10.7 Run the audit

Exactly the Part 9 procedure, written out: move the PPG **earlier** by 0, 2, 4 and 6 samples
(0 to 48 ms), which makes PAT shorter by that much; predict BP for each version; and for each
patient, fit a line of predicted BP against the change in PAT.

First, what one shift looks like:
""")
code(r"""
def shift_ppg(X, shift):
    # Move the PPG (channel 1) `shift` samples earlier; repeat the last value at the end.
    n = X.shape[1]                                       # 1250
    out = X.copy()                                       # the ECG (channel 0) is copied unchanged
    if shift > 0:
        out[:, 0 : n - shift, 1] = X[:, shift : n, 1]    # slots 0..1243 get samples 6..1249 (shift 6)
        out[:, n - shift : n, 1] = X[:, n - 1 : n, 1]    # slots 1244..1249 get the last sample
    return out

seg = X1[:1]
fig, ax = plt.subplots(2, 1, figsize=(9, 3.4), sharex=True)
ax[0].plot(time_s, seg[0, :, 0], color="k", lw=0.8); ax[0].set_ylabel("ECG")
ax[1].plot(time_s, seg[0, :, 1], color="C0", lw=1, label="original")
ax[1].plot(time_s, shift_ppg(seg, 6)[0, :, 1], color="C1", lw=1, label="PPG 48 ms earlier = PAT 48 ms shorter")
ax[1].set(xlim=(0, 3), xlabel="Time (s)", ylabel="PPG"); ax[1].legend(frameon=False, fontsize=7, loc="upper right")
plt.tight_layout(); plt.show()
""")
md(r"""
Now the audit. Each line below is **one patient**: their average predicted BP as PAT is made
shorter (read right to left). A model that uses the textbook law predicts **higher BP for
shorter PAT**, so its lines rise to the left, and the slope is negative. The dashed line is the
law we injected.
""")
code(r"""
shifts = [6, 4, 2, 0]                                    # samples earlier (8 ms each at 125 Hz)
dpat = np.array([-1000 * k / FS for k in shifts])        # change in PAT: -48, -32, -16, 0 ms
print("PAT changes (ms):", dpat)

def audit_sim(trained, X, pid):
    # 1. predict BP for every shifted copy: one column per shift
    P = np.zeros((len(X), len(shifts)))
    for j, k in enumerate(shifts):
        P[:, j] = predict_bp(trained, shift_ppg(X, k))
    # 2. per patient: average over their segments, then fit a line against the PAT change
    curves = []
    slopes = []
    for p in np.unique(pid):
        curve = P[pid == p].mean(0)                      # this patient's mean BP at each shift
        curves.append(curve)
        slopes.append(np.polyfit(dpat, curve, 1)[0])     # mmHg per ms
    return np.array(slopes), np.array(curves)

audit_res = {}
fig, ax = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
for j, alpha in enumerate(sims):
    X_, bp_, pat_, pid_ = sims[alpha]
    slopes, curves = audit_sim(trained[alpha], X_[n_tr:], pid_[n_tr:])
    audit_res[alpha] = slopes
    print(f"α = {alpha}: {len(slopes)} patients, first 3 slopes {slopes[:3].round(3)}")
    for curve in curves:
        ax[j].plot(dpat, curve - curve[-1], color="C0", lw=0.8, alpha=0.5)   # relative to no shift
    ax[j].plot(dpat, alpha * -0.22 * dpat, "k--", lw=1.2, label="injected law")
    ax[j].set(title=f"α = {alpha}: {np.mean(slopes < 0):.0%} of patients negative", xlabel="Change in PAT (ms)")
ax[0].set_ylabel("Change in predicted BP (mmHg)"); ax[0].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.show()
""")
md(r"""
At α = 1 and 0.5 every patient's line rises to the left and sits on the injected law. At α = 0
the lines are flat. Finally, the one plot that summarises it: **injected slope against audit
slope**, with the saved full run (`python synth_waveform_audit.py`: five dial settings, three
training seeds each) in blue and your three models in red.
""")
code(r"""
for alpha in audit_res:
    slopes = audit_res[alpha]
    injected = alpha * -0.22 + 0.0
    audit = np.median(slopes)
    print(f"α = {alpha}: injected {injected:+.3f}, audit {audit:+.3f} mmHg/ms, "
          f"{np.mean(slopes < 0):.0%} of patients negative")
    if injected != 0:
        print(f"         recovered {audit / injected:.0%} of the injected slope")

sw = load_json("synth_waveform_audit.json")
al = sorted(sw["alphas"], key=float)
inj = [float(a) * -0.22 + 0.0 for a in al]
mu_ = [np.mean(sw["alphas"][a]["slopes"]) for a in al]; sd_ = [np.std(sw["alphas"][a]["slopes"]) for a in al]
fig, ax = plt.subplots(figsize=(4.2, 3.4))
ax.errorbar(inj, mu_, yerr=sd_, fmt="o", capsize=3, label="saved full run (3 seeds)")
ax.plot([-0.24, 0.02], [-0.24, 0.02], "k--", lw=0.8, label="perfect recovery")
ax.plot([a * -0.22 for a in audit_res], [np.median(s_) for s_ in audit_res.values()], "s", color="C3",
        label="this notebook")
ax.set(xlabel="Injected slope (mmHg/ms)", ylabel="Audit slope (mmHg/ms)"); ax.legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.show()
print(f"saved full run: r(alpha, audit slope) = {sw['slope_vs_alpha_r']:.3f}; detector r = {sw['estimator']['r_true']:.3f}")
""")
md(r"""
### ✍️ By hand 10.2
At α = 1 the saved run's audit slope is −0.197 mmHg/ms, and −0.22 was injected. What fraction
of the injected slope did the audit recover?
""")
code(r"""
my_102 = None
check("10.2 recovery", my_102, 0.197 / 0.22)
""")
solution(r"""
0.197 / 0.22 ≈ **0.90**, so about 90%. Across α = 0.25 to 1 it is 89–95%, and at α = 0 the audit gives ≈ 0 (+0.003), as it should.
""")
md(r"""
**Reading it.**
- The points sit close to the diagonal: **more PAT in the data → more PAT in the audit**,
  in proportion. r(α, audit slope) ≈ −1.00.
- At α = 0 the audit reports ≈ 0: it does not invent a PAT effect that is not there.
- It recovers ~90%, not 100%, because a trained network only approximates the law (finite data,
  noise, a patient offset it cannot see). The size is slightly shrunk; the **direction** is
  never wrong. The project's claims rest on the direction (the share of patients with the
  textbook sign), so that is what has to be right.

**So:** when the same audit, on the same architecture, says the VitalDB DBP models go the
wrong way, the instrument is not the explanation. The models do.

## 10.8 Three checks, three different questions

It is easy to blur these together. They answer different questions, and the committee may ask
which is which.

| Check | What it is | The question it answers | Result |
|---|---|---|---|
| ± controls (Part 9.4) | a formula, BP = ±3 / PAT, using the detector's PAT; **no network** | does the counting and sign bookkeeping work? | 100% and 0% |
| **Simulation (this part)** | a **trained network** on waveforms where we set the law | can the audit detect a law a network has actually learned? | reads back the dial, ~90% |
| The data (Part 5) | within-patient BP vs PAT in VitalDB itself | is the law there to be learned? | ~74% (DBP) to ~87% (SBP) of patients |

The controls are a weaker test than the simulation: they skip the network entirely, so they
cannot show that a *learned* use of PAT shows up in the audit. The simulation does.

The simulation validates the **method**; it says nothing about physiology. Whether real models
*should* follow the law comes from the data (third row), not from the simulation.

### 🗣️ Say it in 30 seconds

> "Before trusting the audit on real patients, I tested it where I know the answer. I made
> synthetic ECG and PPG with a dial controlling how strongly blood pressure depends on arrival
> time, trained the same model, and ran the same audit. Turn the dial up, and the audit reports
> a stronger dependence: it recovers about 90% of what I put in, and zero when I put in zero.
> So when the audit says the real models go the wrong way, that's the models, not the method."

<details><summary>🗣️ <b>Committee question:</b> "Is the simulation realistic enough to mean anything?"</summary>

It is built to fail in the same ways as the real data: beat shapes fitted to VitalDB, tall T waves, the 400 ms PPG channel offset that broke the old detector, and monitor-style filtering and noise. The detector recovers the injected PAT at r = 0.997 on it. It does not need to be a perfect copy of VitalDB; it needs to contain the things that could make the audit fail, and it does.
</details>

<details><summary>🗣️ <b>Committee question:</b> "Why 90% and not 100%?"</summary>

The network is trained on finite, noisy data and only approximates the law, so the slope it learns is a little smaller than the true one. What matters is that the audit tracks the dial in proportion (r ≈ −1.00), never flips the sign, and gives zero when nothing was injected. The claims are about sign, not exact size.
</details>

<details><summary>🗣️ <b>Committee question:</b> "You already have positive and negative controls. Why do you need a simulation?"</summary>

The controls are formulas that read the detector's PAT directly; no network is involved, so they only check the counting. The question about the backwards result is whether a network that has learned the law would *show* it under the audit. Only a network trained on data with a known law can answer that, which is what the simulation is.
</details>

<details><summary>🗣️ <b>Committee question:</b> "Does the simulation show real models should follow the law?"</summary>

No, and it is not meant to. It shows the audit would detect the law if a model had learned it. Whether the law is in real data is answered separately, within patients in VitalDB (Part 5): longer PAT goes with lower BP in ~74% (DBP) to ~87% (SBP) of patients.
</details>
""")

# =============================================================================================
md(r"""
---
# Part 11 — Probes: decodable is not the same as used

A **linear probe** asks whether a quantity (PAT, heart rate, BP) can be read linearly from a
layer's activations. It is ridge regression:
$$\hat\beta_\lambda = (X^\top X + \lambda I)^{-1}X^\top y.$$
Adding $\lambda I$ makes the matrix invertible and well conditioned (Part 6). In SVD terms, with
$X = U\Sigma V^\top$:
$$\hat\beta_\lambda = V\,\operatorname{diag}\!\Big(\frac{\sigma_i}{\sigma_i^2 + \lambda}\Big)U^\top y,$$
so each direction is shrunk by $\sigma_i^2/(\sigma_i^2+\lambda)$. Weak directions (small σ) are
shrunk most.

### ✍️ By hand 11.1
No intercept, $x = (1, 2, 3)$, $y = (2, 4, 6)$. OLS slope? Ridge slope with λ = 1?
Enter `[ols, ridge]`.
""")
code(r"""
my_111 = None
check("11.1 ridge", my_111, [2.0, 28 / 15])
X = rng.normal(size=(50, 5))                 # 50 rows, 5 features
y = rng.normal(size=50)
lam = 3.0

direct = np.linalg.solve(X.T @ X + lam * np.eye(5), X.T @ y)     # (X^T X + lam I)^-1 X^T y
print("ridge, normal equations:", direct.round(4))

U, s, Vt = np.linalg.svd(X, full_matrices=False)
shrink = s / (s ** 2 + lam)                  # how much each direction is kept
print("singular values:", s.round(2))
print("shrink factors: ", shrink.round(4))
via_svd = Vt.T @ np.diag(shrink) @ U.T @ y
print("ridge, SVD formula:     ", via_svd.round(4))
print("same:", np.allclose(direct, via_svd))
""")
solution(r"""
$x^\top y = 2 + 8 + 18 = 28$, $x^\top x = 14$. OLS = 28/14 = **2**. Ridge = 28/(14 + 1) = **1.867**: pulled toward 0.
""")
code(r"""
# 1. run the original DBP model on the test set and keep every block's activations
block_feats = [[], [], [], []]                       # embedding, layer 1, layer 2, layer 3
with torch.no_grad():
    for start in range(0, len(Xte), 256):
        x = torch.tensor(Xte[start : start + 256], device=DEV)
        y_out, acts = orig(x, return_acts=True)      # acts: 4 tensors of (256, 50, 64)
        for b in range(4):
            block_feats[b].append(acts[b].mean(1).cpu().numpy())   # average the 50 tokens -> (256, 64)
feats = []
for b in range(4):
    feats.append(np.concatenate(block_feats[b]))
print("features per block:", [f.shape for f in feats])

# 2. heart rate of each segment, from its R peaks
hr = np.full(len(Xte), np.nan)
for i in range(len(Xte)):
    r = mechlib.r_peaks(Xte[i, :, 0], fs)
    if len(r) > 2:
        hr[i] = 60 / np.median(np.diff(r) / fs)
print("heart rate, first 5:", hr[:5].round())

# 3. fit the probe on half the patients, score it on the other half
patients = np.unique(gte)
fit_patients = rng.permutation(patients)[: len(patients) // 2]
fit_m = np.isin(gte, fit_patients)                   # True for segments of fitting patients
print("fit segments:", fit_m.sum(), "| scored segments:", (~fit_m).sum())

def probe_r2(F, target, lam=1.0):
    ok = np.isfinite(target)
    fit_rows = fit_m & ok
    test_rows = ~fit_m & ok
    mu = F[fit_rows].mean(0)
    sd = F[fit_rows].std(0) + 1e-8
    Z_fit = np.column_stack([np.ones(fit_rows.sum()), (F[fit_rows] - mu) / sd])     # [1, standardised features]
    Z_test = np.column_stack([np.ones(test_rows.sum()), (F[test_rows] - mu) / sd])
    beta = np.linalg.solve(Z_fit.T @ Z_fit + lam * np.eye(Z_fit.shape[1]), Z_fit.T @ target[fit_rows])   # ridge
    pred = Z_test @ beta
    return 1 - np.mean((target[test_rows] - pred) ** 2) / np.var(target[test_rows])  # held-out R²

targets = {"PAT (ms)": base_pat * 1000, "heart rate": hr, "DBP": yte}
fig, ax = plt.subplots(figsize=(5, 2.8))
for name in targets:
    r2 = []
    for F in feats:
        r2.append(probe_r2(F, targets[name]))
    print(f"{name:10s} held-out R² by block:", np.round(r2, 2))
    ax.plot(range(4), r2, "o-", label=name)
ax.set(xticks=range(4), xticklabels=["embed", "L1", "L2", "L3"], ylabel="Held-out R²", title="What is decodable where")
ax.axhline(0, color="k", lw=0.6); ax.legend(frameon=False, fontsize=7); plt.show()
""")
md(r"""
**Reading it.** Information about arrival time and heart rate is present in the activations,
yet the audit says the model's *output* responds to PAT in the wrong direction. A probe shows
that information is *present*; only an intervention shows how it is *used*. This is the
methodological core of the thesis.

<details><summary>🗣️ <b>Committee question:</b> "Why not just use SHAP / saliency?"</summary>

Attribution methods say which inputs the output is sensitive to, but not *in which physiological direction* along a named mechanism, and they are not causal in the do-operator sense. The audit is a direct intervention on a named physiological variable (the ECG–PPG gap) with a known textbook sign, validated against ground truth in simulation and against positive and negative controls.
</details>
""")

# =============================================================================================
md(r"""
---
# Part 12 — Calibration: what a few cuff readings buy

Model of a patient's readings: $y_{ij} = f(x_{ij}) + c_i + \varepsilon_{ij}$. Most of a
calibration-free model's error is the patient offset $c_i$ (Part 2). With $k$ cuff readings, the
least-squares estimate of $c_i$ uses a **single column of ones**:
$$\hat c_i = (\mathbf 1^\top\mathbf 1)^{-1}\mathbf 1^\top r = \bar r,\qquad r = y_{\text{cuff}} - f(x),\qquad \operatorname{Var}(\hat c_i) = \sigma^2/k.$$
A two-parameter equation per patient ($BP = a/\text{PAT} + b$) has columns
$[1/\text{PAT},\ 1]$. It needs $k \ge 2$, and with small $k$ the slope estimate is noisy
($\operatorname{Var}(\hat a) = \sigma^2/S_{xx}$, and $S_{xx}$ is tiny when the anchors' PATs are
close). That is why fitting fewer numbers per patient wins at low $k$.

Note: adding a constant does not change any derivative, so **calibration does not change the
audit slope**. A backwards model stays backwards after calibration.

### ✍️ By hand 12.1
Model predictions at three cuff times: [70, 72, 68]. Cuff: [78, 80, 79]. (a) Offset? (b) A new
segment is predicted at 71; calibrated prediction? (c) If residual SD is 5 mmHg, the SE of the
offset with k = 4? Enter `[offset, calibrated, se]`.
""")
code(r"""
my_121 = None
check("12.1 calibration", my_121, [9.0, 80.0, 2.5])
""")
solution(r"""
(a) Residuals [8, 8, 11], mean **9**. (b) 71 + 9 = **80**. (c) 5/√4 = **2.5 mmHg**.
""")
md(r"""
### 🧪 Code it 12.2 — offset calibration with random anchors
`calibrated_mae(pred, y, g, k, seed)`: for each patient with more than k segments, choose k
random anchor segments, offset = mean(y − pred) over the anchors, then score
|pred + offset − y| on the *other* segments. Return the mean over all scored segments. Random
anchors, not the first k: the first k are adjacent in time and also absorb drift, which
flatters the result.
""")
code(r"""
def calibrated_mae(pred, y, g, k, seed=0):
    # TODO
    return None
""")
solution(r"""
```python
def calibrated_mae(pred, y, g, k, seed=0):
    r = np.random.default_rng(seed)
    errs = []
    for p in np.unique(g):
        idx = r.permutation(np.where(g == p)[0])     # this patient's segments, shuffled
        if len(idx) <= k:
            continue
        anchors = idx[:k]                            # k "cuff readings"
        scored = idx[k:]                             # the rest
        offset = (y[anchors] - pred[anchors]).mean() if k else 0.0
        errs.append(np.abs(pred[scored] + offset - y[scored]))
    return np.concatenate(errs).mean()
```
""")
code(r"""
def _cal_ref(pred, y, g, k, seed=0):
    # the model's prediction plus one offset per patient, from k random anchor segments
    r = np.random.default_rng(seed)
    errs = []
    for p in np.unique(g):
        idx = r.permutation(np.where(g == p)[0])
        if len(idx) <= max(k, 5):                    # same patients scored at every k
            continue
        anchors = idx[:k]
        scored = idx[max(k, 5):]
        offset = (y[anchors] - pred[anchors]).mean() if k else 0.0
        errs.append(np.abs(pred[scored] + offset - y[scored]))
    return np.concatenate(errs).mean()

def _line_ref(pat, y, g, k, lam, seed=0):
    # a per-patient line BP = b + a (PAT - mean anchor PAT) fitted to k anchors;
    # the slope a is shrunk by ridge lam (ms^2); lam = inf means slope 0, i.e. offset only
    r = np.random.default_rng(seed)
    errs = []
    for p in np.unique(g):
        idx = r.permutation(np.where((g == p) & np.isfinite(pat))[0])
        if len(idx) <= 5:
            continue
        anchors = idx[:k]
        scored = idx[5:]
        x_a = pat[anchors] - pat[anchors].mean()
        if np.isinf(lam):
            slope = 0.0
        else:
            slope = (x_a @ (y[anchors] - y[anchors].mean())) / (x_a @ x_a + lam)
        pred = y[anchors].mean() + slope * (pat[scored] - pat[anchors].mean())
        errs.append(np.abs(pred - y[scored]))
    return np.concatenate(errs).mean()

pat_ms = base_pat * 1000
print(f"{'k':>3s} {'transformer+offset':>19s} {'line, lam=0':>12s} {'lam=1000':>9s} {'offset only (flat)':>19s}")
for k in [1, 2, 3, 5]:
    seeds = range(20)                                # average over 20 random choices of anchors
    model_offset = np.mean([_cal_ref(p_orig, yte, gte, k, s) for s in seeds])
    with np.errstate(all="ignore"):
        line_free = np.mean([_line_ref(pat_ms, yte, gte, k, 0, s) for s in seeds])
        line_ridge = np.mean([_line_ref(pat_ms, yte, gte, k, 1000, s) for s in seeds])
        flat = np.mean([_line_ref(pat_ms, yte, gte, k, np.inf, s) for s in seeds])
    print(f"{k:3d} {model_offset:19.2f} {line_free:12.2f} {line_ridge:9.2f} {flat:19.2f}")

if calibrated_mae(p_orig, yte, gte, 1) is not None:
    print("your calibrated_mae at k=3:", round(calibrated_mae(p_orig, yte, gte, 3), 2))
""")
md(r"""
**Reading it: bias versus variance.** Over a patient's narrow PAT range, $a/\text{PAT} + b$ is
essentially a straight line, so the per-patient equation is a two-column least-squares problem.
With λ = 0 and k = 2 the line goes exactly through two noisy points, so its slope is mostly
noise and the error explodes (k = 1 cannot be fitted at all). Shrinking the slope (ridge,
λ = 1000 ms²) helps. Fixing it at zero (offset only) is better still at small k: each extra
number fitted per patient costs variance that a handful of cuff readings cannot pay for. The
transformer plus an offset does better than the flat line because its within-patient
predictions carry a little information (mostly not PAT, given Part 9).
""")
code(r"""
cc = load_json("calibration_cost.json")
if cc:
    rows_cc = []
    for row in cc:
        if row["gap"] == 0 and row["k"] >= 1:     # gap 0: score from the segment right after the k anchors
            rows_cc.append(row)
    k_list = [row["k"] for row in rows_cc]
    print("cuff readings k:", k_list)
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.plot(k_list, [row["offset"] for row in rows_cc], "o-", ms=3, label="offset (1 number)")
    ax.plot(k_list, [row["equation"] for row in rows_cc], "o-", ms=3, label="equation a/PTT² + b (ridge)")
    ax.plot(k_list, [row["head"] for row in rows_cc], "o-", ms=3, label="ridge on the model's features")
    ax.axhline(rows_cc[0]["uncalibrated"], color="grey", ls="--", label="uncalibrated")
    ax.set(xscale="log", xlabel="Cuff readings per patient (k)", ylabel="DBP MAE (mmHg)",
           title="Saved calibration-cost curve (calibration_cost.json)"); ax.legend(frameon=False, fontsize=7)
    plt.show()
""")

# =============================================================================================
md(r"""
---
# Part 13 — Out of distribution: MIMIC-BP

MIMIC-BP is ICU data from a different hospital, monitor and population. Two different questions:
1. **Absolute error (MAE)** is dominated by **bias**, i.e. the mean shift between datasets.
2. **Within-patient tracking**: subtract each patient's mean from both prediction and truth,
   then correlate. This is what calibration cannot fake, and where the law lives.

### ✍️ By hand 13.1
Patient A: true [60, 64], predicted [70, 72]. Patient B: true [80, 84], predicted [71, 75].
(a) Raw MAE. (b) Within-patient correlation: subtract each patient's own mean from truth and
prediction, pool the four points, compute $r = \sum xy / \sqrt{\sum x^2 \sum y^2}$.
Enter `[mae, r]`.
""")
code(r"""
my_131 = None
check("13.1 MAE vs tracking", my_131, [9.0, 12 / np.sqrt(160)])
""")
solution(r"""
(a) |70−60| = 10, |72−64| = 8, |71−80| = 9, |75−84| = 9 → MAE **9**.
(b) Truth centred: A (−2, 2), B (−2, 2). Prediction centred: A (−1, 1), B (−2, 2). Σxy = 2 + 2 + 4 + 4 = 12; Σx² = 16; Σy² = 10. r = 12/√160 = **0.949**.
Poor MAE, excellent tracking. A model like this is exactly what a single cuff reading would fix.
""")
code(r"""
mo = load_json("mimic_ood_runs.json")
print("MIMIC-BP:", mo["n_segments"], "segments,", mo["n_patients"], "patients")
print("predict-the-VitalDB-mean MAE: DBP", round(mo["dbp_mean_predictor_mae"], 2), "| SBP", round(mo["sbp_mean_predictor_mae"], 2))
print()
print(f"{'run':14s} {'ID MAE':>7s} {'OOD MAE':>8s} {'bias':>6s} {'within r':>9s} {'textbook sign':>14s}")
for name, run in mo["runs"].items():
    print(f"{name:14s} {run['id_mae']:7.2f} {run['ood_mae']:8.2f} {run['ood_bias']:+6.1f} "
          f"{run['ood_within_r']:9.2f} {run['expected_sign']:14.0%}")

ot = load_json("mimic_offset_test.json")
if ot:
    print("\nPPG timing offset test (mimic_offset_test.json): shift applied to MIMIC's PPG (ms):")
    for name, value in ot["shift_ms"].items():
        print(f"  {name}: {round(value)}")
""")
md(r"""
**Reading it.** No VitalDB model does meaningfully better than predicting the VitalDB mean on MIMIC (the best DBP run
ties it); all overestimate
by several mmHg, and within-patient tracking is near zero (r ≈ 0.1). So whether following the
law predicts OOD performance cannot be tested on these near-mean models (TODO §1). The MIMIC
data itself shows the law (SBP ~80%, DBP ~72% of patients; `pat_bp_quality_mimic.json`), and its
PPG lags the arterial line by ~279 ms against VitalDB's ~476 ms, a device difference a
VitalDB-trained model has never seen.
""")

# =============================================================================================
md(r"""
---
# Part 14 — Results at a glance

The final story figure (`fig_story.py`, panel d) plots, for each trained configuration, the
improvement over predicting the mean against the share of patients with the textbook sign. The
cell below rebuilds it from `focus_models.json`.
""")
code(r"""
fm = load_json("focus_models.json")             # one entry per training run

groups = {}                                     # (target, depth, data size) -> list of (gain, % textbook)
for name, run in fm.items():
    if "_lr" in name:                           # skip the learning-rate side runs
        continue
    key = (run["target"], run["depth"], run["data"])
    gain = run["calfree"]["mean_mae"] - run["calfree"]["mae"]          # mmHg better than the mean
    textbook = 100 * run["calfree"]["ppg"]["frac_negative"]            # % patients with the textbook sign
    if key not in groups:
        groups[key] = []
    groups[key].append((gain, textbook))

fig, ax = plt.subplots(figsize=(6, 3.4))
for key in sorted(groups):
    target, depth, size = key
    v = np.array(groups[key])                   # (runs, 2)
    print(f"{target.upper()} {depth}L {size:>4s}: {len(v)} runs, gain over mean {v[:, 0].mean():.2f} mmHg, "
          f"textbook sign {v[:, 1].mean():.0f}% of patients")
    ax.errorbar(v[:, 0].mean(), v[:, 1].mean(), xerr=v[:, 0].std(), yerr=v[:, 1].std(),
                fmt="^" if depth == 1 else "o", color="C3" if target == "sbp" else "C0",
                mfc="none" if size == "12k" else None, capsize=2)
ax.axhline(50, color="grey", lw=0.6, ls=":")
q = load_json("pat_bp_quality_vitaldb.json")["L2 ABP agrees"]
for tg, c in (("SBP", "C3"), ("DBP", "C0")):
    ax.axhline(100 * q[f"PPG_{tg}"]["frac_negative"], color=c, lw=0.8, ls="--")
ax.set(xlabel="MAE improvement over predicting the mean (mmHg)", ylabel="Patients with textbook sign (%)",
       title="Red SBP, blue DBP; ▲ 1-layer, ● 3-layer; hollow 12k, filled 140k; dashed: the data")
ax.title.set_fontsize(7.5); plt.tight_layout(); plt.show()
""")
md(r"""
### Key numbers to know cold

| | Value | Source |
|---|---|---|
| Sampling | 125 Hz, 8 ms/sample, 10 s = 1250 samples | PulseDB |
| Detector vs injected PTT (simulation) | r = 0.997 | `synth_waveform_audit.json` |
| Audit slope vs injected slope | r ≈ −1.00, ~90% recovered (−0.20 vs −0.22) | `synth_waveform_audit.json` |
| R-peak count matches arterial line, old → new detector | 32% → 92% | TODO.md |
| PAT measurable, old → new | 45% → 97% | TODO.md |
| VitalDB PPG minus arterial-line delay | 476 ms (IQR 461–490) | `pat_bp_quality_vitaldb.json` |
| MIMIC-BP PPG minus arterial-line delay | 279 ms | `pat_bp_quality_mimic.json` |
| Data's law, share negative SBP / DBP (clean) | 87% / 74% (ceilings ~95% / ~89%) | `pat_bp_quality_vitaldb.json` |
| DBP transformers, textbook sign | 11–43% of 142 patients (6 runs) | `real_audit_controls.json` |
| SBP transformers, textbook sign | 54–70% (5 runs) | `real_audit_controls_sbp.json` |
| DBP MAE, transformers vs mean | 9.1–9.4 vs 9.45 mmHg | `real_audit_controls.json` |
| MIMIC: models vs VitalDB mean | DBP 10.6–11.3 vs 10.6; within-patient r ≈ 0.1 | `mimic_ood_runs.json` |
""")

# =============================================================================================
md(r"""
---
# Part 15 — Question bank

Answer out loud first, then open the box.

<details><summary><b>1. In one sentence, what is the thesis contribution?</b></summary>
An intervention audit that tests whether a cuffless-BP model uses pulse arrival time in the physiological direction, validated where the answer is known, benchmarked against the data's own within-patient law, and showing that accuracy-matched transformers can read it backwards.
</details>

<details><summary><b>2. Why is a within-patient slope the right unit, and not a pooled correlation?</b></summary>
The law (Moens–Korteweg + Hughes) relates changes in pressure to changes in transit time *within one arterial tree*. Between patients, vessel size, age, path length, PEP and (in VitalDB) channel offsets shift the levels and can reverse the pooled sign (Simpson's paradox, Part 4). Demeaning per patient (fixed effects) removes all constant per-patient terms, including the instrument offset.
</details>

<details><summary><b>3. What went wrong with the original detector?</b></summary>
It searched 50–350 ms after each R peak. VitalDB's PPG lags by ~476 ms (beyond one RR at high heart rates), so it paired feet with the *next* R peak, or found T waves and diastolic dips. Only 32% of segments matched the arterial-line beat count. The fix: Pan–Tompkins R peaks, PPG feet found on the PPG alone by intersecting tangent, and pairing by the delay that is constant across beats (92% match, 97% measurable). Conclusions drawn with the old detector, e.g. "PAT carries almost no BP information", are retired.
</details>

<details><summary><b>4. How do you know the audit works?</b></summary>
(i) Simulation with an injected law: the audit slope tracks the injected slope (r ≈ −1.00, ~90% recovered) through the full pipeline, including the 400 ms offset and filtering. (ii) Analytic controls on real data: BP = a/PAT from the detector's PAT gives 100%, the inverted version 0%. (iii) Specificity: ECG-later = PPG-earlier, both-shifted ≈ 0, small in-range shifts agree.
</details>

<details><summary><b>5. Why only negative (earlier) PPG shifts?</b></summary>
On the negative arm the detector tracks the shift 1:1. Positive shifts can push the foot past the pairing window and saturate or slip a beat (`audit_subject.py` documents −48 → −45.7 ms vs +48 → +26 ms). Segments whose measured change differs from the nominal by > 20 ms are dropped.
</details>

<details><summary><b>6. What is the noise ceiling and why report it?</b></summary>
The share of patients expected to show a negative slope if every patient had exactly the median slope and only estimation noise (their SE) varied: mean of Φ(−β/SE_i). It tells you how far an observed share (e.g. 87%) is from what a perfect law could produce (~95%), so lower shares are not over-read as "the law fails in 13%".
</details>

<details><summary><b>7. Why does DBP show the law less clearly than SBP?</b></summary>
Consistent with PEP inside PAT hiding more of the DBP relation (Payne et al. 2006). SBP is near its noise ceiling, DBP is not. Framing this is still listed as open in TODO.md.
</details>

<details><summary><b>8. The models barely beat the mean. Isn't the audit then auditing noise?</b></summary>
The claim is scoped: "a model that learned little learned PAT backwards". The sign result is consistent across 6 seeds, significant by sign test (e.g. 32% of 142 patients, p ≈ 3e-5), specific to the ECG–PPG gap and present at small shifts. Testing models that clearly beat the mean (full-data, CalBased) is the next step (TODO §1).
</details>

<details><summary><b>9. Does calibration fix a backwards model?</b></summary>
No. Offset calibration adds a constant per patient, which does not change any derivative, so the audit slope is unchanged. Calibration fixes level, not mechanism. For within-patient tracking after calibration you need a model whose within-patient response is right.
</details>

<details><summary><b>10. Explain attention in one breath, and why positions matter here.</b></summary>
Each token builds a query, key and value by linear maps; it scores every other token by scaled dot product, softmaxes the scores into weights, and outputs the weighted sum of values. Without positional embeddings this is permutation-equivariant, so timing between tokens is invisible. Here timing *is* the mechanism, so the model must learn it through `pos` and within-patch content (200 ms patches, 8 ms resolution).
</details>

<details><summary><b>11. Why z-score each segment? What does it remove?</b></summary>
Amplitude and baseline, which depend on sensor contact, gain and the device rather than BP. It keeps timing and shape. It also removes any chance of using PPG amplitude as a shortcut. Note it also removes ABP-like level information, which is why calibration-free models struggle with the between-patient offset.
</details>

<details><summary><b>12. Probe R² is high for PAT. Doesn't that mean the model uses PAT?</b></summary>
No: present ≠ used. A linear probe shows the information can be read from the activations; the downstream head may ignore it or use it with either sign. Only an intervention on the input (or on that activation subspace: `mechlib.donor_swap`) tests use and direction.
</details>

<details><summary><b>13. What's next (the PTT chapter)?</b></summary>
SBP first. Baselines: per-patient BP = a/PAT^p + b with calibration, a PAT-only tree, and deep models, in and out of distribution. Treat the instrument offset as part of the patient intercept, deal with PEP (HR adjustment, PPG-only timing features, ECG → arterial-foot as an upper bound), and report the data's own law as the reference.
</details>
""")

# =============================================================================================
md(r"""
---
# Part 16 — Cheat sheet

| Concept | Formula | Where in the project |
|---|---|---|
| Moens–Korteweg | $\text{PWV} = \sqrt{Eh/(\rho d)}$ | why PAT should track BP |
| Hughes | $E = E_0 e^{\gamma P}$ ⇒ $P = c - \tfrac2\gamma \ln \text{PTT}$ | the law's functional form |
| Arrival time | PAT = PEP + PTT (+ offset) | what the ECG→PPG delay measures |
| Convolution | $y = Hx$, $H$ Toeplitz | every filter; Pan–Tompkins moving integration |
| Zero-phase filter | forward + backward ⇒ no delay | timing preserved at 8 ms scale |
| Tangent foot | $k - (p_k - p_{\min})/p'_k$ | `ppg_feet` |
| Least squares | $\hat\beta = (X^\top X)^{-1}X^\top y$ | slopes, probes, calibration |
| Hat matrix | $H = X(X^\top X)^{-1}X^\top$, $H^2 = H$, $\operatorname{tr}H = p$ | residual ⟂ columns |
| Centering | $C = I - \tfrac1n\mathbf{1}\mathbf{1}^\top$ | within-patient (fixed-effects) slopes |
| FWL | slope with patient dummies = slope on demeaned data | why within-patient removes offsets |
| SE of slope | $\hat\sigma/\sqrt{S_{xx}}$ | noise ceiling $\overline{\Phi(-\beta/\text{SE}_i)}$ |
| Sign test | $k\sim\text{Bin}(N, 1/2)$ | share of patients negative |
| Theil–Sen | median pairwise slope | robust check |
| GD stability | $\eta < 2/\lambda_{\max}$; speed ∝ condition number | why standardise inputs |
| Patch embedding | Conv1d(k = s = 25) = reshape + matmul | 50 tokens × 200 ms |
| Attention | $\operatorname{softmax}(QK^\top/\sqrt d)V$ | permutation-equivariant without `pos` |
| LayerNorm | per-token z-score, then scale/shift | encoder layers |
| Ridge | $(X^\top X+\lambda I)^{-1}X^\top y$ = shrink by $\sigma^2/(\sigma^2+\lambda)$ | linear probes |
| Offset calibration | $\hat c = \bar r$, $\operatorname{Var} = \sigma^2/k$ | cuff readings per patient |
| Total variance | within + between | why calibration matters; why within-patient |
| Audit sign | slope of predicted BP vs ΔPAT < 0 ⇒ textbook | `real_audit_controls.py` |
""")

for i, c in enumerate(cells):
    c["id"] = f"cell-{i:03d}"
NB.write_text(json.dumps({"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
print(f"wrote {NB} ({len(cells)} cells)")
