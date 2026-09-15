"""bp_model.py -- blood pressure from pulse transit time, calibrated against cuff readings.

Why this module exists
----------------------
A transit time is not a pressure. Turning one into the other needs two things the camera cannot
see: a physiological FORM for the mapping, and THIS subject's constants, because the elastic
modulus and wall geometry that set the mapping are not observable from video. So the cuff is not
a validation afterthought here -- it is an input.

The form
--------
Moens-Korteweg gives the wave speed in an elastic tube,

    PWV = sqrt(E h / (2 rho r))

and Hughes' empirical law makes the modulus rise exponentially with distending pressure,
E = E0 exp(zeta P). Substituting and solving for pressure leaves a mapping that is LINEAR IN
ln(PWV):

    P = (2 / zeta) ln(PWV) + const

That is the whole reason this module regresses on ln(PWV) and not on PTT, 1/PTT or 1/PTT^2. Those
are all local linearisations of this same curve; they agree near the calibration point and
diverge away from it, which is exactly where a hand-raise or post-exercise condition puts you.

zeta for human arteries sits near 0.017-0.018 mmHg^-1, so the slope 2/zeta is about
110-120 mmHg per natural-log unit of wave speed. That value is used as a POPULATION PRIOR below,
which is what lets a single cuff reading already produce a usable estimate.

Using ln(PWV) rather than ln(1/PTT) is what makes the path length earn its place: PWV = L/PTT, so
a subject with a 20% longer arm has a 20% higher wave speed at the same pressure. Feeding the
pose-derived L in removes that between-subject term from the intercept instead of leaving it to be
absorbed by calibration.

Heart rate is a second term, not a leak
---------------------------------------
HR is included deliberately. MAP = CO x TPR and CO = HR x SV, so heart rate is a genuine term in
mean pressure, and it is sympathetically co-driven with the pressure changes this rig provokes.
Excluding it to keep the model "purely mechanical" would be the arbitrary choice, not including
it. If you want a Bramwell-Hill-only arm, fit with `hr_term=False` and label it as such.

Honesty about what this can deliver
-----------------------------------
At 30 fps one frame is 33 ms while face-to-hand transit is 20-50 ms, so a single-window PTT is
sub-frame and its scatter is comparable to its value (see rppg_live.py). Propagated through a
110 mmHg/ln-unit slope, a 10% PTT error is about 11 mmHg of pressure. This module therefore
always returns an uncertainty beside the estimate, refuses to return an estimate at all when the
calibration cannot support one, and never reports a sigma below SIGMA_FLOOR no matter how neatly
a handful of points happen to line up. A pressure without its spread would be the dishonest part.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# --- physiological priors -------------------------------------------------------------------
ZETA = 0.018                       # Hughes' exponent for human arteries, mmHg^-1
SLOPE_PRIOR = 2.0 / ZETA           # ~111 mmHg per ln-unit of wave speed

# Acute pressure change per bpm. DBP tracks rate less than SBP does, because it is set more by
# diastolic runoff time, which shortens as rate rises and partly cancels the rise.
HR_PRIOR = {"sbp": 0.30, "dbp": 0.15}
HR_REF = 70.0                      # centring only; the free intercept absorbs the choice

# The prior is worth this many calibration points. The fitted slope is then roughly
# (n * data + PRIOR_WEIGHT * prior) / (n + PRIOR_WEIGHT), so 3 cuff readings NUDGE the slope
# instead of inventing one, and the fit degrades toward physiology rather than toward zero.
PRIOR_WEIGHT = 4.0

# --- validity gates ------------------------------------------------------------------------
# A non-positive lag means the fingertip pulse arrived before the face pulse, which is not a
# transit time. It is rejected rather than abs()'d: taking the magnitude would launder a failed
# cross-correlation into a plausible-looking number, which is the one thing this rig must not do.
MIN_PTT_MS, MAX_PTT_MS = 3.0, 250.0
# The DIFFERENTIAL path (see hand_sites.differential_path_cm), not the anatomical face-to-hand
# route, so the range is wider at the bottom: a short subject with long legs can differential
# down near 25 cm without anything being wrong.
PATH_MIN_CM, PATH_MAX_CM = 20.0, 130.0
HR_MIN, HR_MAX = 30.0, 220.0
# Nominal differential path: half biacromial (~18) + arm to fingertip (~74) - shoulder to ear
# (~20). Used only when pose gives no path, and folds into the intercept, so a wrong nominal
# costs calibration accuracy only for uncalibrated between-subject comparisons.
NOMINAL_PATH_CM = 72.0

# No calibration of this kind beats the AAMI bar (5 mmHg bias, 8 mmHg SD), and a 4-point fit that
# reports 2 mmHg has simply overfitted 4 points. Never claim better than this.
SIGMA_FLOOR = 5.0
# With one cuff reading the slope is ASSUMED, so error grows with distance from that point. A
# 0.3 ln-unit excursion against a 40%-wrong slope is ~13 mmHg, which is the floor for this mode.
OFFSET_SIGMA = 13.0

TARGETS = ("sbp", "dbp")
MODE_MIN = {"offset": 1, "slope": 3, "full": 5}


def slug(s):
    """Filename-safe subject id. Anything exotic becomes '-' rather than a path surprise.

    This runs on operator-typed text that goes straight into a filename, so a stray '/' would
    otherwise write outside data/ and a space would make the shell examples in the docs wrong.
    """
    keep = "".join(c if (c.isalnum() or c in "-_") else "-" for c in str(s or "").strip())
    while "--" in keep:                     # 's01 / pilot #2' -> 's01-pilot-2', not 's01---pilot--2'
        keep = keep.replace("--", "-")
    return keep.strip("-_")[:24]


# ================================================================================= features
def ln_pwv(ptt_ms, path_cm=None):
    """ln of wave speed in m/s -- the feature the physiology is linear in.

    Returns nan for any input the mapping cannot stand on, so a bad frame drops out of the fit
    rather than contributing a wrong pressure.
    """
    if ptt_ms is None:
        return float("nan")
    ptt = float(ptt_ms)
    if not np.isfinite(ptt) or not (MIN_PTT_MS <= ptt <= MAX_PTT_MS):
        return float("nan")
    L = NOMINAL_PATH_CM
    if path_cm is not None and np.isfinite(path_cm) and PATH_MIN_CM < float(path_cm) < PATH_MAX_CM:
        L = float(path_cm)
    return float(np.log((L / 100.0) / (ptt / 1000.0)))


def features(ptt_ms, hr=None, path_cm=None):
    """[ln PWV, centred HR] or None when the transit time is unusable.

    A missing or implausible HR centres to zero rather than killing the row: no rate information
    means no rate contribution, which is the right default and keeps a good PTT usable.
    """
    x = ln_pwv(ptt_ms, path_cm)
    if not np.isfinite(x):
        return None
    h = float(hr) if (hr is not None and np.isfinite(hr) and HR_MIN < float(hr) < HR_MAX) else HR_REF
    return np.array([x, h - HR_REF], float)


def pwv_ms(ptt_ms, path_cm=None):
    """Wave speed in m/s, for display and for the 4-12 m/s plausibility check."""
    x = ln_pwv(ptt_ms, path_cm)
    return float(np.exp(x)) if np.isfinite(x) else float("nan")


# ====================================================================================== fit
def _ridge(X, y, prior, weight):
    """Ridge pulled toward `prior` rather than toward zero, with a free intercept.

    The penalty is scaled by each feature's own variance, so `weight` means "the prior is worth
    this many calibration points" regardless of the units the feature happens to be in. Without
    that scaling the same lambda would crush the ln(PWV) slope (range ~0.2) while leaving the HR
    slope (range ~20 bpm) untouched.
    """
    X = np.atleast_2d(X)
    Xc, yc = X - X.mean(0), y - y.mean()
    lam = weight * (X.var(0) + 1e-12)
    coef = np.linalg.solve(Xc.T @ Xc + np.diag(lam), Xc.T @ yc + lam * prior)
    return float(y.mean() - coef @ X.mean(0)), coef


def _fit_free(X, y, prior, free):
    """Fit only the columns in `free`; the rest contribute at their prior value."""
    prior = np.asarray(prior, float)
    fixed = np.array([j for j in range(X.shape[1]) if j not in free], int)
    resid = y - (X[:, fixed] @ prior[fixed] if fixed.size else 0.0)
    coef = prior.copy()
    if not free:                                   # offset mode: slope assumed, intercept fitted
        return float(np.mean(resid)), coef
    b0, cf = _ridge(X[:, free], resid, prior[free], PRIOR_WEIGHT)
    coef[free] = cf
    return b0, coef


def _loo_rmse(X, y, prior, free):
    """Leave-one-out error -- the only honest accuracy claim available at n < 10.

    In-sample residuals of a 5-point fit understate the error badly; LOO does not, and it is
    cheap at these sample sizes.
    """
    if len(y) < 3:
        return float("nan")
    errs = []
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        try:
            b0, cf = _fit_free(X[m], y[m], prior, free)
        except np.linalg.LinAlgError:
            continue
        errs.append(y[i] - (b0 + cf @ X[i]))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


class Fit:
    """A calibrated PTT->pressure mapping, with an account of what it rests on."""

    def __init__(self, target, mode, intercept=0.0, coef=None, sigma=float("nan"), n=0, note=""):
        self.target, self.mode, self.n, self.note = target, mode, n, note
        self.intercept = float(intercept)
        self.coef = np.zeros(2) if coef is None else np.asarray(coef, float)
        self.sigma = float(sigma)

    @property
    def ok(self):
        return self.mode != "none"

    def predict(self, ptt_ms, hr=None, path_cm=None):
        """(mmHg, sigma). Both nan when this fit cannot speak to this sample."""
        if not self.ok:
            return float("nan"), float("nan")
        f = features(ptt_ms, hr, path_cm)
        if f is None:
            return float("nan"), float("nan")
        return float(self.intercept + self.coef @ f), self.sigma

    def describe(self):
        if not self.ok:
            return "no calibration yet"
        what = {"offset": "population slope, intercept from cuff",
                "slope": "PWV slope fitted (prior-pulled), HR at prior",
                "full": "PWV and HR slopes fitted (prior-pulled)"}[self.mode]
        s = f"n={self.n}  {what}"
        if np.isfinite(self.sigma):
            s += f"  +/-{self.sigma:.0f} mmHg"
        return s + (f"  [{self.note}]" if self.note else "")


def fit(points, target="sbp", hr_term=True):
    """Calibrate on whatever cuff readings exist, and say which mode that bought.

    Modes escalate with the data, so the caller never has to special-case an empty session:

      none    no usable point -> no estimate at all, rather than a population guess
      offset  1-2 points: population slope, intercept from the cuff (classic one-point cal)
      slope   3-4 points: PWV slope fitted but prior-pulled; HR held at its prior
      full    5+ points:  both slopes fitted, both prior-pulled

    Set hr_term=False for a Bramwell-Hill-isolation arm; the HR column is then pinned to zero
    instead of to its prior, and the mode string is marked so the two arms cannot be confused.
    """
    rows, ys = [], []
    for p in points:
        f = features(p.get("ptt_ms"), p.get("hr"), p.get("path_cm"))
        y = p.get(target)
        if f is None or y is None or not np.isfinite(float(y)):
            continue
        rows.append(f)
        ys.append(float(y))
    n = len(ys)
    if n == 0:
        return Fit(target, "none")
    X, y = np.array(rows, float), np.array(ys, float)
    prior = np.array([SLOPE_PRIOR, HR_PRIOR.get(target, 0.0) if hr_term else 0.0], float)

    if n >= MODE_MIN["full"]:
        mode, free = "full", ([0, 1] if hr_term else [0])
    elif n >= MODE_MIN["slope"]:
        mode, free = "slope", [0]
    else:
        mode, free = "offset", []

    b0, coef = _fit_free(X, y, prior, free)
    sigma = _loo_rmse(X, y, prior, free)
    if mode == "offset":
        # Spread across duplicate readings is real information, but it cannot shrink the error
        # below what an assumed slope costs away from the calibration point.
        obs = float(np.std(y - X @ coef)) if n > 1 else float("nan")
        sigma = max(OFFSET_SIGMA, obs if np.isfinite(obs) else 0.0)
    sigma = max(SIGMA_FLOOR, sigma) if np.isfinite(sigma) else float("nan")
    note = "" if hr_term else "no-HR arm"
    return Fit(target, mode, b0, coef, sigma, n, note)


def fit_all(points, hr_term=True):
    """{'sbp': Fit, 'dbp': Fit} -- what the readout needs in one call."""
    return {t: fit(points, t, hr_term) for t in TARGETS}


# ================================================================================== session
class Session:
    """The lab session's irreplaceable record: sync marks and cuff readings.

    Saved on EVERY mutation, not at the end. A recording can be repeated; a session with a
    subject and a cuff already on their arm cannot, so a crash must never cost the marks.

    Two clocks, on purpose. Intervals come from perf_counter, which is monotonic and cannot jump;
    absolute times are reconstructed from ONE wall-clock anchor taken at session start. Stamping
    each mark with time.time() directly would let an NTP correction mid-session silently shift
    marks relative to each other, which is precisely the alignment this file exists to protect.
    """

    def __init__(self, name=None, subject=""):
        self.stamp = name or time.strftime("%Y%m%d_%H%M%S")
        self.subject = slug(subject)
        self.t0_wall = time.time()
        self.t0_mono = time.perf_counter()
        self.marks, self.calib = [], []
        self.hr_term = True

    @property
    def name(self):
        """Timestamp first so sessions sort chronologically, subject second so they group."""
        return f"{self.stamp}_{self.subject}" if self.subject else self.stamp

    @property
    def path(self):
        return DATA / f"session_{self.name}.json"

    def set_subject(self, subject):
        """Fold a subject id into the session name, MOVING any file already written.

        The operator usually types the subject id after the app is already open, by which time
        marks may exist. Writing a fresh file under the new name would leave those marks orphaned
        in the old one, so the existing record is moved rather than abandoned.
        """
        s = slug(subject)
        if s == self.subject:
            return self.path
        old = self.path
        self.subject = s
        if old.exists() and old != self.path:
            old.replace(self.path)
        self.save()
        return self.path

    def _stamp(self, t_mono=None):
        t = time.perf_counter() if t_mono is None else float(t_mono)
        rel = t - self.t0_mono
        wall = self.t0_wall + rel
        return {"t_rel_s": round(rel, 4), "t_wall": wall,
                "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(wall))
                       + f".{int((wall % 1) * 1000):03d}"}

    def mark(self, label="cuff", t_mono=None, note="", **extra):
        """Record a synchronisation event. `t_mono` should be stamped at the keypress itself.

        The caller stamps, not this method, because the stamp must be taken in the handler the
        human triggered -- any queueing between the press and here would be added to the offset
        the mark exists to pin down.
        """
        m = {"label": label, "note": note, **self._stamp(t_mono), **extra}
        self.marks.append(m)
        self.save()
        return m

    def add_calib(self, sbp, dbp, ptt_ms=None, hr=None, path_cm=None, spread_ms=None,
                  label="", t_mono=None, **extra):
        """Pair a cuff reading with the live features at that moment."""
        p = {"label": label, "sbp": float(sbp), "dbp": float(dbp),
             "ptt_ms": None if ptt_ms is None or not np.isfinite(ptt_ms) else float(ptt_ms),
             "hr": None if hr is None or not np.isfinite(hr) else float(hr),
             "path_cm": None if path_cm is None or not np.isfinite(path_cm) else float(path_cm),
             "spread_ms": None if spread_ms is None or not np.isfinite(spread_ms)
                          else float(spread_ms),
             **self._stamp(t_mono), **extra}
        self.calib.append(p)
        self.save()
        return p

    def drop_calib(self, i):
        if 0 <= i < len(self.calib):
            self.calib.pop(i)
            self.save()

    def usable(self, target="sbp"):
        """Cuff readings that can actually enter a fit -- what the UI should count."""
        return [p for p in self.calib
                if features(p.get("ptt_ms"), p.get("hr"), p.get("path_cm")) is not None
                and p.get(target) is not None]

    def fits(self):
        return fit_all(self.calib, self.hr_term)

    def to_dict(self):
        return {"name": self.name, "stamp": self.stamp, "subject": self.subject,
                "t0_wall": self.t0_wall,
                "t0_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.t0_wall)),
                "hr_term": self.hr_term, "marks": self.marks, "calib": self.calib}

    def save(self):
        DATA.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, default=float))
        tmp.replace(self.path)          # atomic: a crash mid-write cannot truncate the record
        return self.path

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        # `stamp` fell back to `name` for sessions written before the subject was split out of it.
        s = cls(d.get("stamp") or d.get("name"), d.get("subject", ""))
        s.t0_wall = float(d.get("t0_wall", s.t0_wall))
        s.marks = list(d.get("marks", []))
        s.calib = list(d.get("calib", []))
        s.hr_term = bool(d.get("hr_term", True))
        return s

    @classmethod
    def latest(cls):
        """Most recent session on disk, or None -- so a relaunch can resume a calibration."""
        got = sorted(DATA.glob("session_*.json"))
        return cls.load(got[-1]) if got else None


# ================================================================================= selftest
def _selftest():
    """Recover a known law from synthetic points, and check the modes degrade in the right order."""
    rng = np.random.default_rng(0)
    B0, B1, B2 = -60.0, 120.0, 0.35
    L = 78.0
    xs = np.log(np.linspace(4.5, 9.0, 40))           # ln PWV over a physiological span
    hrs = rng.uniform(55, 95, xs.size)
    sbp = B0 + B1 * xs + B2 * (hrs - HR_REF) + rng.normal(0, 3.0, xs.size)
    pts = [{"sbp": float(s), "dbp": float(s) - 40.0, "hr": float(h),
            "ptt_ms": float((L / 100.0) / np.exp(x) * 1000.0), "path_cm": L}
           for x, h, s in zip(xs, hrs, sbp)]

    f = fit(pts, "sbp")
    assert f.mode == "full" and f.n == 40, f.describe()
    assert abs(f.coef[0] - B1) < 25, f"slope {f.coef[0]:.0f} vs {B1}"
    assert f.sigma < 12, f.sigma
    got, sig = f.predict(pts[5]["ptt_ms"], pts[5]["hr"], L)
    assert abs(got - pts[5]["sbp"]) < 3 * sig

    assert fit([], "sbp").mode == "none"
    assert not fit([], "sbp").ok
    assert np.isnan(fit([], "sbp").predict(30.0, 70, L)[0])
    assert fit(pts[:1], "sbp").mode == "offset"
    assert fit(pts[:3], "sbp").mode == "slope"
    assert fit(pts[:1], "sbp").sigma >= OFFSET_SIGMA
    assert fit(pts, "sbp").sigma >= SIGMA_FLOOR

    # a one-point calibration must still land near the cuff at its own operating point
    one = fit(pts[20:21], "sbp")
    got, _ = one.predict(pts[20]["ptt_ms"], pts[20]["hr"], L)
    assert abs(got - pts[20]["sbp"]) < 1e-6, got

    # nonphysical inputs are refused, not laundered
    assert np.isnan(ln_pwv(-30.0)), "negative transit must be rejected"
    assert np.isnan(ln_pwv(0.0))
    assert np.isnan(ln_pwv(float("nan")))
    assert np.isnan(ln_pwv(1e4))
    assert features(-30.0, 70, L) is None
    # a missing HR must not kill an otherwise good row
    assert features(30.0, None, L) is not None and features(30.0, None, L)[1] == 0.0
    # an absent path falls back to nominal rather than nan
    assert np.isfinite(ln_pwv(30.0)) and np.isfinite(ln_pwv(30.0, float("nan")))
    # the no-HR arm must pin the rate column, not merely shrink it
    assert fit(pts, "sbp", hr_term=False).coef[1] == 0.0

    print("bp_model selftest ok")
    print(" ", fit(pts, "sbp").describe())
    print(" ", fit(pts[:3], "sbp").describe())
    print(" ", fit(pts[:1], "sbp").describe())


if __name__ == "__main__":
    _selftest()
