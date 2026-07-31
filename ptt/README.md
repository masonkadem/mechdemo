# Camera pulse transit time

Self-contained. Everything this project reads or writes lives in this folder: `data/`,
`figures/`, `models/`. Nothing outside it is required, and it imports nothing from the
blood-pressure faithfulness work in the parent directory.

## Run it

Double-click **`run_app.command`** (macOS). First launch fetches a 5.5 MB pose model.

The window opens in **preview**, which runs the whole pipeline and saves nothing. Wait until
the sites you need are lit and a pulse is visible in the panel, then press **Record**. A clean
exit is not evidence of a good recording, so the app says explicitly when nothing was saved.

Camera permission is per-application. If it cannot open the camera, enable Terminal under
System Settings → Privacy & Security → Camera, then **quit Terminal fully (Cmd-Q)** and reopen —
the grant only takes effect on a fresh launch.

## What to record, in order

| tag | what | prediction |
|---|---|---|
| `rest` | hand at heart level | baseline |
| `rest2` | **a repeat of rest** | none — this is the control |
| `hand_up` | hand 30–40 cm above the heart | transit LENGTHENS |
| `hand_down` | hand below the heart | transit SHORTENS |
| `post_exer` | straight after ~30 s effort | transit SHORTENS |

`rest2` is not optional. The gap between two identical recordings is this rig's noise floor,
and every other result has to beat it. Without it a condition difference cannot be distinguished
from run-to-run variation.

## Then check whether it is real

```bash
python rppg_robust.py            # the three plots
python rppg_shutter.py --check   # did the ROI move rows between conditions?
```

1. **Arrival vs distance.** The discriminator. A fixed instrument delay is *constant* in
   distance; propagation is *linear* in it. The slope is the physiology, and its reciprocal is
   pulse wave velocity with a known 4–12 m/s upper-limb range. Plotted with a bootstrap CI —
   if that CI includes zero, no propagation has been shown.
2. **Test–retest.** `rest` vs `rest2` is the noise floor.
3. **Condition distributions.** Per-window spread, not bars of means: a difference in means
   with overlapping spreads cannot classify one new recording.

## Honest limits

At 30 fps a frame is 33 ms while neck-to-hand transit is 20–50 ms, so **per-beat transit is not
resolvable** — what is measured is a beat-averaged cross-correlation offset, refined below the
sample interval. Its absolute value also contains any fixed instrument delay, so only the
**change between conditions** carries evidence.

One fixed delay is not fixed: rolling shutter samples image rows at different times, and the
hand_up/hand_down protocol moves the hand to a different row *by design*. At a 24 ms readout a
180-row move fabricates ~9 ms of apparent transit shift, with the same sign dependence as the
predicted physiology — so it passes the "hand_down predicts the opposite" control too.
`rppg_shutter.py --check` bounds it from ROI geometry with no calibration; `rppg_shutter.py`
measures it directly. Calibrations that fail physical bounds are refused rather than applied.

For real per-beat timing, record 240 fps phone slow-motion and use `rppg_video.py`.

## Files

| | |
|---|---|
| `app_ptt.py` | desktop console (PySide6) — preview, record, results |
| `rppg_cam.py` | platform-aware camera open; reports the *delivered* frame rate |
| `rppg_pose.py` | pose-tracked multi-site capture and analysis |
| `rppg_live.py` | live pulse / HR / transit panel |
| `rppg_multi.py` | skin masking, incl. the shadow-tolerant `SkinModel` |
| `rppg_two_site.py` | two-site neck→hand capture, CHROM, sub-frame lag |
| `rppg_robust.py` | the three validity plots |
| `rppg_shutter.py` | rolling-shutter calibration and bound |
| `rppg_video.py` | offline 240 fps video path |
| `rppg_sota.py`, `rppg_multi.py`, `rppg_compare.py`, `rppg_gui.py` | earlier instruments, kept |
