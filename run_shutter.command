#!/bin/bash
# Double-click this in Finder to run the rolling-shutter calibration.
# macOS counterpart of run_gui.bat / run_webcam.bat. A .command file is what Finder will open
# in Terminal on a double-click; a plain .sh is not associated with Terminal by default.

# Finder starts double-clicked scripts in $HOME, so anchor to this file's own directory.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

clear
cat <<'BANNER'
==========================================================
  Rolling-shutter calibration
==========================================================

Measures the fake transit time your camera invents between
two image rows, so a hand that moves up or down the frame
cannot masquerade as a real change in pulse transit time.

BEFORE YOU START
  1. Dim the room. The screen must be the brightest thing
     that is changing.
  2. Point the camera at your face, or better, a plain wall
     that the screen lights up.
  3. A green fullscreen flash will pulse for ~40 s. Do not
     move. Press q in the video window to stop early.

The first run will ask for camera permission for Terminal.
If you see "cannot open camera", grant it in
System Settings > Privacy & Security > Camera, then re-run.

==========================================================
BANNER

read -r -p "Press Return to start (Ctrl-C to cancel) ... "
echo

CAL="data/rppg_shutter_cal.json"
before=$(stat -f %m "$CAL" 2>/dev/null || echo 0)

"$PY" rppg_shutter.py "$@"
status=$?
after=$(stat -f %m "$CAL" 2>/dev/null || echo 0)

echo
if [ $status -ne 0 ]; then
  echo "Exited with status $status -- nothing was saved."
  echo "To check the maths without a camera:  ./.venv/bin/python rppg_shutter.py --selftest"
elif [ "$after" != "$before" ]; then
  echo "Calibration written to $CAL"
  echo "Recordings made from now on are corrected automatically."
else
  echo "Finished, but no calibration was written (nothing to correct with yet)."
fi

echo
read -r -p "Press Return to close this window ... "
