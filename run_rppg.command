#!/bin/bash
# Double-click in Finder: pose-tracked rPPG with a live view of every sampling site.
# macOS counterpart of run_webcam.bat / run_gui.bat.

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

MODEL="models/pose_landmarker.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

clear
cat <<'BANNER'
==========================================================
  Pose-tracked rPPG  --  neck to hand
==========================================================

A live window shows the camera with a coloured dot on every
sampling site. The sites FOLLOW your body, so you do not
have to hold still inside a fixed box:

    neck (carotid)  shoulder  upper arm  forearm  hand

Non-skin pixels are dimmed, so you can see the skin mask
working. Press q in the video window to stop early.

FOR A GOOD RECORDING
  * Bright, steady light on your face and arm. Avoid
    flicker: daylight or an LED lamp, not a dim room.
  * Keep your neck AND your whole right arm in frame.
  * Hold still. The first 3 s are discarded while the
    camera's auto-exposure settles.

==========================================================
BANNER

# The pose model is a 5.5 MB download, fetched once.
if [ ! -f "$MODEL" ]; then
  echo "Downloading the pose model (5.5 MB, one time) ..."
  mkdir -p models
  if ! curl -fL# -o "$MODEL" "$MODEL_URL"; then
    echo "Download failed. Check your connection and try again."
    read -r -p "Press Return to close ... "; exit 1
  fi
  echo
fi

echo "Which condition?"
echo "   1) rest           hand at heart level  (record this first)"
echo "   2) hand_up        hand well ABOVE the heart   -> PTT should LENGTHEN"
echo "   3) hand_down      hand hanging BELOW the heart -> PTT should SHORTEN"
echo "   4) post_exercise  straight after ~30 s effort  -> PTT should SHORTEN"
echo
read -r -p "Number [1]: " choice
case "${choice:-1}" in
  2) TAG="hand_up"   ;;
  3) TAG="hand_down" ;;
  4) TAG="post_exer" ;;
  *) TAG="rest"      ;;
esac

read -r -p "Seconds [60]: " SECS
SECS="${SECS:-60}"

echo
echo "Recording '$TAG' for ${SECS}s. The video window opens in a moment ..."
echo

OUT="data/rppg_pose_${TAG}.json"
before=$(stat -f %m "$OUT" 2>/dev/null || echo 0)

"$PY" rppg_pose.py --seconds "$SECS" --tag "$TAG" --stages
status=$?
after=$(stat -f %m "$OUT" 2>/dev/null || echo 0)

echo
if [ $status -eq 0 ] && [ "$after" = "$before" ]; then
  # The script exits 0 on its own quality gates ("too few accepted points"), so a clean exit
  # code is not evidence anything was written. Check the file.
  echo "Finished, but NOTHING WAS SAVED -- the run did not pass its quality gates."
  echo "Scroll up for the [qc] lines. Usually: more light on the face and forearm,"
  echo "keep still, and make sure the hand stays in frame."
elif [ $status -eq 0 ]; then
  echo "Saved: $OUT   figures/fig_rppg_stages_${TAG}.png"
  echo
  echo "Record 'rest' plus at least one other condition, then check whether"
  echo "moving your hand also moved its image row, which fakes a PTT shift:"
  echo "    $PY rppg_shutter.py --check"
else
  echo "Exited with status $status -- nothing was saved."
  echo "If it said 'cannot open camera', grant Camera to Terminal in"
  echo "System Settings > Privacy & Security > Camera, then QUIT Terminal"
  echo "completely (Cmd-Q) and re-run. Permission is per-app and is only"
  echo "picked up on a fresh launch."
fi

echo
read -r -p "Press Return to close this window ... "
