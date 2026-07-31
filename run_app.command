#!/bin/bash
# Double-click in Finder to open the desktop app.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

MODEL="models/pose_landmarker.task"
if [ ! -f "$MODEL" ]; then
  echo "Fetching the pose model (5.5 MB, one time) ..."
  mkdir -p models
  curl -fL# -o "$MODEL" \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" \
    || { echo "Download failed."; read -r -p "Return to close ... "; exit 1; }
fi

"$PY" app_ptt.py
status=$?
if [ $status -ne 0 ]; then
  echo
  echo "Exited with status $status."
  echo "If it could not open the camera: System Settings > Privacy & Security > Camera,"
  echo "enable Terminal, then QUIT Terminal fully (Cmd-Q) and reopen. Camera permission is"
  echo "per-app and only takes effect on a fresh launch."
  read -r -p "Press Return to close ... "
fi
