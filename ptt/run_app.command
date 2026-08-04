#!/bin/bash
# Double-click in Finder to open the desktop app.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

# The virtualenv lives at the repo root, one level up from this self-contained project.
PY="../.venv/bin/python"
[ -x "$PY" ] || PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

# Both models are needed, and both are gitignored binaries that a clone cannot carry.
# The hand model is not optional-in-practice: pose landmark 20 is the index KNUCKLE, so
# without it the fingertips -- the strongest rPPG signal on the body -- are never sampled.
mkdir -p models
fetch () {   # name, path, url, size
  [ -f "$2" ] && return 0
  echo "Fetching the $1 model ($4, one time) ..."
  curl -fL# -o "$2" "$3" && return 0
  rm -f "$2"                       # never leave a truncated .task behind
  echo "Download failed for the $1 model."
  return 1
}
fetch pose models/pose_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" \
  "5.5 MB" || { read -r -p "Return to close ... "; exit 1; }
fetch hand models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
  "7.5 MB" || echo "  -> fingertip sites will be unavailable; the app will say so."

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
