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
# fetch: name, path, url, min_bytes
#
# Downloads via a .part file and moves it into place only once it is verified. The previous
# version pointed curl straight at the final path and deleted it on any non-zero exit, which is
# fragile here for two reasons:
#
#   - this tree lives in OneDrive, which locks and re-writes files while it syncs. A 9 MB write
#     into a syncing folder can be interrupted, and the old code responded by deleting the
#     partial file and reporting a download failure for what was really a sync collision.
#   - curl exits 0 on a truncated body when the server sends no Content-Length, so a short file
#     could be kept and then fail much later inside MediaPipe with an unrelated-looking error.
#
# So: retry, verify a minimum size, confirm the ZIP magic that every .task carries, and only then
# publish the file. A model that is already present and valid is left alone.
fetch () {
  local name="$1" path="$2" url="$3" min="$4" tmp="$2.part" try rc sz
  if [ -f "$path" ]; then
    sz=$(wc -c < "$path" | tr -d ' ')
    [ "$sz" -ge "$min" ] && return 0
    echo "Existing $name model is only ${sz} bytes (expected >= ${min}); refetching."
    rm -f "$path"
  fi
  for try in 1 2 3; do
    echo "Fetching the $name model (attempt $try of 3) ..."
    rm -f "$tmp"
    curl -fL# --retry 2 --retry-delay 1 --connect-timeout 20 -o "$tmp" "$url"
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "  curl exited $rc"
      sleep 2; continue
    fi
    sz=$(wc -c < "$tmp" | tr -d ' ')
    if [ "$sz" -lt "$min" ]; then
      echo "  got only ${sz} bytes, expected >= ${min} -- treating as truncated"
      sleep 2; continue
    fi
    # Every .task bundle is a ZIP; the magic sits a couple of bytes in. Cheaper and more
    # reliable than trusting the byte count alone.
    if ! head -c 16 "$tmp" | grep -q "PK"; then
      echo "  not a .task bundle (no ZIP magic) -- server may have returned an error page"
      sleep 2; continue
    fi
    mv -f "$tmp" "$path" && return 0
    echo "  could not move into place (OneDrive may be holding the file)"
    sleep 2
  done
  rm -f "$tmp"
  echo "Download failed for the $name model after 3 attempts."
  echo "Fetch it by hand with:"
  echo "  curl -fL -o \"$path\" \"$url\""
  return 1
}
# Pose model variant. `full` by default rather than `lite`: the landmarks place every sampling
# patch, so landmark jitter becomes ROI jitter, which becomes signal noise -- and the world
# landmarks also set the path length that scales the wave speed. lite is the least accurate of
# the three MediaPipe offers and was costing accuracy in exactly the two places it matters.
#   POSE_MODEL=lite   5.5 MB   fastest, weakest
#   POSE_MODEL=full   9.0 MB   the default here
#   POSE_MODEL=heavy   29 MB   best landmarks, noticeably more CPU per frame
POSE_MODEL="${POSE_MODEL:-full}"
# Minimum plausible size per variant, in bytes -- the truncation check above compares against it.
case "$POSE_MODEL" in
  lite)  POSE_MIN=5000000  ;;   # ~5.8 MB actual
  full)  POSE_MIN=9000000  ;;   # ~9.4 MB actual
  heavy) POSE_MIN=25000000 ;;   # ~29 MB actual
  *) echo "POSE_MODEL must be lite, full or heavy (got '$POSE_MODEL')"; exit 1 ;;
esac
# Variant is part of the filename, so switching does not silently reuse the previous model and
# leave you comparing recordings made with different landmarkers.
POSE_TASK="models/pose_landmarker_${POSE_MODEL}.task"
if ! fetch "pose ($POSE_MODEL)" "$POSE_TASK" \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${POSE_MODEL}/float16/1/pose_landmarker_${POSE_MODEL}.task" \
  "$POSE_MIN"; then
  # A failed pose fetch is not fatal if ANY pose model is already on disk: the app falls back to
  # whatever is present, which is far better than refusing to open with a subject in the chair.
  if ls models/pose_landmarker*.task >/dev/null 2>&1; then
    echo "  -> continuing with the pose model already on disk; accuracy may be lower."
    unset POSE_MODEL
  else
    read -r -p "Return to close ... "; exit 1
  fi
fi
export POSE_MODEL
fetch hand models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
  7000000 || echo "  -> fingertip sites will be unavailable; the app will say so."

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
