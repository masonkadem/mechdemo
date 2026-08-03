@echo off
REM run_app.bat -- Windows equivalent of run_app.command. Double-click to open the desktop app.
REM
REM run_app.command is a bash script with mac-only paths (../.venv/bin/python, curl) and will not
REM run here. app_ptt.py itself is PySide6 and cross-platform, so only the launcher differs.

setlocal
cd /d "%~dp0"
set PY=C:\Users\mason\miniconda3\envs\bp\python.exe

if not exist "models\pose_landmarker.task" (
  echo Fetching the pose model ^(5.8 MB, one time^) ...
  mkdir models 2>nul
  "%PY%" -c "import urllib.request as u;open('models/pose_landmarker.task','wb').write(u.urlopen('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',timeout=90).read())"
  if errorlevel 1 (
    echo Download failed.
    pause
    exit /b 1
  )
)

"%PY%" app_ptt.py
if errorlevel 1 (
  echo.
  echo Exited with an error.
  echo If the camera would not open, close any other app using it ^(Teams, Zoom, the
  echo browser^) and check Settings ^> Privacy ^& security ^> Camera is enabled for
  echo desktop apps.
  pause
)
