@echo off
REM run_webcam.bat -- one-click two-site rPPG capture (neck + hand).
REM
REM Usage, from Explorer: double-click this file.
REM Usage, from a terminal:
REM     run_webcam.bat                 60 s, tagged "rest", with live preview
REM     run_webcam.bat 30 hand_up      30 s, tagged "hand_up"
REM
REM The protocol that actually produces evidence is TWO runs:
REM     run_webcam.bat 60 rest         hand at heart level
REM     run_webcam.bat 60 hand_up      hand raised well above the heart
REM Hydrostatic pressure falls when the hand is raised, so PTT should LENGTHEN.
REM A reproducible shift BETWEEN conditions is meaningful; a single absolute number is not,
REM because a fixed camera/ROI processing delay would look identical.

setlocal
set PY=C:\Users\mason\miniconda3\envs\bp\python.exe
set SECONDS=%1
set TAG=%2
if "%SECONDS%"=="" set SECONDS=60
if "%TAG%"=="" set TAG=rest

cd /d "%~dp0"

echo.
echo  Two-site rPPG capture
echo  ---------------------
echo   duration : %SECONDS%s
echo   tag      : %TAG%
echo.
echo  Position: face the camera, then raise your palm to chest height so BOTH
echo  your neck and your hand are visible. Bright, steady light. Hold still.
echo  Press q in the preview window to stop early.
echo.
pause

"%PY%" rppg_two_site.py --seconds %SECONDS% --tag %TAG% --show

echo.
echo  Saved to data\rppg_two_site_%TAG%.json
echo.
pause
