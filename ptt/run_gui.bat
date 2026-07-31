@echo off
REM run_gui.bat -- two-site rPPG experiment console. Double-click to launch.
cd /d "%~dp0"
start "" "C:\Users\mason\miniconda3\envs\bp\pythonw.exe" rppg_gui.py
