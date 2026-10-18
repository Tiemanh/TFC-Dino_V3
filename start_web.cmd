@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" run_web.py
) else if exist "..\TFC-pretraining-dinov3\.venv\Scripts\python.exe" (
  "..\TFC-pretraining-dinov3\.venv\Scripts\python.exe" run_web.py
) else (
  python run_web.py
)
pause
