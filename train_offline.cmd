@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" train_offline.py %*
) else if exist "..\TFC-pretraining-dinov3\.venv\Scripts\python.exe" (
  "..\TFC-pretraining-dinov3\.venv\Scripts\python.exe" train_offline.py %*
) else (
  python train_offline.py %*
)
pause
