# READ_ME

This file is provided as requested and mirrors the main project overview.

For full details, use `README.md` in this folder.

## Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Verify all tasks automatically:
```powershell
.\run_all_tasks.ps1
```
3. Verify and launch all Streamlit apps:
```powershell
.\run_all_tasks.ps1 -LaunchApps
```

## Notes
- API keys are loaded from root `.env` for Task 4.
- Optional heavy training for Task 5:
```powershell
.\run_all_tasks.ps1 -RunTask5Training
```
