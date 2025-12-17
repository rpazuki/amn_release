@echo off
REM Batch script to run the parallel dataset generation
REM This activates the conda environment and runs the Python script

echo Activating conda environment...
call conda activate C:\Users\rh2310\projects\amn_release\.env

echo Running parallel dataset generation...
python build_dataset_parallel.py

echo.
echo Script completed. Press any key to exit...
pause
