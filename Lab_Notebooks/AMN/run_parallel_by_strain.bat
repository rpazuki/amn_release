@echo off
REM Batch script to run the parallel dataset generation by strain
REM This activates the conda environment and runs the Python script
REM 
REM Usage:
REM   run_parallel_by_strain.bat [size_i] [reduce]
REM 
REM Arguments:
REM   size_i  - Sample size for COBRA training set (default: 1)
REM   reduce  - Reduce dataset: true/false (default: true)
REM 
REM Examples:
REM   run_parallel_by_strain.bat          - Use defaults (size_i=1, reduce=true)
REM   run_parallel_by_strain.bat 10       - Use size_i=10, reduce=true
REM   run_parallel_by_strain.bat 10 false - Use size_i=10, reduce=false

echo Activating conda environment...
call conda activate C:\Users\rh2310\projects\amn_release\.env

echo Running parallel dataset generation by strain...

REM Build the command with optional arguments
set CMD=python build_dataset_parallel_by_strain.py
if not "%1"=="" set CMD=%CMD% --size_i %1
if not "%2"=="" set CMD=%CMD% --reduce %2
if not "%3"=="" set CMD=%CMD% --levels %3
if not "%4"=="" set CMD=%CMD% --max_val %4
if not "%5"=="" set CMD=%CMD% --original_medium %5

echo Command: %CMD%
%CMD%

echo.
echo Script completed. Press any key to exit...
pause
