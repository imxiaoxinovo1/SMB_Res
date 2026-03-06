@echo off
REM Run all steps of the elevation band data processing pipeline

echo ======================================================================
echo ELEVATION BAND DATA PROCESSING PIPELINE
echo ======================================================================

REM Activate conda environment
call G:\Study\Anaconda\Scripts\activate
call conda activate smb

REM Run all steps
python run_all_steps.py

pause
