@echo off
REM Run Random Forest Model with Leave-One-Year-Out Cross-Validation

echo ======================================================================
echo Running Random Forest Model - LOOCV
echo ======================================================================

REM Activate conda environment
call G:\Study\Anaconda\Scripts\activate
call conda activate smb

REM Run model
python RF_model_LOOCV.py

pause
