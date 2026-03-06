@echo off
REM Run Step 2: Extract ERA5-Land climate data

echo ======================================================================
echo Running Step 2: Extract ERA5-Land climate data
echo ======================================================================

REM Activate conda environment
call G:\Study\Anaconda\Scripts\activate
call conda activate smb

REM Run step 2
python step2_extract_ERA5_data.py

pause
