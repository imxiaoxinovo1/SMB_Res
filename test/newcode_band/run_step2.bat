@echo off
REM Run Step 2: Extract ERA5-Land climate data for elevation bands

echo ======================================================================
echo Running Step 2: Extract ERA5-Land Climate Data
echo ======================================================================

REM Activate conda environment
call G:\Study\Anaconda\Scripts\activate
call conda activate smb

REM Run step 2
python step2_extract_ERA5_band_data.py

pause
