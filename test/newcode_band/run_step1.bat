@echo off
REM Run Step 1: Extract WGMS elevation band data

echo ======================================================================
echo Running Step 1: Extract WGMS Elevation Band Data
echo ======================================================================

REM Activate conda environment
call G:\Study\Anaconda\Scripts\activate
call conda activate smb

REM Run step 1
python step1_extract_WGMS_band_data.py

pause
