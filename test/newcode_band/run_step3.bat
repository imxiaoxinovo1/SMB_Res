@echo off
REM Run Step 3: Merge WGMS and ERA5 elevation band data

echo ======================================================================
echo Running Step 3: Merge WGMS and ERA5 Band Data
echo ======================================================================

REM Activate conda environment
call G:\Study\Anaconda\Scripts\activate
call conda activate smb

REM Run step 3
python step3_merge_WGMS_ERA5_band.py

pause
