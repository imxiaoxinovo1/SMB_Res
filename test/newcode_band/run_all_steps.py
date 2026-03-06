"""
Master script to run all 3 steps of elevation band data processing pipeline

This script runs the complete workflow:
1. Extract WGMS elevation band mass balance data
2. Extract ERA5-Land climate data
3. Merge datasets and perform quality control

Author: Claude Code
Date: 2025-12-29
"""

import subprocess
import sys
import os

def run_step(step_num, script_name, description):
    """Run a single step of the pipeline"""
    print("\n" + "=" * 80)
    print(f"RUNNING STEP {step_num}: {description}")
    print("=" * 80 + "\n")

    result = subprocess.run([sys.executable, script_name], cwd=os.path.dirname(__file__))

    if result.returncode != 0:
        print(f"\n❌ ERROR: Step {step_num} failed with exit code {result.returncode}")
        print(f"Please check the error messages above and fix any issues.")
        sys.exit(1)

    print(f"\n✅ Step {step_num} completed successfully!")
    return True


def main():
    print("=" * 80)
    print("ELEVATION BAND DATA PROCESSING PIPELINE")
    print("=" * 80)
    print("\nThis script will run all 3 steps:")
    print("  Step 1: Extract WGMS elevation band mass balance data")
    print("  Step 2: Extract ERA5-Land climate data")
    print("  Step 3: Merge datasets and perform QC")
    print("\nEstimated time: ~5-10 minutes depending on system performance")

    input("\nPress Enter to start, or Ctrl+C to cancel...")

    # Run all steps
    run_step(1, "step1_extract_WGMS_band_data.py", "Extract WGMS Elevation Band Data")
    run_step(2, "step2_extract_ERA5_band_data.py", "Extract ERA5-Land Climate Data")
    run_step(3, "step3_merge_WGMS_ERA5_band.py", "Merge and Quality Control")

    print("\n" + "=" * 80)
    print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nOutput files created:")
    print("  📄 H:/Code/SMB/test/result_data_band/wgms_region02_band_clean.csv")
    print("  📄 H:/Code/SMB/test/result_data_band/era5_climate_band_data.csv")
    print("  📄 H:/Code/SMB/test/result_data_band/wgms_era5_band_merged_final.csv")
    print("  📄 H:/Code/SMB/test/result_data_band/data_quality_report_band.txt")
    print("\nYou can now use the final merged dataset for machine learning modeling!")


if __name__ == "__main__":
    main()
