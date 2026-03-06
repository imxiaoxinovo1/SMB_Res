"""
Master script to run all data processing steps

This script executes the complete data preparation pipeline:
    Step 1: Extract WGMS mass balance data for Region 02
    Step 2: Extract ERA5-Land climate data for glacier locations
    Step 3: Merge WGMS and ERA5 datasets

Author: Claude Code
Date: 2025-12-29
"""

import subprocess
import sys
import os
from datetime import datetime

NEWCODE_DIR = r"H:\Code\SMB\test\newcode"

STEPS = [
    {
        'name': 'Step 1: Extract WGMS Data',
        'script': 'step1_extract_WGMS_data.py',
        'description': 'Extract and clean WGMS mass balance data for Region 02'
    },
    {
        'name': 'Step 2: Extract ERA5 Data',
        'script': 'step2_extract_ERA5_data.py',
        'description': 'Extract ERA5-Land climate variables for glacier locations'
    },
    {
        'name': 'Step 3: Merge Datasets',
        'script': 'step3_merge_WGMS_ERA5.py',
        'description': 'Merge WGMS mass balance with ERA5 climate data'
    }
]


def run_step(step_info):
    """Run a single processing step"""
    print("\n" + "=" * 80)
    print(f"{step_info['name']}")
    print("=" * 80)
    print(f"Description: {step_info['description']}")
    print(f"Script: {step_info['script']}")
    print("")

    script_path = os.path.join(NEWCODE_DIR, step_info['script'])

    if not os.path.exists(script_path):
        print(f"ERROR: Script not found: {script_path}")
        return False

    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        # Print output
        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print("STDERR:", result.stderr)

        # Check if successful
        if result.returncode == 0:
            print(f"\n✓ {step_info['name']} completed successfully")
            return True
        else:
            print(f"\n✗ {step_info['name']} failed with return code {result.returncode}")
            return False

    except Exception as e:
        print(f"\n✗ Error running {step_info['name']}: {str(e)}")
        return False


def main():
    """Run all processing steps"""
    print("=" * 80)
    print("WGMS + ERA5-Land Data Processing Pipeline")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {NEWCODE_DIR}")
    print("")

    # Track success
    results = []

    # Run each step
    for i, step in enumerate(STEPS, 1):
        print(f"\n{'#' * 80}")
        print(f"# Running Step {i}/{len(STEPS)}")
        print(f"{'#' * 80}")

        success = run_step(step)
        results.append({
            'step': step['name'],
            'success': success
        })

        if not success:
            print(f"\n{'!' * 80}")
            print(f"! Pipeline stopped at Step {i} due to error")
            print(f"{'!' * 80}")
            break

    # Print summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    for i, result in enumerate(results, 1):
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"  Step {i}: {result['step']:50s} {status}")

    # Overall status
    all_success = all(r['success'] for r in results)
    print("")
    if all_success:
        print("=" * 80)
        print("✓ ALL STEPS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nFinal output file:")
        print("  H:/Code/SMB/test/result_data/wgms_era5_merged_final.csv")
        print("\nYou can now proceed with machine learning model training.")
    else:
        print("=" * 80)
        print("✗ PIPELINE INCOMPLETE - PLEASE CHECK ERRORS ABOVE")
        print("=" * 80)

    return all_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
