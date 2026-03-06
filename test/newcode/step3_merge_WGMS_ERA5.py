"""
Step 3: Merge WGMS mass balance data with ERA5-Land climate data

Input:
    - H:/Code/SMB/test/result_data/wgms_region02_clean.csv (from Step 1)
    - H:/Code/SMB/test/result_data/era5_climate_data.csv (from Step 2)
Output:
    - H:/Code/SMB/test/result_data/wgms_era5_merged_final.csv

Author: Claude Code
Date: 2025-12-29
"""

import pandas as pd
import numpy as np
import os
import sys

# ================= Configuration =================

# Input paths
WGMS_FILE = r"H:\Code\SMB\test\result_data\wgms_region02_clean.csv"
CLIMATE_FILE = r"H:\Code\SMB\test\result_data\era5_climate_data.csv"

# Output path
OUTPUT_DIR = r"H:\Code\SMB\test\result_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "wgms_era5_merged_final.csv")
QC_REPORT_FILE = os.path.join(OUTPUT_DIR, "data_quality_report.txt")

# ================= Functions =================

def load_data():
    """Load WGMS and climate data"""
    print(f"\nLoading input files...")

    # Check files exist
    if not os.path.exists(WGMS_FILE):
        print(f"  ERROR: WGMS file not found: {WGMS_FILE}")
        print(f"  Please run step1_extract_WGMS_data.py first")
        sys.exit(1)

    if not os.path.exists(CLIMATE_FILE):
        print(f"  ERROR: Climate file not found: {CLIMATE_FILE}")
        print(f"  Please run step2_extract_ERA5_data.py first")
        sys.exit(1)

    df_wgms = pd.read_csv(WGMS_FILE)
    df_climate = pd.read_csv(CLIMATE_FILE)

    print(f"  WGMS data: {len(df_wgms):,} records")
    print(f"  Climate data: {len(df_climate):,} records")

    return df_wgms, df_climate


def merge_datasets(df_wgms, df_climate):
    """Merge WGMS and climate data on WGMS_ID and YEAR"""
    print(f"\nMerging datasets...")
    print(f"  Merge keys: WGMS_ID, YEAR")

    # Ensure merge keys have same data type
    df_wgms['WGMS_ID'] = df_wgms['WGMS_ID'].astype(int)
    df_wgms['YEAR'] = df_wgms['YEAR'].astype(int)
    df_climate['WGMS_ID'] = df_climate['WGMS_ID'].astype(int)
    df_climate['YEAR'] = df_climate['YEAR'].astype(int)

    # Perform inner join (keep only records with both mass balance and climate data)
    df_merged = pd.merge(
        df_wgms,
        df_climate,
        on=['WGMS_ID', 'YEAR'],
        how='inner'
    )

    print(f"  Merged: {len(df_merged):,} records")
    print(f"  Lost from WGMS: {len(df_wgms) - len(df_merged):,} records")
    print(f"  Match rate: {len(df_merged)/len(df_wgms)*100:.1f}%")

    return df_merged


def add_derived_features(df):
    """Add derived geographical and climatic features"""
    print(f"\nAdding derived features...")

    derived_count = 0

    # Elevation range (if we have upper and lower bounds from RGI)
    # Note: These columns might not exist yet, will be added when merging RGI data
    # We'll prepare the logic here

    # For now, add simple derived features from available data

    # Example: Convert Kelvin to Celsius if temperature is in Kelvin
    temp_cols = [c for c in df.columns if 'temp' in c.lower() and not 'temp_celsius' in c]
    for col in temp_cols:
        if df[col].min() > 100:  # Likely in Kelvin
            new_col = col.replace('temp', 'temp_celsius')
            df[new_col] = df[col] - 273.15
            derived_count += 1

    print(f"  Added {derived_count} derived features")

    return df


def quality_control(df):
    """Perform data quality checks"""
    print(f"\nPerforming quality control...")

    qc_report = []
    qc_report.append("=" * 70)
    qc_report.append("DATA QUALITY CONTROL REPORT")
    qc_report.append("=" * 70)
    qc_report.append("")

    # Check 1: Missing values
    qc_report.append("1. Missing Values Check:")
    critical_cols = ['WGMS_ID', 'YEAR', 'LATITUDE', 'LONGITUDE', 'ANNUAL_BALANCE']
    for col in critical_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                qc_report.append(f"   WARNING: {col} has {missing} missing values")
            else:
                qc_report.append(f"   OK: {col} has no missing values")

    # Check 2: Value ranges
    qc_report.append("\n2. Value Range Check:")

    # Latitude should be between -90 and 90
    if 'LATITUDE' in df.columns:
        out_of_range = ((df['LATITUDE'] < -90) | (df['LATITUDE'] > 90)).sum()
        if out_of_range > 0:
            qc_report.append(f"   WARNING: {out_of_range} records with invalid latitude")
        else:
            qc_report.append(f"   OK: All latitudes in valid range")

    # Longitude should be between -180 and 180
    if 'LONGITUDE' in df.columns:
        out_of_range = ((df['LONGITUDE'] < -180) | (df['LONGITUDE'] > 180)).sum()
        if out_of_range > 0:
            qc_report.append(f"   WARNING: {out_of_range} records with invalid longitude")
        else:
            qc_report.append(f"   OK: All longitudes in valid range")

    # Annual balance should be reasonable (typically -5 to +2 m w.e.)
    if 'ANNUAL_BALANCE' in df.columns:
        extreme_negative = (df['ANNUAL_BALANCE'] < -5).sum()
        extreme_positive = (df['ANNUAL_BALANCE'] > 2).sum()
        if extreme_negative > 0:
            qc_report.append(f"   INFO: {extreme_negative} records with mass balance < -5 m w.e. (extreme loss)")
        if extreme_positive > 0:
            qc_report.append(f"   INFO: {extreme_positive} records with mass balance > +2 m w.e. (extreme gain)")

    # Check 3: Duplicates
    qc_report.append("\n3. Duplicate Check:")
    duplicates = df.duplicated(subset=['WGMS_ID', 'YEAR']).sum()
    if duplicates > 0:
        qc_report.append(f"   WARNING: {duplicates} duplicate WGMS_ID-YEAR combinations")
    else:
        qc_report.append(f"   OK: No duplicates found")

    # Check 4: Data completeness by variable type
    qc_report.append("\n4. Data Completeness by Variable Type:")

    var_groups = {
        'Mass Balance': ['ANNUAL_BALANCE', 'WINTER_BALANCE', 'SUMMER_BALANCE'],
        'Temperature': [c for c in df.columns if 'temp' in c.lower()],
        'Precipitation': [c for c in df.columns if 'precip' in c.lower() or 'snowfall' in c.lower()],
        'Snow': [c for c in df.columns if 'snow' in c.lower() and 'snowfall' not in c.lower()],
        'Radiation': [c for c in df.columns if 'radiation' in c.lower()],
        'Hydrology': [c for c in df.columns if 'runoff' in c.lower() or 'evaporation' in c.lower()]
    }

    for group_name, cols in var_groups.items():
        existing_cols = [c for c in cols if c in df.columns]
        if existing_cols:
            completeness = []
            for col in existing_cols:
                comp = (1 - df[col].isna().sum() / len(df)) * 100
                completeness.append(comp)
            avg_completeness = np.mean(completeness)
            qc_report.append(f"   {group_name:20s}: {avg_completeness:5.1f}% complete ({len(existing_cols)} variables)")

    # Print to console
    print("\n".join(qc_report))

    return qc_report


def organize_final_columns(df):
    """Organize columns in a logical order"""
    print(f"\nOrganizing column order...")

    # Define column groups
    metadata_cols = [
        'WGMS_ID', 'NAME', 'SHORT_NAME', 'GLIMS_ID', 'RGI60_ID', 'RGI70_ID'
    ]

    geography_cols = [
        'LATITUDE', 'LONGITUDE', 'COUNTRY', 'REGION'
    ]

    glacier_props_cols = [
        'AREA', 'ELA', 'AAR'
    ]

    time_cols = [
        'YEAR', 'TIME_SYSTEM', 'BEGIN_DATE', 'MIDSEASON_DATE', 'END_DATE'
    ]

    mass_balance_cols = [
        'ANNUAL_BALANCE', 'ANNUAL_BALANCE_UNC',
        'WINTER_BALANCE', 'WINTER_BALANCE_UNC',
        'SUMMER_BALANCE', 'SUMMER_BALANCE_UNC',
        'ELA_POSITION', 'ELA_UNC'
    ]

    # Climate variables (all ending in _year or _summer)
    climate_year_cols = sorted([c for c in df.columns if c.endswith('_year')])
    climate_summer_cols = sorted([c for c in df.columns if c.endswith('_summer')])

    # Reference cols
    reference_cols = [
        'INVESTIGATORS', 'AGENCIES', 'REFERENCES', 'REMARKS'
    ]

    # Combine all groups
    all_priority_cols = (
        metadata_cols +
        geography_cols +
        time_cols +
        mass_balance_cols +
        glacier_props_cols +
        climate_year_cols +
        climate_summer_cols +
        reference_cols
    )

    # Keep only columns that exist
    existing_priority = [c for c in all_priority_cols if c in df.columns]

    # Add any remaining columns
    remaining = [c for c in df.columns if c not in existing_priority]

    final_order = existing_priority + remaining
    df = df[final_order]

    print(f"  Organized {len(existing_priority)} columns in priority order")
    print(f"  Additional columns: {len(remaining)}")

    return df


def generate_final_summary(df):
    """Generate comprehensive summary statistics"""
    print(f"\n" + "=" * 70)
    print(f"FINAL MERGED DATASET SUMMARY")
    print(f"=" * 70)

    print(f"\nDataset Size:")
    print(f"  Total records: {len(df):,}")
    print(f"  Total variables: {len(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    print(f"\nTemporal Coverage:")
    print(f"  Start year: {int(df['YEAR'].min())}")
    print(f"  End year: {int(df['YEAR'].max())}")
    print(f"  Time span: {int(df['YEAR'].max() - df['YEAR'].min() + 1)} years")

    print(f"\nSpatial Coverage:")
    print(f"  Number of glaciers: {df['WGMS_ID'].nunique()}")
    print(f"  Countries: {df['COUNTRY'].nunique()} ({', '.join(sorted(df['COUNTRY'].unique()))})")
    print(f"  Latitude range: {df['LATITUDE'].min():.2f}° to {df['LATITUDE'].max():.2f}°")
    print(f"  Longitude range: {df['LONGITUDE'].min():.2f}° to {df['LONGITUDE'].max():.2f}°")

    print(f"\nMass Balance Statistics:")
    print(f"  Mean annual balance: {df['ANNUAL_BALANCE'].mean():.3f} m w.e.")
    print(f"  Std deviation: {df['ANNUAL_BALANCE'].std():.3f} m w.e.")
    print(f"  Median: {df['ANNUAL_BALANCE'].median():.3f} m w.e.")
    print(f"  Range: [{df['ANNUAL_BALANCE'].min():.3f}, {df['ANNUAL_BALANCE'].max():.3f}] m w.e.")

    print(f"\nVariable Categories:")
    var_types = {
        'WGMS Metadata': len([c for c in df.columns if c in ['WGMS_ID', 'NAME', 'GLIMS_ID', 'RGI60_ID', 'RGI70_ID']]),
        'Geography': len([c for c in df.columns if c in ['LATITUDE', 'LONGITUDE', 'COUNTRY', 'REGION']]),
        'Mass Balance': len([c for c in df.columns if 'BALANCE' in c or c in ['ELA', 'AAR']]),
        'Climate (Annual)': len([c for c in df.columns if c.endswith('_year')]),
        'Climate (Summer)': len([c for c in df.columns if c.endswith('_summer')]),
        'Reference': len([c for c in df.columns if c in ['INVESTIGATORS', 'AGENCIES', 'REFERENCES', 'REMARKS']])
    }

    for category, count in var_types.items():
        print(f"  {category:20s}: {count:3d} variables")

    print(f"\nTop 5 Glaciers by Observations:")
    # Group by WGMS_ID only (NAME may not exist after Step 1 cleaning)
    top5 = df.groupby('WGMS_ID').size().sort_values(ascending=False).head(5)
    for wgms_id, count in top5.items():
        # Try to get name if it exists
        if 'NAME' in df.columns:
            name = df[df['WGMS_ID'] == wgms_id]['NAME'].iloc[0] if 'NAME' in df.columns else 'N/A'
        elif 'SHORT_NAME' in df.columns:
            name = df[df['WGMS_ID'] == wgms_id]['SHORT_NAME'].iloc[0]
        else:
            name = f"Glacier {wgms_id}"
        print(f"  {name:30s} (ID={wgms_id:5d}): {count:3d} years")


# ================= Main Execution =================

def main():
    print("=" * 70)
    print("STEP 3: Merge WGMS and ERA5-Land Data")
    print("=" * 70)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load data
    df_wgms, df_climate = load_data()

    # Step 2: Merge datasets
    df_merged = merge_datasets(df_wgms, df_climate)

    # Step 3: Add derived features
    df_enhanced = add_derived_features(df_merged)

    # Step 4: Quality control
    qc_report = quality_control(df_enhanced)

    # Step 5: Organize columns
    df_final = organize_final_columns(df_enhanced)

    # Step 6: Generate final summary
    generate_final_summary(df_final)

    # Step 7: Save outputs
    print(f"\n" + "=" * 70)
    print(f"Saving outputs...")

    # Save merged data
    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"  Main file: {OUTPUT_FILE}")
    print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")

    # Save QC report
    with open(QC_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(qc_report))
    print(f"  QC report: {QC_REPORT_FILE}")

    # Show preview
    print(f"\nFirst 3 rows preview (selected columns):")
    preview_cols = ['WGMS_ID', 'YEAR', 'LATITUDE', 'LONGITUDE',
                    'ANNUAL_BALANCE', 'temp_2m_year', 'precip_total_year', 'snowfall_year']
    # Add NAME or SHORT_NAME if exists
    if 'NAME' in df_final.columns:
        preview_cols.insert(1, 'NAME')
    elif 'SHORT_NAME' in df_final.columns:
        preview_cols.insert(1, 'SHORT_NAME')

    preview_cols = [c for c in preview_cols if c in df_final.columns]
    print(df_final[preview_cols].head(3).to_string(index=False))

    print(f"\n" + "=" * 70)
    print(f"STEP 3 COMPLETED SUCCESSFULLY")
    print(f"=" * 70)
    print(f"\nFinal merged dataset ready for machine learning!")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"\nNext steps:")
    print(f"  - Review quality control report: {QC_REPORT_FILE}")
    print(f"  - Run exploratory data analysis")
    print(f"  - Train machine learning models")


if __name__ == "__main__":
    main()
