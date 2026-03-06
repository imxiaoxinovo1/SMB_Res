"""
Step 3: Merge WGMS elevation band mass balance data with ERA5-Land climate data

This script merges elevation band mass balance observations with climate data.
Each elevation band record is matched with climate data for its glacier-year.

Input:
    - H:/Code/SMB/test/result_data_band/wgms_region02_band_clean.csv (from Step 1)
    - H:/Code/SMB/test/result_data_band/era5_climate_band_data.csv (from Step 2)
Output:
    - H:/Code/SMB/test/result_data_band/wgms_era5_band_merged_final.csv
    - H:/Code/SMB/test/result_data_band/data_quality_report_band.txt

Author: Claude Code
Date: 2025-12-29
"""

import pandas as pd
import numpy as np
import os
import sys

# ================= Configuration =================

# Input paths
WGMS_BAND_FILE = r"H:\Code\SMB\test\result_data_band\wgms_region02_band_clean.csv"
CLIMATE_FILE = r"H:\Code\SMB\test\result_data_band\era5_climate_band_data.csv"

# Output path
OUTPUT_DIR = r"H:\Code\SMB\test\result_data_band"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "wgms_era5_band_merged_final.csv")
QC_REPORT_FILE = os.path.join(OUTPUT_DIR, "data_quality_report_band.txt")

# ================= Functions =================

def load_data():
    """Load WGMS band and climate data"""
    print(f"\nLoading input files...")

    if not os.path.exists(WGMS_BAND_FILE):
        print(f"  ERROR: WGMS band file not found: {WGMS_BAND_FILE}")
        sys.exit(1)

    if not os.path.exists(CLIMATE_FILE):
        print(f"  ERROR: Climate file not found: {CLIMATE_FILE}")
        sys.exit(1)

    df_wgms = pd.read_csv(WGMS_BAND_FILE)
    df_climate = pd.read_csv(CLIMATE_FILE)

    print(f"  WGMS band data: {len(df_wgms):,} records")
    print(f"  Climate data: {len(df_climate):,} records")

    return df_wgms, df_climate


def merge_datasets(df_wgms, df_climate):
    """Merge WGMS band and climate data on WGMS_ID and YEAR"""
    print(f"\nMerging datasets...")
    print(f"  Merge keys: WGMS_ID, YEAR")

    # Ensure merge keys have same data type
    df_wgms['WGMS_ID'] = df_wgms['WGMS_ID'].astype(int)
    df_wgms['YEAR'] = df_wgms['YEAR'].astype(int)
    df_climate['WGMS_ID'] = df_climate['WGMS_ID'].astype(int)
    df_climate['YEAR'] = df_climate['YEAR'].astype(int)

    # Perform inner join
    # Each elevation band gets the same climate data for its glacier-year
    df_merged = pd.merge(
        df_wgms,
        df_climate,
        on=['WGMS_ID', 'YEAR'],
        how='inner'
    )

    print(f"  Merged: {len(df_merged):,} elevation band records")
    print(f"  Lost from WGMS: {len(df_wgms) - len(df_merged):,} records")
    print(f"  Match rate: {len(df_merged)/len(df_wgms)*100:.1f}%")
    print(f"  Glacier-year combinations: {df_merged.groupby(['WGMS_ID', 'YEAR']).ngroups:,}")

    return df_merged


def add_derived_features(df):
    """Add derived features specific to elevation band analysis"""
    print(f"\nAdding derived features...")

    derived_count = 0

    # Temperature at 2m in Celsius (from Kelvin)
    temp_cols = [c for c in df.columns if 'temp' in c.lower() and '_celsius' not in c]
    for col in temp_cols:
        if df[col].min() > 100:  # Likely in Kelvin
            new_col = col.replace('temp', 'temp_celsius')
            df[new_col] = df[col] - 273.15
            derived_count += 1

    # Elevation-normalized features (useful for elevation band analysis)
    # Elevation in km (for normalization)
    df['ELEVATION_KM'] = df['ELEVATION_MIDPOINT'] / 1000.0

    # Elevation band normalized by glacier range
    # This will be calculated after we have glacier-level min/max elevations
    glacier_elev_stats = df.groupby('WGMS_ID').agg({
        'LOWER_ELEVATION': 'min',
        'UPPER_ELEVATION': 'max'
    }).rename(columns={
        'LOWER_ELEVATION': 'GLACIER_MIN_ELEV',
        'UPPER_ELEVATION': 'GLACIER_MAX_ELEV'
    })

    df = df.merge(glacier_elev_stats, left_on='WGMS_ID', right_index=True, how='left')

    # Normalized elevation position (0 = lowest band, 1 = highest band)
    df['ELEVATION_NORMALIZED'] = (df['ELEVATION_MIDPOINT'] - df['GLACIER_MIN_ELEV']) / \
                                  (df['GLACIER_MAX_ELEV'] - df['GLACIER_MIN_ELEV'])

    derived_count += 2

    print(f"  Added {derived_count} derived feature groups")

    return df


def quality_control(df):
    """Perform data quality checks"""
    print(f"\nPerforming quality control...")

    qc_report = []
    qc_report.append("=" * 70)
    qc_report.append("DATA QUALITY CONTROL REPORT - ELEVATION BAND DATA")
    qc_report.append("=" * 70)
    qc_report.append("")

    # Check 1: Missing values
    qc_report.append("1. Missing Values Check:")
    critical_cols = ['WGMS_ID', 'YEAR', 'LATITUDE', 'LONGITUDE',
                    'LOWER_ELEVATION', 'UPPER_ELEVATION', 'ANNUAL_BALANCE']
    for col in critical_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                qc_report.append(f"   WARNING: {col} has {missing} missing values")
            else:
                qc_report.append(f"   OK: {col} has no missing values")

    # Check 2: Value ranges
    qc_report.append("\n2. Value Range Check:")

    # Latitude
    if 'LATITUDE' in df.columns:
        out_of_range = ((df['LATITUDE'] < -90) | (df['LATITUDE'] > 90)).sum()
        if out_of_range > 0:
            qc_report.append(f"   WARNING: {out_of_range} records with invalid latitude")
        else:
            qc_report.append(f"   OK: All latitudes in valid range")

    # Longitude
    if 'LONGITUDE' in df.columns:
        out_of_range = ((df['LONGITUDE'] < -180) | (df['LONGITUDE'] > 180)).sum()
        if out_of_range > 0:
            qc_report.append(f"   WARNING: {out_of_range} records with invalid longitude")
        else:
            qc_report.append(f"   OK: All longitudes in valid range")

    # Elevation bands
    invalid_bands = (df['UPPER_ELEVATION'] <= df['LOWER_ELEVATION']).sum()
    if invalid_bands > 0:
        qc_report.append(f"   WARNING: {invalid_bands} bands with upper <= lower elevation")
    else:
        qc_report.append(f"   OK: All elevation bands valid")

    # Annual balance extreme values
    if 'ANNUAL_BALANCE' in df.columns:
        extreme_negative = (df['ANNUAL_BALANCE'] < -10).sum()
        extreme_positive = (df['ANNUAL_BALANCE'] > 5).sum()
        if extreme_negative > 0:
            qc_report.append(f"   INFO: {extreme_negative} bands with mass balance < -10 m w.e. (extreme loss)")
        if extreme_positive > 0:
            qc_report.append(f"   INFO: {extreme_positive} bands with mass balance > +5 m w.e. (extreme gain)")

    # Check 3: Duplicates
    qc_report.append("\n3. Duplicate Check:")
    duplicates = df.duplicated(subset=['WGMS_ID', 'YEAR', 'LOWER_ELEVATION', 'UPPER_ELEVATION']).sum()
    if duplicates > 0:
        qc_report.append(f"   WARNING: {duplicates} duplicate band records")
    else:
        qc_report.append(f"   OK: No duplicates found")

    # Check 4: Elevation band coverage per glacier-year
    qc_report.append("\n4. Elevation Band Coverage:")
    bands_per_gy = df.groupby(['WGMS_ID', 'YEAR']).size()
    qc_report.append(f"   Average bands per glacier-year: {bands_per_gy.mean():.1f}")
    qc_report.append(f"   Min bands: {bands_per_gy.min()}")
    qc_report.append(f"   Max bands: {bands_per_gy.max()}")

    # Check 5: Data completeness by variable type
    qc_report.append("\n5. Data Completeness by Variable Type:")

    var_groups = {
        'Mass Balance': ['ANNUAL_BALANCE', 'WINTER_BALANCE', 'SUMMER_BALANCE'],
        'Elevation Info': ['LOWER_ELEVATION', 'UPPER_ELEVATION', 'ELEVATION_MIDPOINT'],
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
    metadata_cols = ['WGMS_ID', 'NAME', 'SHORT_NAME', 'GLIMS_ID', 'RGI60_ID', 'RGI70_ID']
    geography_cols = ['LATITUDE', 'LONGITUDE', 'COUNTRY', 'REGION']
    time_cols = ['YEAR']
    elevation_cols = [
        'LOWER_ELEVATION', 'UPPER_ELEVATION', 'ELEVATION_MIDPOINT',
        'ELEVATION_RANGE', 'ELEVATION_KM', 'ELEVATION_NORMALIZED',
        'GLACIER_MIN_ELEV', 'GLACIER_MAX_ELEV'
    ]
    mass_balance_cols = [
        'ANNUAL_BALANCE', 'ANNUAL_BALANCE_UNC',
        'WINTER_BALANCE', 'WINTER_BALANCE_UNC',
        'SUMMER_BALANCE', 'SUMMER_BALANCE_UNC',
        'BAND_AREA'
    ]
    climate_year_cols = sorted([c for c in df.columns if c.endswith('_year')])
    climate_summer_cols = sorted([c for c in df.columns if c.endswith('_summer')])
    reference_cols = ['REMARKS']

    # Combine all groups
    all_priority_cols = (
        metadata_cols +
        geography_cols +
        time_cols +
        elevation_cols +
        mass_balance_cols +
        climate_year_cols +
        climate_summer_cols +
        reference_cols
    )

    # Keep only columns that exist
    existing_priority = [c for c in all_priority_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in existing_priority]

    final_order = existing_priority + remaining
    df = df[final_order]

    print(f"  Organized {len(existing_priority)} columns in priority order")
    print(f"  Additional columns: {len(remaining)}")

    return df


def generate_final_summary(df):
    """Generate comprehensive summary statistics"""
    print(f"\n" + "=" * 70)
    print(f"FINAL MERGED ELEVATION BAND DATASET SUMMARY")
    print(f"=" * 70)

    print(f"\nDataset Size:")
    print(f"  Total elevation band records: {len(df):,}")
    print(f"  Total variables: {len(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    print(f"\nTemporal Coverage:")
    print(f"  Start year: {int(df['YEAR'].min())}")
    print(f"  End year: {int(df['YEAR'].max())}")
    print(f"  Time span: {int(df['YEAR'].max() - df['YEAR'].min() + 1)} years")

    print(f"\nSpatial Coverage:")
    print(f"  Number of glaciers: {df['WGMS_ID'].nunique()}")
    print(f"  Glacier-year combinations: {df.groupby(['WGMS_ID', 'YEAR']).ngroups:,}")
    try:
        countries = df['COUNTRY'].value_counts()
        print(f"  Countries: {len(countries)} ({', '.join(countries.index)})")
    except:
        pass
    print(f"  Latitude range: {df['LATITUDE'].min():.2f}deg to {df['LATITUDE'].max():.2f}deg")
    print(f"  Longitude range: {df['LONGITUDE'].min():.2f}deg to {df['LONGITUDE'].max():.2f}deg")

    print(f"\nElevation Band Statistics:")
    print(f"  Elevation range: {int(df['LOWER_ELEVATION'].min())}-{int(df['UPPER_ELEVATION'].max())} m a.s.l.")
    print(f"  Average band midpoint: {df['ELEVATION_MIDPOINT'].mean():.1f} m")
    print(f"  Average band width: {df['ELEVATION_RANGE'].mean():.1f} m")
    print(f"  Bands per glacier-year (avg): {len(df) / df.groupby(['WGMS_ID', 'YEAR']).ngroups:.1f}")

    print(f"\nMass Balance Statistics (All Bands):")
    print(f"  Mean annual balance: {df['ANNUAL_BALANCE'].mean():.3f} m w.e.")
    print(f"  Std deviation: {df['ANNUAL_BALANCE'].std():.3f} m w.e.")
    print(f"  Median: {df['ANNUAL_BALANCE'].median():.3f} m w.e.")
    print(f"  Range: [{df['ANNUAL_BALANCE'].min():.3f}, {df['ANNUAL_BALANCE'].max():.3f}] m w.e.")

    print(f"\nVariable Categories:")
    var_types = {
        'WGMS Metadata': len([c for c in df.columns if c in ['WGMS_ID', 'NAME', 'GLIMS_ID', 'RGI60_ID', 'RGI70_ID']]),
        'Geography': len([c for c in df.columns if c in ['LATITUDE', 'LONGITUDE', 'COUNTRY', 'REGION']]),
        'Elevation Bands': len([c for c in df.columns if 'ELEVATION' in c or 'BAND' in c]),
        'Mass Balance': len([c for c in df.columns if 'BALANCE' in c]),
        'Climate (Annual)': len([c for c in df.columns if c.endswith('_year')]),
        'Climate (Summer)': len([c for c in df.columns if c.endswith('_summer')])
    }

    for category, count in var_types.items():
        print(f"  {category:20s}: {count:3d} variables")

    print(f"\nTop 5 Glaciers by Band Records:")
    top5 = df.groupby('WGMS_ID').size().sort_values(ascending=False).head(5)
    for wgms_id, count in top5.items():
        try:
            if 'SHORT_NAME' in df.columns:
                name = df[df['WGMS_ID'] == wgms_id]['SHORT_NAME'].iloc[0]
            else:
                name = f"Glacier {wgms_id}"
        except:
            name = f"Glacier {wgms_id}"
        print(f"  {name:30s} (ID={wgms_id:5d}): {count:4d} band records")


# ================= Main Execution =================

def main():
    print("=" * 70)
    print("STEP 3: Merge WGMS Band and ERA5-Land Data")
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

    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"  Main file: {OUTPUT_FILE}")
    print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")

    with open(QC_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(qc_report))
    print(f"  QC report: {QC_REPORT_FILE}")

    # Show preview
    print(f"\nFirst 5 elevation bands preview:")
    preview_cols = ['WGMS_ID', 'SHORT_NAME', 'YEAR', 'LOWER_ELEVATION', 'UPPER_ELEVATION',
                   'ELEVATION_MIDPOINT', 'ANNUAL_BALANCE', 'temp_2m_year', 'precip_total_year']
    preview_cols = [c for c in preview_cols if c in df_final.columns]
    print(df_final[preview_cols].head(5).to_string(index=False))

    print(f"\n" + "=" * 70)
    print(f"STEP 3 COMPLETED SUCCESSFULLY")
    print(f"=" * 70)
    print(f"\nFinal elevation band dataset ready for machine learning!")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"\nNext steps:")
    print(f"  - Review quality control report: {QC_REPORT_FILE}")
    print(f"  - Analyze mass balance gradients with elevation")
    print(f"  - Train ML models using elevation band features")


if __name__ == "__main__":
    main()
