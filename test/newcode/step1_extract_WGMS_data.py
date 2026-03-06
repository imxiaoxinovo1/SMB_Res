"""
Step 1: Extract and prepare WGMS glacier mass balance data for Region 02 (Western Canada & USA)

Input:
    - WGMS FoG database (glacier.csv, mass_balance.csv)
Output:
    - H:/Code/SMB/test/result_data/wgms_region02_clean.csv

Author: Claude Code
Date: 2025-12-29
"""

import pandas as pd
import numpy as np
import os
import sys

# ================= Configuration =================

# Input paths
WGMS_DATA_DIR = r"H:\Code\SMB\test\data\WGMS\FoG_DataBase\DOI-WGMS-FoG-2025-02b\data"
GLACIER_FILE = os.path.join(WGMS_DATA_DIR, "glacier.csv")
MASS_BALANCE_FILE = os.path.join(WGMS_DATA_DIR, "mass_balance.csv")

# Output path
OUTPUT_DIR = r"H:\Code\SMB\test\result_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "wgms_region02_clean.csv")

# Target region
TARGET_REGION = "02_western_canada_usa"

# ================= Functions =================

def load_csv_safe(file_path, description):
    """Safely load CSV with encoding handling"""
    print(f"\nLoading {description}...")
    print(f"  Path: {file_path}")

    if not os.path.exists(file_path):
        print(f"  ERROR: File not found!")
        sys.exit(1)

    try:
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        print(f"  Warning: UTF-8 failed, trying latin1 encoding...")
        df = pd.read_csv(file_path, encoding='latin1', low_memory=False)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
    return df


def extract_region_glaciers(df_glacier, region_code):
    """Extract glaciers from target region"""
    print(f"\nFiltering glaciers by region: {region_code}")

    region_col = 'gtng_region'
    if region_col not in df_glacier.columns:
        print(f"  ERROR: Column '{region_col}' not found!")
        print(f"  Available columns: {list(df_glacier.columns[:10])}")
        sys.exit(1)

    # Filter by region (case-insensitive)
    mask = df_glacier[region_col].astype(str).str.lower() == region_code.lower()
    df_filtered = df_glacier[mask].copy()

    print(f"  Found {len(df_filtered):,} glaciers in {region_code}")

    if df_filtered.empty:
        print(f"  ERROR: No glaciers found!")
        print(f"  Available regions: {df_glacier[region_col].unique()[:10]}")
        sys.exit(1)

    return df_filtered


def merge_mass_balance_data(df_glacier, df_mb):
    """Merge glacier metadata with mass balance observations"""
    print(f"\nMerging mass balance data...")

    # Rename ID columns for consistency
    df_glacier = df_glacier.rename(columns={'id': 'glacier_id'})

    print(f"  Glacier metadata: {len(df_glacier):,} glaciers")
    print(f"  Mass balance records: {len(df_mb):,} observations")

    # Inner join to keep only records with both metadata and observations
    df_merged = pd.merge(
        df_mb,
        df_glacier,
        on='glacier_id',
        how='inner',
        suffixes=('_mb', '_glacier')
    )

    print(f"  Merged: {len(df_merged):,} mass balance records")

    # Check for duplicate column names
    duplicate_cols = [col for col in df_merged.columns if col.endswith('_mb') or col.endswith('_glacier')]
    if duplicate_cols:
        print(f"  Note: {len(duplicate_cols)} duplicate columns resolved with suffixes")

    return df_merged


def clean_and_standardize(df):
    """Clean and standardize column names and data types"""
    print(f"\nCleaning and standardizing data...")

    # Define standard column names (uppercase for consistency with your existing code)
    column_mapping = {
        'glacier_id': 'WGMS_ID',
        'name_glacier': 'NAME',
        'name_mb': 'NAME',
        'short_name': 'SHORT_NAME',
        'latitude': 'LATITUDE',
        'longitude': 'LONGITUDE',
        'year': 'YEAR',
        'annual_balance': 'ANNUAL_BALANCE',
        'winter_balance': 'WINTER_BALANCE',
        'summer_balance': 'SUMMER_BALANCE',
        'ela': 'ELA',
        'ela_position': 'ELA_POSITION',
        'aar': 'AAR',
        'area': 'AREA',
        'country_glacier': 'COUNTRY',
        'country_mb': 'COUNTRY',
        'gtng_region': 'REGION',
        'time_system': 'TIME_SYSTEM',
        'begin_date': 'BEGIN_DATE',
        'end_date': 'END_DATE',
        'midseason_date': 'MIDSEASON_DATE',
        'annual_balance_unc': 'ANNUAL_BALANCE_UNC',
        'winter_balance_unc': 'WINTER_BALANCE_UNC',
        'summer_balance_unc': 'SUMMER_BALANCE_UNC',
        'ela_unc': 'ELA_UNC',
        'investigators': 'INVESTIGATORS',
        'agencies': 'AGENCIES',
        'references_glacier': 'REFERENCES',
        'references_mb': 'REFERENCES',
        'remarks_glacier': 'REMARKS',
        'remarks_mb': 'REMARKS',
        'glims_id': 'GLIMS_ID',
        'rgi60_ids': 'RGI60_ID',
        'rgi70_ids': 'RGI70_ID'
    }

    # Rename columns that exist
    rename_dict = {}
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            rename_dict[old_col] = new_col

    df = df.rename(columns=rename_dict)

    print(f"  Renamed {len(rename_dict)} columns to standard names")

    # Convert data types
    if 'YEAR' in df.columns:
        df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce').astype('Int64')

    if 'ANNUAL_BALANCE' in df.columns:
        df['ANNUAL_BALANCE'] = pd.to_numeric(df['ANNUAL_BALANCE'], errors='coerce')

    if 'LATITUDE' in df.columns:
        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')

    if 'LONGITUDE' in df.columns:
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')

    if 'AREA' in df.columns:
        df['AREA'] = pd.to_numeric(df['AREA'], errors='coerce')

    # Remove rows with missing critical values
    critical_cols = ['WGMS_ID', 'YEAR', 'LATITUDE', 'LONGITUDE', 'ANNUAL_BALANCE']
    missing_before = len(df)
    df = df.dropna(subset=critical_cols)
    missing_after = len(df)

    if missing_before > missing_after:
        print(f"  Removed {missing_before - missing_after} rows with missing critical data")

    return df


def organize_columns(df):
    """Organize column order for better readability"""
    print(f"\nOrganizing column order...")

    # Define priority column order
    priority_cols = [
        'WGMS_ID', 'NAME', 'SHORT_NAME', 'YEAR',
        'LATITUDE', 'LONGITUDE', 'COUNTRY', 'REGION',
        'ANNUAL_BALANCE', 'ANNUAL_BALANCE_UNC',
        'WINTER_BALANCE', 'WINTER_BALANCE_UNC',
        'SUMMER_BALANCE', 'SUMMER_BALANCE_UNC',
        'ELA', 'ELA_POSITION', 'ELA_UNC', 'AAR', 'AREA',
        'TIME_SYSTEM', 'BEGIN_DATE', 'MIDSEASON_DATE', 'END_DATE',
        'GLIMS_ID', 'RGI60_ID', 'RGI70_ID',
        'INVESTIGATORS', 'AGENCIES', 'REFERENCES', 'REMARKS'
    ]

    # Keep only columns that exist
    existing_priority = [col for col in priority_cols if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_priority]

    final_order = existing_priority + remaining_cols
    df = df[final_order]

    print(f"  Organized {len(existing_priority)} priority columns")

    return df


def generate_summary_stats(df):
    """Generate summary statistics"""
    print(f"\n" + "=" * 70)
    print(f"SUMMARY STATISTICS")
    print(f"=" * 70)

    print(f"\nData Coverage:")
    print(f"  Total records: {len(df):,}")
    print(f"  Unique glaciers: {df['WGMS_ID'].nunique()}")
    print(f"  Time period: {int(df['YEAR'].min())}-{int(df['YEAR'].max())}")
    # Handle COUNTRY column - it might be a weird type
    if 'COUNTRY' in df.columns:
        try:
            country_col = df['COUNTRY']
            if hasattr(country_col, 'iloc'):  # It's a Series
                unique_countries = country_col.dropna().astype(str).drop_duplicates().tolist()
            else:  # It's something else
                unique_countries = list(set(str(x) for x in country_col if pd.notna(x)))
            countries_str = ', '.join(sorted(unique_countries))
            print(f"  Countries: {len(unique_countries)} ({countries_str})")
        except:
            print(f"  Countries: Unable to parse")
    else:
        print(f"  Countries: N/A")

    print(f"\nMass Balance Statistics:")
    print(f"  Mean annual balance: {df['ANNUAL_BALANCE'].mean():.3f} m w.e.")
    print(f"  Std deviation: {df['ANNUAL_BALANCE'].std():.3f} m w.e.")
    print(f"  Min: {df['ANNUAL_BALANCE'].min():.3f} m w.e.")
    print(f"  Max: {df['ANNUAL_BALANCE'].max():.3f} m w.e.")

    if 'AREA' in df.columns:
        print(f"\nGlacier Size:")
        print(f"  Mean area: {df['AREA'].mean()/1e6:.2f} km^2")
        print(f"  Total area: {df['AREA'].sum()/1e6:.2f} km^2")

    print(f"\nTop 10 Glaciers by Number of Observations:")
    # Check which name column exists
    name_col = None
    for col_name in ['NAME', 'SHORT_NAME', 'glacier_name']:
        if col_name in df.columns:
            name_col = col_name
            break

    if name_col:
        top_glaciers = df.groupby(['WGMS_ID', name_col]).size().sort_values(ascending=False).head(10)
        for (wgms_id, name), count in top_glaciers.items():
            print(f"  {name:30s} (ID={wgms_id:5d}): {count:3d} years")
    else:
        top_glaciers = df.groupby('WGMS_ID').size().sort_values(ascending=False).head(10)
        for wgms_id, count in top_glaciers.items():
            print(f"  ID={wgms_id:5d}: {count:3d} years")

    print(f"\nData Completeness:")
    key_cols = ['ANNUAL_BALANCE', 'WINTER_BALANCE', 'SUMMER_BALANCE', 'ELA', 'AAR', 'AREA']
    for col in key_cols:
        if col in df.columns:
            completeness = (1 - df[col].isna().sum() / len(df)) * 100
            print(f"  {col:25s}: {completeness:5.1f}%")


# ================= Main Execution =================

def main():
    print("=" * 70)
    print("STEP 1: Extract and Prepare WGMS Mass Balance Data")
    print("=" * 70)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load data
    df_glacier = load_csv_safe(GLACIER_FILE, "Glacier metadata")
    df_mb = load_csv_safe(MASS_BALANCE_FILE, "Mass balance observations")

    # Step 2: Filter by region
    df_glacier_filtered = extract_region_glaciers(df_glacier, TARGET_REGION)

    # Step 3: Merge datasets
    df_merged = merge_mass_balance_data(df_glacier_filtered, df_mb)

    # Step 4: Clean and standardize
    df_clean = clean_and_standardize(df_merged)

    # Step 5: Organize columns
    df_final = organize_columns(df_clean)

    # Step 6: Generate summary
    generate_summary_stats(df_final)

    # Step 7: Save output
    print(f"\n" + "=" * 70)
    print(f"Saving output...")
    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"  File saved: {OUTPUT_FILE}")
    print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")

    # Show preview
    print(f"\nFirst 5 rows preview:")
    preview_cols = ['WGMS_ID', 'NAME', 'YEAR', 'LATITUDE', 'LONGITUDE', 'ANNUAL_BALANCE']
    preview_cols = [c for c in preview_cols if c in df_final.columns]
    print(df_final[preview_cols].head().to_string(index=False))

    print(f"\n" + "=" * 70)
    print(f"STEP 1 COMPLETED SUCCESSFULLY")
    print(f"=" * 70)
    print(f"\nNext step: Run step2_extract_ERA5_data.py")


if __name__ == "__main__":
    main()
