"""
Step 1: Extract WGMS elevation band mass balance data for RGI Region 02

This script processes WGMS Fluctuations of Glaciers (FoG) database to extract
elevation band mass balance data for western Canada and USA (RGI Region 02).

Input:
    - H:/Code/SMB/test/data/WGMS/FoG_DataBase/DOI-WGMS-FoG-2025-02b/data/
      - glacier.csv
      - mass_balance_band.csv
Output:
    - H:/Code/SMB/test/result_data_band/wgms_region02_band_clean.csv

Author: Claude Code
Date: 2025-12-29
"""

import pandas as pd
import numpy as np
import os
import sys

# ================= Configuration =================

# Input paths
DATA_DIR = r"H:\Code\SMB\test\data\WGMS\FoG_DataBase\DOI-WGMS-FoG-2025-02b\data"
GLACIER_FILE = os.path.join(DATA_DIR, "glacier.csv")
BAND_FILE = os.path.join(DATA_DIR, "mass_balance_band.csv")

# Output path
OUTPUT_DIR = r"H:\Code\SMB\test\result_data_band"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "wgms_region02_band_clean.csv")

# Target region
TARGET_REGION = "02_western_canada_usa"

# ================= Helper Functions =================

def load_csv_safe(file_path, description):
    """Safely load CSV file with error handling"""
    print(f"\nLoading {description}...")
    print(f"  Path: {file_path}")

    if not os.path.exists(file_path):
        print(f"  ERROR: File not found!")
        sys.exit(1)

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"  Loaded: {len(df):,} records")
        return df
    except Exception as e:
        print(f"  ERROR: Failed to load file: {e}")
        sys.exit(1)


def extract_region_glaciers(df_glacier, region_code):
    """Extract glaciers from target region"""
    print(f"\nFiltering glaciers by region: {region_code}")

    # Filter by gtng_region
    df_region = df_glacier[df_glacier['gtng_region'] == region_code].copy()

    print(f"  Found {len(df_region):,} glaciers in region {region_code}")

    return df_region


def merge_band_data(df_glacier_region, df_band):
    """Merge glacier metadata with elevation band mass balance data"""
    print(f"\nMerging glacier data with elevation band mass balance...")

    # Rename glacier_id to id for joining
    df_band_renamed = df_band.rename(columns={'glacier_id': 'id'})

    # Merge on glacier id
    df_merged = pd.merge(
        df_band_renamed,
        df_glacier_region[['id', 'short_name', 'latitude', 'longitude', 'gtng_region',
                          'glims_id', 'rgi60_ids', 'rgi70_ids']],
        on='id',
        how='inner'
    )

    print(f"  Merged records: {len(df_merged):,}")
    print(f"  Unique glaciers: {df_merged['id'].nunique()}")
    print(f"  Year range: {int(df_merged['year'].min())}-{int(df_merged['year'].max())}")

    return df_merged


def clean_and_standardize(df):
    """Clean data and standardize column names"""
    print(f"\nCleaning and standardizing data...")

    # Rename columns to uppercase standard names
    rename_map = {
        'id': 'WGMS_ID',
        'glacier_name': 'NAME',
        'short_name': 'SHORT_NAME',
        'country': 'COUNTRY',
        'year': 'YEAR',
        'latitude': 'LATITUDE',
        'longitude': 'LONGITUDE',
        'gtng_region': 'REGION',
        'glims_id': 'GLIMS_ID',
        'rgi60_ids': 'RGI60_ID',
        'rgi70_ids': 'RGI70_ID',
        'lower_elevation': 'LOWER_ELEVATION',
        'upper_elevation': 'UPPER_ELEVATION',
        'area': 'BAND_AREA',
        'winter_balance': 'WINTER_BALANCE',
        'winter_balance_unc': 'WINTER_BALANCE_UNC',
        'summer_balance': 'SUMMER_BALANCE',
        'summer_balance_unc': 'SUMMER_BALANCE_UNC',
        'annual_balance': 'ANNUAL_BALANCE',
        'annual_balance_unc': 'ANNUAL_BALANCE_UNC',
        'remarks': 'REMARKS'
    }

    df = df.rename(columns=rename_map)

    # Convert data types
    df['WGMS_ID'] = df['WGMS_ID'].astype(int)
    df['YEAR'] = df['YEAR'].astype(int)
    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
    df['LOWER_ELEVATION'] = pd.to_numeric(df['LOWER_ELEVATION'], errors='coerce')
    df['UPPER_ELEVATION'] = pd.to_numeric(df['UPPER_ELEVATION'], errors='coerce')
    df['ANNUAL_BALANCE'] = pd.to_numeric(df['ANNUAL_BALANCE'], errors='coerce')

    # Add derived feature: elevation band midpoint
    df['ELEVATION_MIDPOINT'] = (df['LOWER_ELEVATION'] + df['UPPER_ELEVATION']) / 2

    # Add elevation band width
    df['ELEVATION_RANGE'] = df['UPPER_ELEVATION'] - df['LOWER_ELEVATION']

    # Remove records with missing critical values
    initial_count = len(df)
    df = df.dropna(subset=['WGMS_ID', 'YEAR', 'LATITUDE', 'LONGITUDE',
                           'LOWER_ELEVATION', 'UPPER_ELEVATION', 'ANNUAL_BALANCE'])
    removed = initial_count - len(df)

    if removed > 0:
        print(f"  Removed {removed:,} records with missing critical values")

    # Sort by glacier ID, year, and elevation
    df = df.sort_values(['WGMS_ID', 'YEAR', 'LOWER_ELEVATION']).reset_index(drop=True)

    print(f"  Final cleaned records: {len(df):,}")

    return df


def generate_summary(df):
    """Generate summary statistics"""
    print(f"\n" + "=" * 70)
    print(f"DATA SUMMARY")
    print(f"=" * 70)

    print(f"\nDataset Size:")
    print(f"  Total elevation band records: {len(df):,}")
    print(f"  Unique glaciers: {df['WGMS_ID'].nunique()}")
    print(f"  Unique glacier-year combinations: {df.groupby(['WGMS_ID', 'YEAR']).ngroups:,}")

    print(f"\nTemporal Coverage:")
    print(f"  Year range: {int(df['YEAR'].min())}-{int(df['YEAR'].max())}")
    print(f"  Time span: {int(df['YEAR'].max() - df['YEAR'].min() + 1)} years")

    print(f"\nSpatial Coverage:")
    try:
        unique_countries = df['COUNTRY'].value_counts()
        print(f"  Countries: {len(unique_countries)}")
        for country, count in unique_countries.items():
            print(f"    {country}: {count:,} band records")
    except:
        print(f"  Country information not available")

    print(f"  Latitude range: {df['LATITUDE'].min():.2f}deg to {df['LATITUDE'].max():.2f}deg")
    print(f"  Longitude range: {df['LONGITUDE'].min():.2f}deg to {df['LONGITUDE'].max():.2f}deg")

    print(f"\nElevation Band Statistics:")
    print(f"  Elevation range: {int(df['LOWER_ELEVATION'].min())}-{int(df['UPPER_ELEVATION'].max())} m a.s.l.")
    print(f"  Average band midpoint: {df['ELEVATION_MIDPOINT'].mean():.1f} m a.s.l.")
    print(f"  Average band width: {df['ELEVATION_RANGE'].mean():.1f} m")
    print(f"  Bands per glacier-year (avg): {len(df) / df.groupby(['WGMS_ID', 'YEAR']).ngroups:.1f}")

    print(f"\nMass Balance Statistics (All Bands):")
    print(f"  Mean annual balance: {df['ANNUAL_BALANCE'].mean():.3f} m w.e.")
    print(f"  Std deviation: {df['ANNUAL_BALANCE'].std():.3f} m w.e.")
    print(f"  Median: {df['ANNUAL_BALANCE'].median():.3f} m w.e.")
    print(f"  Range: [{df['ANNUAL_BALANCE'].min():.3f}, {df['ANNUAL_BALANCE'].max():.3f}] m w.e.")

    # Check winter/summer balance availability
    winter_available = df['WINTER_BALANCE'].notna().sum()
    summer_available = df['SUMMER_BALANCE'].notna().sum()
    print(f"\nSeasonal Balance Availability:")
    print(f"  Winter balance: {winter_available:,} records ({winter_available/len(df)*100:.1f}%)")
    print(f"  Summer balance: {summer_available:,} records ({summer_available/len(df)*100:.1f}%)")

    print(f"\nTop 5 Glaciers by Elevation Band Records:")
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
    print("STEP 1: Extract WGMS Elevation Band Mass Balance Data")
    print("=" * 70)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load glacier metadata
    df_glacier = load_csv_safe(GLACIER_FILE, "glacier metadata")

    # Step 2: Load elevation band mass balance data
    df_band = load_csv_safe(BAND_FILE, "elevation band mass balance")

    # Step 3: Filter glaciers by region
    df_glacier_region = extract_region_glaciers(df_glacier, TARGET_REGION)

    # Step 4: Merge datasets
    df_merged = merge_band_data(df_glacier_region, df_band)

    # Step 5: Clean and standardize
    df_clean = clean_and_standardize(df_merged)

    # Step 6: Generate summary
    generate_summary(df_clean)

    # Step 7: Save output
    print(f"\n" + "=" * 70)
    print(f"Saving output...")
    df_clean.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"  File saved: {OUTPUT_FILE}")
    print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")

    # Show preview
    print(f"\nFirst 10 rows preview:")
    preview_cols = ['WGMS_ID', 'SHORT_NAME', 'YEAR', 'LOWER_ELEVATION',
                   'UPPER_ELEVATION', 'ELEVATION_MIDPOINT', 'ANNUAL_BALANCE']
    preview_cols = [c for c in preview_cols if c in df_clean.columns]
    print(df_clean[preview_cols].head(10).to_string(index=False))

    print(f"\n" + "=" * 70)
    print(f"STEP 1 COMPLETED SUCCESSFULLY")
    print(f"=" * 70)
    print(f"\nNext step: Run step2_extract_ERA5_band_data.py")


if __name__ == "__main__":
    main()
