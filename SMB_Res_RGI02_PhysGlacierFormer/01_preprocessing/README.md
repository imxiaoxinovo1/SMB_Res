# Preprocessing

This folder will contain only reproducible data-building scripts for the clean RGI02 experiment.

Planned order:

1. `step01_filter_wgms.py`: filter WGMS annual SMB records for RGI02.
2. `step02_match_rgi.py`: match WGMS glaciers to RGI terrain attributes.
3. `step03_extract_era5.py`: extract monthly ERA5-Land climate sequences.
4. `step04_build_tabular.py`: build the tabular baseline dataset.
5. `step05_build_sequences.py`: build normalized sequence tensors.
6. `step06_build_qc_sequences.py`: remove poor WGMS-to-RGI matches and rebuild tensors.
7. `step07_build_hypsometry_sequences.py`: add elevation-band features.
8. `step08_build_hugonnet_weak_labels.py`: map Hugonnet geodetic mass-change rates to RGI02 weak labels.
9. Future remote-sensing step: add albedo or snow-cover features.
