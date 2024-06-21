# Internship project at Empa

This folder contains 2 main things:
1) Script to compare ICON-ART and CAM-Chem output to TROPOMI NO2 observations
2) Script to compare ICON-ART to CAM-Chem output

Additionally, it contains a land-sea mask for a 13x13 ICON European grid, and a script to interpolate CAM-Chem 6-hourly files to daily files.

## 1) ICON-ART/CAM-Chem comparison to TROPOMI observations

### What does the script do?

It compares ICON-ART and CAM-Chem output to TROPOMI NO2 vertical tropospheric columns, after recalculating the AMF in the TROPOMI product by replacing the vertical a priori vertical NO2 columns from the TM5 model by our ICON-ART vertical profiles. Since TROPOMI data is on a different grid than ICON-ART/CAM-Chem, the script remmaps the data to match, and allows for a comparison.

Step 1. Remap ICON to TROPOMI
Step 2. Recalculate AMF + TROPOMI NO2 columns
Step 3. Remap TROPOMI to ICON grid
Step 4. Calculate ICON NO2 columns

Folders are automatically created in which the intermediate results are stored

I documented the whole theory behind all the steps in a report, which is available at Empa or ask by sending an email to alba.v.mols@hotmail.com

### Usage

Files needed:
- TROPOMI NO2 raw files (can be downloaded using e.g. the ddeq package)
- ICON-ART output files
- OR CAM-Chem output files -> if not in hourly resolution, can be interpolated using the script "interpolate_camchem_hourly.ipynb"

## 2) Script to compare ICON-ART to CAM-Chem output

### What do the scripts do?

The first script, "vertical_interp.ipynb" extracts the model output at the right height.
The second script "generate_maps.ipynb" generates maps of the two models at different heights.

### Usage

Files needed:
- ICON-ART output files
- CAM-Chem output files

First run the "vertical_interp.ipynb" script to extract the model output at the right height. Then run "generate_maps.ipynb" to make maps of the two models and compare them, at different heights.