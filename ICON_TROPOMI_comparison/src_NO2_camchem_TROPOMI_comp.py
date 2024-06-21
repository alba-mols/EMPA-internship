# Import packages
import xarray as xr
import numpy as np
import subprocess
import pandas as pd
import glob
import os
import sys
import ddeq
from datetime import datetime as dt

# Import packages for plotting
from matplotlib.collections import PolyCollection 
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt 
import cartopy.feature as cfeature
import cartopy.crs as ccrs


def find_closest_full_hour_TROPOMI(TROPOMI_file):
    # Extract start and end times of the overpass as datetime objects
    # Extract the part after '____'
    file_parts = TROPOMI_file.split('____')[1]

    # Split the file parts by '_'
    time_parts = file_parts.split('_')

    # Extract start and end times
    start_time_str = time_parts[0][9:]  # Extract substring after 'T' in the first part
    end_time_str = time_parts[1][9:]    # Extract substring after 'T' in the second part
    start_time = dt.strptime(start_time_str, '%H%M%S').time()
    end_time = dt.strptime(end_time_str, '%H%M%S').time()

    # Calculate the average time
    average_time = dt.combine(dt.today(), start_time) + (dt.combine(dt.today(), end_time) - dt.combine(dt.today(), start_time)) / 2

    # Round the average time to the nearest hour
    rounded_hour = average_time.replace(minute=0, second=0, microsecond=0)
    closest_hour = rounded_hour.hour

    return(closest_hour)

def camchem_to_TROPOMI(ICON_file, chamchem_file, TROPOMI_file, output_path, date, overpass_nr):
    '''
    Regrid ICON data to TROPOMI grid

    Parameters
    ----------
    ICON_file : str
        Contains the path to the ICON file

    camchem_file : str
        Contains the path to the camchem file

    TROPOMI_file : str
        Contains the path to the TROPOMI file

    output_path : str
        Contains the path to the output folder

    date : str
        Contains the date

    overpass_nr : int
        Since there can be multiple overpasses over europe per day, define the overpass number

    Returns
    -------
    outfile: nc file
        Contains the camchem parameters mapped on the TROPOMI grid

    '''
    cam_chem_ds = xr.open_dataset(chamchem_file)
    ICON_ds = xr.open_dataset(ICON_file) 

    cam_chem_ds['pres_ifc'] = cam_chem_ds['hyai']*cam_chem_ds['P0'] + cam_chem_ds['hybi']*cam_chem_ds['PS'] # Seperating levels
    cam_chem_ds['pres'] = cam_chem_ds['hyam']*cam_chem_ds['P0'] + cam_chem_ds['hybm']*cam_chem_ds['PS'] # level on which the variables are defined
    cam_chem_ds['pres_ifc'] = cam_chem_ds['pres_ifc'].transpose('time', 'ilev', 'ncells')
    cam_chem_ds['pres'] = cam_chem_ds['pres'].transpose('time', 'lev', 'ncells')

    cam_chem_ds['clon_bnds'] = ICON_ds['clon_bnds']
    cam_chem_ds['clat_bnds'] = ICON_ds['clat_bnds']

    cam_chem_ds['tempv'] = cam_chem_ds.T * (1 + 0.608 * cam_chem_ds.Q)

    # Drop unthe necessary variables from camchem ds  
    cam_chem_ds = cam_chem_ds.drop_vars(['hyai', 'hyam','hybi','hybm'])

    output_filename = f"{output_path}reduced_camchem/camchem_{date}_reduced.nc"

    # Delete the file if it already exists, so a new one can be created
    if os.path.exists(output_filename):
        os.remove(output_filename)  # Delete the existing file

    cam_chem_ds.to_netcdf(output_filename)


    class TROPOMIGridProcessor:
        def __init__(self, filename):
            self.filename = filename
            self._read_grid_specification()

        def _read_grid_specification(self):
            dat = xr.Dataset()
            with xr.open_dataset(self.filename, group='PRODUCT/SUPPORT_DATA/GEOLOCATIONS') as nc_file:
                dat['lonc'] = nc_file['longitude_bounds'][0].copy()
                dat['latc'] = nc_file['latitude_bounds'][0].copy()
            self.data = xr.Dataset(
                data_vars=dict(
                    latc=(["nobs", "nrows", "corner"], dat['latc'].values),
                    lonc=(["nobs", "nrows", "corner"], dat['lonc'].values),
                ),
                coords=dict(
                    lat=(['nobs', 'nrows'], dat['latc'].values.mean(-1)),
                    lon=(['nobs', 'nrows'], dat['lonc'].values.mean(-1)),
                ),
                attrs=dict(description="TROPOMI grid specifics"),
            )

        def generate_unstructured_grid_file(self, grid_file):
            self.unstructured = True
            self.grid_file = grid_file
            lon = self.data.lon.values.flatten()
            lat = self.data.lat.values.flatten()
            lonc = self.data.lonc.values.reshape(-1, 4)
            latc = self.data.latc.values.reshape(-1, 4)
            latc = latc[~np.isnan(lon), :]
            lonc = lonc[~np.isnan(lon), :]
            lat = lat[~np.isnan(lon)]
            lon = lon[~np.isnan(lon)]
            npoints = lon.shape[0]

            with open(grid_file, "w") as file1:
                file1.write("# Grid for interpolation \n \n")
                file1.write("gridtype = unstructured \n")
                file1.write("gridsize = %i \n" % (npoints))
                file1.write("nvertex = %i \n" % (4))
                file1.write("# Longitudes \n")
                file1.write("xvals = ")
                for ilon in lon:
                    file1.write(f" {ilon}")
                    file1.write("\n        ")
                file1.write("\n")
                file1.write("# Longitudes of cell corners \n")
                file1.write("xbounds = \n")
                for yy in range(npoints):
                    xbounds = lonc[yy]
                    file1.write(" ".join(map(str, xbounds.astype(float))))
                    file1.write("\n")
                file1.write("\n")
                file1.write("# Latitudes \n")
                file1.write("yvals = ")
                for ilat in lat:
                    file1.write(f" {ilat}")
                    file1.write("\n        ")
                file1.write("\n")
                file1.write("# Latitudes of cell corners \n")
                file1.write("ybounds = \n")
                for yy in range(npoints):
                    ybounds = latc[yy]
                    file1.write(" ".join(map(str, ybounds.astype(float))))
                    file1.write("\n")

        def generate_structured_grid_file(self, grid_file):
            self.unstructured = False
            self.grid_file = grid_file
            lons = self.data.lon.values
            lats = self.data.lat.values
            lonc = self.data.lonc.values
            latc = self.data.latc.values
            ny = self.data.dims["nrows"]
            nx = self.data.dims["nobs"]

            with open(grid_file, "w") as file1:
                file1.write("# Grid for interpolation \n \n")
                file1.write("gridtype = curvilinear  \n")
                file1.write("gridsize = %i \n" % (nx * ny))
                file1.write("xsize = %i \n" % (ny))
                file1.write("ysize = %i \n" % (nx))
                file1.write("# Longitudes \n")
                file1.write("xvals = ")
                for ilon in lons:
                    file1.write(" ".join(map(str, ilon.astype(float))))
                    file1.write("\n        ")
                file1.write("\n")
                file1.write("# Longitudes of cell corners \n")
                file1.write("xbounds = \n")
                for yy in lonc:
                    for xx in yy:
                        file1.write(" ".join(map(str, xx.astype(float))))
                        file1.write("\n        ")
                file1.write("\n")
                file1.write("# Latitudes \n")
                file1.write("yvals = ")
                for ilat in lats:
                    file1.write(" ".join(map(str, ilat.astype(float))))
                    file1.write("\n        ")
                file1.write("\n")
                file1.write("# Latitudes of cell corners \n")
                file1.write("ybounds = \n")
                for yy in latc:
                    for xx in yy:
                        file1.write(" ".join(map(str, xx.astype(float))))
                        file1.write("\n        ")
                    
        def apply_cdo(self, camchem_file, out_file, remapper = 'remapcon', data=['NO2', 'tempv', 'pres', 'pres_ifc', 'clon_bnds', 'clat_bnds', 'T', 'PS']):
            '''
            remapper can be remapcon, remapnn, remapdis, ..., see https://code.mpimet.mpg.de/projects/cdo/embedded/index.html#x1-6900002.12
            data can be string (e.g., "CO2") or a list of strings (e.g., ["CO2", "SO2"]) in which case only these fields will be selected. If left None, all fields are remapped
            '''
            self.out_file = out_file
            apply = f"module load daint-mc; module load CDO; cdo -{remapper},{self.grid_file} {'-select,name=' + data if isinstance(data, str) else '-select,name=' + ','.join(data) if isinstance(data, list) else ''} {camchem_file} {out_file}"
            subprocess.run(apply, shell=True)


        def postprocess(self, final_output_file):
            '''
            Some remapping and merging (bit sketchy and prone to breaking!)
            '''
            if self.unstructured:
                with xr.open_dataset(self.out_file) as remapped:
                    # if fields is not None:
                    #     remapped = remapped[ fields ]
                    cell_idx = np.where(~np.isnan(self.data.lon.values.flatten()))
                    ncells_to_nobsnrows = np.unravel_index(cell_idx, self.data.lon.values.shape)
                    remapped.coords['ncells'] = pd.MultiIndex.from_tuples(
                        list(zip(ncells_to_nobsnrows[0][0], ncells_to_nobsnrows[1][0])),
                        names=['nobs', 'nrows'])
                    remapped = remapped.unstack('ncells')
            else:
                with xr.open_dataset(self.out_file) as remapped:
                    remapped = xr.merge( [remapped.rename({'x': 'nrows', 'y': 'nobs', 'nv4': 'corner'}), self.data] ).load()
            # Fix any encoding issues...
            for item in remapped:
                remapped[item].encoding = {}
            remapped.to_netcdf(final_output_file)
            
    # Example usage
    cam_chem_file_reduced = f"{output_path}reduced_camchem/camchem_{date}_reduced.nc"
    outfile = f"{output_path}camchem_on_TROPOMI/camchem_on_TROPOMI_{date}_{overpass_nr}.nc"

    if os.path.exists(outfile):
        os.remove(outfile)  # Delete the existing file

    processor = TROPOMIGridProcessor(TROPOMI_file)
    processor.generate_structured_grid_file('grid_camchem')
    processor.apply_cdo(cam_chem_file_reduced, outfile)
    processor.postprocess(outfile)



def recalc_AMF(TROPOMI_file, output_path, overpass_nr, date, min_lon, max_lon, min_lat, max_lat):
    '''
    Recalculate the TROPOMI air mass factor (and tropospheric columns) using the camchem vertical NO2 profiles

    Parameters
    ----------

    TROPOMI_file : str
        Contains the path to the TROPOMI file

    output_path : str
        Contains the path to the output folder

    date : str
        Contains the date

    overpass_nr : int
        Since there can be multiple overpasses over europe per day, define the overpass number

    min_lon, max_lon, min_lat, max_lat : int
        Define domain   

    Returns
    -------
    outfile: nc file
        Contains the TROPOMI file including the recalculated NO2 columns

    '''

    ###########################################
    ###### Prepare the TROPOMI and camchem data
    ###########################################

    # Open the TROPOMI and camchem on TROPOMI files
    data_S5p = ddeq.download_S5P.open_netCDF(TROPOMI_file)
    camchem_on_TROPOMI = xr.open_dataset(f"{output_path}camchem_on_TROPOMI/camchem_on_TROPOMI_{date}_{overpass_nr}.nc")

    # Add NO2 profiles to TROPOMI (currently not working)
    # data_S5p = ddeq.download_S5P.add_NO2_profile(data_S5p, path=data_path_raw, delete=False)

    ## Crop file to given domain
    data_S5p = data_S5p.set_coords(("latitude_bounds", "longitude_bounds"))
    mask_lon = (data_S5p.lon >= min_lon) & (data_S5p.lon <= max_lon)
    mask_lat = (data_S5p.lat >= min_lat) & (data_S5p.lat <= max_lat)
    data_S5p_cropped = data_S5p.where(mask_lon & mask_lat, drop=True)

    # Also crop the camchem file to domain
    # This is a work-around using the same TROPOMI mask, because making a seperate camchem mask was not working for all dates...
    mask_arr = (mask_lon & mask_lat)[0].values 
    mask_arr = xr.Dataset({"mask_arr": (["nobs", "nrows"], mask_arr)})
    camchem_on_TROPOMI = camchem_on_TROPOMI.where(mask_arr['mask_arr'], drop=True)

    # Calculate and add pressure levels to the TROPOMI data
    data_S5p_cropped=ddeq.download_S5P.calc_pressure(data_S5p_cropped, 'surface_pressure')

    # Apply quality filter on TROPOMI data
    mask_qa = data_S5p_cropped['qa_value'] < 0.75
    data_S5p_cropped = data_S5p_cropped.where(~mask_qa, drop=True)
    mask_clouds = data_S5p_cropped["cloud_fraction_crb"] > 0.5
    data_S5p_cropped = data_S5p_cropped.where(~mask_clouds, drop=True)

    # Select only needed TROPOMI variables, drop the others
    data_S5p_cropped = data_S5p_cropped.drop_vars([var for var in data_S5p_cropped.variables if var not in ['scanline', 'ground_pixel',
    'layer', 'vertices', 'lat', 'lon', 'latitude_bounds', 'longitude_bounds', 'nitrogendioxide_tropospheric_column', 'averaging_kernel', 
    'air_mass_factor_troposphere', 'air_mass_factor_total', 'tm5_tropopause_layer_index', 'tm5_constant_a', 'tm5_constant_b','surface_altitude', 
    'surface_pressure', 'NO2_profile', 'height_profile', 'mid_pressure_prior_NO2', 'pressure', 'corner']])

    ################################
    ######### Recalculate the AMF
    ################################

    # Using the methodology of interpolation and stretching described in:
    # https://amt.copernicus.org/articles/2/401/2009/amt-2-401-2009.pdf

    #### Step 1: Calculate the NO2 partial columns on the camchem vertical levels 
    # Do this using: Partial column[mol/m2/layer] = Concentration[mol/m3] * layer_depth[m/layer]
    # Where: Concentration[mol/m3] = camchem_NO2[mol/mol] / Mair[kg/mol] * Rho[kg/m3]

    ### Determine the depth of each camchem layer, using the hypsometric equation
    g = 9.8 # Gravity constant [m/s2]
    Rd = 278 # Gas constant [J/kg·K]
    depth = Rd/g * (camchem_on_TROPOMI['tempv'][0]) * np.log(camchem_on_TROPOMI["pres_ifc"][0,1:,:].values/camchem_on_TROPOMI["pres_ifc"][0,0:-1,:].values) # Height in meters

    # Calculate NO2 concentration in each layer [mol/m3]
    # # Using air pressure -- this needs to be used for cam-chem since there is no air density data
    camchem_on_TROPOMI['NO2_conc'] = camchem_on_TROPOMI.NO2 * camchem_on_TROPOMI["pres"] / (8.31*camchem_on_TROPOMI['T']) # Concentration in mol/m3

    # Calculate the NO2 partial columns [mol/m2/layer]
    camchem_on_TROPOMI['NO2_partial_col'] = camchem_on_TROPOMI['NO2_conc']*depth


    #### Step 2: Interpolate the NO2 on the 61 camchem levels to the 34 TM5 levels, using the camchem surface pressure and 
    #### TM5 hybrid pressure level coefficients

    # Add camchem surface pressure to camchem_on_TROPOMI dataset
    data_S5p_cropped['pres_sfc_camchem'] = xr.DataArray(camchem_on_TROPOMI['PS'].values, dims=('time', 'scanline', 'ground_pixel'))

    # Calculate TM5 levels with camchem surface pressure and TM5 hybrid pressure level coefficients (TM5_camchem levels)
    data_S5p_cropped["TM5_p_camchem_full"] = data_S5p_cropped.tm5_constant_a + data_S5p_cropped.tm5_constant_b * data_S5p_cropped['pres_sfc_camchem']

    # Calculate the mid-TM5 levels, on which the TM5 data is actually stored (page 33 of https://sentinel.esa.int/documents/247904/2476257/Sentinel-5P-TROPOMI-ATBD-NO2-data-products)
    data_S5p_cropped["TM5_p_camchem_mid"] = (data_S5p_cropped.variables['TM5_p_camchem_full'][:, 0, 0]+data_S5p_cropped.variables['TM5_p_camchem_full'][:, 1, 0])/2 #For the camchem TM5 levels
    data_S5p_cropped["pressure_mid"] = (data_S5p_cropped.variables['pressure'][:, 0, 0]+data_S5p_cropped.variables['pressure'][:, 1, 0])/2 #Same for the original TM5 pressure levels

    # Interpolate the camchem vertical profiles to the TM5_camchem levels
    # Dimension order: level, scanline, ground_pixel
    camchem_levels = np.flip(camchem_on_TROPOMI.pres_ifc[0,:,:,:].values,axis=0) # Flip camchem levels to start with 0, to be consistent with TM5, and drop time dimension
    target_pressures = data_S5p_cropped["TM5_p_camchem_full"][:,0,0,:,:].values

    # Finds the first index where model lies below target level (so model pressure is higher than target pressure)
    # this function gives a 0 when the sample lies below the lowest model level, and a -1 if the sample lies above the highest model level                   
    above_target = camchem_levels[:,np.newaxis] <= target_pressures
    lev_count=(camchem_levels[:,np.newaxis]>=target_pressures).sum(axis=0)
    first_negative_indices = np.where((lev_count == len(camchem_levels)) | (lev_count == len(camchem_levels) - 1), -1, np.argmax(above_target, axis=0))

    # Select indices of 2 closest neighbouring levels
    vertical_indices = np.stack([first_negative_indices, first_negative_indices-1], axis=0) # Second index thus lies /above/ the target
    vertical_indices[:,first_negative_indices==0] = 0 # If the sample lies below the lowest model level. Set it to the lowest model level !!Note: this also asigns a 0 if the pressure is nan, but for our case, these points are filtered out later anyways
    vertical_indices[:,first_negative_indices==-1] = len(camchem_levels)-2 # If the sample lies above the highest model level. Set it to the highest model level

    # Select pressure values of 2 closest neighbouring levels
    pk_low = np.take_along_axis((camchem_levels), vertical_indices[1], axis=0) #pk, this is the pressure of the lower level (higher pressure)
    pk_up = np.take_along_axis((camchem_levels), vertical_indices[0], axis=0) #pk-1, this is the pressure of the higher level (lower pressure) 

    # Compute the weights of both pressure levels (using the fact that only ln(P) is linear with height)
    alpha_low = (np.log(target_pressures/pk_up))/(np.log(pk_low/pk_up)) # Weights for the lower closest neighbour
    alpha_complete = np.stack([alpha_low, 1-alpha_low], axis=0) # These are now the complete weights for the vertical_indices

    # Correct alpha for levels lower or higher than the model levels
    alpha_complete[:,first_negative_indices==0] = 0 # If sample lies below lowest model level, give all weight to the lowest model level
    alpha_complete[:,first_negative_indices==-1] = 0 # If sample lies above the highest model level, give all weight to the highest model level


    #### Calculate which camchem levels lie fully within the TM5 level
    # Compute the number of full camchem levels falling within the TM5_camchem level
    full_count_levels = (vertical_indices[1,1:]-vertical_indices[0,:-1])
    full_count_levels = np.concatenate((full_count_levels, np.zeros((1, *full_count_levels.shape[1:]))), axis=0)
    full_count_levels = full_count_levels.astype(int)

    # Calculate the maximum number of full_count_levels
    max_levels = np.max(full_count_levels)

    # Initialize an empty list to store arrays
    full_count_levels_arrays = []
    weights_full_levels = []

    # Loop through each level
    for i in range(0, max_levels):
        # Create a new array based on the condition
        new_array = np.where(full_count_levels > i, vertical_indices[0] + i, 0)
        new_array2 = np.where(full_count_levels > i, 1, 0)
        # Append the new array to the list
        full_count_levels_arrays.append(new_array)
        weights_full_levels.append(new_array2)

    # Flip camchem NO2 to match TM5 vertical levels and drop the time dimension
    camchem_subset= np.flip(camchem_on_TROPOMI.NO2_partial_col[0].values, axis=0) 

    # Perform the interpolation using the determined weights
    NO2_interp = np.take_along_axis(camchem_subset, vertical_indices[1], axis=0) * alpha_complete[0]
    # Loop through each level array and its corresponding weight array
    for i in range(0, max_levels):
        interpolated_array = np.take_along_axis(camchem_subset, full_count_levels_arrays[i], axis=0) * weights_full_levels[i]
        NO2_interp += interpolated_array
    # Add the last part of the interpolation
    NO2_interp += np.take_along_axis(camchem_subset, (np.concatenate((vertical_indices[1][1:], np.zeros((1, *vertical_indices[1][1:].shape[1:]))), axis=0)).astype(int), axis=0) * (np.concatenate((alpha_complete[1][1:], np.zeros((1, *alpha_complete[1][1:].shape[1:]))), axis=0)).astype(np.float64) 

    # Add the interpolated camchem NO2 to the TROPOMI dataset
    data_S5p_cropped['NO2_camchem_partial_col_interp'] = (('layer', 'scanline', 'ground_pixel'), NO2_interp)


    #### Step 3: Stretch partial camchem columns to actual TM5 levels (with tropomi surface pressure) using eq 6 from https://amt.copernicus.org/articles/2/401/2009/amt-2-401-2009.pdf
    data_S5p_cropped['NO2_partial_col_stretched'] = data_S5p_cropped['NO2_camchem_partial_col_interp']*((data_S5p_cropped.pressure[:,0,0,:,:]-data_S5p_cropped.pressure[:,1,0,:,:])/(data_S5p_cropped.TM5_p_camchem_full[:,0,0,:,:]-data_S5p_cropped.TM5_p_camchem_full[:,1,0,:,:]))


    #### Step 4: Recalculate the air mass factor (AMF) for tropospheric levels only

    # Extract the tropopause index data
    tropopause_index_data = data_S5p_cropped.tm5_tropopause_layer_index[0].values

    # Calculate the pressure of the tropopause
    valid_mask = ~np.isnan(tropopause_index_data) # Create a mask to identify valid tropopause indices
    pressure_tropopause = np.full_like(tropopause_index_data, np.nan) # Initialize the tropopause pressure array with NaNs
    pressure_tropopause[valid_mask] = data_S5p_cropped.pressure_mid.values[tropopause_index_data.astype(int)[valid_mask],valid_mask]

    # Add 'pressure_tropopause' variable to data_S5p_cropped dataset
    data_S5p_cropped['pressure_tropopause'] = (('scanline', 'ground_pixel'), pressure_tropopause)

    # Also calulate the tropopause pressure on the TM5_camchem levels, we need this later for the camchem tropospheric column calculation
    tropopause_index_data = data_S5p_cropped.tm5_tropopause_layer_index[0].values
    valid_mask = ~np.isnan(tropopause_index_data) # Create a mask to identify valid tropopause indices
    pressure_tropopause_camchem = np.full_like(tropopause_index_data, np.nan) # Initialize the tropopause height array with NaNs
    pressure_tropopause_camchem[valid_mask] = data_S5p_cropped.TM5_p_camchem_mid.values[tropopause_index_data.astype(int)[valid_mask],valid_mask]
    data_S5p_cropped['pressure_tropopause_camchem'] = (('scanline', 'ground_pixel'), pressure_tropopause)

    # Create a mask of only the tropospheric levels
    troposphere_mask = data_S5p_cropped.pressure_mid>=data_S5p_cropped['pressure_tropopause']

    # Calculate the tropospheric Averaging kernel, set it to 0 above the tropopause
    data_S5p_cropped['AK_trop'] = xr.where(troposphere_mask,data_S5p_cropped.averaging_kernel * (data_S5p_cropped.air_mass_factor_total / data_S5p_cropped.air_mass_factor_troposphere),0)

    # Recalculate the AMF
    data_S5p_cropped['AMF_recalc_camchem'] = data_S5p_cropped['air_mass_factor_troposphere'] * ((data_S5p_cropped['AK_trop'][:,:,:,0] * data_S5p_cropped['NO2_partial_col_stretched']).sum(dim='layer', skipna=True) / data_S5p_cropped['NO2_partial_col_stretched'].where(troposphere_mask).sum(dim='layer', skipna=True))

    # Recalculate the vertical NO2 columns
    data_S5p_cropped['NO2_recalc_camchem'] = (data_S5p_cropped['air_mass_factor_troposphere']/data_S5p_cropped['AMF_recalc_camchem'])*data_S5p_cropped.nitrogendioxide_tropospheric_column

    # Save file
    outfile = f"{output_path}recalc_TROPOMI_AMF/recalc_AMF_{date}_{overpass_nr}.nc"

    if os.path.exists(outfile):
        os.remove(outfile)  # Delete the existing file

    data_S5p_cropped.to_netcdf(outfile)


def TROPOMI_to_camchem(camchem_file, output_path, date, overpass_nr):
    '''
    Remap TROPOMI (including recalculated columns), to the camchem grid

    Parameters
    ----------

    camchem_file : str
        Contains the path to the camchem file

    output_path : str
        Contains the path to the output folder

    date : str
        Contains the date

    overpass_nr : int
        Since there can be multiple overpasses over europe per day, define the overpass number

    Returns
    -------
    outfile: nc file
        Contains the TROPOMI NO2 columns on the camchem grid

    '''
    TROPOMI_file = f"{output_path}recalc_TROPOMI_AMF/recalc_AMF_{date}_{overpass_nr}.nc"
    outfile = f"{output_path}TROPOMI_on_camchem/TROPOMI_on_camchem_{date}_{overpass_nr}.nc"

    if os.path.exists(outfile):
        os.remove(outfile)  # Delete the existing file

    #  Extract TROPOMI data
    dat = xr.Dataset()
    with xr.open_dataset(TROPOMI_file) as nc_file:
        dat['pressure_tropopause_camchem'] = nc_file['pressure_tropopause_camchem'].copy() # Here, we automatically "inherit" the longitude/latitude coordinates
        dat['NO2'] = nc_file['nitrogendioxide_tropospheric_column'][0].copy() # Here, we automatically "inherit" the longitude/latitude coordinates
        dat['NO2_recalc'] = nc_file['NO2_recalc_camchem'][0].copy() # Here, we automatically "inherit" the longitude/latitude coordinates
        dat['longitude_bounds'] = nc_file['longitude_bounds'][0].copy()
        dat = dat.set_coords('longitude_bounds')
        dat['latitude_bounds'] = nc_file['latitude_bounds'][0].copy()
    dat["lon"] = dat["lon"].assign_attrs(bounds="longitude_bounds") # Here, we specify where the bounds for these variables are set
    dat["lat"] = dat["lat"].assign_attrs(bounds="latitude_bounds")
    dat.to_netcdf('TROPOMI_extracted_camchem.nc')

    camchem_file_reduced = f'/scratch/snx3000/amols/NO2_col_comp_nudged/column_comp_results_camchem/reduced_camchem/camchem_{date}_reduced.nc'

    # Get CDO grid definition
    command = f'module load daint-mc; module load CDO; cdo selgrid,1 {camchem_file_reduced} triangular_grid_camchem.txt'
    subprocess.run(command, shell=True)

    # Remap
    command = f'module load daint-mc; module load CDO; cdo remapcon,triangular_grid_camchem.txt TROPOMI_extracted_camchem.nc {outfile}'
    subprocess.run(command, shell=True)

    

def calc_camchem_col(output_path, overpass_nr, date):
    '''
    Calculate the camchem tropospheric NO2 columns on the camchem grid

    Parameters
    ----------

    output_path : str
        Contains the path to the output folder

    date : str
        Contains the date

    overpass_nr : int
        Since there can be multiple overpasses over europe per day, define the overpass number

    Returns
    -------
    outfile: nc file
        Contains the camchem and TROPOMI NO2 columns on the camchem grid

    '''
    outfile = f"{output_path}TROPOMI_on_camchem/TROPOMI_on_camchem_{date}_{overpass_nr}.nc"
    TROPOMI_interpolated = xr.open_dataset(outfile)

    camchem_ds = xr.open_dataset(f"{output_path}reduced_camchem/camchem_{date}_reduced.nc")

    ### Determine the depth of each camchem layer, using the hypsometric equation
    g = 9.8 # Gravity constant [m/s2]
    Rd = 278 # Gas constant [J/kg·K]

    # depth=camchem_ds['z_ifc'].values[0:-1,:]-camchem_ds['z_ifc'].values[1:,:]
    depth = Rd/g * (camchem_ds['tempv'][0]) * np.log(camchem_ds["pres_ifc"][0,1:,:].values/camchem_ds["pres_ifc"][0,0:-1,:].values) # Height in meters
    camchem_ds['camchem_layer_depth'] =  xr.DataArray(depth, dims=['lev', 'ncells'])

    # Calculate the NO2 concentration
    #NO2_conc = camchem_ds.NO2_full / M_air * camchem_ds.rho
    NO2_conc = camchem_ds.NO2 * camchem_ds.pres / (8.31*camchem_ds.T)

    # Create a mask to select levels below the tropopause pressure
    below_tropopause_mask = camchem_ds.pres[0,:,:].values > TROPOMI_interpolated.pressure_tropopause_camchem.values

    # Apply the mask to the layer depth and NO2 concentration
    layer_depth_below_tropopause = camchem_ds.camchem_layer_depth.where(below_tropopause_mask, 0)
    NO2_conc_below_tropopause = NO2_conc.where(below_tropopause_mask, 0)

    # Calculate the NO2 total column
    column = (layer_depth_below_tropopause * NO2_conc_below_tropopause).sum(dim='lev', skipna=True)

    # Replace zeros with NaNs
    column = column.where(column != 0, other=np.nan)

    # Assign the calculated column to the dataset
    camchem_ds['NO2_camchem'] = column

    # Merge with TROPOMI NO2 column interpolated to the camchem grid and save
    merged_ds = xr.merge([TROPOMI_interpolated, camchem_ds[['NO2_camchem']]])

    outfile = f"{output_path}final_merged_result/column_comp_{date}_{overpass_nr}.nc"

    if os.path.exists(outfile):
        os.remove(outfile)  # Delete the existing file

    merged_ds.to_netcdf(outfile)



def plot_TROPOMI_map(data, var, minlim, maxlim, title, unit, scale=Normalize(), colormap='viridis',
             west_lon=-13, east_lon=28, south_lat=32, north_lat = 68, save = False):    

    '''
    Plot a map on the TROPOMI grid

    Parameters
    ----------
    data : ds
        Contains the dataset of which a variable is plotted (needed to extract the grid)

    var : 2D array
        Contains the values to be plotted (with both scanline and gr_pixel dimension)

    minlim, maxlim : int
        Defines the range of values to be plotted

    title : str
        Defines the title of the plot

    Unit : str
        Defines the unit of the colorbar

    Scale : str
        LogNorm() will put the data on a logarithmic scale

    colormap : str
        Defines the colormap to be plotted   

    west_lon, east_lon, south_lat, north_lat : int
        Define the plot domain   
        
    Returns
    -------

    Shows the plot of the variable

    '''

    fig = plt.figure(figsize=(5,8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    lonc = data.longitude_bounds.values
    latc = data.latitude_bounds.values

    lonc = np.nan_to_num( lonc, nan = np.nanmin(lonc) )
    latc = np.nan_to_num( latc, nan = np.nanmin(latc) )

    n_cells = data.lat.values.size
    corners = np.zeros((n_cells, 4, 2)) 
    corners[:, :, 0] = lonc.reshape(-1,4)
    corners[:, :, 1] = latc.reshape(-1,4)
    
    poly_coll = PolyCollection( 
            corners, 
            cmap=colormap,
            edgecolors="black", 
            linewidth=0.00, 
            antialiased=False,  # AA will help show the line when set to true 
            alpha=1,
            norm=scale,
    )  

    # Add the collection to the ax 
    ax.add_collection(poly_coll) 

    # Add counties
    ax.add_feature(cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_countries',
            scale='10m',
            facecolor='none',
            edgecolor='black',
            linewidth=0.5))

    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, color='grey', linestyle='--')
    gl.top_labels = False  
    gl.right_labels = False 
    gl.xlabel_style = {'size': 12, 'color': 'black'}
    gl.ylabel_style = {'size': 12, 'color': 'black'}

    # Set the field you want to see
    poly_coll.set_array(var.flatten()) 
    poly_coll.set_clim(vmin=minlim, vmax=maxlim)
    cbar = plt.colorbar(poly_coll,fraction=0.046, pad=0.04)

    cbar.set_label(unit, rotation =270)
    ax.set_title(title)

    ax.set_xlim(west_lon, east_lon)
    ax.set_ylim(south_lat, north_lat)

    # if save == True


def plot_camchem_map(data, var, minlim, maxlim, title, unit, scale=Normalize, colormap='viridis',
             west_lon=-13, east_lon=28, south_lat=32, north_lat = 68, save=False, save_as=None, output_path=None,
             date=None, overpass_nr=None):
    '''
    Plot a map on the camchem grid

    Parameters
    ----------
    data : ds
        Contains the dataset of which a variable is plotted (needed to extract the grid)

    var : 2D array
        Contains the values to be plotted (with both scanline and gr_pixel dimension)

    minlim, maxlim : int
        Defines the range of values to be plotted

    title : str
        Defines the title of the plot

    Unit : str
        Defines the unit of the colorbar

    Scale : str
        LogNorm() will put the data on a logarithmic scale

    colormap : str
        Defines the colormap to be plotted   

    west_lon, east_lon, south_lat, north_lat : int
        Define the plot domain   
        
    Returns
    -------

    Shows the plot of the variable

    '''

    fig = plt.figure(figsize=(5,8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ### Plot results
    camchem_grid = data

    lonc = camchem_grid.clon_bnds.values
    latc = camchem_grid.clat_bnds.values

    n_cells = camchem_grid.clat.values.size
    corners = np.zeros((n_cells, 3, 2)) 
    corners[:, :, 0] = lonc
    corners[:, :, 1] = latc
    corners = np.rad2deg(corners)

    poly_coll = PolyCollection( 
            corners, 
            cmap=colormap, 
            norm=scale(vmin=minlim, vmax=maxlim), 
            edgecolors="red", 
            linewidth=0.0, 
            antialiased=False,  # AA will help show the line when set to true 
            alpha=1, 
    ) 

    # Add the collection to the ax 
    ax.add_collection(poly_coll) 

    # Add counties
    ax.add_feature(cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_countries',
            scale='10m',
            facecolor='none',
            edgecolor='black',
            linewidth=0.5))

    # Set the field you want to see
    poly_coll.set_array(var)  
    cbar = plt.colorbar(poly_coll,fraction=0.046, pad=0.04)
    cbar.set_label(unit, rotation =270)
    ax.set_title(title)
    ax.set_xlim([west_lon, east_lon])
    ax.set_ylim([south_lat, north_lat])

    if save == True:
        plt.savefig(f'{output_path}figs/{save_as}_{date}_{overpass_nr}.png', dpi=300, bbox_inches='tight')
