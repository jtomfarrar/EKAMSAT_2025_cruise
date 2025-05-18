# Plot processed UCTD data from EKAMSAT 2025 Leg-1 cruise on the R/V Thompson

# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs                   # import projections
import cartopy
import geopandas as gpd # for the EEZ shapefile
import functions
import pandas as pd

import cartopy.crs as ccrs                   # import projections
import cartopy
import geopandas as gpd # for the EEZ shapefile
from matplotlib.patheffects import Stroke
# %%
# Change to this directory
home_dir = os.path.expanduser("~")

# To work for Tom and other people
if os.path.exists(home_dir + '/Python/EKAMSAT_2025_cruise/src'):
    os.chdir(home_dir + '/Python/EKAMSAT_2025_cruise/src')

# %%
# %matplotlib inline
%matplotlib widget
# %matplotlib qt5
plt.rcParams['figure.figsize'] = (6,6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 400

savefig = True # set to true to save plots as file

__figdir__ = '../img/UCTD/'
os.system('mkdir  -p ' + __figdir__) #make directory if it doesn't exist
savefig_args = {'bbox_inches':'tight', 'pad_inches':0.2}
kml_savefig_args = {'bbox_inches':'tight', 'pad_inches':0, 'transparent':True}
plotfiletype='png'

# %%
data_dir_uctd = '/mnt/c/d_drive/EKAMSAT_2025_cruise_data/science_share/UCTD/Processed_Data/'
data_dir = '../data/external'

url = 'https://psl.noaa.gov/thredds/fileServer/Datasets/noaa.oisst.v2.highres/sst.day.mean.2025.nc'


sst_filename = os.path.join(data_dir, os.path.basename(url))


# %%
# Make a map of A76A positions and its shards
# Add new waypoints to wpt
wpt = dict(lon=[88.83, 88.83, 86.48, 86.50, 86.50, 86.50, 87.5, 88.5], lat=[12.33, 15, 13.87, 13.32, 12.80, 12.30, 13.25, 12.0])

# WG recovery waypoint
wpt['lon'] += [87.67]
wpt['lat'] += [13.25]


# %%
# Read theEEZ  shapefile
shapefile = data_dir + '/World_EEZ_v12_20231025/eez_v12.shp'
gdf = gpd.read_file(shapefile)


# %%


# Get ship position from this url:
# https://www.ocean.washington.edu/files/thompson.txt
ship_pos_url = 'https://www.ocean.washington.edu/files/thompson.txt'
ship_pos = pd.read_csv(ship_pos_url, sep=',', header=None, names=['asset', 'DD-MM-YYYY', 'HH:MM:SS', 'lat', 'lon', 'T', 'NULL1', 'NULL2', 'BPR', 'NULL3', 'NULL4', 'NULL5', 'NULL6', 'NULL7', 'NULL8', 'NULL9', 'Voyage'])
print(ship_pos.head())
latitudes = ship_pos['lat']
longitudes = ship_pos['lon']
# %%

# %%
zoom = True
domovie = False
if zoom:
    xmin, xmax = (85,90)
    ymin, ymax = (10.5,15)
else:
    xmin, xmax = (70,100)
    ymin, ymax = (0, 25)
# %%
xmin_op, xmax_op = (70,100)
ymin_op, ymax_op = (0, 25)
######################
# %%
# Load the data

data = xr.open_dataset(sst_filename)

# %% Shift sst to have longitude from -180 to 180
data = data.assign_coords(lon=(data.lon + 180) % 360 - 180)
data = data.sortby(data.lon)
# Make one plot for the last time in the file
t1 = data.time[-1].values
fstr = t1.astype('datetime64[D]')
t1 = str(fstr)
sst = data.sst.sel(lon=slice(xmin,xmax), lat=slice(ymin,ymax),time=t1)

# %%
# Load the UCTD data
ds = xr.open_dataset(data_dir_uctd + 'ASTRAL25_binneduCTD_combined.nc')
# %%
## Plot the data
def plot_map(data, levels, xmin, xmax, ymin, ymax, xmin_op, xmax_op, ymin_op, ymax_op, ax=None):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})  # Create a new axis if none is provided
    coast = cartopy.feature.GSHHSFeature(scale="full")
    ax.add_feature(coast, zorder=3, facecolor=[.6, .6, .6], edgecolor='black')

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    # Plot the data
    cs = ax.pcolormesh(data.lon, data.lat, data, vmin=levels[0], vmax=levels[-1], cmap='turbo', transform=ccrs.PlateCarree())
    cb = plt.colorbar(cs, ax=ax, fraction=0.022, extend='both')
    cb.set_label('SST [$\\circ$C]', fontsize=10)
    ax.axis('scaled')

    # Add country boundaries
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5, zorder=10)
    ax.add_feature(cartopy.feature.NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='50m', facecolor='none'), zorder=10)
    ax.add_feature(cartopy.feature.RIVERS, edgecolor='blue', zorder=10, alpha=0.25)
    gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=2, alpha=0.9, transform=ccrs.PlateCarree(), zorder=2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # Plot the data
    ax.axis('scaled')

    # Create an inset GeoAxes showing the location of the operating area.
    sub_ax = plt.axes([0.74, 0.64, 0.2, 0.2], projection=ccrs.PlateCarree())
    sub_ax.set_extent([xmin_op, xmax_op, ymin_op, ymax_op])

    # Make a nice border around the inset axes.
    effect = Stroke(linewidth=2, foreground='wheat', alpha=0.5)
    #sub_ax.outline_patch.set_path_effects([effect])

    # Add the land, coastlines and the extent of the operating area.
    sub_ax.add_feature(cartopy.feature.LAND)
    sub_ax.coastlines()
    # sub_ax.add_feature(cartopy.feature.STATES, zorder=3, linewidth=0.5)
    coord = [[xmin,ymin], [xmax,ymin], [xmax,ymax], [xmin,ymax]]
    coord.append(coord[0]) #repeat the first point to create a 'closed loop'
    xs, ys = zip(*coord) #create lists of x and y values
    sub_ax.plot(xs,ys, transform=ccrs.PlateCarree(), linewidth=2)
    #plot_ops_area(sub_ax,transform=ccrs.PlateCarree(),color='k')
    gdf.plot(ax=sub_ax, color='none', edgecolor='black', linewidth=0.5, alpha=0.9, transform=ccrs.PlateCarree(), zorder=2)


    # return both axes

    return ax, sub_ax
ds_ssh = xr.open_dataset('../data/external/aviso.nc')

# %%
# Reverted the add_vel_quiver function to its original state

def add_vel_quiver(tind, ax=plt.gca()):
    if ax is None:
        ax = plt.gca()

    u = np.squeeze(ds_ssh.ugos.isel(time=tind))  # dtype=object
    v = np.squeeze(ds_ssh.vgos.isel(time=tind))
    skip = 2
    scalefac = 10
    q = ax.quiver(ds_ssh.longitude.values[::skip], ds_ssh.latitude.values[::skip], u.values[::skip, ::skip], v.values[::skip, ::skip], scale=scalefac, transform=ccrs.PlateCarree())
    x0 = 81.5
    y0 = 17.33
    ax.quiverkey(q, x0, y0, 0.25, '0.25 m/s', zorder=5, transform=ccrs.PlateCarree())

# %%\
# Plot a map
levels = np.arange(29.5, 32, 0.25)
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})  # Create a new axis if none is provided
ax, sub_ax = plot_map(sst, levels, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xmin_op=xmin_op, xmax_op=xmax_op, ymin_op=ymin_op, ymax_op=ymax_op, ax=ax)
# Add the ship's track
ax.plot(ds.lon, ds.lat, '.', markersize=0.75, color='k', transform=ccrs.PlateCarree(), zorder=4)
ax.plot(0,0, color='k', label='UCTD locations', transform=ccrs.PlateCarree(), zorder=4)
 
# Also plot it in the inset axis
sub_ax.plot(ds.lon, ds.lat, '.', markersize=0.1, color='k', transform=ccrs.PlateCarree(), zorder=4)
# set axes limits
ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())
# Add EEZ boundaries
# restore axis limits
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.legend(loc='upper left', fontsize=12, frameon=True)
tind=-1
add_vel_quiver(tind, ax=ax)
plt.show()

if savefig:
    plt.savefig(__figdir__ + 'UCTD_map.' + plotfiletype, **savefig_args)
    print('Saved figure to ' + __figdir__ + 'UCTD_map.' + plotfiletype)

# %%
# Find large time gaps in the data
time_diff = np.diff(ds.dday.values)
large_gaps = np.where(time_diff > np.timedelta64(1, 'h'))[0]

# Make the depth profile at the beginning and end of those gaps NaNs
for gap in large_gaps:
    ds.T[:, gap] = np.nan  # Beginning of the gap
    ds.T[:, gap + 1] = np.nan  # End of the gap
    ds.S[:, gap] = np.nan
    ds.S[:, gap + 1] = np.nan
    ds.sig0[:, gap] = np.nan
    ds.sig0[:, gap + 1] = np.nan

# %%
# Plot a depth-time section of the UCTD data
fig, ax = plt.subplots(2,1,figsize=(8, 6),sharex=True, sharey=True)
levels = np.arange(20, 35, 1)

c = ax[0].pcolormesh(ds.dday, ds.depth, ds.T, cmap='turbo', vmin=levels[0], vmax=levels[-1], shading='auto')
#reverse the y-axis
ax[0].set_ylim(200, ds.depth.min())
ax[0].set_ylabel('Depth [m]')
ax[0].set_title('Temperature [°C]')
plt.colorbar(c, ax=ax[0], label='Temperature [°C]')
# make the time axis look nicer
levels = np.arange(32.5, 35, .05)
fig.autofmt_xdate()
C2 = ax[1].pcolormesh(ds.dday, ds.depth, ds.S, cmap='viridis', vmin=levels[0], vmax=levels[-1], shading='auto')
ax[1].set_ylim(200, ds.depth.min())
ax[1].set_ylabel('Salinity [PSU]')
ax[1].set_title('Salinity [PSU]')
plt.colorbar(C2, ax=ax[1], label='Salinity [PSU]')
fig.tight_layout()


# %%
# Plot the time from 05-12-2025 05:00 to 05-12-2025 23:30, but against longitude

# Find the index of the time range
start_time = np.datetime64('2025-05-12T05:00:00')
end_time = np.datetime64('2025-05-12T23:30:00')
start_index = np.where(ds.dday.values > start_time)[0][0]
end_index = np.where(ds.dday.values < end_time)[-1][-1]
# Get the data for the time range
ds_subset = ds.isel(dday=slice(start_index, end_index))
# Plot the data
fig, ax = plt.subplots(2,1,figsize=(8, 6),sharex=True, sharey=True)
levels = np.arange(20, 35, 1)
c = ax[0].pcolormesh(ds_subset.lon, ds_subset.depth, ds_subset.T, cmap='turbo', vmin=levels[0], vmax=levels[-1], shading='auto')
#reverse the y-axis
ax[0].set_ylim(200, ds_subset.depth.min())
ax[0].set_ylabel('Depth [m]')
ax[0].set_title('Temperature [°C]')
plt.colorbar(c, ax=ax[0], label='Temperature [°C]')
# make the time axis look nicer
levels = np.arange(32.5, 35, .05)
fig.autofmt_xdate()
C2 = ax[1].pcolormesh(ds_subset.lon, ds_subset.depth, ds_subset.S, cmap='viridis', vmin=levels[0], vmax=levels[-1], shading='auto')
ax[1].set_ylim(200, ds_subset.depth.min())
ax[1].set_ylabel('Salinity [PSU]')
ax[1].set_title('Salinity [PSU]')
ax[1]
plt.colorbar(C2, ax=ax[1], label='Salinity [PSU]')
fig.tight_layout()

if savefig:
    plt.savefig(__figdir__ + 'UCTD_section_lon.' + plotfiletype, **savefig_args)
    print('Saved figure to ' + __figdir__ + 'UCTD_section_lon.' + plotfiletype)
# %%
# Plot density on the section
fig, ax = plt.subplots(2,1,figsize=(8, 6),sharex=True, sharey=True)
levels = np.arange(19.5, 21.5, .05)
c = ax[0].pcolormesh(ds_subset.lon, ds_subset.depth, ds_subset.sig0, cmap='turbo', vmin=levels[0], vmax=levels[-1], shading='auto')
#reverse the y-axis
ax[0].set_ylim(200, ds_subset.depth.min())
ax[0].set_ylabel('Depth [m]')
ax[0].set_title('Potential density [kg/m^3]')
plt.colorbar(c, ax=ax[0], label='$\sigma_0$ [kg/m^3]')
# make the time axis look nicer
levels = np.arange(32.5, 35, .05)
C2 = ax[1].pcolormesh(ds_subset.lon, ds_subset.depth, ds_subset.S, cmap='viridis', vmin=levels[0], vmax=levels[-1], shading='auto')
ax[1].set_ylim(200, ds_subset.depth.min())
ax[1].set_ylabel('Salinity [PSU]')
ax[1].set_title('Salinity [PSU]')
ax[1]
plt.colorbar(C2, ax=ax[1], label='Salinity [PSU]')
fig.tight_layout()
if savefig:
    plt.savefig(__figdir__ + 'UCTD_section_lon_density.' + plotfiletype, **savefig_args)
    print('Saved figure to ' + __figdir__ + 'UCTD_section_lon_density.' + plotfiletype)
# %%
