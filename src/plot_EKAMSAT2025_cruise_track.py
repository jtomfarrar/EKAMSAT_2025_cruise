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
# ALoad waypoints from EKAMSAT_2025_cruise/data/external/waypoints_cleaned_degmin_decimal_rounded.xlsx
waypoints = pd.read_excel('../data/external/waypoints_cleaned_degmin_decimal_rounded.xlsx', sheet_name='Sheet1')
wpt = dict(lon=waypoints['Longitude (decimal)'].values, lat=waypoints['Latitude (decimal)'].values, number=waypoints['Waypoint number'].values)


# %%
# Read theEEZ  shapefile
shapefile = data_dir + '/World_EEZ_v12_20231025/eez_v12.shp'
gdf = gpd.read_file(shapefile)


# %%
# Load ADCP file for ship positions
ds_adcp = xr.open_dataset(data_dir + '/os75nb.nc')
# add Phuket as the first and last lat/lon/time,     7.819318°   98.406202°
# Note lon and lat are variables not coordinates
phuket1 = dict(lon=98.406202, lat=7.819318, time=pd.Timestamp('2025-05-03'))
phuket2 = dict(lon=98.406202, lat=7.819318, time=pd.Timestamp('2025-05-15'))
# Create a new dataset for Phuket data
phuket_ds1 = xr.Dataset(
    {
        "lon": ("time", [phuket1["lon"]]),
        "lat": ("time", [phuket1["lat"]]),
    },
    coords={"time": [phuket1["time"]]}
)
phuket_ds2 = xr.Dataset(
    {
        "lon": ("time", [phuket2["lon"]]),
        "lat": ("time", [phuket2["lat"]]),
    },
    coords={"time": [phuket2["time"]]}
)
# Concatenate the Phuket data with your existing dataset
ds_adcp = xr.concat([phuket_ds1, ds_adcp], dim="time")
ds_adcp = xr.concat([ds_adcp, phuket_ds2], dim="time")

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
    xmin, xmax = (82,92)
    ymin, ymax = (10,19)
else:
    xmin, xmax = (70,100)
    ymin, ymax = (0, 25)
# %%
xmin_op, xmax_op = (70,105)
ymin_op, ymax_op = (0, 23)
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
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5, zorder=3)
    ax.add_feature(cartopy.feature.NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='50m', facecolor='none'), zorder=3)
    ax.add_feature(cartopy.feature.RIVERS, edgecolor='blue', zorder=3, alpha=0.25)
    gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=2, alpha=0.9, transform=ccrs.PlateCarree(), zorder=2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # Plot the data
    ax.axis('scaled')

    # Create an inset GeoAxes showing the location of the operating area.
    sub_ax = plt.axes([0.6, 0.6, 0.25, 0.25], projection=ccrs.PlateCarree())
    sub_ax.set_extent([xmin_op, xmax_op, ymin_op, ymax_op])

    # Make a nice border around the inset axes.
    effect = Stroke(linewidth=2, foreground='wheat', alpha=0.5)
    #sub_ax.outline_patch.set_path_effects([effect])

    # Add the land, coastlines and the extent of the operating area.
    sub_ax.add_feature(cartopy.feature.LAND)
    sub_ax.coastlines()
    sub_ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5, zorder=5)

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
    skip = 3
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
ax.plot(ds_adcp.lon, ds_adcp.lat, color='m', transform=ccrs.PlateCarree(), zorder=10)
#ax.plot(0,0, color='k', label='UCTD locations', transform=ccrs.PlateCarree(), zorder=4)
h_wpt = ax.plot(wpt['lon'], wpt['lat'], 'o', color='grey', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='Waypoints',zorder=9)
# add waypoint number directly on each waypoint
#for i, txt in enumerate(wpt['number']):
#    ax.annotate(txt, (wpt['lon'][i], wpt['lat'][i]), textcoords="offset points", xytext=(0, -1), ha='center', fontsize=8, color='k', zorder=6)
# Also plot it in the inset axis
# sub_ax.plot(ds.lon, ds.lat, '.', markersize=0.1, color='k', transform=ccrs.PlateCarree(), zorder=4)
sub_ax.plot(ds_adcp.lon, ds_adcp.lat, color='m', transform=ccrs.PlateCarree(), zorder=6)
# set axes limits
ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())
# Add EEZ boundaries
# restore axis limits
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
sst_timestr = sst.time.values.astype('datetime64[D]')
sst_timestr = str(sst_timestr)
ax.set_title('EKAMSAT 2025 Cruise Track (SST from '+sst_timestr+')', fontsize=14)
ax.legend(loc='upper left', fontsize=12, frameon=True, edgecolor='k', facecolor='w', shadow=True, fancybox=True, framealpha=0.9)
tind=-1
add_vel_quiver(tind, ax=ax)
plt.show()

if savefig:
    plt.savefig(__figdir__ + 'cruise_track_map.' + plotfiletype, **savefig_args)
    print('Saved figure to ' + __figdir__ + 'cruise_track_map.' + plotfiletype)

# %%
