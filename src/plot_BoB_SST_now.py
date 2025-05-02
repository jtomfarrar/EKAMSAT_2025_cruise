# Plot OISST for the Bay of Bengal (EKAMSAT 2025 cruise)
#

# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs                   # import projections
import cartopy
import geopandas as gpd # for the EEZ shapefile

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

__figdir__ = '../img/SST_movie/'
sst_figdir = '../img/SST_movie/'
os.system('mkdir  -p ' + __figdir__) #make directory if it doesn't exist
savefig_args = {'bbox_inches':'tight', 'pad_inches':0.2}
plotfiletype='png'

# %%
# clear the directory
os.system('rm -f ' + __figdir__ + '*')


# %%
# Download the data if needed
# https://psl.noaa.gov/thredds/catalog/Datasets/noaa.oisst.v2.highres/catalog.html?dataset=Datasets/noaa.oisst.v2.highres/sst.day.mean.2023.nc

# download the data to ../data/external if it does not already exist there
data_dir = '../data/external'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

#url = 'https://psl.noaa.gov/thredds/fileServer/Datasets/noaa.oisst.v2.highres/sst.day.mean.2023.nc'
url = 'https://psl.noaa.gov/thredds/fileServer/Datasets/noaa.oisst.v2.highres/sst.day.mean.2025.nc'


filename = os.path.join(data_dir, os.path.basename(url))
if not os.path.exists(filename):
    import urllib.request
    urllib.request.urlretrieve(url, filename)
else: # check if the file is older than 0.5 days
    # Get the last modified time of the file
    import datetime
    import time
    import urllib.request
    # Get the last modified time of the file
    last_modified_time = os.path.getmtime(filename)
    # Convert the last modified time to a human-readable format
    last_modified_time = datetime.datetime.fromtimestamp(last_modified_time)
    print("Last modified time:", last_modified_time)
    print("Time now:", datetime.datetime.now())
    # compute age of file:
    age = datetime.datetime.now() - last_modified_time
    print("Age of file:", age)
    # If the file is older than 0.5 days, download it again
    if age.total_seconds() > 43200:  # 0.5 days in seconds
        print("File is older than 0.5 days, downloading again")
        urllib.request.urlretrieve(url, filename)



# %%

# Nominal location: 7°57'47.83" N  87°38'25.56" E
mooring = dict(
    lon = [86],
    lat = [12])
# %%
# Load the data
ds = xr.open_dataset(filename)

# %% Shift ds to have longitude from -180 to 180
ds = ds.assign_coords(lon=(ds.lon + 180) % 360 - 180)
ds = ds.sortby(ds.lon)


# %%
zoom = True
domovie = False
if zoom:
    xmin, xmax = (80,95)
    ymin, ymax = (5,20)
else:
    xmin, xmax = (70,100)
    ymin, ymax = (0, 25)
# %%
ds_ssh = xr.open_dataset('../data/external/aviso.nc')

# %%
def add_vel_quiver(tind,ax=plt.gca()):
    if ax is None:
        ax = plt.gca()

    u = np.squeeze(ds_ssh.ugos.isel(time=tind)) #dtype=object
    v = np.squeeze(ds_ssh.vgos.isel(time=tind))
    skip = 3
    scalefac = 10
    q = ax.quiver(ds_ssh.longitude.values[::skip], ds_ssh.latitude.values[::skip], u.values[::skip,::skip], v.values[::skip,::skip], scale=scalefac, transform=ccrs.PlateCarree())
    x0 = 81.5
    y0 = 17.33
    ax.quiverkey(q,x0,y0,0.25, '0.25 m/s', zorder=3, transform=ccrs.PlateCarree())
    #ax.quiver(np.array([x0]), np.array([y0]), -np.array([0.25/np.sqrt(2)],), np.array([0.25/np.sqrt(2)]), scale=scalefac, transform=ccrs.PlateCarree(),zorder=3)
    #ax.text(x0+3/60, y0+.15/60, '0.25 m/s', fontsize=6,transform=ccrs.PlateCarree())


######################

# %%
## Plot the data
def plot_map(data, levels, title='', outfile='', savefig=False, ax=None):
    if title is None:
        title = ''
    if outfile is None:
        outfile = title
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})  # Create a new axis if none is provided
    coast = cartopy.feature.GSHHSFeature(scale="full")
    ax.add_feature(coast, zorder=3, facecolor=[.6, .6, .6], edgecolor='black')

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Add country boundaries
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5, zorder=10)
    ax.add_feature(cartopy.feature.NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='50m', facecolor='none'), zorder=10)
    ax.add_feature(cartopy.feature.RIVERS, edgecolor='blue', zorder=10, alpha=0.25)

    # Plot the data
    cs = ax.pcolormesh(data.lon, data.lat, data, vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree())
    cb = plt.colorbar(cs, ax=ax, fraction=0.022, extend='both')
    cb.set_label('SST [$\\circ$C]', fontsize=10)
    ax.axis('scaled')
    ax.set_title(title)

    # Add TAO mooring locations
    ax.plot(mooring['lon'], mooring['lat'], 'o', color='k', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='2024 site')

    if savefig:
        outfile2 = outfile.replace(' ', '_')
        plt.savefig(__figdir__ + outfile2 + '.' + plotfiletype, **savefig_args)




# %%
# Make one plot for the last time in the file
t1 = ds.time[-1].values
fstr = ds.time[-1].astype('datetime64[D]').values
t1 = str(fstr)
sst = ds.sst.sel(lon=slice(xmin,xmax), lat=slice(ymin,ymax),time=t1)




# %%
levels = np.arange(29, 31, 0.25)
plot_map(sst, levels, title='SST,' + t1, savefig=False)
# add WG pts at lat = -0.5, 0.5, 1.0 and lon =-140.5 and -139.5
# BD08: 17.817 N, 89.175 E
# BD09: 17.460 N, 89.124 E
# BD10: 16.322 N, 87.991 E
# BD13: 13.99 N, 87.00 E
# RAMA: 15.04 N, 89.04 E
# RAMA: 12.01 N, 88.51 E



pts_lon = [89.175, 89.124, 87.991, 87.00, 89.04, 88.51]
pts_lat = [17.817, 17.460, 16.322, 13.99, 15.04, 12.01]
plt.plot(pts_lon, pts_lat, 'o', color='m',markeredgecolor='k', markersize=4, transform=ccrs.PlateCarree(),label='BD/RAMA moorings')
plt.legend()

plt.savefig(__figdir__+'SST_WG_array_example.' +plotfiletype,**savefig_args)
# %%
# Add EEZ
# Download the file to ../data/external/
# https://www.marineregions.org/downloads.php#eez

EEZ_file = '../data/external/World_EEZ_v12_20231025.zip'
if not os.path.exists(EEZ_file):
    import urllib.request
    url = 'https://www.marineregions.org/downloads.php#eez'
    urllib.request.urlretrieve(url, EEZ_file)
# %%
# Unzip the file
if not os.path.exists(data_dir + '/World_EEZ_v12_20231025'):
    import zipfile
    # Unzip the file
    data_dir = '../data/external'
    EEZ_file = data_dir + '/World_EEZ_v12_20231025.zip'
    with zipfile.ZipFile(EEZ_file, 'r') as zip_ref:
        zip_ref.extractall(data_dir)


# Read the shapefile
shapefile = data_dir + '/World_EEZ_v12_20231025/eez_v12.shp'
gdf = gpd.read_file(shapefile)

# %%
# Plot the shapefile on the map
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})  # Define ax with cartopy projection
plt.set_cmap(cmap=plt.get_cmap('turbo'))
levels = np.arange(29, 31, 0.25)

# Plot the SST map on the same axis
plot_map(sst, levels, title='SST,' + t1, savefig=False, ax=ax)

# Plot the BD/RAMA moorings
ax.plot(pts_lon, pts_lat, 'o', color='m', markeredgecolor='k', markersize=8, transform=ccrs.PlateCarree(), label='BD/RAMA moorings')

# Plot the shapefile
gdf.plot(ax=ax, color='lightgrey', edgecolor='black', alpha=0.3, transform=ccrs.PlateCarree(), zorder=2)

ax.grid(True)

# Add titles and labels
ax.set_title('SST,' + t1)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

plt.legend(framealpha=0.8)
plt.show()


tind=-1
add_vel_quiver(tind, ax=ax)

if savefig:
    outfile2 = 'SST_UV_map_' + t1.replace(' ', '_')
    plt.savefig(__figdir__ + outfile2 + '.' + plotfiletype, **savefig_args)



# %%
# Calculate the SST difference over the last week
# Get the time range for the last week
t2 = (ds.time[-7].values).astype('datetime64[D]').astype(str)
t1 = (ds.time[-1].values).astype('datetime64[D]').astype(str)
sst_2 = ds.sst.sel(time=t1).sel(lon=slice(75, 100), lat=slice(0, 25))
sst_1 = ds.sst.sel(time=t2).sel(lon=slice(75, 100), lat=slice(0, 25))
sst_diff = sst_2 - sst_1

# Plot the SST difference
levels_diff = np.linspace(-1, 1, 21)  # Adjust levels as needed
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
plt.set_cmap(cmap=plt.get_cmap('RdBu_r'))  # Diverging colormap for differences

cs = ax.pcolormesh(sst_diff.lon, sst_diff.lat, sst_diff, vmin=levels_diff[0], vmax=levels_diff[-1], transform=ccrs.PlateCarree())
cb = plt.colorbar(cs, ax=ax, fraction=0.022, extend='both')
cb.set_label('SST Difference (°C)', fontsize=10)

# Add map features
coast = cartopy.feature.GSHHSFeature(scale="full")
ax.add_feature(coast, zorder=3, facecolor=[.6, .6, .6], edgecolor='black')
ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5, zorder=10)
ax.add_feature(cartopy.feature.RIVERS, edgecolor='blue', alpha=0.25, zorder=10)

# Add gridlines
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Add titles and labels
ax.set_title(f'SST Difference: {t1} minus {t2}')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
# add the eez boundaries
gdf.plot(ax=ax, color='lightgrey', edgecolor='black', alpha=0.4, transform=ccrs.PlateCarree(), zorder=2)
# Add BD/RAMA moorings
ax.plot(pts_lon, pts_lat, 'o', color='m', markeredgecolor='k', markersize=4, transform=ccrs.PlateCarree(), label='BD/RAMA moorings')
ax.plot(mooring['lon'], mooring['lat'], 'o', color='k', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='2024 site')

add_vel_quiver(tind, ax=ax)
plt.legend(framealpha=0.8)

plt.show()

if savefig:
    outfile2 = 'Delta_SST_UV_map_' + t1.replace(' ', '_')
    plt.savefig(__figdir__ + outfile2 + '.' + plotfiletype, **savefig_args)




# %%
