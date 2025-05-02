# %% [markdown]
# # Download, plot near-real-time DUACS SSH product
# 
# Tom Farrar, started 10/9/2022
# 
# * Download with motuclient
# * Plot latest map
# * make movie of longer time
# * extract U, V time series at some point

# %%
# mamba install conda-forge::copernicusmarine --yes
import copernicusmarine
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import cartopy.crs as ccrs                   # import projections
import cartopy
import gsw
import pandas as pd

# import cftime

# %%
# Run the following line to create the login file for Copernicus Marine Service
# copernicusmarine.login()


# %%
# Change to this directory
home_dir = os.path.expanduser("~")

# To work for Tom and other people
if os.path.exists(home_dir + '/Python/EKAMSAT_2025_cruise/src'):
    os.chdir(home_dir + '/Python/EKAMSAT_2025_cruise/src')

# %%
%matplotlib widget
plt.rcParams['figure.figsize'] = (5,4)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 400
plt.close('all')

__figdir__ = '../img/' + 'SSH_plots/'
savefig_args = {'bbox_inches':'tight', 'pad_inches':0.2}
plotfiletype='png'

# %%
savefig = False
zoom = True
domovie = False
if zoom:
    xmin, xmax = (80,95)
    ymin, ymax = (5,20)
    levels = np.linspace(-.2,.2,11)
else:
    xmin, xmax = (70,100)
    ymin, ymax = (0, 25)
    levels = np.linspace(-.3,.3,11)

# %%
# A shell script using the motuclient, https://help.marine.copernicus.eu/en/articles/4796533-what-are-the-motu-client-motuclient-and-python-requirements
# Shell script from Ben Greenwood (email 9/21/2022)
'''
echo "$(date -u) download_aviso.sh" >> ./aviso_download.log

start="2022-09-15 00:00:00"
end="$(date -u "+%Y-%m-%d %H:%M:%S")"
out_dir='./'

# download data
motuclient --motu https://nrt.cmems-du.eu/motu-web/Motu --service-id SEALEVEL_GLO_PHY_L4_NRT_OBSERVATIONS_008_046-TDS --product-id dataset-duacs-nrt-global-merged-allsat-phy-l4 --longitude-min -140 --longitude-max -120 --latitude-min 34 --latitude-max 43 --date-min "$start" --date-max "$end" --variable adt --variable err_ugosa --variable err_vgosa --variable sla --variable ugos --variable ugosa --variable vgos --variable vgosa --out-dir "$out_dir" --out-name aviso.nc --user ***** --pwd *****
'''

'''
if not os.path.exists('../data/external/aviso.nc'):
    ! bash ~/Python/download_aviso.sh
    ! cp ./aviso.nc ../data/external/aviso.nc
'''
# %%
if not os.path.exists('../data/external/aviso.nc'):
    print('Need to download the data')
    ds = copernicusmarine.open_dataset(
        dataset_id = "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
        minimum_longitude = xmin, maximum_longitude = xmax,
        minimum_latitude = ymin, maximum_latitude = ymax,
        minimum_depth = 0., maximum_depth = 10., 
        start_datetime = "2025-04-21 00:00:00",    
        end_datetime = "2025-05-15 23:59:59", 
        variables = ['adt', 'sla', 'ugos', 'vgos'], 
        )
    ds.to_netcdf('../data/external/aviso.nc')
else:
    ds = xr.open_dataset('../data/external/aviso.nc')
    # check when that file was written
    print('File exists, check the date')
    print('File created: ', ds.time[0].values)
    print('File last modified: ', ds.time[-1].values)
    
    # download a small set of the data to see if there are any problems
    last_time = str(pd.to_datetime(ds.time[-1].values))
    ds_test = copernicusmarine.open_dataset(
        dataset_id = "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
        minimum_longitude = xmin, maximum_longitude = xmax,
        minimum_latitude = ymin, maximum_latitude = ymax,
        minimum_depth = 0., maximum_depth = 10., 
        start_datetime = last_time,    
        end_datetime = "2025-05-16 23:59:59", 
        variables = ['adt', 'sla', 'ugos', 'vgos'], 
        )
    # check if the last time of the new file is more recent than the last time of the old file
    if ds_test.time[-1].values > ds.time[-1].values:
        print('New data is more recent, appending to the old file')
        # append the new data to the old dataset
        ds = xr.concat([ds, ds_test], dim='time')
        # ds.to_netcdf('../data/external/aviso.nc')
    else:
        print('Existing data is up to date, no need to append.')
        # ds_test.to_netcdf('../data/external/aviso.nc', mode='a')

# %%


# %%
def plot_SSH_map(tind):
    plt.clf()
    ax = plt.axes(projection = ccrs.PlateCarree(central_longitude=85))  # Orthographic
    extent = [xmin, xmax, ymin, ymax]
    day_str = np.datetime_as_string(ds.time[tind], unit='D')
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_title('Sea level anomaly (DUACS), '+ day_str,size = 10.)

    #plt.set_cmap(cmap=plt.get_cmap('nipy_spectral'))
    plt.set_cmap(cmap=plt.get_cmap('turbo'))
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    #gl.xlocator = matplotlib.ticker.MaxNLocator(10)
    #gl.xlocator = matplotlib.ticker.AutoLocator
    # gl.xlocator = matplotlib.ticker.FixedLocator(np.arange(0, 360 ,30))

    cs = ax.contourf(ds.longitude,ds.latitude,np.squeeze(ds.sla.isel(time=tind)), levels, extend='both', transform=ccrs.PlateCarree())
    # cs = ax.pcolormesh(ds.longitude,ds.latitude,np.squeeze(ds.sla), vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree())
    # cb = plt.colorbar(cs,ax=ax,shrink=.8,pad=.05)
    cb = plt.colorbar(cs,fraction = 0.022,extend='both')
    cb.set_label('SLA [m]',fontsize = 10)
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND, zorder=3, facecolor=[.6,.6,.6], edgecolor='black')

    # Add the 2024 site
    ax.plot(86, 12, 'ko', markersize=3, transform=ccrs.PlateCarree(), zorder=4, label='2024 site')
    #ax.text(86.1, 12.1, '2024 site', fontsize=6, transform=ccrs.PlateCarree(), zorder=4)
   
    # Add a 10 km scale bar
    km_per_deg_lat=gsw.geostrophy.distance((121.7,121.7), (37,38))/1000
    deg_lat_equal_10km=10/km_per_deg_lat
    x0 = 87 
    y0 = 12
    ax.plot(x0+np.asarray([0, 0]),y0+np.asarray([0.,deg_lat_equal_10km[0]]),transform=ccrs.PlateCarree(),color='k',zorder=3)
    ax.text(x0+1/60, y0+.15/60, '10 km', fontsize=6,transform=ccrs.PlateCarree())

    u = np.squeeze(ds.ugos.isel(time=tind)) #dtype=object
    v = np.squeeze(ds.vgos.isel(time=tind))
    skip = 5
    scalefac = 10
    # ax.quiver(ds.longitude.values[::skip], ds.latitude.values[::skip], u.values[::skip,::skip], v.values[::skip,::skip], scale=scalefac, transform=ccrs.PlateCarree())
    x0 = 80.5
    y0 = 17.33
    #ax.quiver(np.array([x0]), np.array([y0]), -np.array([0.25/np.sqrt(2)],), np.array([0.25/np.sqrt(2)]), scale=scalefac, transform=ccrs.PlateCarree(),zorder=3)
    # ax.text(x0+3/60, y0+.15/60, '0.25 m/s', fontsize=6,transform=ccrs.PlateCarree())

    ax.legend(loc='upper right', fontsize=6, frameon=True)

    if savefig:
        plt.savefig(__figdir__+'SLA'+str(tind)+'.'+plotfiletype,**savefig_args)

    return ax

# %%
def add_vel_quiver(tind,ax=plt.gca()):
    if ax is None:
        ax = plt.gca()

    u = np.squeeze(ds.ugos.isel(time=tind)) #dtype=object
    v = np.squeeze(ds.vgos.isel(time=tind))
    skip = 5
    scalefac = 10
    q = ax.quiver(ds.longitude.values[::skip], ds.latitude.values[::skip], u.values[::skip,::skip], v.values[::skip,::skip], scale=scalefac, transform=ccrs.PlateCarree())
    x0 = 81.5
    y0 = 17.33
    ax.quiverkey(q,x0,y0,0.25, '0.25 m/s', zorder=3, transform=ccrs.PlateCarree())
    #ax.quiver(np.array([x0]), np.array([y0]), -np.array([0.25/np.sqrt(2)],), np.array([0.25/np.sqrt(2)]), scale=scalefac, transform=ccrs.PlateCarree(),zorder=3)
    #ax.text(x0+3/60, y0+.15/60, '0.25 m/s', fontsize=6,transform=ccrs.PlateCarree())




# %%
ds

# %%
fig = plt.figure()
tind=-1
ax = plot_SSH_map(tind)

# %%
add_vel_quiver(tind, ax=ax)

# %%
# Make a movie of the SSH
if domovie:
    fig = plt.figure()
    for tind in range(len(ds.time)):
        ax = plot_SSH_map(tind)
        add_vel_quiver(tind, ax=ax)


# %%

# %%
# !ffmpeg -i SLA%d.png -r 10 SSH_April_20.avi



# %%



