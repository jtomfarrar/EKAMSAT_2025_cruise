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
import datetime
import matplotlib.dates as mdates
# %% Mounting google drive
# sudo mkdir /mnt/g
# sudo mount -t drvfs G: /mnt/g

# %%
# %matplotlib inline
%matplotlib widget
# %matplotlib qt5
plt.rcParams['figure.figsize'] = (6,6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 400
# %% Acess the files
directory = "../../../../mnt/g/Shared\ drives/AirSeaLab_Shared/ASTRAL_2025/PAYLOAD/MAT/";
Assets = ['IDA','PLANCK','WHOI43','WHOI1102'];
Extra = '_PLD_DATA_ALL.mat';
to_file = "../data/external/"
!cp {"../../../../../mnt/g/Shared\ drives/AirSeaLab_Shared/ASTRAL_2025/PAYLOAD/MAT/IDA_PLD_DATA_ALL.mat"} {to_file}
!cp {"../../../../../mnt/g/Shared\ drives/AirSeaLab_Shared/ASTRAL_2025/PAYLOAD/MAT/WHOI43_PLD_DATA_ALL.mat"} {to_file}
!cp {"../../../../../mnt/g/Shared\ drives/AirSeaLab_Shared/ASTRAL_2025/PAYLOAD/MAT/PLANCK_PLD_DATA_ALL.mat"} {to_file}
!cp {"../../../../../mnt/g/Shared\ drives/AirSeaLab_Shared/ASTRAL_2025/PAYLOAD/MAT/WHOI1102_PLD_DATA_ALL.mat"} {to_file}

# %% Read .mat files
import scipy.io as sio
date_start = datetime.datetime(2025,5,7,0,0,0)
IDA = sio.loadmat(to_file+Assets[0]+Extra,struct_as_record=False, squeeze_me=True)
PLANCK = sio.loadmat(to_file+Assets[1]+Extra,struct_as_record=False, squeeze_me=True)
WHOI43 = sio.loadmat(to_file+Assets[2]+Extra,struct_as_record=False, squeeze_me=True)
WHOI1102 = sio.loadmat(to_file+Assets[3]+Extra,struct_as_record=False, squeeze_me=True)
# %% Read the datenums as datetime
def datenum_to_datetime(datenum):
    # MATLAB's datenum = days since 0000-01-00
    # Python's datetime starts from year 1, so we subtract the difference
    days = datenum % 1
    return np.datetime64(datetime.datetime.fromordinal(int(datenum)) + datetime.timedelta(days=days) - datetime.timedelta(days=366))

IDA_datetime = [datenum_to_datetime(dn) for dn in IDA["IDA"].PLD2_TAB1.time]

PLANCK_datetime = [datenum_to_datetime(dn) for dn in PLANCK["PLANCK"].PLD2_TAB1.time]

WHOI43_datetime = [
    datenum_to_datetime(dn) 
    for dn in WHOI43["WHOI43"].PLD2_TAB1.time 
    if not np.isnan(dn)
]
time_array = WHOI43["WHOI43"].PLD2_TAB1.time
non_nan_indices = np.where(~np.isnan(time_array))[0]

WHOI1102_datetime = [datenum_to_datetime(dn) for dn in WHOI1102["WHOI1102"].PLD2_TAB1.time]

datetime_concat = {
    'IDA_datetime': IDA_datetime,
    'PLANCK_datetime': PLANCK_datetime,
    'WHOI43_datetime': WHOI43_datetime,
    'WHOI1102_datetime': WHOI1102_datetime
}
# %% Plotting the maps
#variables = ["TrueWindSpeed_Avg","relative_humidity_Avg","atmospheric_temperature_Avg", "rain_intensity_Avg","SMP21_SW_Flux_Avg", "SGR4_LW_Flux_Avg"]
variables = ["TrueWindSpeed_Avg","relative_humidity_Avg","atmospheric_temperature_Avg", "rain_intensity_Avg","SMP21_SW_Flux_Avg", "SGR4_LW_Flux_Avg"]
colors = ["red","blue","teal","orange"]
ylabel = ["wS (m/s)","Rh (%)","AT($^o$C)","Rain(mm/h)","SWR(W/m$^2$)","LWR(W/m$^2$)"]
fig, ax = plt.subplots(6, 1, figsize=(12,14))
for i in range(len(variables)):
    for j in range(len(Assets)):
        Struct = globals()[Assets[j]][Assets[j]];
        data = Struct.PLD2_TAB1;

        time_key = list(datetime_concat.keys())[j]
        time_series = datetime_concat[time_key]

        ax[i].set_xlim([datetime.datetime(2025,5,7,12,0,0),time_series[-1]])
        ax[i].set_ylabel(ylabel[i],fontsize=14)
        ax[i].grid(True)
        ax[i].axvline(datetime.datetime(2025,5,10,2,0,0),linewidth=1.5,linestyle="--",color="teal")
        ax[i].axvline(datetime.datetime(2025,5,10,12,0,0),linewidth=1.5,linestyle="--",color="teal")
        ax[i].axvline(datetime.datetime(2025,5,10,20,0,0),linewidth=1.5,linestyle="--",color="orange")
        ax[i].axvline(datetime.datetime(2025,5,11,4,0,0),linewidth=1.5,linestyle="--",color="orange")
        if (i==4 or i==5) and j==1:
            continue;
        else:
            if j==2:
                ax[i].plot(time_series,getattr(data, variables[i])[non_nan_indices],label = Assets[j],color=colors[j])
            else:
                ax[i].plot(time_series,getattr(data, variables[i]),label = Assets[j],color=colors[j])

ax[0].set_ylim([0,10])
ax[3].legend(loc='upper center',ncols=2,fontsize=12)
ax[1].set_ylim([60,100])
ax[2].set_ylim([26,33])
ax[3].set_ylim([0,40])
#ax[4].set_ylim([30,34])
ax[4].set_ylim([0,1200])
ax[5].set_ylim([410,470])

# %% Call in the RAMA mooring data
directory = '../data/external/RAMA_12/';
variables_RAMA = ["WS_401","RH_910","AT_21","RN_485","RD_495"]
variables_WHOI1102 = ["TrueWindSpeed_Avg","HC2A_RH","HC2A_ATMP", "rain_intensity_Avg","SMP21_SW_Flux_Avg"]
# variables_WHOI1102 = ["TrueWindSpeed_Avg","relative_humidity_Avg","atmospheric_temperature_Avg", "rain_intensity_Avg","SMP21_SW_Flux_Avg"]
WG = ["WHOI1102"]
ylabel = ["wS (m/s)","RH (%)","AT($^\circ$C)","Rain(mm/h)","SWR(W/m$^2$)"]
Time = xr.open_dataset(directory+'met12n90e_hr.cdf');
# list = [directory+'airt12n90e_hr.cdf',directory+'rad12n90e_hr.cdf', directory+'rh12n90e_hr.cdf',directory+"rain12n90e_hr.cdf", directory+'sst12n90e_hr.cdf',directory+'w12n90e_hr.cdf'];
list = [directory+'met12n90e_hr.cdf',directory+'rad12n90e_hr.cdf', directory+"rain12n90e_hr.cdf", directory+'sss12n90e_hr.cdf'];
ds_combined = xr.open_mfdataset(list, combine='by_coords')
All_vars = ds_combined.set_coords(['time','depth', 'lat', 'lon'])
depth_values = [-4,-3,-3,-4,-4]
fig, ax = plt.subplots(5,1,figsize=(12,12), sharex=True, sharey=False)
for i in range(len(variables_RAMA)):
    ax[i].plot(Time["time"],All_vars[variables_RAMA[i]].sel(lat=12,lon=90).squeeze(),color='black',label="RAMA-12$^\circ$N (hourly)")
    Struct = WHOI1102["WHOI1102"];
    if i==0:
        data = Struct.PLD1_TAB1
    else:
        data = Struct.PLD2_TAB1
    ax[i].set_xlim([datetime.datetime(2025,5,7,12,0,0),time_series[-1]])
    ax[i].set_ylabel(ylabel[i],fontsize=14)
    ax[i].grid(True)
    if i==0:
        time0 = [datenum_to_datetime(dn) for dn in WHOI1102["WHOI1102"].PLD2_TAB1.time if not np.isnan(dn)]
        ax[i].plot(time0[0:-1],getattr(Struct.PLD1_TAB1, variables_WHOI1102[i]),label = "WHOI1102 (15 min)",color="orange")
    else:
        ax[i].plot(time_series,getattr(data, variables_WHOI1102[i]),label = "WHOI1102 (15 min)",color="orange")
    ax[i].set_ylabel(ylabel[i],fontsize=14)
    ax[i].axvline(datetime.datetime(2025,5,10,20,0,0),linewidth=1.5,linestyle="--",color="orange")
    ax[i].axvline(datetime.datetime(2025,5,11,4,0,0),linewidth=1.5,linestyle="--",color="orange")
ax[0].legend(loc='upper center',ncols=2,fontsize=12)
ax[0].set_ylim([0,17])
ax[1].set_ylim([60,100])
ax[2].set_ylim([26,33])
ax[3].set_ylim([0,30])
#ax[4].set_ylim([30,34])
ax[4].set_ylim([0,1200])


# %% load the surface T(z) data from IDA and WHOI43
levels=np.arange(30,32,0.1)
fig, ax = plt.subplots(2, 1, figsize=(10,8),sharex=True)
mesh = ax[0].contourf(IDA_datetime,IDA["IDA"].PLD2_TAB1.z_tchn_temp[:,0],
                 IDA["IDA"].PLD2_TAB1.tchn_temp,levels=levels,cmap='turbo',extend="both");
ax[0].set_xlim([datetime.datetime(2025,5,7,12,0,0),IDA_datetime[-1]]);
cbar = fig.colorbar(mesh, ax=ax[0])
cbar.set_label("Temperature (°C)")
#fig.autofmt_xdate()



start_date = np.datetime64(datetime.datetime(2025, 5, 10,2,0,0))
end_date = np.datetime64(datetime.datetime(2025, 5, 10,12,0,0))
depth_top = 0
depth_bottom = -50
width = end_date - start_date
mask_time = (WHOI43_datetime >= start_date) & (WHOI43_datetime <= end_date)
WHOI43["WHOI43"].PLD2_TAB1.tchn_temp[:,non_nan_indices][:, mask_time] = np.nan
mesh = ax[1].contourf(WHOI43_datetime,WHOI43["WHOI43"].PLD2_TAB1.z_tchn_temp[:,0],
                 WHOI43["WHOI43"].PLD2_TAB1.tchn_temp[:,non_nan_indices],levels=levels,cmap='turbo',extend="both");
ax[1].set_xlim([datetime.datetime(2025,5,7,12,0,0),WHOI43_datetime[-1]]);
cbar = fig.colorbar(mesh, ax=ax[1])
cbar.set_label("Temperature (°C)")
fig.autofmt_xdate()

ax[0].set_ylim([-50,0])
ax[1].set_ylim([-50,0])
ax[0].set_title("IDA T(z)")
ax[1].set_title("WHOI43 T(z)")
ax[0].set_ylabel("Depth (m)")
ax[1].set_ylabel("Depth (m)")

# %% plot the RBR sensor from 
# %% PLANCK RBR data
Temp_RbR = PLANCK["PLANCK"].PLD2_TAB4.rbr_temp_Avg;
Sal_RbR = PLANCK["PLANCK"].PLD2_TAB4.rbr_sali_Avg;
Depth_RbR =PLANCK["PLANCK"].PLD2_TAB4.rbr_z_Avg;
Density_RbR = PLANCK["PLANCK"].PLD2_TAB4.rbr_dens_Avg;
RbR_datetime = [datenum_to_datetime(dn) for dn in PLANCK["PLANCK"].PLD2_TAB4.time]
 # Let's bin them
RbR_datetime = pd.to_datetime(np.array(RbR_datetime))
#dates_ts = [date.timestamp() for date in RbR_datetime]
#depth_bins = np.arange(np.nanmin(Depth_RbR), np.nanmax(Depth_RbR) + 7, 7)  # 3 m bins
#time_bins = pd.date_range(start=np.datetime64(datetime.datetime(2025,5,7,12,0,0)), end=np.max(RbR_datetime), freq='15min')  # 1 min bins

#dates_ts = [date.timestamp() for date in RbR_datetime]
#dates_bin_ts= [date.timestamp() for date in time_bins]

# Digitize to get bin indices
#depth_idx = np.digitize(Depth_RbR, depth_bins) - 1
#time_idx = np.digitize(dates_ts, dates_bin_ts) - 1
# Create 2D array to hold binned data
#temp_binned = np.full((len(depth_bins)-1, len(time_bins)-1), np.nan)
#counts = np.zeros_like(temp_binned)

# Bin the data by averaging
#for d, t, v in zip(depth_idx, time_idx, Temp_RbR):
#    if 0 <= d < temp_binned.shape[0] and 0 <= t < temp_binned.shape[1]:
#        if np.isnan(temp_binned[d, t]):
#            temp_binned[d, t] = v
#            counts[d, t] = 1
#        else:
#            temp_binned[d, t] += v
#            counts[d, t] += 1

# Average where we have multiple entries
#mask = counts > 0
#temp_binned[mask] /= counts[mask]

# Plot
#plt.figure(figsize=(12, 6))
#X, Y = np.meshgrid(time_bins[:-1], -depth_bins[:-1])
#pcm = plt.contourf(X, Y, temp_binned, shading='auto', cmap='turbo',levels=levels)
#plt.gca().invert_yaxis()
#plt.colorbar(pcm, label='Temperature (°C)')
#plt.xlabel('Time')
#plt.ylabel('Depth (m)')
#plt.title('Temperature binned every 6 m and 15 minute')
#plt.tight_layout()
#plt.show()
#fig, ax = plt.subplots(1,1,figsize=(8,6))
#plot = ax.scatter(RbR_datetime,Depth_RbR,c=Temp_RbR,cmap='turbo')
#cbar = fig.colorbar(plot,ax=ax)
# %% Plot the locations of IDA, WHOI43 and PLANCK
savefig = False
zoom = True
domovie = False
if zoom:
    #xmin, xmax = (88.45,88.55)
    #ymin, ymax = (12.05,11.95)
    xmin, xmax = (87.4,87.8)
    ymin, ymax = (13.1,13.5)
    levels = np.linspace(-.3,.3,51)
else:
    xmin, xmax = (70,100)
    ymin, ymax = (0, 25)
    levels = np.linspace(-.3,.3,51)
ds = xr.open_dataset('../data/external/aviso.nc')
# %% EEZ
'''import scipy.io
mat_data = scipy.io.loadmat('../data/external/World_EEZ/eez.mat')
eez = mat_data['eez'];
eez = np.array(eez)
'''
# %% Plotting script
def plot_SSH_map(tind):
    plt.clf()
    ax = plt.axes(projection = ccrs.PlateCarree(central_longitude=85))  # Orthographic
    extent = [xmin, xmax, ymin, ymax]
    day_str = np.datetime_as_string(ds.time[tind], unit='D')
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_title('Current Velocity magnitude (DUACS), '+ day_str,size = 10.)
#    ax.plot(eez[:,0], eez[:,1], 'w-', linewidth=1, transform=ccrs.PlateCarree(), zorder=3,label='EEZ')

    #plt.set_cmap(cmap=plt.get_cmap('nipy_spectral'))
    plt.set_cmap(cmap=plt.get_cmap('turbo'))
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    #gl.xlocator = matplotlib.ticker.MaxNLocator(10)
    #gl.xlocator = matplotlib.ticker.AutoLocator
    # gl.xlocator = matplotlib.ticker.FixedLocator(np.arange(0, 360 ,30))
    u = np.squeeze(ds.ugos) #dtype=object
    v = np.squeeze(ds.vgos)
    uu = np.sqrt(u**2+v**2);
    cs = ax.contourf(ds.longitude,ds.latitude,np.squeeze(uu.isel(time=tind)), levels, extend='both', transform=ccrs.PlateCarree())
    # cs = ax.pcolormesh(ds.longitude,ds.latitude,np.squeeze(ds.sla), vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree())
    # cb = plt.colorbar(cs,ax=ax,shrink=.8,pad=.05)
    cb = plt.colorbar(cs,fraction = 0.022,extend='both')
    cb.set_label('Current mag. [m/s]',fontsize = 10)
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND, zorder=3, facecolor=[.6,.6,.6], edgecolor='black')

    # Add the 2024 site
    #ax.plot(86, 12, 'ko', markersize=3, transform=ccrs.PlateCarree(), zorder=4, label='2024 site')
    #ax.text(86.1, 12.1, '2024 site', fontsize=6, transform=ccrs.PlateCarree(), zorder=4)
    pts_lon = [89.175, 89.124, 87.991, 87.00, 89.04, 88.51]
    pts_lat = [17.817, 17.460, 16.322, 13.99, 15.04, 12.01]
    #plt.plot(pts_lon, pts_lat, 'o', color='m',markeredgecolor='k', markersize=6, transform=ccrs.PlateCarree(),label='BD/RAMA moorings')
    # Add a 10 km scale bar
    km_per_deg_lat=gsw.geostrophy.distance((121.7,121.7), (37,38))/1000
    deg_lat_equal_10km=10/km_per_deg_lat
    x0 = 87
    y0 = 12
    ax.plot(x0+np.asarray([0, 0]),y0+np.asarray([0.,deg_lat_equal_10km[0]]),transform=ccrs.PlateCarree(),color='k',zorder=3)
    #ax.text(x0+1/60, y0+.15/60, '10 km', fontsize=6,transform=ccrs.PlateCarree())
    u = np.squeeze(ds.ugos.isel(time=tind)) #dtype=object
    v = np.squeeze(ds.vgos.isel(time=tind))
    skip = 1
    scalefac = 1
    ax.quiver(ds.longitude.values[::skip], ds.latitude.values[::skip], u.values[::skip,::skip], v.values[::skip,::skip], scale=scalefac, transform=ccrs.PlateCarree())
    x0 = 80.5
    y0 = 17.33
    ax.quiver(np.array([x0]), np.array([y0]), -np.array([0.25/np.sqrt(2)],), np.array([0.25/np.sqrt(2)]), scale=scalefac, transform=ccrs.PlateCarree(),zorder=3)
    #ax.text(x0+3/60, y0+.15/60, '0.25 m/s', fontsize=6,transform=ccrs.PlateCarree())
    #ax.legend(loc='upper right', fontsize=6, frameon=False)

    ax.scatter(IDA["IDA"].PLD2_TAB1.longitude_Avg[1000:-1],IDA["IDA"].PLD2_TAB1.latitude_Avg[1000:-1],color='red',s=25,label='IDA',transform=ccrs.PlateCarree(),zorder=5)
    ax.scatter(PLANCK["PLANCK"].PLD2_TAB1.longitude_sitex_Avg[1000:-1],PLANCK["PLANCK"].PLD2_TAB1.latitude_sitex_Avg[1000:-1],color='blue',s=25,label='PLANCK',transform=ccrs.PlateCarree(),zorder=5)
    ax.scatter(WHOI43["WHOI43"].PLD2_TAB1.longitude_Avg[1000:-1],WHOI43["WHOI43"].PLD2_TAB1.latitude_Avg[1000:-1],color='teal',s=25,label='WHOI43',transform=ccrs.PlateCarree(),zorder=5)
    #ax.scatter(WHOI1102["WHOI1102"].PLD2_TAB1.longitude_Avg[150:-1],WHOI1102["WHOI1102"].PLD2_TAB1.latitude_Avg[150:-1],color='black',s=15,label='WHOI1102',transform=ccrs.PlateCarree(),zorder=5)

    plt.legend(loc='upper left', fontsize=8, frameon=True)

    directory = '../data/processed/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = 'SLA.png'
    filepath = os.path.join(directory, filename)
    plt.savefig(filepath)

# %% Plot
fig = plt.figure()
tind=-1
plot_SSH_map(tind)
# %% Plotting planck
PLANCK_datetime = [datenum_to_datetime(dn) for dn in PLANCK["PLANCK"].PLD2_TAB4.time]
import gsw
fig, ax = plt.subplots(1, 1, figsize=(5,8))
mesh=ax.scatter(PLANCK["PLANCK"].PLD2_TAB4.rbr_sali_Avg,
           PLANCK["PLANCK"].PLD2_TAB4.rbr_temp_Avg,15,c=PLANCK["PLANCK"].PLD2_TAB4.time,cmap="jet")
ax.set_xlim([32,35])
ax.set_ylim([20,32])

s_grid = np.linspace(32, 35, 100)
t_grid = np.linspace(20, 32, 200)
S_grid, T_grid = np.meshgrid(s_grid, t_grid)
SA_grid = gsw.SA_from_SP(S_grid, 0, lon=0, lat=0)
CT_grid = gsw.CT_from_t(SA_grid, T_grid, 0)
rho_grid = gsw.rho(SA_grid, CT_grid, 0)
ax.grid(True)
# Draw isopycnals
cs = ax.contour(S_grid, T_grid, rho_grid-1000, colors='k', levels=np.arange(20, 30, 0.5), linewidths=0.8)
label_positions = []
for collection in cs.collections:
    for path in collection.get_paths():
        verts = path.vertices
        if len(verts) > 0:
            mid_idx = len(verts) // 2
            label_positions.append(verts[mid_idx])

# Add labels at the calculated midpoint positions
plt.clabel(cs, fmt='%1.1f', fontsize=12, manual=label_positions)
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Day of May, 2025")

cbar_ticks = np.linspace(PLANCK["PLANCK"].PLD2_TAB4.time.min(), PLANCK["PLANCK"].PLD2_TAB4.time.max(), 6)
cbar_ticklabels = [datenum_to_datetime(dn) for dn in cbar_ticks]

def day_extract(date64):
    return(date64.astype('datetime64[D]').astype(object).day)

Day_array = [day_extract(i) for i in cbar_ticklabels]
cbar.ax.set_yticks(cbar_ticks)
cbar.ax.set_yticklabels(Day_array)
ax.set_xlabel('Temperature ($^circC$)')
ax.set_ylabel('Salinity')
# %%
# Plot Planck CTD versus time
fig, ax = plt.subplots(3, 1, figsize=(8,5),sharex=True)
# set the color scale for the scatter plot to span temperatures of 20 to 32 degrees Celsius
levels = np.arange(20, 32, 0.5)
mesh = ax[0].scatter(PLANCK_datetime,PLANCK["PLANCK"].PLD2_TAB4.rbr_depth_Avg.T, 1, c=PLANCK["PLANCK"].PLD2_TAB4.rbr_temp_Avg, vmin=levels[0], vmax=levels[-1], cmap='turbo')
cbar = fig.colorbar(mesh, ax=ax[0])
cbar.set_label("T ($^\circ C$)")

ax[0].set_ylabel('Depth (m)')
ax[0].set_title('Planck winched CTD')
ax[0].set_ylim([0, 150])
ax[0].invert_yaxis()

# Salinity
levels = np.arange(32, 35, 0.1)
mesh = ax[1].scatter(PLANCK_datetime,PLANCK["PLANCK"].PLD2_TAB4.rbr_depth_Avg.T, 1, c=PLANCK["PLANCK"].PLD2_TAB4.rbr_sali_Avg, vmin=32, vmax=35, cmap='viridis')
cbar = fig.colorbar(mesh, ax=ax[1])
cbar.set_label("Salinity (g/kg)")
ax[1].set_ylabel('Depth (m)')
ax[1].set_ylim([0, 150])
ax[1].invert_yaxis()
# Potential density
SA = gsw.SA_from_SP(PLANCK["PLANCK"].PLD2_TAB4.rbr_sali_Avg, PLANCK["PLANCK"].PLD2_TAB4.rbr_depth_Avg.T, lon=88, lat=13)
CT = gsw.CT_from_t(SA, PLANCK["PLANCK"].PLD2_TAB4.rbr_temp_Avg, PLANCK["PLANCK"].PLD2_TAB4.rbr_depth_Avg.T)
# Calculate potential density at 0 dbar
sigma = gsw.density.sigma0(SA, CT)
levels = np.arange(20, 25, 0.5)

mesh = ax[2].scatter(PLANCK_datetime,PLANCK["PLANCK"].PLD2_TAB4.rbr_depth_Avg.T, 1, c=sigma, vmin=levels[0], vmax=levels[-1], cmap='turbo')
cbar = fig.colorbar(mesh, ax=ax[2])
cbar.set_label("$\sigma_0$ (kg/m$^3$)")
ax[2].set_ylabel('Depth (m)')
ax[2].set_ylim([0, 150])
ax[2].invert_yaxis()
fig.autofmt_xdate()

# %%
