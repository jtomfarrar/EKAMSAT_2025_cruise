# Ersen'S Joseph wrote this to plot the SST and SSS data from the RAMA mooring and the Wave Gliders (using the "upper CTD", confusingly called a UCTD)

# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io as sio
import datetime as datetime 
import pandas as pd
import os
from netCDF4 import Dataset


# %%
# sudo mount -t drvfs G: /mnt/g # use the terminal to mount the drive

# %%
# call in the RAMA mooring data
directory = "../astral_2025/external/RAMA_12/"
files = os.listdir(directory)
files

# %%
file_path = "../astral_2025/external/RAMA_12/sst12n90e_hr.cdf"
nc = Dataset(file_path, mode='r')

# print the variable names
print("Variables in file:")
print(nc.variables.keys())

# close the file
nc.close()

# %%
sst_ds = xr.open_dataset("../astral_2025/external/RAMA_12/sst12n90e_hr.cdf")
sss_ds = xr.open_dataset("../astral_2025/external/RAMA_12/sss12n90e_hr.cdf")


# %%
# convert time to datetime
sst_time = pd.to_datetime(sst_ds['time'].values)
sss_time = pd.to_datetime(sss_ds['time'].values)

# list of variables in the SST and SSS file
print(sst_ds.data_vars)  
print(sss_ds.data_vars) 

# %%
# %% Acess the files
directory = "../../../../mnt/g/Shared\ drives/AirSeaLab_Shared/ASTRAL_2025/PAYLOAD/MAT/";
Assets = ['IDA','PLANCK','WHOI43','WHOI1102'];
Extra = '_PLD_DATA_ALL.mat';
to_file = "../astral_2025/external/"
!cp {"../../../../../mnt/g/Shared\ drives/AirSeaLab_Shared/ASTRAL_2025/PAYLOAD/MAT/IDA_PLD_DATA_ALL.mat"} {to_file}
!cp {"../../../../../mnt/g/Shared\ drives/AirSeaLab_Shared/ASTRAL_2025/PAYLOAD/MAT/WHOI43_PLD_DATA_ALL.mat"} {to_file}
!cp {"../../../../../mnt/g/Shared\ drives/AirSeaLab_Shared/ASTRAL_2025/PAYLOAD/MAT/PLANCK_PLD_DATA_ALL.mat"} {to_file}
!cp {"../../../../../mnt/g/Shared\ drives/AirSeaLab_Shared/ASTRAL_2025/PAYLOAD/MAT/WHOI1102_PLD_DATA_ALL.mat"} {to_file}

# %% Read .mat files

date_start = datetime.datetime(2025,5,7,0,0,0)
IDA = sio.loadmat(to_file+Assets[0]+Extra,struct_as_record=False, squeeze_me=True)
PLANCK = sio.loadmat(to_file+Assets[1]+Extra,struct_as_record=False, squeeze_me=True)
WHOI43 = sio.loadmat(to_file+Assets[2]+Extra,struct_as_record=False, squeeze_me=True)
WHOI1102 = sio.loadmat(to_file+Assets[3]+Extra,struct_as_record=False, squeeze_me=True)

# %%
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

# %%
uctd_temp_IDA = IDA["IDA"].PLD2_TAB1.uctd_temp_Avg
uctd_temp_PLANCK = PLANCK["PLANCK"].PLD2_TAB1.uctd_temp_Avg
uctd_temp_WHOI1102 = WHOI1102["WHOI1102"].PLD2_TAB1.uctd_temp_Avg
uctd_temp_WHOI43 = WHOI43["WHOI43"].PLD2_TAB1.uctd_temp_Avg

uctd_sali_IDA = IDA["IDA"].PLD2_TAB1.uctd_sali_Avg
uctd_sali_PLANCK = PLANCK["PLANCK"].PLD2_TAB1.uctd_sali_Avg
uctd_sali_WHOI1102 = WHOI1102["WHOI1102"].PLD2_TAB1.uctd_sali_Avg
uctd_sali_WHOI43 = WHOI43["WHOI43"].PLD2_TAB1.uctd_sali_Avg

# %%
# extract data
rama_sst = sst_ds['T_25'].values.squeeze()  # squeeze in case it's 2D
rama_sss = sss_ds['S_41'].values.squeeze()

# extract SST and SSS at surface level (depth=0), and first lat/lon
#rama_sst = sst_ds['T_25'].isel(depth=0, lat=0, lon=0).values
#rama_sss = sss_ds['S_41'].isel(depth=0, lat=0, lon=0).values


# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# temperature plots
axes[0].plot(IDA_datetime, uctd_temp_IDA, lw=1.2, label='IDA')
axes[0].plot(PLANCK_datetime, uctd_temp_PLANCK, lw=1.2, label='PLANCK')
axes[0].plot(WHOI1102_datetime, uctd_temp_WHOI1102, lw=1.2, label='WHOI1102')
axes[0].plot(WHOI43_datetime, uctd_temp_WHOI43[non_nan_indices], lw=1.2, label='WHOI43')

axes[0].plot(sst_time, rama_sst, lw=1.2, label='RAMA SST', color='gray')

axes[0].set_xlim(datetime.datetime(2025,5,7,12,0,0),IDA_datetime[-1])
axes[0].set_ylim(30, 34)
axes[0].set_ylabel("SST (°C)")
#axes[0].set_title("UCTD Temperature Over Time")
axes[0].legend(loc='upper right', fontsize=8, frameon=True)
axes[0].grid(True)

# salinity plots
axes[1].plot(IDA_datetime, uctd_sali_IDA, lw=1.2, label='IDA')
axes[1].plot(PLANCK_datetime, uctd_sali_PLANCK, lw=1.2, label='PLANCK')
axes[1].plot(WHOI1102_datetime, uctd_sali_WHOI1102, lw=1.2, label='WHOI1102')
axes[1].plot(WHOI43_datetime, uctd_sali_WHOI43[non_nan_indices], lw=1.2, label='WHOI43')

axes[1].plot(sss_time, rama_sss, lw=1.2, label='RAMA SSS', color='gray')

axes[0].set_xlim(datetime.datetime(2025,5,7,12,0,0),IDA_datetime[-1])
axes[1].set_ylim(32, 34)
axes[1].set_ylabel("SSS")
axes[1].set_xlabel("Date")
#axes[1].set_title("UCTD Salinity Over Time")
axes[1].legend(loc='upper right', fontsize=8, frameon=True)
axes[1].grid(True)

plt.tight_layout()
plt.show()

fig.savefig('sst_sali_astral_2025.pdf', format="pdf", dpi=300)#, bbox_inches="tight")



