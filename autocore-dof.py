from __future__ import print_function
print('--------------------------------------------------------------------------------------------------------')
print('Importing')
import os
import datetime
import subprocess
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
print('   done\n')

path = '/scratch/zmanthos/thesis/';
outpath = '/homes/zmanthos/thesis/plots/autocore/'

#era5 = xr.open_dataset(path+'era5.2t.C-A.lat25-55N.lon130-50W.nc', decode_times=True, decode_cf=True)
era5 = xr.open_dataset('/scratch/zmanthos/thesis/gpcp.lat25-55N.lon130-50W.nc', decode_times=True, decode_cf=True)
print(era5)

# era5 = era5['anoms']

# elat = era5['lat']
# elon = era5['lon']

era5 = era5['precip']

elat = era5['latitude']
elon = era5['longitude']
factors = [];
for year in range(1997,2019):
	year = str(year)
	print('Loop year: ',year)
	
	era51 = era5.sel(time=slice(year+'-01-01',year+'-12-30'))
	era52 = era5.sel(time=slice(year+'-01-02',year+'-12-31'))
	length1 =len(era51['time'])
	length2 =len(era52['time'])
	print('lengths: ',length1,length2)
	loc1 = range(0,length1)
	loc2 = range(0,length2)
	era51 = era51.assign_coords(time=(loc1))
	era52 = era52.assign_coords(time=(loc2))
	
	eacorr = xr.corr(era51,era52,dim='time')
	
	era5corradj = eacorr.where(eacorr>0)

	era5log = np.log(era5corradj)
	
	#meanera5log = era5log.mean(dim=['lat','lon'],skipna=True)
	meanera5log = era5log.mean(dim=['latitude','longitude'],skipna=True)
	factor = -1/meanera5log.values;
	factors.append(factor)
	print('DOF Factors:')
	print('   ERA5: ',(factor),' meanlog: ',meanera5log.values)
	print('\n')

mean = np.mean(factors)

print('DOF Factor Average: ',mean)




