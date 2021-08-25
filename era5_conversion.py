## This code takes the ERA5 Land only data and converts to standard lat lon grid
## Created by: Zachary Manthos

from __future__ import print_function
print('--------------------------------------------------------------------------------------------------------')
print('Importing')
import os
import datetime
import subprocess
import numpy as np
import xarray as xr
import pandas as pd
import dask
print('    done\n')

path = '/shared/land/ERA5/daily/global/'
file = 'compressed_sfc_fc_daily_*.nc4'

# dataset that is used to convert the dimensions
convert = xr.open_dataset('/shared/land/ERA5/daily/index.nc',decode_times=True, decode_cf=True)

# Selecting a specific area to obtain the data, if want all comment out line 28
# vary prone to run out of memory recommend cutting or asking for access to cpu or atlas servers
lowlat = 25;
highlat = 55;
leftlon = 130;
rightlon = 50;
convert = convert.sel(lat=slice(lowlat,highlat), lon=slice(360-leftlon,360-rightlon))

print('Compiling data')
data = xr.open_mfdataset(path+file, combine='by_coords', decode_coords=True,
	decode_times=True, decode_cf=True, engine='netcdf4')

print(data,'\n')

# selecting time span to work with, if want all comment out line 31 
startdate = 19970101 # integer year mon day
enddate = 20181231
data = data.sel(time=slice(startdate,enddate)) 

# selecting the variable wanted
datavar = '2t';
data = data[datavar] 

# triggering the load of the data, can take awhile
print('Loading Data')
print('    ',datetime.datetime.now().ctime())
data = data.load()
print('    ',datetime.datetime.now().ctime(),'\n')
print(data,'\n')
print('data collected\n')

# obtaining lat and lon info
lon = convert['lon']
l_lon = len(lon)
lat = convert['lat']
l_lat = len(lat)

# obtaining time span
time = data['time'].values
print(time)
time = pd.to_datetime(time,format='%Y%m%d')
l_time = len(time)
print('Time dimension: \n',time)

# creating new map to project values on to
map = np.zeros((l_lat,l_lon,l_time))

# loop converting dimensions
print('\nConverting')
for x in range(0, l_lat):
	print(' Working on lat: ',x,end='\r')
	for y in range(0, l_lon):
		index = convert['index'][x,y].values
		if np.isnan(index):
			map[x,y,:]=np.nan;
			continue; 
		else: 
			index=int(index);
		point = data.sel(lgrid=index).values
		map[x,y,:] = point[:];
print() # for output spacing

# creating xarray dataArray
darray = xr.DataArray(map, dims=("lat", "lon", "time"), 
	coords={'lon': (['lon'],lon), 'lat':(['lat'],lat), 'time':(['time'],time)})

# converting to dataset, needed for saving
dset = darray.to_dataset(name=datavar)

print('\nConverted dataset\n',dset)

# saving dataset to netcdf file
# outpath =  # directory path were the file will be saved start and end with "/"
# outfile =  # name of file ".nc" needed at end
# dset.to_netcdf(outpath+outfile)
# print('\nSave Location: ',(outpath+outfile),'\n')



