# Reading and slicing Data
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

t1 = datetime.datetime.now().ctime()
print('Start:',t1,'\n')

cont = True;
while cont == True:
	print('Data set: 1 = GPCP  2 = ERA5');
	answer = str(input()); 					## changed for nohup
	if answer == '1' or answer == '2':
		ans = int(answer)-1;
		print('\n')
		cont = False;
	else:
		print('invalid')

print('Handling Data')
paths = ['/scratch/zmanthos/GPCP/','/shared/land/ERA5/daily/global/']
files = ['gpcp_v01r03_daily.*.nc','compressed_sfc_fc_daily_*.nc4']

if ans == 0:
	data = xr.open_mfdataset(paths[0]+files[0], combine='by_coords', decode_times=True, decode_cf=True)
	data = data.sel(latitude=slice(25,55), longitude=slice(360-130,360-50), time=slice('1997-01-01','2018-12-31'))
	data.to_netcdf('/scratch/zmanthos/thesis/gpcp.lat25-55N.lon130-50W.nc')
	t = datetime.datetime.now().ctime()
	print('climo ',t)
	climo = data['precip'].groupby('time.dayofyear').mean(dim='time',skipna=True)
	t = datetime.datetime.now().ctime()
	print('anoms ',t)
	anoms = data['precip'].groupby('time.dayofyear')-climo;
	t = datetime.datetime.now().ctime()
	print('done ',t,'\n',anoms)
else:
	convert = xr.open_dataset('/shared/land/ERA5/daily/index.nc',decode_times=True, decode_cf=True)
	convert = convert.sel(lat=slice(25,55), lon=slice(360-130,360-50))
	
	data = xr.open_mfdataset(paths[1]+files[1], data_vars=['2t'], combine='by_coords', decode_coords=True,
		decode_times=True, decode_cf=True, engine='netcdf4')
	data = data.sel(time=slice(19970101,20181231)) 
	data = data['2t']
	print('  Loading Data')
	print('    ',datetime.datetime.now().ctime())
	data = data.load()
	print('    ',datetime.datetime.now().ctime())
	print(data,'\n')
	print('    data collected\n')
	print(data.attrs)
	quit()
	
	lon = convert['lon']
	l_lon = len(lon)
	lat = convert['lat']
	l_lat = len(lat)
	time = data['time']
	time = pd.to_datetime(time,foramt='%Y%m%d')
	print(time)
	l_time = len(time)
	print(l_lat,l_lon,l_time,'\n')
	
	map = np.zeros((l_lat,l_lon,l_time))
	#print(map)

	#print(convert,'\n')
	for x in range(0, 107):
		print('\033[F',x)
		for y in range(0, 285):
			index = convert['index'][x,y].values
			if np.isnan(index):
				map[x,y,:]=np.nan;
				continue; 
			else: 
				index=int(index);
			point = data.sel(lgrid=index).values
			map[x,y,:] = point[:];

	
	
	tps = xr.DataArray(map, dims=("lat", "lon", "time"), 
		coords={'lon': (['lon'],lon), 'lat':(['lat'],lat), 'time':(['time'],time)})
	temp = tps.to_dataset(name='2t')
	print(temp)
	# climo = temp['2t'].groupby('time.dayofyear').mean(dim='time',skipna=True)
	# temp['climo'] = climo;
	# anoms = temp['2t'].groupby('time.dayofyear')-temp['climo'];
	# tmep['anoms'] = anoms;
	# print(temp)
	temp.to_netcdf('/scratch/zmanthos/thesis/data.era5.2t.lat25-55N.lon130-50W.nc')
	
t = datetime.datetime.now().ctime()
print('Timestamps:\n   ', t1,'\n   ',t,'\n')