from __future__ import print_function
print('--------------------------------------------------------------------------------------------------------')
print('Importing')
import datetime
import numpy as np
import xarray as xr
import pandas as pd

path = '/scratch/zmanthos/thesis/'
file = 'data.era5.2t.lat25-55N.lon130-50W.nc'
data = xr.open_dataset(path+file,decode_times=True,decode_cf=True)

time = data['time'].values

datestr = str(time[0])[0:8]
print(datestr)
date = datetime.datetime.strptime(datestr, '%Y%m%d')
date = date.date()
newtime = [date];

for x in range(1,len(time)):
	datestr = str(time[x])[0:8]
	date = datetime.datetime.strptime(datestr, '%Y%m%d')
	date = date.date()
	newtime.append(date);

newtime = pd.to_datetime(newtime)
data['time'] = newtime;
print(data)

outfile = 'era5.2t.lat25-55N.lon130-50W.nc'

data.to_netcdf(path+outfile)