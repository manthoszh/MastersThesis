from __future__ import print_function
print('--------------------------------------------------------------------------------------------------------')
print('Importing')
import datetime
import numpy as np
import xarray as xr
import pandas as pd

path = '/data/vortex/scratch/'
files = ['norm.daily.nao.index.b500101.current.ascii','norm.daily.pna.index.b500101.current.ascii']

which = 0;

data = pd.read_csv(path+files[which],header=None, delim_whitespace=True)

year = data[0]
month = data[1]
day = data[2]
index_vals = data[3]
length = len(year)

datestr = str(year[0])+'-'+str(month[0])+'-'+str(day[0])[0:2]
date = datetime.datetime.strptime(datestr, '%Y-%m-%d')
date = date.date()
time = [date];

for x in range(1,length):
	datestr = str(year[x])+'-'+str(month[x])+'-'+str(day[x])[0:2]
	date = datetime.datetime.strptime(datestr, '%Y-%m-%d')
	date = date.date()
	time.append(date);

index_vals = index_vals.to_numpy()
time = pd.to_datetime(time)

index = xr.DataArray(index_vals, dims=("time"), 
		coords={'time':(['time'],time)})

name = ['nao','pna']
loc = ['nao','pna']

ds_index = index.to_dataset(name=name[which])
ds_index.attrs['Info'] = 'Standardized '+name[which]+' Index'
ds_index.attrs['Source'] = 'https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/'+loc[which]+'.shtml'
ds_index.attrs['By'] = 'Zak Manthos'

print(ds_index)
outfile = '/homes/zmanthos/thesis/index/'+loc[which]+'.index.nc'
ds_index.to_netcdf(outfile)
	
