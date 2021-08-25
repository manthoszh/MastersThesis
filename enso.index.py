from __future__ import print_function
print('--------------------------------------------------------------------------------------------------------')
print('Importing')
import datetime
import numpy as np
import xarray as xr

file = 'sst.wkmean.1990-present.nc'
path = '/shared/obs/gridded/OISSTv2/weekly/'

data = xr.open_dataset(path+file,decode_times=True,decode_cf=True)
timedata = xr.open_dataset('/scratch/zmanthos/thesis/gpcp.lat25-55N.lon130-50W.nc',decode_times=True, decode_cf=True)
newtime = timedata['time']

## Nino 3.4 area http://www.bom.gov.au/climate/ahead/about-ENSO-outlooks.shtml

data = data.sel(time=slice('1997-01-01', '2018-12-31'), lat=slice(5,-5), lon=slice(360-170,360-120))

mean = data['sst'].mean(dim='time')
anoms = data['sst'] - mean;
index = anoms.mean(dim=['lat','lon'])

sig = index.std().values
sig = np.round(sig,decimals=6)
print(sig,'\n')

index = index/sig;
index = np.round(index, decimals=6)

lat = data['lat']
lon = data['lon']
map = np.zeros(len(newtime))

newdata = xr.DataArray(map, dims=("time"), coords={'time':(['time'],timedata['time'])})

n=0;
for x in newtime:
	val = index.sel(time=x,method='nearest').values
	val = np.round(val,decimals=6)
	newdata[n] = val;
	n+=1;
	print('\033[FLoop iteration: ',n)

ds_index = newdata.to_dataset(name='enso')
ds_index.attrs['Title'] = 'Nino 3.4 Standardized index' 
ds_index.attrs['About'] = 'Area avg of weekly SST anoms Center of week starts on Jan 5th 1997'
ds_index.attrs['Area'] = 'Nino 3.4 area http://www.bom.gov.au/climate/ahead/about-ENSO-outlooks.shtml'
ds_index.attrs['Sigma'] = str(sig)
ds_index.attrs['By'] = 'Zak Manthos'
print(ds_index)

ds_index.to_netcdf('/homes/zmanthos/thesis/index/enso.nin34.97-18.nc')


