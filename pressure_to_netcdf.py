from __future__ import print_function
print('--------------------------------------------------------------------------------------------------------')
print('Importing')
import os
import fnmatch
import datetime
import subprocess
import numpy as np
import xarray as xr
import dask
print('   done\n')

outpath = '/scratch/zmanthos/thesis/'

years = np.arange(1997,2019,1)
print(years)
days=[32,29,32,31,32,31,32,32,31,32,31,32]
month = [1,2,3,4,5,6,7,8,9,10,11,12]
mons = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

print('convert(0) or compile(1)?')
ans = int(input())


nextyear = years[0];
if ans == 0:
	print('Compiling File Names')
	files=[];
	for y in years:
		for m in mons:
			for d in range(1,days[int(m)-1]):
				if int(d) < 10:
					d = '0'+str(d);
				else:
					d = str(d)
				path = '/shared/working/rean/era-interim/daily/data/'+str(int(y))+'/';
				filename = 'ei.oper.an.pl.regn128cm.'+str(int(y))+m+d+'00';
				file = path+filename
				files.append(path+filename)
	print('Files shape: ',np.shape(files),'\n')

	i=1;
	f = files[0]
	data = xr.open_dataset(f,engine='cfgrib',backend_kwargs={'indexpath':''})
	print(data['isobaricInhPa'])
	quit()
	geoh = data.z.sel(isobaricInhPa=500,latitude=slice(55,25), longitude=slice(360-130,360-50))
	gpot = geoh / 9.80665;
	gpot = gpot.to_dataset(name='z500')
	
	uwind = data.u.sel(isobaricInhPa=850,latitude=slice(55,25), longitude=slice(360-130,360-50))
	vwind = data.v.sel(isobaricInhPa=850,latitude=slice(55,25), longitude=slice(360-130,360-50))
	
	gpot['u'] = uwind;
	gpot['v'] = vwind;
	print(gpot,'\n')
	print('First file: ',f)
	i=0;
	for file in files:
		if i==0:
			i=1;
			print('File skipped: ',file)
			continue
		
		t = datetime.datetime.now().ctime()
		print('Timestamp: ', t,'File: ',file[-10:-2])  #Sat Feb 29 12:36:33 2020
		#print('  Getting data for: ',file)
		data = xr.open_dataset(file,engine='cfgrib',backend_kwargs={'indexpath':''})  #,backend_kwargs={'indexpath': ' ','filter_by_keys':{'name': 'Geopotential'}})
		geoh = data.z.sel(isobaricInhPa=500,latitude=slice(55,25), longitude=slice(360-130,360-50))
		geoh = geoh / 9.80665;
		geoh = geoh.to_dataset(name='z500')
		
		uwind = data.u.sel(isobaricInhPa=850,latitude=slice(55,25), longitude=slice(360-130,360-50))
		vwind = data.v.sel(isobaricInhPa=850,latitude=slice(55,25), longitude=slice(360-130,360-50))
	
		geoh['u'] = uwind;
		geoh['v'] = vwind;
			
		gpot = xr.combine_nested([gpot,geoh],concat_dim='time')
		gpot.attrs['Last File'] = file[-10:-2];
		#print(gpot)
		gfile = outpath + 'era-interim.geoh.500.u.v.850.nc'
		
		yyeeaarr = file[-10:-6]
		if int(yyeeaarr) == nextyear:
			gpot.to_netcdf(gfile)
			nextyear +=1;
			print("*** --- SAVED --- ***")
	print('Saving')
	gfile = outpath + 'era-interim.geoh.500.u.v.850.nc'
	gpot.to_netcdf(gfile)
	print('Saved\n')
	print(gpot)

else:
	path = outpath;	11
	data1 = xr.open_dataset(path+'era-interim.geoh.u.v.500.nc',decode_times=True,decode_cf=True)
	
	data2 = xr.open_dataset(path+'era-interim.geoh.u.v.500.1.nc',decode_times=True,decode_cf=True)
	#print(data2['time'])
	#data3 = xr.open_dataset(path+'era-interim.geoh.500.2.nc',decode_times=True,decode_cf=True)
	#print(data3['time'])
	#print(data1['time'])
	print(data2['time'])
	#print(data3['time'])
	#print(data1['z500'].values)
	#print(data2['z500'].values)
	#print(data3['z500'].values)
	end1 = data1['time'][-1].values
	#strt3 = data2['time'][0].values
	#data2 = data2.sel(time=slice(end1,strt3))

	# d2time = data2['time'][1:-1]
	# strt = d2time[0]
	# end = d2time[-1]
	# data2 = data2.sel(time=slice(strt,end))
	
	end2 = data2['time'][-1].values
	data2 = data2.sel(time=slice(end1,end2))
	
	print(data1['time'],'\n',data2['time'])
	
	gpot = xr.combine_nested([data1,data2],concat_dim='time') # ,data3
	
	clima = gpot['z500'].groupby('time.dayofyear').mean(dim='time',skipna=True)
	anoms = gpot['z500'].groupby('time.dayofyear') - clima;
	
	gpot['clima'] = clima;
	gpot['anoms'] = anoms;
	
	print(gpot)
	gfile = outpath + 'era-interim.geoh.u.v.500.97-18.nc'
	gpot.to_netcdf(gfile)
	



