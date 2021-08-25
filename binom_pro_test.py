from __future__ import print_function
print('--------------------------------------------------------------------------------------------------')
print('Importing')
import os
import datetime
import subprocess
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import scipy.stats as sp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
print('    done\n')

def conplot(data,min,max,space):
	fig = plt.figure(figsize=(10,6))
	ax = plt.axes(projection=ccrs.PlateCarree())
	levels = np.arange(min,max,space)
	print(levels)
	plt.contourf( lon, lat, data, extend='both', cmap='terrain_r', levels=levels, transform=ccrs.PlateCarree())
	ax.coastlines(resolution='110m')
	plt.colorbar(ax=ax, shrink=0.6, orientation='horizontal')
	ax.gridlines(color="black", linestyle="dotted")
	plt.show()

## question to pick out scenario
cont = True;
while cont == True:
	print('Data set? 1:GPCP  2:ERA5');
	answer1 = str(int(input())-1);
	if answer1 == '0' or answer1 == '1':
		answer1 = int(answer1)
		print()
		cont = False;
	else:
		print('invalid')
cont = True;
while cont == True:
	print('Season? 1: Summer May-Aug  2: Winter Oct-Apr');
	answer2 = str(int(input())-1);
	if answer2 == '0' or answer2 == '1':
		sea = int(answer2)
		print()
		cont = False;
	else:
		print('invalid')
if sea == 0:
	seaname = 'Summer May-Aug'
	fs1 = 'summer'
else:
	seaname = 'Winter Oct-Apr'
	fs1 = 'winter'
	
cont = True;
ans = [0,0]
while cont == True:
	print('Multi Index: 1 = Yes  2 = No');
	answer3 = str(input());
	print()
	if answer3 == '1':
		print('Which 2? 1:MLSO 2:ENSO 3:NAO 4:PNA - spearate by comma & no spaces')
		inputed = str(input())
		multi = True;
		ans[0] = int(inputed[0])-1;
		ans[1] = int(inputed[2])-1;
		print()
		cont = False;
	elif answer3 == '2':
		print('Which? 1:MLSO 2:ENSO 3:NAO 4:PNA')
		ans = int(input())-1;
		multi = False;
		print()
		cont = False;
	else:
		print('invalid')
ans1 = ans;

if multi:
	cont = True;
	while cont == True:
		print('In or Out of Phase? 1:IN 2:OUT');
		phase = int(input())-1;
		ans1 = str(ans[0])+' '+str(ans[1])+' '+str(phase);
		if phase == 0:
			phasename = 'inphase';
			print()
			cont=False;
		elif phase == 1:
			phasename = 'outphase';
			print()
			cont=False;
		else:
			print('invalid entry')

## opening data file
dfiles = ['/scratch/zmanthos/thesis/gpcp.lat25-55N.lon130-50W.nc','/scratch/zmanthos/thesis/era5.2t.C-A.lat25-55N.lon130-50W.nc']
data = xr.open_dataset(dfiles[answer1], decode_times=True, decode_cf=True)
print('Data time length: ',len(data['time']),'\n')

## setting up variables and stuff 
if answer1 == 0:
	## gpcp
	fn1 = 'gpcp'
	lon = data['longitude']
	l_lon = len(lon)
	lat = data['latitude']
	l_lat = len(lat)
	datavar = 'precip'
	varnames = ['-wet','-dry']
	threshold = 0;
	data = data[datavar].where(data[datavar]>=0.01,other=0)
	data = data.to_dataset(name=datavar);
	#print(data)
	#data = data[datavar]
	#data = data.where(data>0,other=np.nan)
	#print(data.values)
	#clima = data.groupby('time.dayofyear').mean(skipna=True)
	#clima = clima.rolling(dayofyear=5, center=True).mean()
	# ## PLOT ##

	# lat = 30
	# lon = 100
	
	# xdat = clima.sel(latitude=lat,longitude=(360-lon),method='nearest')
	# y = xdat.values
	# x = clima['dayofyear'].values
	# plt.plot(x,y)
	# plt.ylabel(datavar)
	# plt.xlabel('Day of year')
	# plt.title('Clima Lat: '+str(lat)+'N Lon: '+str(lon)+'W '+fs1)
	# plt.show()
	# quit()
	
	
	#anoms = data.groupby('time.dayofyear')-clima;
	#data = anoms.to_dataset(name=datavar);
	
else:
	## era5
	fn1 = 'era5'
	lon = data['lon']
	l_lon = len(lon)
	lat = data['lat']
	l_lat = len(lat)
	datavar = 'anoms';
	varnames = ['-warm','-cold']
	threshold = 0;
	
	## PLOT ##
	
	lat = 30
	lon = 100
	
	xdat = data['clima'].sel(lat=lat,lon=(360-lon),method='nearest')
	y = xdat.values
	x = data['dayofyear'].values
	plt.plot(x,y)
	plt.ylabel('temp')
	plt.xlabel('Day of year')
	plt.title('Clima Lat: '+str(lat)+'N Lon: '+str(lon)+'W '+fs1)
	plt.show()
	quit()
	
	data = data.drop_vars(['2t','clima'])
	data = data.drop_dims('dayofyear')



# ## Histogram
# print('lat?')
# lat = int(input())
# print('lon?')
# lon = int(input())
# print(data)
# histdata = data[datavar].sel(latitude=lat,longitude=(360-lon),method='nearest')
# hdata = histdata.values
# print(len(hdata))
# #print(hdata)
# plt.hist(hdata,100,rwidth=0.75)
# plt.axvline(np.nanmean(hdata),color='k',linestyle='dashed')
# plt.axvline(np.nanmedian(hdata),color='k')
# plt.axvline(0,color='red')
# plt.xlabel('Precip mm/day')
# plt.ylabel('Frequency')
# plt.title('Lat: '+str(lat)+'N Lon: '+str(lon)+'W ')
# plt.show()
# quit()


## Selecting Indices
path = '/homes/zmanthos/thesis/index/'
ifiles = ['enso.nin34.97-18.nc', 'mlso.index.01011979-08312019.nc', 'nao.index.nc', 'pna.index.nc']

names = [];
if np.isin(ans,0).any():
	mlso = xr.open_dataset(path+ifiles[1],decode_times=True,decode_cf=True)
	mlso = mlso.sel(time=slice('1997-01-01','2018-12-31'))
	names.append('mlso')
	print('MLSO dims:',len(mlso['time']))

if np.isin(ans,1).any():
	enso = xr.open_dataset(path+ifiles[0],decode_times=True,decode_cf=True)
	enso = enso.sel(time=slice('1997-01-01','2018-12-31'))
	names.append('enso')
	print('ENSO dims:',len(enso['time']))
	
if np.isin(ans,2).any():
	nao = xr.open_dataset(path+ifiles[2],decode_times=True,decode_cf=True)
	nao = nao.sel(time=slice('1997-01-01','2018-12-31'))
	names.append('nao')
	print('NAO dims:',len(nao['time']))
	
if np.isin(ans,3).any():
	pna = xr.open_dataset(path+ifiles[3],decode_times=True,decode_cf=True)
	pna = pna.sel(time=slice('1997-01-01','2018-12-31'))
	names.append('pna')
	print('PNA dims:',len(pna['time']))
print()

print('Filtering Data\n')
## -- Data Slicing -- ##
map = np.zeros((l_lat,l_lon),dtype=int)
map2 = np.zeros((l_lat,l_lon),dtype=int)

ratio = xr.DataArray(map, dims=("lat", "lon"), 
		coords={'lon': (['lon'],lon), 'lat':(['lat'],lat)})
ratios = ratio.to_dataset(name='rmax')


rmin = xr.DataArray(map2, dims=("lat", "lon"), 
		coords={'lon': (['lon'],lon), 'lat':(['lat'],lat)})
ratios['rmin'] = rmin;


ratios.attrs['Season'] = seaname;

if multi:
	ratios.attrs['Phase state'] = phasename;
ratios.attrs['Threshold'] = threshold;

if multi:
	limits=[10000,0.75,0,-0.75,-10000]
	index1 = globals()[names[0]][names[0]]
	index2 = globals()[names[1]][names[1]]
	
	for x in range(0,(len(limits)-1)):
		lim1 = limits[x+1];
		lim2 = limits[x];
		print('Limits:')
		print(lim1,lim2)

		sigs = index1.where(index1>lim1).dropna(dim='time')
		#print(sigs)
		sigs = sigs.where(sigs<lim2).dropna(dim='time')
		#print(sigs)
		
		timecut = sigs['time'].values
		indexcut = index2.where(index2['time'].isin(timecut)).dropna(dim='time',how='all')

		if phase == 0:
			sigs2 = indexcut.where(indexcut>lim1).dropna(dim='time')
			#print(sigs2)
			sigs2 = sigs2.where(sigs2<lim2).dropna(dim='time')
			#print(sigs2)
			
			timecut2 = sigs2['time'].values
			indexcut = indexcut.where(indexcut['time'].isin(timecut2)).dropna(dim='time',how='all')
		else:
			lim11 = -1*lim2;
			lim22 = -1*lim1;
			print(lim11,lim22)
			sigs2 = indexcut.where(indexcut>lim11).dropna(dim='time')
			sigs2 = sigs2.where(sigs2<lim22).dropna(dim='time')
			
			timecut2 = sigs2['time'].values
			indexcut = indexcut.where(indexcut['time'].isin(timecut2)).dropna(dim='time',how='all')
	
		dates = pd.to_datetime(indexcut['time'].values)
		mon = dates.month
		season = [[1,2,3,10,11,12],[4,5,6,7,8,9]]
		test = np.isin(mon,season[sea])
		#print(indexcut['time'],test)
		timecut_season = ma.masked_array(indexcut['time'],test)
		timecut_season = timecut_season.compressed()
		#print(timecut_season)
		print(' ---- Days after filter:  ',len(timecut_season),' ----')
		
		datacut = data.where(data['time'].isin(timecut_season)).dropna(dim='time',how='all')
		totdays = len(datacut['time'])
		print(' ---- Days after slicing: ',totdays,' ----\n')
		
		wetcount = datacut.where(datacut[datavar]>threshold).count(dim='time')
		drycount = totdays-wetcount[datavar];
		ratio1 = wetcount[datavar]/drycount;
		
		mmean = ratio1.mean()
		print('Mean Ratio: ',mmean)
		conplot(ratio1,0.1,0.6,0.01)
		#print(ratio1)
		eman = 'bin'+str(x+1)+'-ratio';
		eman1 = 'bin'+str(x+1)+varnames[0];
		eman2 = 'bin'+str(x+1)+varnames[1];
		ratios[eman] = ratio1;
		ratios[eman1] = wetcount[datavar];
		ratios[eman2] = drycount;
		ratios.attrs['bin'+str(x+1)+' days'] = totdays;
		ratios.attrs['bin'+str(x+1)+' - '+names[0]+' limits'] = [lim2,lim1]
		if phase == 1:
			ratios.attrs['bin'+str(x+1)+' - '+names[1]+' limits'] = [lim22,lim11]
		#print(ratios)
	quit()
	## picking highest ratio ##
	print("Finding Highest Ratio")
	nans=0;
	for x in range(0,l_lat):
		for y in range(0,l_lon):
			print("Working ",x,y,'   nans: ',nans,'   ',end='\r')
			nums = [ratios['bin1-ratio'][x,y].values,ratios['bin2-ratio'][x,y].values,
				ratios['bin3-ratio'][x,y].values,ratios['bin4-ratio'][x,y].values]
			
			binmax = int(nums.index(max(nums)))
			binmin = int(nums.index(min(nums)))
			if nums[binmax]==0 and nums[binmin]==0:
				ratios['rmax'][x,y] = 0;
				ratios['rmin'][x,y] = 0;
				nans+=1;
			else:
				ratios['rmax'][x,y] = binmax+1;
				ratios['rmin'][x,y] = binmin+1;
			
	## Binomial Proportion Test ##
	print("\n\nStatistics")
	pmap1 = np.zeros((l_lat,l_lon))
	pmap2 = np.zeros((l_lat,l_lon))
	tmap1 = np.zeros((l_lat,l_lon))
	tmap2 = np.zeros((l_lat,l_lon))
	n=1;
	
	for r in ['rmax','rmin']:
		print('   ',r)
		skips=0;
		for x in range(0,l_lat):
			for y in range(0,l_lon):
				print("Working ",x,y,'  Skip count: ',skips,'   ',end='\r')
				maxbin = ratios[r][x,y].values;
				if maxbin == 0:
					skips+=1;
					pval = np.nan;
					globals()['pmap'+str(n)][x,y]=pval;
					continue
				
				treatA = ratios['bin'+str(maxbin)+varnames[0]][x,y].values
				treatC = ratios['bin'+str(maxbin)+varnames[1]][x,y].values
				treatM = treatA+treatC;
				
				controlB=0;
				controlD=0;
				for f in range(1,5):
					if f == maxbin:
						continue
					controlB = controlB+ratios['bin'+str(f)+varnames[0]][x,y].values
					controlD = controlD+ratios['bin'+str(f)+varnames[1]][x,y].values
				controlN = controlB+controlD;
				
				totalN = treatM+controlN;
				
				if answer1 == 1:
					DOF = (totalN-5)/3.395556;
					ratios.attrs['dof factor'] = '1/3.395556';
				else:
					DOF = totalN-5;
					ratios.attrs['dof factor'] = 'none';
				
				part1 = treatA*controlD-controlB*treatC;
				part2 = controlN*treatA*treatC+treatM*controlB*controlD
				part3 = (totalN*(part2))
				part4 = totalN-DOF;
				t_val = (part1)*((part4)/part3)**0.5;
							
				p_val =1-sp.t.cdf(abs(t_val),df=DOF)
				#print(part1,part4,part3,t_val,p_val)
				globals()['pmap'+str(n)][x,y]=abs(p_val)*2;
				globals()['tmap'+str(n)][x,y]=abs(t_val);
		
		globals()['pmap'+str(n)] = xr.DataArray(globals()['pmap'+str(n)], dims=("lat", "lon"), 
			coords={'lon': (['lon'],lon), 'lat':(['lat'],lat)})
		globals()['tmap'+str(n)] = xr.DataArray(globals()['tmap'+str(n)], dims=("lat", "lon"), 
			coords={'lon': (['lon'],lon), 'lat':(['lat'],lat)})
		ratios[r+'-p_val'] = globals()['pmap'+str(n)];
		ratios[r+'-t_val'] = globals()['tmap'+str(n)];
		n+=1;
		print()





## SOLO ##
else:
	limits=[10000,0.75,0,-0.75,-10000]
	index1 = globals()[names[0]][names[0]]
	## -- Data Slicing -- ##
	
	n = 1;
	for x in range(0,(len(limits)-1)):
		lim1 = limits[x+1];
		lim2 = limits[x];
	
		sigs = index1.where(index1>lim1).dropna(dim='time')
		sigs = sigs.where(sigs<lim2).dropna(dim='time')
		
		timecut = sigs['time'].values
		indexcut = index1.where(index1['time'].isin(timecut)).dropna(dim='time',how='all')
			
		dates = pd.to_datetime(indexcut['time'].values)
		
		mon = dates.month
		season = [[1,2,3,10,11,12],[4,5,6,7,8,9]]
		test = np.isin(mon,season[sea])
		#print(indexcut['time'],test)
		timecut_season = ma.masked_array(indexcut['time'],test)
		timecut_season = timecut_season.compressed()
		#print(timecut_season)
		print('\n ---- Days after filter:  ',len(timecut_season),' ----')
		
		datacut = data.where(data['time'].isin(timecut_season)).dropna(dim='time',how='all')
		totdays = len(datacut['time'])
		print(' ---- Days after slicing: ',totdays,' ----\n')
		
		wetcount = datacut.where(datacut[datavar]>threshold).count(dim='time')
		drycount = totdays-wetcount[datavar];
		ratio1 = wetcount[datavar]/drycount;

		mmean = ratio1.mean()
		print('Mean Ratio: ',mmean)
		#print(ratio1)
		#conplot(ratio1,0.1,0.6,0.01)
		
		
		eman = 'bin'+str(x+1)+'-ratio';
		eman1 = 'bin'+str(x+1)+varnames[0];
		eman2 = 'bin'+str(x+1)+varnames[1];
		ratios[eman] = ratio1;
		ratios[eman1] = wetcount[datavar];
		ratios[eman2] = drycount;
		ratios.attrs['bin'+str(x+1)+' days'] = totdays;
		ratios.attrs['bin'+str(x+1)+' limits'] = [lim2,lim1]
		#print(ratios)
	

	## picking highest ratio ##
	print("Finding Highest Ratio")
	nans=0;
	for x in range(0,l_lat):
		for y in range(0,l_lon):
			print("Working ",x,y,'   nans: ',nans,'   ',end='\r')
			nums = [ratios['bin1-ratio'][x,y].values,ratios['bin2-ratio'][x,y].values,
				ratios['bin3-ratio'][x,y].values,ratios['bin4-ratio'][x,y].values]
			
			binmax = int(nums.index(max(nums)))
			binmin = int(nums.index(min(nums)))
			if nums[binmax]==0 and nums[binmin]==0:
				ratios['rmax'][x,y] = 0;
				ratios['rmin'][x,y] = 0;
				nans+=1;
			else:
				ratios['rmax'][x,y] = binmax+1;
				ratios['rmin'][x,y] = binmin+1;
			
			
	## Binomial Proportion Test ##
	print("\n\nStatistics")
	pmap1 = np.zeros((l_lat,l_lon))
	pmap2 = np.zeros((l_lat,l_lon))
	tmap1 = np.zeros((l_lat,l_lon))
	tmap2 = np.zeros((l_lat,l_lon))
	n=1;
	for r in ['rmax','rmin']:
		print('   ',r)
		skips = 0;
		for x in range(0,l_lat):
			for y in range(0,l_lon):
				print("Working ",x,y,'  Skip count: ',skips,'   ',end='\r')
				
				maxbin = ratios[r][x,y].values;
				if maxbin == 0:
					skips+=1;
					pval = np.nan;
					globals()['pmap'+str(n)][x,y]=pval;
					continue
				
				treatA = ratios['bin'+str(maxbin)+varnames[0]][x,y].values
				treatC = ratios['bin'+str(maxbin)+varnames[1]][x,y].values
				treatM = treatA+treatC;
				
				controlB=0;
				controlD=0;
				for f in range(1,5):
					if f == maxbin:
						continue
					controlB = controlB+ratios['bin'+str(f)+varnames[0]][x,y].values
					controlD = controlD+ratios['bin'+str(f)+varnames[1]][x,y].values
				controlN = controlB+controlD;
				
				totalN = treatM+controlN;
				
				if answer1 == 1:
					DOF = (totalN-5)/3.395556;
					ratios.attrs['dof factor'] = '1/3.395556';
				else:
					DOF = totalN-5;
					ratios.attrs['dof factor'] = 'none';
				
				part1 = treatA*controlD-controlB*treatC;
				part2 = controlN*treatA*treatC+treatM*controlB*controlD
				part3 = (totalN*(part2))
				part4 = totalN-DOF;
				t_val = (part1)*((part4)/part3)**0.5;
							
				p_val =1-sp.t.cdf(abs(t_val),df=totalN-DOF)
				#print(part1,part4,part3,t_val,p_val)
				globals()['pmap'+str(n)][x,y]=abs(p_val)*2;
				globals()['tmap'+str(n)][x,y]=abs(t_val);
			
		globals()['pmap'+str(n)] = xr.DataArray(globals()['pmap'+str(n)], dims=("lat", "lon"), 
			coords={'lon': (['lon'],lon), 'lat':(['lat'],lat)})
		globals()['tmap'+str(n)] = xr.DataArray(globals()['tmap'+str(n)], dims=("lat", "lon"), 
			coords={'lon': (['lon'],lon), 'lat':(['lat'],lat)})
		ratios[r+'-p_val'] = globals()['pmap'+str(n)];
		ratios[r+'-t_val'] = globals()['tmap'+str(n)];
		n+=1;
		print()		


print('\n',ratios,'\n')
print('Min Pvals: ',ratios['rmax-p_val'].min(skipna=True).values,ratios['rmin-p_val'].min(skipna=True).values)
print('Max Tvals: ',ratios['rmax-t_val'].max(skipna=True).values,ratios['rmin-t_val'].max(skipna=True).values)

outpath = '/scratch/zmanthos/thesis/outdata/'

fi1 = '';
for f in names:
	fi1 = fi1+f+'.';

if multi:
	fname = fn1+'.'+fi1+phasename+'.'+fs1+'.bpt.nc'
else:
	fname = fn1+'.'+fi1+fs1+'.bpt.nc'
print('file: ',fname)

ratios.to_netcdf(outpath+fname)







