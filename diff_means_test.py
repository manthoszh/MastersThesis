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
from scipy.stats import ranksums
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
print('    done\n')

def conplot(data):
	fig = plt.figure(figsize=(10,6))
	ax = plt.axes(projection=ccrs.PlateCarree())
	plt.contourf( lon, lat, data, extend='both', transform=ccrs.PlateCarree())
	ax.coastlines(resolution='110m')
	plt.colorbar(ax=ax, shrink=0.6, orientation='horizontal')
	ax.gridlines(color="black", linestyle="dotted")
	plt.show()
	

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

ext1 = '';
tail=0;
cont = True;
while cont == True:
	print('Extremes? yes(1) or no(2)')
	ext = int(input())
	if ext == 1:
		print('Upper(1) or Lower(2)')
		tail = int(input())
		if tail == 1:
			qt = 'upper'
			percent = 0.95;
		elif tail == 2:
			qt = 'lower'
			percent = 0.05;
		fs1 = fs1+'.'+qt+'-extreme';
		ext1 = str(ext)+' '+str(tail)
		cont = False;
	elif ext == 2:
		cont = False;
	else:
		print('invaild answer')
print()


dfiles = ['/scratch/zmanthos/thesis/gpcp.lat25-55N.lon130-50W.nc','/scratch/zmanthos/thesis/era5.2t.C-A.lat25-55N.lon130-50W.nc']
data = xr.open_dataset(dfiles[answer1], decode_times=True, decode_cf=True)

### threshold stuff ###
# print(data)
# count = data['precip'].where(data['precip']<0.011).count().values
# print(count)
# count1 = data['precip'].where(data['precip']==0).count().values
# print(count1)
# print(count-count1)
# quit()
########################

print('Data time length: ',len(data['time']),'\n')

if answer1 == 0:
	## gpcp
	fn1 = 'gpcp'
	lon = data['longitude']
	l_lon = len(lon)
	lat = data['latitude']
	l_lat = len(lat)
	datavar = 'precip'
	varnames = ['-avg']
	threshold = 0;
	data = data[datavar]
	##
	data = data.where(data>0,other=np.nan)
	print(data.values)
	clima = data.groupby('time.dayofyear').mean(skipna=True)
	clima = clima.rolling(dayofyear=5, center=True).mean()
	##
	anoms = data.groupby('time.dayofyear')-clima;
	data = anoms.to_dataset(name=datavar);
else:
	## era5
	fn1 = 'era5'
	lon = data['lon']
	l_lon = len(lon)
	lat = data['lat']
	l_lat = len(lat)
	datavar = 'anoms';
	varnames = ['-avg']
	threshold = 0;
	data = data.drop_vars(['2t','clima'])
	data = data.drop_dims('dayofyear')

if ext == 1:
	print('\nHandling Extremes')
	quant = data[datavar].quantile(percent,dim='time')
	if tail == 1:
		data = data.where(data[datavar]>quant)
	elif tail ==2:
		data = data.where(data[datavar]<quant)
	

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


print('Filtering Data')
## -- Data Slicing -- ##
map = np.zeros((l_lat,l_lon),dtype=int)
map2 = np.zeros((l_lat,l_lon),dtype=int)

aavg = xr.DataArray(map, dims=("lat", "lon"), 
		coords={'lon': (['lon'],lon), 'lat':(['lat'],lat)})

avgs = aavg.to_dataset(name='avgmax')

amin = xr.DataArray(map2, dims=("lat", "lon"), 
		coords={'lon': (['lon'],lon), 'lat':(['lat'],lat)})

avgs['avgmin'] = amin;	
avgs.attrs['Season'] = seaname;

if multi:
	avgs.attrs['Phase state'] = phasename;

avgs.attrs['Threshold'] = threshold;
if ext == 'y':
	avgs.attrs['Quantile'] = str(int(percent*100))+'th';

if answer3 == '1':
	limits=[10000,0.75,0,-0.75,-10000]
	index1 = globals()[names[0]][names[0]]
	index2 = globals()[names[1]][names[1]]
	
	for x in range(0,(len(limits)-1)):
		lim1 = limits[x+1];
		lim2 = limits[x];
	
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
		totdays =len(datacut['time'])
		print(' ---- Days after slicing: ',totdays,' ----\n')
		

		# if answer1 == 0:
			# globals()['datacut'+str(x+1)] = datacut.where(datacut[datavar]>=threshold)
		# else:
		globals()['datacut'+str(x+1)] = datacut;
		count = datacut.where(datacut[datavar]>=threshold).count(dim='time')
		avg = globals()['datacut'+str(x+1)].mean(dim='time',skipna=True)
		#print(avg)
		
		eman = 'bin'+str(x+1)+varnames[0]
		eman1 = 'bin'+str(x+1)+'-passthresh';
		avgs[eman] = avg[datavar];
		avgs[eman1] = count[datavar];
		avgs.attrs['bin'+str(x+1)+' days'] = totdays;
		avgs.attrs['bin'+str(x+1)+' limits'] = [lim2,lim1]
		if phase == 1:
			avgs.attrs['bin'+str(x+1)+' phased limits'] = [lim22,lim11]


	## picking highest Avg ##
	nans = 0;
	print("Finding Highest Averages")
	for x in range(0,l_lat):
		for y in range(0,l_lon):
			print("Working ",x,y,' number of nans: ',nans,'  ',end='\r')
			nums = [avgs['bin1-avg'][x,y].values,avgs['bin2-avg'][x,y].values,
				avgs['bin3-avg'][x,y].values,avgs['bin4-avg'][x,y].values]
			binmax = int(nums.index(max(nums)))+1;
			binmin = int(nums.index(min(nums)))+1;
			if np.isnan(nums[binmax-1]):
				avgs['avgmax'][x,y] = 0;
				avgs['avgmin'][x,y] = 0;
				nans = nans+1;
			else:
				avgs['avgmax'][x,y] = binmax;
				avgs['avgmin'][x,y] = binmin;
			
	
	## Difference of Means Test ##
	print("\n\nStatistics")
	pmap1 = np.zeros((l_lat,l_lon))
	pmap2 = np.zeros((l_lat,l_lon))
	n=1;
	
	for a in ['avgmax','avgmin']:
		print('   ',a)
		skips=0;
		for x in range(0,l_lat):
			for y in range(0,l_lon):
				print("Working ",x,y,'  Skip count: ',skips,'   ',end='\r')
				
				maxbin = int(avgs[a][x,y].values)
				if maxbin == 0:
					skips+=1;
					pval = np.nan;
					globals()['pmap'+str(n)][x,y]=pval;
					continue
					
				
				outs = [];
				for q in range(1,len(limits)):
					if q != maxbin:
						outs.append(q)

				if answer1 == 0:
					group1a = globals()['datacut'+str(maxbin)][datavar][:,x,y].values
				else:
					group1a = globals()['datacut'+str(maxbin)][datavar][x,y,:].values
				
				group1 = [];
				for ft in group1a:
					if np.isnan(ft) == False:
						group1.append(ft)
				
				group2 = [];
				for z in outs:
					if answer1 == 0:
						vals = globals()['datacut'+str(z)][datavar][:,x,y].values
					else:
						vals = globals()['datacut'+str(z)][datavar][x,y,:].values
					for v in vals:
						if np.isnan(v) == False:
							group2.append(v)
				
				
				wilcox = ranksums(group1,group2);
				pval = wilcox.pvalue
				globals()['pmap'+str(n)][x,y]=abs(pval)
				
		globals()['pmap'+str(n)] = xr.DataArray(globals()['pmap'+str(n)], dims=("lat", "lon"), 
			coords={'lon': (['lon'],lon), 'lat':(['lat'],lat)})
		avgs[a+'-p_val'] = globals()['pmap'+str(n)];
		n+=1;
		print()


else:
	#limits=[10000,1.5,0.75,0,-0.75,-1.5,-10000]
	limits=[10000,0.75,0,-0.75,-10000]
	index1 = globals()[names[0]][names[0]]
	## -- Data Slicing -- ##
	
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
		print(' ---- Days after filter:  ',len(timecut_season),' ----')
		
		datacut = data.where(data['time'].isin(timecut_season)).dropna(dim='time',how='all')
		totdays =len(datacut['time'])
		print(' ---- Days after slicing: ',totdays,' ----\n')
		
		# ## Histogram
		
		# if x == 3:
			# print('lat?')
			# lat = int(input())
			# print('lon?')
			# lon = int(input())
			
			# histdata = datacut.sel(lat=lat,lon=(360-lon),method='nearest')
			# hdata = histdata['anoms'].values
			# print(histdata)
			# plt.hist(hdata,50)
			# plt.axvline(np.nanmean(hdata),color='k',linestyle='dashed')
			# plt.axvline(np.nanmedian(hdata),color='k')
			# plt.axvline(0,color='red')
			# plt.xlabel('Temperature C')
			# plt.ylabel('Frequency')
			# plt.title('Lat: '+str(lat)+'N Lon: '+str(lon)+'W')
			# plt.show()
			# quit()
			
		# ##
		
		# if answer1 == 0:
			# globals()['datacut'+str(x+1)] = datacut.where(datacut[datavar]>=threshold)
		# else:
		globals()['datacut'+str(x+1)] = datacut;
		count = datacut.where(datacut[datavar]>=threshold).count(dim='time')
		avg = globals()['datacut'+str(x+1)].mean(dim='time',skipna=True)
		
		#conplot(avg[datavar])
		
		
		eman = 'bin'+str(x+1)+varnames[0]
		eman1 = 'bin'+str(x+1)+'-passthresh';
		avgs[eman] = avg[datavar];
		avgs[eman1] = count[datavar];
		avgs.attrs['bin'+str(x+1)+' days'] = totdays;
		avgs.attrs['bin'+str(x+1)+' limits'] = [lim2,lim1]
	#quit()
	#print(avgs)
	
	## picking highest Avg ##
	print("Finding Highest Averages")
	maxskips=0;
	minskips=0;
	for x in range(0,l_lat):
		for y in range(0,l_lon):
			print("Working ",x,y,' ',maxskips,minskips,'  ',end='\r')
			# nums = [avgs['bin1-avg'][x,y].values,avgs['bin2-avg'][x,y].values,
				# avgs['bin3-avg'][x,y].values,avgs['bin4-avg'][x,y].values,
				# avgs['bin5-avg'][x,y].values,avgs['bin6-avg'][x,y].values]
			nums = [avgs['bin1-avg'][x,y].values,avgs['bin2-avg'][x,y].values,
				avgs['bin3-avg'][x,y].values,avgs['bin4-avg'][x,y].values]
			max = np.nanmax(nums)
			
			if np.isnan(max):
				avgs['avgmax'][x,y] = 0;
				maxskips+=1;
			else:
				binmax = int(nums.index(max));
				avgs['avgmax'][x,y] = binmax+1;
			
			min = np.nanmin(nums)
			
			if np.isnan(min):
				avgs['avgmin'][x,y] = 0;
				minskips+=1;
			else:
				binmin = int(nums.index(min));
				avgs['avgmin'][x,y] = binmin+1;
	

	## Diff of means test ##
	print("\n\nStatistics")
	pmap1 = np.zeros((l_lat,l_lon))
	pmap2 = np.zeros((l_lat,l_lon))
	n=1;
	
	for a in ['avgmax','avgmin']:
		print('   ',a)
		skips=0;
		for x in range(0,l_lat):
			for y in range(0,l_lon):
				print("Working ",x,y,'  Skip count: ',skips,'   ',end='\r')
				maxbin = avgs[a][x,y].values;
				if maxbin == 0:
					skips+=1;
					pval = np.nan;
					globals()['pmap'+str(n)][x,y]=pval;
					continue
					
				
				outs = [];
				for q in range(1,len(limits)):
					if q != maxbin:
						outs.append(q)

				if answer1 == 0:
					group1a = globals()['datacut'+str(maxbin)][datavar][:,x,y].values
				else:
					group1a = globals()['datacut'+str(maxbin)][datavar][x,y,:].values
				
				group1 = [];
				for ft in group1a:
					if np.isnan(ft) == False:
						group1.append(ft)
				
				group2 = [];
				for z in outs:
					if answer1 == 0:
						vals = globals()['datacut'+str(z)][datavar][:,x,y].values
					else:
						vals = globals()['datacut'+str(z)][datavar][x,y,:].values
					for v in vals:
						if np.isnan(v) == False:
							group2.append(v)
				
				wilcox = ranksums(group1,group2);
				pval = wilcox.pvalue
				globals()['pmap'+str(n)][x,y]=abs(pval);

				
		globals()['pmap'+str(n)] = xr.DataArray(globals()['pmap'+str(n)], dims=("lat", "lon"), 
			coords={'lon': (['lon'],lon), 'lat':(['lat'],lat)})
		avgs[a+'-p_val'] = globals()['pmap'+str(n)];
		n+=1;
		print()


print(avgs,'\n')
print('Min pvals: ',avgs['avgmax-p_val'].min(skipna=True).values,avgs['avgmin-p_val'].min(skipna=True).values)

outpath = '/scratch/zmanthos/thesis/outdata/'

fi1 = '';
for f in names:
	fi1 = fi1+f+'.';
if multi:
	fname = fn1+'.'+fi1+phasename+'.'+fs1+'.dmt.nc'
else:
	fname = fn1+'.'+fi1+fs1+'.dmt.rolling.5.nc'
print('file: ',fname)

avgs.to_netcdf(outpath+fname)