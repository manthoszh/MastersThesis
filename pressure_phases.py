from __future__ import print_function
print('--------------------------------------------------------------------------------------------------------')
print('Importing')
import os
import datetime
import subprocess
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
print('    done\n')

def linplotter(x,y,show=False):
	plt.plot(x,y)
	if show == True:
		plt.show()

def conplotter(data,x,y,levels,cmap,title,suptitle,save):
	fig = plt.figure(figsize=(10,6))
	ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=230))
	plt.contourf( lon, lat, data,levels=levels, extend='both', cmap=cmap, transform=ccrs.PlateCarree())
	ax.coastlines(resolution='110m')
	plt.colorbar(ax=ax, shrink=0.6, orientation='horizontal',label='meters')
	ax.gridlines(color="black", linestyle="dotted")
	plt.title(title)
	fig.suptitle(suptitle)
	plt.show()
	fig.savefig(save)

def plotstyle(x):
	ax[x].set_xticks(np.arange(-130,-49,10), crs=ccrs.PlateCarree())
	lon_formatter = cticker.LongitudeFormatter()
	ax[x].xaxis.set_major_formatter(lon_formatter)
        
	ax[x].set_yticks(np.arange(25,56,10), crs=ccrs.PlateCarree())
	lat_formatter = cticker.LatitudeFormatter()
	ax[x].yaxis.set_major_formatter(lat_formatter)
	
	ax[x].coastlines()
	ax[x].gridlines(linestyle='--')


cont = True;
while cont == True:
	print('Season? 1: Summer May-Aug  2: Winter Oct-Apr');
	answer2 = str(int(input())-1);
	if answer2 == '0' or answer2 == '1':
		sea = int(answer2)
		print('\n')
		cont = False;
	else:
		print('invalid')
if sea == 0:
	seaname = 'Summer Apr-Sept'
	fseason = 'summer'
else:
	seaname = 'Winter Oct-Mar'
	fseason = 'winter'
	
cont = True;
ans = [0,0]
while cont == True:
	print('Multi Index: 1 = Yes  2 = No');
	answer3 = str(input());
	print('\n')
	if answer3 == '1':
		print('Which 2? 1:MLSO 2:ENSO 3:NAO 4:PNA - spearate by comma & no spaces')
		inputed = str(input())
		
		multi = True;
		ans[0] = int(inputed[0])-1;
		
		if ans[0] == 0:
			modpath = 'mlso'
		elif ans[0] == 1:
			modpath = 'enso'
		elif ans[0] == 2:
			modpath = 'nao'
		elif ans[0] == 3:
			modpath = 'pna'
	
		ans[1] = int(inputed[2])-1;

		if ans[1] == 0:
			modpath = modpath+'-mlso/'
		elif ans[1] == 1:
			modpath = modpath+'-enso/'
		elif ans[1] == 2:
			modpath = modpath+'-nao/'
		elif ans[1] == 3:
			modpath = modpath+'-pna/'
		print('\n')
		outfile=fseason+'.'+modpath[0:-1]
		
		cont = False;
	elif answer3 == '2':
		print('Which? 1:MLSO 2:ENSO 3:NAO 4:PNA')
		ans = int(input())-1;
		if ans == 0:
			modpath = 'mlso/';
		elif ans == 1:
			modpath = 'enso/';
		elif ans == 2:
			modpath = 'nao/';
		elif ans == 3:
			modpath = 'pna/';
		
		multi = False;
		outfile=fseason+'.'+modpath[0:-1]
		modpath = 'single/'
		print()
		cont = False;
	else:
		print('invalid')

if multi:
	cont = True;
	while cont == True:
		print('In or Out of Phase? 1:IN 2:OUT');
		phase = int(input())-1;
		if phase == 0:
			phasename = 'inphase';
			print('\n')
			cont=False;
		elif phase == 1:
			phasename = 'outphase';
			print('\n')
			cont=False;
		else:
			print('invalid entry')
	outfile = outfile+'.'+phasename
plotpath = '/homes/zmanthos/thesis/plots/pressure/'+modpath


dfiles = '/scratch/zmanthos/thesis/era-interim.geoh.u.v.500.97-18.nc'
data = xr.open_mfdataset(dfiles, decode_times=True, decode_cf=True)
data = data.drop_vars(['clima'])
data = data.drop_dims('dayofyear')

#print(data)

lat = data['latitude']
lon = data['longitude']

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
print('\n')


if multi == True:
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
		print('\n ---- Days after filter:  ',len(timecut_season),' ----')
		
		datacut = data.where(data['time'].isin(timecut_season)).dropna(dim='time',how='all')
		totdays =len(datacut['time'])
		print(' ---- Days after slicing: ',totdays,' ----\n')
		
		globals()['pdata'+str(x)] = datacut['anoms'].mean(dim='time',skipna=True)
		tnames=['Strong Positive','Weak Positive','Weak Negative','Strong Negative']
		globals()['title'+str(x)] = tnames[x]+'  Days Included: '+str(totdays)
		
		
else:
	limits=[10000,0.75,0,-0.75,-10000]
	index1 = globals()[names[0]][names[0]]
	## -- Data Slicing -- ##

	for x in range(0,(len(limits)-1)):
		lim1 = limits[x+1];
		lim2 = limits[x];
	
		sigs = index1.where(index1>lim1).dropna(dim='time')
		sigs = sigs.where(sigs<lim2).dropna(dim='time')
		print(lim1,lim2,'\n',sigs['time'])
		
		timecut = sigs['time'].values
		indexcut = index1.where(index1['time'].isin(timecut)).dropna(dim='time',how='all')
		
		dates = pd.to_datetime(indexcut['time'].values)
		#print(dates)

		
		mon = dates.month
		season = [[1,2,3,10,11,12],[4,5,6,7,8,9]]
		test = np.isin(mon,season[sea])
		#print(indexcut['time'],test)
		timecut_seasons = ma.masked_array(indexcut['time'],test)
		timecut_season = timecut_seasons.compressed()
		#print(timecut_season)
		print('\n ---- Days after filter:  ',len(timecut_season),' ----')

		
		datacut = data.where(data['time'].isin(timecut_season)).dropna(dim='time',how='all')
		totdays =len(datacut['time'])

		print(' ---- Days after slicing: ',totdays,' ----\n')

		globals()['pdata'+str(x)] = datacut['anoms'].mean(dim='time',skipna=True)
		

		tnames=['Strong Positive','Weak Positive','Weak Negative','Strong Negative']
		globals()['title'+str(x)] = tnames[x]+'  Days Included: '+str(totdays)
		
		
fig, ax = plt.subplots(nrows=(len(limits)-1),ncols=1,
      subplot_kw={'projection': ccrs.PlateCarree()},
      figsize=(8,11.5))
	
plon = lon;
plat = lat;

if sea == 0:
	levels = np.arange(-20,20.1,1)
	if multi:
		levels = np.arange(-30,30.1,1)
else:
	levels = np.arange(-35,35.1,3)
	if multi:
		levels = np.arange(-30,30.1,1)

cmap = 'RdBu_r'
for x in range(0,(len(limits)-1)):
	cs = ax[x].contourf(plon, plat, globals()['pdata'+str(x)],
		levels=levels, transform = ccrs.PlateCarree(),
		cmap=cmap,extend='both')
				
	plotstyle(x)
	ax[x].set_title(globals()['title'+str(x)])
	
cax = fig.add_axes([0.855,0.33, 0.03, 0.33]) #cax = fig.add_axes([left, bottom, width, height])
fig.colorbar(cs,cax=cax, orientation='vertical', label = 'Meters')
if multi:
	indices = 'Indices: '+names[0]+'-'+names[1]
else:
	indices = 'Index: '+names[0]
plt.suptitle('Geopotential Height Anomalies - '+indices+'  '+seaname,fontsize='large',weight='bold')
fig.subplots_adjust(bottom=0.02, top=0.93, left=0.05, right=0.95, wspace=0.05, hspace=0.28)

plt.show()
	
outname = plotpath+'z500.'+outfile+'.new.png'
print(outname)
fig.savefig(outname)
	
# ## LINE PLOT ##
	# print('plotting')
	# colors=['coral','green','grey','cyan']
	
	# index = ma.asarray(index1.values)
	# xs = index1['time']
	# ys0 = ma.masked_outside(index,10,0.75)
	# ys1 = ma.masked_outside(index,0.75,0)
	# ys2 = ma.masked_outside(index,0,-0.75)
	# ys3 = ma.masked_outside(index,-0.75,-10)

	# title = names[0].upper()+' index bins colored'
	# fig = plt.figure(figsize=(12,6))
	# plt.plot(xs,ys0,label='Strong Positive',color=colors[0])
	# plt.plot(xs,ys1,label='Weak Positive',color=colors[1])
	# plt.plot(xs,ys2,label='Weak Negative',color=colors[2])
	# plt.plot(xs,ys3,label='Strong Negative',color=colors[3])
	# years = ['1997','1998','1999','2000','2001','2002','2003','2004',
		# '2005','2006','2007','2008','2009','2010','2011','2012',
		# '2013','2014','2015','2016','2017','2018','2019'];
	# plt.xticks(years,years,fontsize='small')
	# plt.xlabel('Year')
	# plt.ylabel('Index Value')
	# plt.grid(which='major',axis='x',linestyle='--')
	
	# plt.title(title)
	# plt.legend(bbox_to_anchor=(.95, 1), loc='upper left')
	# plt.show()
	# save = '/homes/zmanthos/thesis/plots/'+names[0]+'.index.bin-color.1.png'
	# fig.savefig(save)
	# print(save)







