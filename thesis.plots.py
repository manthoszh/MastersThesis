from __future__ import print_function
print('--------------------------------------------------------------------------------------------------------')
print('Importing')
import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from scipy.stats import rankdata
from scipy.stats.stats import spearmanr
print('    done\n')

def conplot(data):
	fig = plt.figure(figsize=(10,6))
	ax = plt.axes(projection=ccrs.PlateCarree())
	levels = np.arange(0,0.1,0.01)
	plt.contourf( lon, lat, data, extend='both', cmap='Reds', levels=levels, transform=ccrs.PlateCarree())
	ax.coastlines(resolution='110m')
	plt.colorbar(ax=ax, shrink=0.6, orientation='horizontal')
	ax.gridlines(color="black", linestyle="dotted")
	plt.show()

def plotstyle(x):
	ax[x].set_xticks(np.arange(-130,-49,10), crs=ccrs.PlateCarree())
	lon_formatter = cticker.LongitudeFormatter()
	ax[x].xaxis.set_major_formatter(lon_formatter)
        
	ax[x].set_yticks(np.arange(25,56,10), crs=ccrs.PlateCarree())
	lat_formatter = cticker.LatitudeFormatter()
	ax[x].yaxis.set_major_formatter(lat_formatter)
	
	ax[x].coastlines()
	ax[x].gridlines(linestyle='--')

def fdrtest(pvalues,alpha=0.05):
	## ----- FDR Test ----- ##
	pvals = pvalues.values
	#print(pvals)
	shape = pvals.shape
	#print(pvals.shape,'\n')
	length = shape[0]*shape[1];
	listpval = np.reshape(pvals,length,order='C')

	ranks = rankdata(listpval)
	unique = np.unique(ranks)

	passed = np.zeros(length)
	n=1;
	testing=0;
	test2 = 0;
	cont=True;
	pfdr = 0;
	print('First Pval Thresh: ',(alpha * n/length))
	for x in unique:
		locations = np.where(ranks == x);
		location = locations[0].tolist()
		p = listpval[location[0]];
		
		test = alpha * n/length;
		n+= len(location);
		#print(location,p,test,n)
		if p < test and cont==True:
			pfdr=p;
			for loc in location:
				passed[loc] = 1;
				testing+=1;
		else: 
			cont = False;
		if p<alpha:
			test2+= len(location);
		#print('Rank: ',x, end='\r'),
	print()
	if fndata =='era5':
		length=length-12817;
	fdrpass = round((testing/length),4)
	sigpass = round((test2/length),4)
	print('FDR Passed: ',testing,'/',length,' = ',fdrpass)
	print(alpha,' Passed: ',test2,'/',length,' = ',sigpass,'\n')

	mask = np.reshape(passed,shape, order='C')
	return mask,fdrpass

##### QUESTIONS #####

cont = True;
while cont == True:
	print('Data set? 1:GPCP  2:ERA5');
	fndata = str(int(input())-1);
	if fndata == '0' or fndata == '1':
		fndata = int(fndata)
		print()
		cont = False;
	else:
		print('invalid')

if fndata == 0:
	fndata = 'gpcp';
else:
	fndata = 'era5';


cont = True;
while cont == True:
	print('Season? 1: Summer May-Aug  2: Winter Oct-Apr');
	fnseas = str(int(input())-1);
	if fnseas == '0' or fnseas == '1':
		sea = int(fnseas)
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
fnindexes = [0,0];
indexes = ['mlso','enso','nao','pna']
while cont == True:
	print('Multi Index: 1 = Yes  2 = No');
	fnindex = str(input());
	print()
	if fnindex == '1':
		print('Which 2? 1:MLSO 2:ENSO 3:NAO 4:PNA - spearate by comma & no spaces')
		inputed = str(input())
		multi = True;
		fnindexes[0] = indexes[int(inputed[0])-1]
		fnindexes[1] = indexes[int(inputed[2])-1]
		print()
		cont = False;
	elif fnindex == '2':
		print('Which? 1:MLSO 2:ENSO 3:NAO 4:PNA')
		fnindexes = indexes[int(input())-1]
		multi = False;
		print()
		cont = False;
	else:
		print('invalid')

if multi:
	cont = True;
	while cont == True:
		print('In or Out of Phase? 1:IN 2:OUT');
		fnphase = int(input())-1;
		if fnphase == 0:
			phasename = 'inphase';
			print()
			cont=False;
		elif fnphase == 1:
			phasename = 'outphase';
			print()
			cont=False;
		else:
			print('invalid entry')
			

cont = True;
while cont == True:
	print('Extremes? yes(1) or no(2)')
	ext = int(input())
	if ext == 1:
		print('Upper(1) or Lower(2)')
		tail = int(input())
		if tail == 1:
			qt = 'upper'

		elif tail == 2:
			qt = 'lower'

		fs1 = fs1+'.'+qt+'-extreme';
		cont = False;
	elif ext == 2:
		cont = False;
	else:
		print('invaild answer')
print()

cont = True;
while cont == True:
	print('BPT or DMT? ratio(1) or average(2)')
	testtype = int(input())
	if testtype == 1:
		ttype = 'bpt'
		cont = False;
	elif testtype == 2:
		ttype = 'dmt'
		cont = False;
	else:
		print('invaild answer')
print()

if multi:
	fin1 = '';
	for f in fnindexes:
		fin1 = fin1+f+'.';
	fname = fndata+'.'+fin1+phasename+'.'+fs1;
	stitle = ttype.upper()+' '+fnindexes[0].upper()+' & '+fnindexes[1].upper()+' '+phasename+' '+seaname
	binnum = 4;
	levels = [0.5,1.5,2.5,3.5,4.5]
	llevels = [1,2,3,4]

else:
	fin1 = fnindexes
	fname = fndata+'.'+fin1+'.'+fs1;
	stitle = ttype.upper()+' '+fin1.upper()+' '+seaname
	binnum = 6;
	# levels = [0.5,1.5,2.5,3.5,4.5,5.5,6.5]
	# llevels = [1,2,3,4,5,6]
	# llabels = ['E-Pos','Pos','N-Pos','N-Neg','Neg','E-Neg']
	levels = [0.5,1.5,2.5,3.5,4.5]
	llevels = [1,2,3,4]
	
llabels = ['Strong Positive','Weak Positive','Weak Negative','Strong Negative']

l2labels = ['Strong\nPositive','Weak\nPositive','Weak\nNegative','Strong\nNegative']

if testtype == 1:
	file = fname+'.bpt.nc';
	stype = '.bpt.'
else:
	file = fname+'.dmt.nc';
	stype = '.dmt.'

print('File name: ',file,'\n')
path = '/scratch/zmanthos/thesis/outdata/';

totdata = xr.open_dataset(path+file,decode_times=True,decode_cf=True)

#print(totdata)

if fndata == 'gpcp':
	## gpcp
	lon = totdata['longitude']
	l_lon = len(lon)
	lat = totdata['latitude']
	l_lat = len(lat)
	varnames = ['-wet','-dry']
	titles1 = 'Wet/Dry';
	titles2 = 'mm/Day';
	alpha = 0.1;
else:
	## era5
	lon = totdata['lon']
	l_lon = len(lon)
	lat = totdata['lat']
	l_lat = len(lat)
	varnames = ['-warm','-cold']
	titles1 = 'Warm/Cold';
	titles2 = 'Temp Anom';
	alpha = 0.05;


print('Plotting')
if testtype == 1:
	dnames = ['rmax','rmin']
	titles = ['Bins w/ Highest Ratio '+titles1,'Bins w/ Lowest Ratio '+titles1]
else:
	dnames = ['avgmax','avgmin']
	titles = ['Bins w/ Highest Avg '+titles2,'Bins w/ Lowest Avg '+titles2]

fig, ax = plt.subplots(nrows=len(dnames),ncols=1,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(8,6))

cmap_name='zak'
colors=['coral','green','grey','cyan']
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=4)
n=0;
for name in dnames:
	data = totdata[name]
	
	cs = ax[n].contourf(lon, lat, data, levels=levels, transform = ccrs.PlateCarree(),
		cmap=cm)
	
	plotstyle(n)
	
	pdata = totdata[name+'-p_val']

	fdr,rate = fdrtest(pdata,alpha=alpha)
	ax[n].contourf(lon,lat,fdr,[0,1],transform = ccrs.PlateCarree(),colors='None',
			hatches=['','....'],extend='both',alpha=0)
	ax[n].set_title(titles[n]+' Passed FDR: '+str(round(rate*100,2))+'% a='+str(alpha))
	n+=1;

fig.subplots_adjust(bottom=0.05, top=0.90, left=0.05, right=0.85, wspace=0.1, hspace=0.25)
cax = fig.add_axes([0.87,0.2, 0.03, 0.6]) #cax = fig.add_axes([left, bottom, width, height])
cbar = fig.colorbar(cs, cax=cax, orientation='vertical',ticks=llevels) #
cbar.ax.invert_yaxis()
cbar.ax.set_yticklabels(l2labels)
plt.suptitle(stitle,fontsize='x-large',weight='bold')
#plt.show()

outpath = '/homes/zmanthos/thesis/plots/analysis/'
if multi:
	outpath = outpath +fnindexes[0]+'-'+fnindexes[1]+'/'
else:
	outpath = outpath + fnindexes +'/';
	
print('Saving')
fig.savefig(outpath+fname+stype+'test.png')
print((outpath+fname+stype+'test.png'))

print('Plotting Bins')

cmap='coolwarm'
if testtype == 1:
	dname = '-ratio'
	if fndata == 'era5':
		levels = np.arange(0.4,1.6,0.05)
	else:
		levels = np.arange(0,2,0.05)
		cmap = 'terrain_r'
	tit = 'Ratio'
	tit1 = titles1
else:
	dname = '-avg'
	if fndata == 'era5':
		levels = np.arange(-0.7,0.7,0.05)
	else:
		levels = np.arange(-1.5,1.5,0.05)
		cmap = 'terrain_r'
		
	tit = 'Avg'
	tit1 = titles2
	
fig, ax = plt.subplots(nrows=4,ncols=1,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(6,11))

n=0;
for x in range(1,5):
	data = totdata['bin'+str(x)+dname]
	if fndata == 'era5' and ttype == 'bpt':
		data = data.where(data >= 0.1)
		
	cs = ax[n].contourf(lon, lat, data, levels=levels, transform = ccrs.PlateCarree(),
		cmap=cmap,extend='both')
	
	plotstyle(n)
	
	titles = llabels[n]
	ax[n].set_title(titles)
	n+=1;


fig.subplots_adjust(bottom=0.10, top=0.93, left=0.05, right=0.95, wspace=0.1, hspace=0.28)
cax = fig.add_axes([0.2,0.04, 0.6, 0.02]) #cax = fig.add_axes([left, bottom, width, height])
cbar = fig.colorbar(cs, cax=cax, orientation='horizontal', label = tit+' '+tit1)
cbar.ax.locator_params(nbins=9)
plt.suptitle(stitle,fontsize='x-large',weight='bold')
plt.show()

fig.savefig(outpath+fname+stype+'bins.test.png')
print((outpath+fname+stype+'bins.test.png'))