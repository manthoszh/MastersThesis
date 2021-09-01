# MastersThesis
 Programs used to complete Masters Thesis, All of which were written and debugged by Zachary Manthos
 
 Short Descriptions:
 - autocore-dof.py: Finds the 1 day autocorrelation of a dataset and calculates a DOF, degrees of freedom, factor to be use in statistical tests to compensate for autocorrelation in the data
 - binom_pro_test.py: Runs a Binomial Proportion Test
 - data.thesis.py: Initial slicing of the datasets for easier use
 - datetime.convert.py: Convert datetime structure of Era5 dataset to fit the other datasets used
 - diff_means_test.py: Runs a difference of means test, Wilcoxon Ranksum test
 - enso.index.py: Creates a simple ENSO index
 - era5_conversion.py: Converts a single dimension, X only, dataset to a 2 dimension, X & Y, dataset
 - index.nao-pna.py: Converts NAO & PNA indexes, simple text file, to Xarray, netcdf, datasets
 - pressure_phases.py: Cuts and organizes Era-Interim dataset for plotting pressure maps
 - pressure_to_netcdf.py: Converts Era-Interim dataset from grib to netcdf
 - thesis.plots.py: Main plotting program for datasets calculated in other programs

Thesis PREZ.pdf: This is the presentation used to defend my masters thesis. It did not convert to pdf well, presentation included animations, so some of the oddities are due to this fact.
