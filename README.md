# Upper-Air-Maps
Script for download upper air data and plotting upper air maps.

Requirements:

Python 3.6+
Metpy 0.11+
Siphon 0.8+
Cartopy 0.17.0
Matplotlib 3.1.2
XArray 0.14.1

Note: If you install Metpy using the conda-forge channel, it will install all the necessary dependencies in order to run this script accept for Siphon, which will be required to be installed after the Metpy installation. 

This script will plot upperair maps for 250,300,500,700,850 and 925mb. The script will automatically download all the data from the University of Wyoming using Siphon. The 18z and 06z GFS T+06 model data is used as the objective analysis for the contours of Geopotential Height, temperature and theta-e. 

The script is designed to run via the command line. Example:

python uamaps.py --am 

This will plot the 12z maps. To plot the 00z maps use the option --pm. 

Note: the default moisture parameter for 850/925mb is dewpoint depression. In order to plot dewpoint for both of these levels use the following:

python uamaps.py --am --td


