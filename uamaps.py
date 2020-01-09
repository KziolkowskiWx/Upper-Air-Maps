#!/home/ziolkowskik/anaconda3/envs/analysis/bin/python
from datetime import datetime, timedelta
from siphon.simplewebservice.wyoming import WyomingUpperAir
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as feat
#Uncomment the two lines below if running in cron 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable
from metpy.plots import simple_layout, StationPlot, StationPlotLayout
from metpy.calc import equivalent_potential_temperature as te
from metpy.calc import relative_humidity_from_specific_humidity as calc_rh
from metpy.calc import dewpoint_rh as calc_td
from metpy.units import units
import metpy.calc as mpcalc
import numpy as np
import xarray as xr
from optparse import OptionParser
import multiprocessing as mp
import time
from pathlib import Path
from os import path



def main():
    usage="usage: %prog [--am or --pm] \n example usage for 12z maps: python uamaps.py --am"
    parser = OptionParser(conflict_handler="resolve", usage=usage, version="%prog 1.0 By Kyle Ziolkowski")
    # parser = OptionParser(conflict_handler="resolve")
    parser.add_option("--h", "--help", dest="help", help="--am or --pm for 12z or 00z obs")
    parser.add_option("--am", "--am", dest="am", action="store_true", help="Get 12z obs")
    parser.add_option("--pm", "--pm", dest="pm",  action="store_true", help="Get 00z obs")
    parser.add_option("--td", dest='td', action="store_true", help="Plot dewpoint instead of dewpoint depression")
    (opt, arg) = parser.parse_args()

    #Some Variables to start
    start = time.time()
    home = Path.home()
    station_file = home / 'wxcentre/scripts/uaStations.xlsx' #Can the string of this location. Full path is not required.
    save_dir = home / 'wxcentre/maps/uamaps/today' #Change the string to choose where to save the file. 
    td_option = False #default to dewpoint depression

    if (opt.am == True and opt.pm == True) or (opt.am == None and opt.pm == None):
        parser.error('No option selected or too many arguments. Choose nbm12z or nbm00z. Example: python nbm_parse.py --nbm12z or type python nbm_parse.py -h for help.')
    if opt.am:
        hh = 12
    if opt.pm:
        hh = 0
    if opt.td:
        td_option = True #change default to dewpoint

    levels = [250, 300, 500, 700, 850, 925]
    uadata, stations = getData(station_file, hh)
    today = datetime.utcnow()
    date = datetime(today.year, today.month, today.day, hh) - timedelta(hours=6)
        #Get GFS data
    ds = xr.open_dataset('https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_{0:%Y%m%d}_{0:%H}00.grib2'.format(date)).metpy.parse_cf()
    print('Working on maps.....')
    with mp.Pool(processes=6) as pool:
        for level in levels:
            pool.apply_async(uaPlot, args=(generateData(uadata, stations, level), level, date, save_dir, ds, td_option,))
        pool.close()
        pool.join()
    end = time.time()
    total_time = round(end-start, 2)
    print('Process Complete..... Total time = {}s'.format(total_time))
    


def getData(station_file, hh):
    """
    This function will make use of Siphons Wyoming Upperair utility. Will pass
    the function a list of stations and data will be downloaded as pandas a 
    pandas dataframe and the corresponding lats and lons will be placed into
    a dictionary along with the data. 
    """
    
    print ('Getting station data...')
    station_data = pd.read_excel(station_file)
    stations, lats, lons = station_data['station_id'].values, station_data['lat'].values, station_data['lon'].values
    stations = list(stations)    
    date = datetime.utcnow()
    date = datetime(date.year, date.month, date.day, hh)
    data = {} #a dictionary for our data
    station_list = []
    
    for station, lat, lon in zip(stations, lats, lons):
        try:
            df = WyomingUpperAir.request_data(date, station)
            data[station] = [df, lat, lon]
            station_list += [station]
        except:
            pass
    print ('Data retrieved...')
    return data, station_list


def generateData(data, stations, level):
    """
    Test the data and put it into a an array to so it can be passed to dataDict
    """
    
    temp_c = []  
    dpt = []
    u = []
    v = []
    h= []
    p = []   
    lats = []
    lons = []
    
    for station in stations:
        t, td, u_wind, v_wind, height = getLevels(data[station][0], level)
        temp_c += [t]
        dpt += [td]
        u += [u_wind]
        v += [v_wind]
        h += [height]
        p += [level]
        lats += [data[station][1]]
        lons += [data[station][2]]
    temp = np.array(temp_c, dtype=float)    
    dewp = np.array(dpt, dtype=float)
    uwind = np.array(u, dtype=float)
    vwind = np.array(v, dtype=float)     
    data_array = np.array([temp, dewp, uwind, vwind, lats, lons, h, p])
        
    return data_array
    
def getLevels(df, level):
    """
    Get the 925, 850, 700, 500, 300, and 250 mb levels called by generateData
    """   

    level = df.loc[df['pressure'] == level]
        
    t, td, u, v, h, p = level['temperature'].values, level['dewpoint'].values, \
                        level['u_wind'].values, level['v_wind'].values, \
                        level['height'].values, level['pressure']
    #check to see if the data exits and create it. If not label as np.nan
    try:
        temp = t[0]
    except:
        temp = np.nan
    try:
        dwpt = td[0]
    except:
        dwpt = np.nan
    try:
        u_wind = u[0]
    except:
        u_wind = np.nan
    try:
        v_wind = v[0]
    except:
        v_wind = np.nan
    try:
        height = h[0]
    except:
        height = np.nan

    return temp, dwpt, u_wind, v_wind, height
    
def dataDict(data_arr):
    """In order to plot the data using MetPy StationPlot, we need to put
    the data into dictionaries, which will be done here. We will also assign
    the data its units here as well."""  
    
    
    #Container for the data
    data = dict()

    data['longitude'] = data_arr[5]
    data['latitude'] = data_arr[4]
    data['air_temperature'] = data_arr[0] * units.degC
    data['dew_point_temperature'] = data_arr[1] * units.degC
    data['eastward_wind'], data['northward_wind'] = data_arr[2] * units('knots') , data_arr[3] * units('knots')
    data['height'] = data_arr[6] * units('meters')
    data['pressure'] = data_arr[7] * units('hPa')
    data['thetae'] = te(data['pressure'], data['air_temperature'], data['dew_point_temperature']) 
    data['tdd'] = (data_arr[0] - data_arr[1]) * units.degC

    return data    

def uaPlot(data, level, date, save_dir, ds, td_option):

    custom_layout = StationPlotLayout()
    custom_layout.add_barb('eastward_wind', 'northward_wind', units='knots')
    custom_layout.add_value('NW', 'air_temperature', fmt='.0f', units='degC', color='darkred')

    # Geopotential height and smooth
    hght = ds.Geopotential_height_isobaric.metpy.sel(vertical=level*units.hPa, time=date, lat=slice(85, 15), lon=slice(360-200, 360-10))
    smooth_hght = mpcalc.smooth_n_point(hght, 9, 10)

    # Temperature, smooth, and convert to Celsius
    tmpk = ds.Temperature_isobaric.metpy.sel(vertical=level*units.hPa, time=date, lat=slice(85, 15), lon=slice(360-200, 360-10))
    smooth_tmpc = (mpcalc.smooth_n_point(tmpk, 9, 10)).to('degC')
    
    #Calculate Theta-e
    rh = ds.Relative_humidity_isobaric.metpy.sel(vertical=level*units.hPa, time=date, lat=slice(85, 15), lon=slice(360-200, 360-10))
    td = mpcalc.dewpoint_from_relative_humidity(tmpk, rh)
    te = mpcalc.equivalent_potential_temperature(level*units.hPa, tmpk, td)
    smooth_te = mpcalc.smooth_n_point(te, 9,10)

                            
    #decide on the height format based on the level
    if level == 250:
        custom_layout.add_value('NE', 'height', fmt=lambda v: format(v, '1')[1:4], units='m', color='black')
        cint = 120
        tint = 5
    if level == 300:
        custom_layout.add_value('NE', 'height', fmt=lambda v: format(v, '1')[1:4], units='m', color='black')
        cint = 120
        tint = 5
    if level == 500:
        custom_layout.add_value('NE', 'height', fmt=lambda v: format(v, '1')[0:3], units='m', color='black')
        cint = 60
        tint =5
    if level == 700:
        custom_layout.add_value('NE', 'height', fmt=lambda v: format(v, '1')[1:4], units='m', color='black')
        custom_layout.add_value('SW', 'tdd', units='degC', color='green')
        custom_layout.add_value('SE', 'thetae', units='degK', color='orange')
        temps = 'Tdd, and Theta-e'
        cint = 30
        tint=4
    if level == 850:
        custom_layout.add_value('NE', 'height', fmt=lambda v: format(v, '1')[1:4], units='m', color='black')
        if td_option == True:
            custom_layout.add_value('SW', 'dew_point_temperature', units='degC', color='green')
            temps = 'Td, and Theta-e'
        if td_option == False:
            custom_layout.add_value('SW', 'tdd', units='degC', color='green')
            temps = 'Tdd, and Theta-e'
        # custom_layout.add_value('SW', 'tdd', units='degC', color='green')
        # temps = 'Tdd, and Theta-e'
        custom_layout.add_value('SE', 'thetae', units='degK', color='orange')
        cint = 30
        tint = 4
    if level == 925:
        custom_layout.add_value('NE', 'height', fmt=lambda v: format(v, '1')[1:4], units='m', color='black') 
        if td_option == True:
            custom_layout.add_value('SW', 'dew_point_temperature', units='degC', color='green')
            temps = 'Td, and Theta-e'
        if td_option == False:
            custom_layout.add_value('SW', 'tdd', units='degC', color='green')
            temps = 'Tdd, and Theta-e'
        custom_layout.add_value('SE', 'thetae', units='degK', color='orange')
        cint = 30
        tint = 4
 
    globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371200.,
                       semiminor_axis=6371200.)
    proj = ccrs.Stereographic(central_longitude=-105., 
                               central_latitude=90., globe=globe,
                               true_scale_latitude=60)
    # Plot the image
    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    state_boundaries = feat.NaturalEarthFeature(category='cultural',
                                            name='admin_1_states_provinces_lines',
                                            scale='10m', facecolor='none')
    coastlines = feat.NaturalEarthFeature('physical', 'coastline', '50m', facecolor='none')
    lakes = feat.NaturalEarthFeature('physical', 'lakes', '50m', facecolor='none')
    countries = feat.NaturalEarthFeature('cultural', 'admin_0_countries', '50m', facecolor='none')
    ax.add_feature(state_boundaries, zorder=2, edgecolor='grey')
    ax.add_feature(lakes, zorder=2, edgecolor='grey')
    ax.add_feature(coastlines, zorder=2, edgecolor='grey')
    ax.add_feature(lakes, zorder=2, edgecolor='grey')
    ax.add_feature(countries, zorder=2, edgecolor='grey') 
    ax.coastlines(resolution='50m', zorder=2, color='grey')    
    ax.set_extent([-132., -70, 26., 80.], ccrs.PlateCarree())

    stationData = dataDict(data) 
    stationplot = StationPlot(ax, stationData['longitude'], stationData['latitude'],
                              transform=ccrs.PlateCarree(), fontsize=22)
    custom_layout.plot(stationplot, stationData)

    # Plot Solid Contours of Geopotential Height
    cs = ax.contour(hght.lon, hght.lat, smooth_hght.m, range(0, 20000, cint), colors='black', transform=ccrs.PlateCarree())
    clabels = plt.clabel(cs, fmt='%d', colors='white', inline_spacing=5, use_clabeltext=True, fontsize=22)

    # Contour labels with black boxes and white text
    for t in clabels:
        t.set_bbox({'facecolor': 'black', 'pad': 4})
        t.set_fontweight('heavy')

    #Check levels for different contours
    if level == 250 or level == 300 or level == 500:
        # Plot Dashed Contours of Temperature
        cs2 = ax.contour(hght.lon, hght.lat, smooth_tmpc.m, range(-60, 51, tint), colors='red', transform=ccrs.PlateCarree())
        clabels = plt.clabel(cs2, fmt='%d', colors='red', inline_spacing=5, use_clabeltext=True, fontsize=22)
        # Set longer dashes than default
        for c in cs2.collections:
            c.set_dashes([(0, (5.0, 3.0))])
        temps = 'T'
    if level == 700 or level == 850 or level == 925:
        # Plot Dashed Contours of Temperature
        cs2 = ax.contour(hght.lon, hght.lat, smooth_te.m, range(210, 360, tint), colors='orange', transform=ccrs.PlateCarree())
        clabels = plt.clabel(cs2, fmt='%d', colors='orange', inline_spacing=5, use_clabeltext=True, fontsize=22)
        # Set longer dashes than default
        for c in cs2.collections:
            c.set_dashes([(0, (5.0, 3.0))])
        
    
    dpi = plt.rcParams['savefig.dpi'] = 255    
    date = date + timedelta(hours = 6)
    text = AnchoredText(str(level) + 'mb Wind, Heights, and '+ temps +' Valid: {0:%Y-%m-%d} {0:%H}:00UTC'.format(date), loc=3, frameon=True, prop=dict(fontsize=22))
    ax.add_artist(text)
    plt.tight_layout()
    save_fname = '{0:%Y%m%d_%H}z_'.format(date) + str(level) +'mb.pdf'
    plt.savefig(save_dir / save_fname, dpi = dpi, bbox_inches='tight')
    #plt.show()

if __name__ == '__main__':
    main()
