import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import pandas as pd
import os
import agents

from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize


# MARK: Setting up the workspace
current_working_dir = os.getcwd()
os.chdir(current_working_dir+"/Data")


def load_pd(file_name, verbosity=False):
    data_frame = pd.read_excel(file_name, sheet_name=0)
    if verbosity: print(data_frame)

    return data_frame


def build_agents() -> None:
    '''
    As of now, we only want to look at the relevant agents,
    :return: None
    '''

    wpp_f = load_pd('wind_power_producers.xls')
    ngpp_f = load_pd('natural_gas_power_producers.xls')

    return wpp_f, ngpp_f


def plot_agents(res='i', lllon=-74.13, lllat=39.74, urlon=-66.58, urlat=47.56) -> object:
    '''
    http://boundingbox.klokantech.com/ on the New England Area
    westlimit=-75.13; southlimit=39.74; eastlimit=-66.58; northlimit=47.56
    '''

    figure, axes = plt.subplots(figsize=(10, 20))
    MAP = Basemap(resolution=res, projection='merc', lat_0=(lllat - urlon)/2, lon_0=(lllat - urlat)/2,
                llcrnrlon=lllon, llcrnrlat=lllat, urcrnrlon=urlon, urcrnrlat=urlat)

    MAP.drawmapboundary(fill_color='#46bcec')
    MAP.fillcontinents(color='#f2f2f2', lake_color='#46bcec')
    MAP.drawcountries()
    MAP.drawcoastlines()
    MAP.drawstates()

    wpp_f, ngpp_f = build_agents()

    wpp_longitudes, wpp_latitudes = MAP(wpp_f['Longitude'].values, wpp_f['Latitude'].values)
    ngpp_longitudes, ngpp_latitudes = MAP(ngpp_f['Longitude'].values, ngpp_f['Latitude'].values)

    MAP.plot(wpp_longitudes, wpp_latitudes, 'bo', markersize=3, label='WPPs')
    MAP.plot(ngpp_longitudes, ngpp_latitudes, 'ro', markersize=3, label='NGPPs')

    plt.title('NGPPs and WPPs in the New England Region')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_agents()


