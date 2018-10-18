import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from mpl_toolkits.basemap import Basemap


# MARK: Setting up the workspace
current_working_dir = os.getcwd()
os.chdir(current_working_dir+"/Data")


def load_pd(file_name, verbosity=False):
    data_frame = pd.read_excel(file_name, sheet_name=0)
    if verbosity:
        print(data_frame)

    return data_frame


def build_agents() -> None:
    wpp_f = load_pd('wind_power_producers.xls')
    ngpp_f = load_pd('natural_gas_power_producers.xls')

    return wpp_f, ngpp_f


def plot_agents(wpp_f, ngpp_f, des_ngpps=[], des_wpps=[], res='i', lllon=-74.13, lllat=39.74,
                urlon=-66.58, urlat=47.56) -> object:
    '''
    Plot the desired agents and annotate them . . .
    http://boundingbox.klokantech.com/ on the New England Area
    westlimit=-75.13; southlimit=39.74; eastlimit=-66.58; northlimit=47.56
    '''

    # figure, axes = plt.subplots(figsize=(10, 20))
    MAP = Basemap(resolution=res, projection='merc', lat_0=(lllat - urlon)/2, lon_0=(lllat - urlat)/2,
                llcrnrlon=lllon, llcrnrlat=lllat, urcrnrlon=urlon, urcrnrlat=urlat)

    MAP.drawmapboundary(fill_color='#46bcec')
    MAP.fillcontinents(color='#f2f2f2', lake_color='#46bcec')
    MAP.drawcountries(); MAP.drawcoastlines(); MAP.drawstates()

    wpp_longitudes, wpp_latitudes = MAP(wpp_f['Longitude'].values[des_wpps], wpp_f['Latitude'].values[des_wpps])
    ngpp_longitudes, ngpp_latitudes = MAP(ngpp_f['Longitude'].values[des_ngpps], ngpp_f['Latitude'].values[des_ngpps])

    MAP.plot(wpp_longitudes, wpp_latitudes, 'bo', markersize=3, label='WPPs')
    MAP.plot(ngpp_longitudes, ngpp_latitudes, 'ro', markersize=3, label='NGPPs')

    plt.title('NGPPs and WPPs in the New England Region')
    plt.legend()

    def annotate_agents(desired_agents_indices, agent_type) -> None:
        if agent_type == 'wind':
            agent_names = wpp_f['Power Plant'][desired_agents_indices]
            agent_coords = wpp_f['Longitude'][desired_agents_indices], wpp_f['Latitude'][desired_agents_indices]

        elif agent_type == 'gas':
            agent_names = ngpp_f['Power Plant'][desired_agents_indices]
            agent_coords = ngpp_f['Longitude'][desired_agents_indices], ngpp_f['Latitude'][desired_agents_indices]
        else:
            raise NameError(f'{agent_type} does not exist!')

        for name, longitude, latitude in zip(agent_names, *agent_coords):
            longitude, latitude = MAP(longitude, latitude)
            plt.annotate(name, (longitude, latitude))

    # annotate desired wind and gas power plants
    annotate_agents(des_wpps, agent_type='wind')
    annotate_agents(des_ngpps, agent_type='gas')
    plt.show()


def main(des_wpps=[], des_ngpps=[], verbosity=False):
    wpp_f, ngpp_f = build_agents()

    if verbosity:
        print(f'WPP dataframe: {wpp_f}')
        print(f'NGPP dataframe: {ngpp_f}')

    if not (des_ngpps or des_ngpps):
        des_ngpps = np.arange(len(ngpp_f))
        des_wpps = np.arange(len(wpp_f))


    # MARK: Perform the filtering of the NGPPs and WPPs
    # filter the NGPPs on the basis of technology
    def filter_tech_ngpp(ngpp, tech_type='Technology Type', des_tech='Combined Cycle') -> np.array:
        criteria = ngpp[tech_type] == des_tech
        return ngpp[criteria]

    ngpp_f = filter_tech_ngpp(ngpp_f)

    # compute the distances between the ngpps and the wpps, find the k closest pairings


    # MARK: Perform the final plotting
    plot_agents(wpp_f=wpp_f, ngpp_f=ngpp_f, des_ngpps=des_ngpps, des_wpps=des_wpps)



if __name__ == '__main__':
    main(verbosity=True)
