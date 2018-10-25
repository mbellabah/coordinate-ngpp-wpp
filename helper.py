import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from mpl_toolkits.basemap import Basemap
###############################################################################


# MARK: Excel manipulation, and plotting
def load_pd(file_name, verbosity=False):
    data_frame = pd.read_excel(file_name, sheet_name=0)
    if verbosity:
        print(data_frame)

    return data_frame


def plot_agents(wpp_f, ngpp_f, p_nodes_f, des_ngpps=[], des_wpps=[], res='i', lllon=-74.13, lllat=39.74, urlon=-66.58, urlat=47.56) -> object:
    '''
    Plot the desired agents and annotate them . . .
    http://boundingbox.klokantech.com/ on the New England Area
    westlimit=-75.13; southlimit=39.74; eastlimit=-66.58; northlimit=47.56
    '''

    assert isinstance(wpp_f, pd.DataFrame), TypeError
    assert isinstance(ngpp_f, pd.DataFrame), TypeError
    assert isinstance(p_nodes_f, pd.DataFrame), TypeError


    MAP = Basemap(resolution=res, projection='merc', lat_0=(lllat - urlon)/2, lon_0=(lllat - urlat)/2, llcrnrlon=lllon, llcrnrlat=lllat, urcrnrlon=urlon, urcrnrlat=urlat)

    MAP.drawmapboundary(fill_color='#46bcec')
    MAP.fillcontinents(color='#f2f2f2', lake_color='#46bcec')
    MAP.drawcountries(); MAP.drawcoastlines(); MAP.drawstates()

    wpp_longitudes, wpp_latitudes = MAP(wpp_f['Longitude'].values[des_wpps], wpp_f['Latitude'].values[des_wpps])
    ngpp_longitudes, ngpp_latitudes = MAP(ngpp_f['Longitude'].values[des_ngpps], ngpp_f['Latitude'].values[des_ngpps])

    MAP.plot(wpp_longitudes, wpp_latitudes, 'bo', markersize=3, label='WPPs')
    MAP.plot(ngpp_longitudes, ngpp_latitudes, 'ro', markersize=3, label='NGPPs')

    p_nodes_longitudes, p_nodes_latitudes = MAP(p_nodes_f['Longitude'].values, p_nodes_f['Latitude'].values)
    MAP.plot(p_nodes_longitudes, p_nodes_latitudes, 'cs', markersize=2, label='Pricing Nodes')


    plt.title('NGPPs, WPPs and Pricing Nodes in the New England Region')
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


# MARK: Functions to do with agents
def build_agents() -> None:
    wpp_f = load_pd('wind_power_producers.xls')
    ngpp_f = load_pd('natural_gas_power_producers.xls')
    p_nodes_f = load_pd('ISO_NE_pnode_coords.xlsx', verbosity=False)

    return wpp_f, ngpp_f, p_nodes_f


def helper_main(des_wpps: np.array =[], des_ngpps: np.array =[], verbosity=False):

    assert isinstance(des_wpps, np.array) and des_wpps.size, 'Provide indices of desired WPPs'
    assert isinstance(des_ngpps, np.array) and des_ngpps.size, 'Provide indices of desired NGPPs'

    wpp_f, ngpp_f, p_nodes_f = build_agents()

    if verbosity:
        print(f'WPP dataframe: {wpp_f}')
        print(f'NGPP dataframe: {ngpp_f}')
        print(f'P_Nodes dataframe: {p_nodes_f}')

    # Currently doesn't work, so do not use
    # if not (des_ngpps or des_ngpps):
    #     des_ngpps = np.arange(len(ngpp_f) - 1)
    #     des_wpps = np.arange(len(wpp_f) - 1)


    # MARK: Perform the filtering of the NGPPs and WPPs
    # filter the NGPPs on the basis of technology
    def filter_tech_ngpp(ngpp, tech_type='Technology Type', des_tech='Combined Cycle') -> np.array:
        criteria = ngpp[tech_type] == des_tech
        return ngpp[criteria]

    # filter the WPPs on the basis of distance to each other (bundle)

    ngpp_f = filter_tech_ngpp(ngpp_f)

    # compute the distances between the ngpps and the wpps, find the k closest pairings

    # MARK: Perform the final plotting
    plot_agents(wpp_f=wpp_f, ngpp_f=ngpp_f, p_nodes_f=p_nodes_f, des_ngpps=des_ngpps, des_wpps=des_wpps)


# -> MARK: Interpretation of the agents
def fractional_capacity():
    # compute the fractional capacity that could engaged
    # in a reliability contract
    return None


if __name__ == '__main__':
    # MARK: Setting up the workspace
    current_working_dir = os.getcwd()
    os.chdir(current_working_dir + "/Data")

    helper_main(des_wpps=np.arange(10), des_ngpps=np.arange(10), verbosity=False)
