import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import logging
import functools
from scipy.spatial.distance import cdist
from geopy.distance import vincenty
from mpl_toolkits.basemap import Basemap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.propagate = True


def trackcalls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.has_been_called = True
        return func(*args, **kwargs)

    wrapper.has_been_called = False

    return wrapper


###############################################################################

# TODO: [x] Bundle the WPPs
# TODO: [] -> check above via coords
# TODO: [] Calculate the relevant data regarding the pricing nodes

################################################################################


# MARK: Excel manipulation, and plotting
def load_pd(file_name, verbosity=False):
    data_frame = pd.read_excel(file_name, sheet_name=0)
    if verbosity:
        print(data_frame)

    return data_frame


def build_agents() -> None:
    wpp_f = load_pd('wind_power_producers.xls')
    ngpp_f = load_pd('natural_gas_power_producers.xls')
    p_nodes_f = load_pd('ISO_NE_pnode_coords.xlsx')

    # bundle the WPPs and NGPPs
    def bundle(dataframe, criteria=['Power Plant']) -> pd.DataFrame:
        dataframe['Operating Capacity'] = dataframe.groupby(criteria)['Operating Capacity'].transform('sum')
        filtered_dataframe = dataframe.drop_duplicates(subset=[*criteria])
        return filtered_dataframe

    # filter the tech type of the NGPPs
    def filter_tech_ngpp(ngpp, tech_type='Technology Type', des_tech='Combined Cycle') -> np.array:
        criteria = ngpp[tech_type] == des_tech
        return ngpp[criteria]

    # Clean the data, i.e. remove na values, and drop duplicates
    p_nodes_f = p_nodes_f.dropna()
    ngpp_f = filter_tech_ngpp(ngpp_f)
    ngpp_f = bundle(ngpp_f)
    wpp_f = bundle(wpp_f)

    return wpp_f, ngpp_f, p_nodes_f


@trackcalls
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


# MARK: Functions to do with agents

def find_closest_pricing_node(pnodesf=None, wppf=None, ngppf=None, verbosity=False):
    assert pnodesf is not None, ValueError('Provide the pricing node file')

    # First find the closest pricing node for each ngpp -- # <Longitude>, <Latitude>, <Plant Key>
    np_wpp = np.array([wppf['Longitude'], wppf['Latitude'], wppf['Power Plant']]).T
    np_ngpp = np.array([ngppf['Longitude'], ngppf['Latitude'], ngppf['Power Plant']]).T
    np_pnodes = np.array([pnodesf['Longitude'], pnodesf['Latitude'], pnodesf['LMP Name']]).T

    # Grab only the coordinates
    wpp_coords = np_wpp[:, 0:2]
    ngpp_coords = np_ngpp[:, 0:2]
    pnodes_coords = np_pnodes[:, 0:2]

    # Grab only the names
    wpp_names = np_wpp[:, 2]
    wpp_names = np.reshape(wpp_names, (wpp_names.size, 1))
    ngpp_names = np_ngpp[:, 2]
    ngpp_names = np.reshape(ngpp_names, (ngpp_names.size, 1))
    pnode_names = np_pnodes[:, 2]

    def closest_point(point, others):
        # return the arguments of the closest coord in others relative to point
        try:
            return cdist([point], others, lambda u, v: vincenty(u, v).kilometers).argmin()
        except KeyError:
            return None

    if verbosity: print('\nComputing the closest pricing nodes . . .')
    wpp_closest_pairs_args = np.apply_along_axis(
        closest_point, 1, wpp_coords, pnodes_coords
    )
    ngpp_closest_pairs_args = np.apply_along_axis(
        closest_point, 1, ngpp_coords, pnodes_coords
    )

    # Stack the pairs, confirm using snl
    def stack_pairs(pair_args, pair_type: str):
        closest_pnodes = np_pnodes[pair_args][:, 2]
        closest_pnodes = np.reshape(closest_pnodes, (closest_pnodes.size, 1))

        if pair_type == 'wind':
            return np.concatenate((wpp_names, closest_pnodes), axis=1)
        elif pair_type == 'gas':
            return np.concatenate((ngpp_names, closest_pnodes), axis=1)
        else:
            raise ValueError(f'{pair_type} does not exist!')

    # Got the pricing node and agent pairs
    wpp_closest_pairs = stack_pairs(wpp_closest_pairs_args, 'wind')
    ngpp_closest_pairs = stack_pairs(ngpp_closest_pairs_args, 'gas')


    logger.info(f'Closest P-nodes to WPPs: {wpp_closest_pairs}')
    logger.info(f'Closest P-nodes to NGPPs: {ngpp_closest_pairs}')


def helper_main(des_wpps: np.array=[], des_ngpps: np.array=[], to_plot=False):

    assert des_wpps.size, 'Provide indices of desired WPPs'
    assert des_ngpps.size, 'Provide indices of desired NGPPs'

    wpp_f, ngpp_f, p_nodes_f = build_agents()

    logger.info(f'WPP dataframe: {wpp_f}')
    logger.info(f'NGPP dataframe: {ngpp_f}')
    logger.info(f'P_Nodes dataframe: {p_nodes_f}')

    # Currently doesn't work, so do not use
    # if not (des_ngpps or des_ngpps):
    #     des_ngpps = np.arange(len(ngpp_f) - 1)
    #     des_wpps = np.arange(len(wpp_f) - 1)

    # TODO: compute the distances between the ngpps and the wpps, find the k closest pairings
    find_closest_pricing_node(p_nodes_f, wppf=wpp_f, ngppf=ngpp_f, verbosity=True)

    if to_plot:
        plot_agents(wpp_f=wpp_f, ngpp_f=ngpp_f, p_nodes_f=p_nodes_f, des_ngpps=des_ngpps, des_wpps=des_wpps)

    if plot_agents.has_been_called:
        plt.show()


# -> MARK: Interpretation of the agents
def fractional_capacity():
    # compute the fractional capacity that could engaged
    # in a reliability contract
    return None


if __name__ == '__main__':
    # MARK: Setting up the workspace
    current_working_dir = os.getcwd()
    os.chdir(current_working_dir + "/Data")

    helper_main(des_wpps=np.arange(10), des_ngpps=np.arange(10), to_plot=False)
