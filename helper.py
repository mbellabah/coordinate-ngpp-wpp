import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import logging
import functools
import collections
import pickle
from scipy.spatial.distance import cdist
from geopy.distance import vincenty
from mpl_toolkits.basemap import Basemap
from typing import List, Dict


# Exception Handling
from xlrd import XLRDError
from pandas.errors import ParserError


def trackcalls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.has_been_called = True
        return func(*args, **kwargs)

    wrapper.has_been_called = False

    return wrapper


def suspendlogging(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.disable(logging.DEBUG)
        try:
            return func(*args, **kwargs)
        finally:
            logging.disable(logging.NOTSET)
    return wrapper

###############################################################################

# TODO: [x] Bundle the WPPs
# TODO: [] -> check above via coords
# TODO: [] Calculate the relevant data regarding the pricing nodes

################################################################################


# MARK: Helper Functions

@suspendlogging
def load_pd(file_name):

    excel_file_type = ('.xls', '.xlsx', '.XLS', '.XLSX')
    csv_file_type = ('.csv', '.CSV')

    log.debug(('load_pd', os.getcwd()))

    if file_name.endswith(excel_file_type):
        try:
            return pd.read_excel(file_name, sheet_name=0)
        except XLRDError:
            return None

    elif file_name.endswith(csv_file_type):
        try:
            return pd.read_csv(file_name, skiprows=[0, 1, 2, 3, 5])     # skip rows to avoid the nonsense at the top
        except ParserError as e:
            log.debug(e)
            return None

    else:
        log.debug('Unknown file type')
        return None


def closest_point(point, others):
    # return the arguments of the closest coord in others relative to point
    try:
        return cdist([point], others, lambda u, v: vincenty(u, v).kilometers).argmin()
    except KeyError:
        return None


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

@suspendlogging
def find_closest_pricing_node_to_agent(pnodesf=None, wppf=None, ngppf=None):
    assert pnodesf is not None, ValueError('Provide the pricing node file')

    # First find the closest pricing node for each agent -- # <Longitude>, <Latitude>, <Plant Key>
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
    pnode_names = np.reshape(pnode_names, (pnode_names.size, 1))

    log.debug('\nComputing the closest pricing nodes . . .')

    wpp_closest_pairs_args = np.apply_along_axis(
        closest_point, 1, wpp_coords, pnodes_coords
    )
    ngpp_closest_pairs_args = np.apply_along_axis(
        closest_point, 1, ngpp_coords, pnodes_coords
    )

    # Stack the pairs, confirm using snl
    def stack_pairs(pair_args, pair_type: str):
        closest_pnodes = np_pnodes[pair_args]
        # closest_pnodes = np.reshape(closest_pnodes, (closest_pnodes.size, 1))

        if pair_type == 'wind':
            return np.concatenate((wpp_names, closest_pnodes), axis=1)
        elif pair_type == 'gas':
            return np.concatenate((ngpp_names, closest_pnodes), axis=1)
        elif pair_type == 'node':
            return np.concatenate((pnode_names, closest_pnodes), axis=1)
        else:
            raise ValueError(f'{pair_type} does not exist!')

        # output: <power plant> <node_long> <node_lat> <node_name>

    # Got the pricing node and agent pairs
    wpp_pricing_node_pairs = stack_pairs(wpp_closest_pairs_args, 'wind')
    ngpp_pricing_node_pairs = stack_pairs(ngpp_closest_pairs_args, 'gas')

    log.debug(f'WPP_Pairs {wpp_pricing_node_pairs}')
    log.debug(f'NGPP_Pairs {ngpp_pricing_node_pairs}')


    ##################################################################################
    pnodes_wpp_names_coords = wpp_pricing_node_pairs[:, 1:3]
    pnodes_ngpp_names_coords = ngpp_pricing_node_pairs[:, 1:3]


    log.debug(f'names_coords_wpp {pnodes_wpp_names_coords}')
    log.debug(f'names_coords_ngpp {pnodes_ngpp_names_coords}')

    pairs_within_selves_args = np.apply_along_axis(
        closest_point, 1,
        pnodes_wpp_names_coords,
        pnodes_ngpp_names_coords
    )

    inter_node_agent_pairings = np.concatenate((wpp_pricing_node_pairs, ngpp_pricing_node_pairs[pairs_within_selves_args]), axis=1)
    log.debug(f'inter_node_agent_pairings {inter_node_agent_pairings}')

    return inter_node_agent_pairings


# @suspendlogging
def helper_main(des_wpps: np.array=[], des_ngpps: np.array=[], to_plot=False):

    assert des_wpps.size, 'Provide indices of desired WPPs'
    assert des_ngpps.size, 'Provide indices of desired NGPPs'

    wpp_f, ngpp_f, p_nodes_f = build_agents()

    log.debug(f'WPP dataframe: {wpp_f}')
    log.debug(f'NGPP dataframe: {ngpp_f}')
    log.debug(f'P_Nodes dataframe: {p_nodes_f}')

    # Currently doesn't work, so do not use
    # if not (des_ngpps or des_ngpps):
    #     des_ngpps = np.arange(len(ngpp_f) - 1)
    #     des_wpps = np.arange(len(wpp_f) - 1)

    inter_node_pairings = find_closest_pricing_node_to_agent(p_nodes_f, wppf=wpp_f, ngppf=ngpp_f)
    construct_desired_tables(inter_node_pairings)

    if to_plot:
        plot_agents(wpp_f=wpp_f, ngpp_f=ngpp_f, p_nodes_f=p_nodes_f, des_ngpps=des_ngpps, des_wpps=des_wpps)

    if plot_agents.has_been_called:
        plt.show()


# @suspendlogging
def construct_year_chart(node_names_list: List[str], write_to_excel: bool = False) -> Dict[str, pd.DataFrame]:
    ''' Returns an excel file with the excel file '''

    if os.getcwd().endswith('2016_realtime_hourly_dataset'):
        pass
    else:
        os.chdir(os.getcwd() + '/2016_realtime_hourly_dataset')

    curr_working_dir = os.getcwd()

    output_df = None
    infer_headers_flag = False
    headers = []
    df_name = None
    desired_key = 'Location Name'

    node_pd_dict = {}

    # assume infer_headers = ['H', 'Date', 'Hour Ending', 'Location ID', 'Location Name', 'Location Type', 'Locational Marginal Price', 'Energy Component', 'Congestion Component', 'Marginal Loss Component']

    files = os.listdir(curr_working_dir)
    number_files = len(files) - 1

    for node_name in node_names_list:
        frames = []

        if f'{node_name}_2016.xlsx' in os.listdir('../individual_nodes'):
            log.debug(f'{node_name} excel already exists, so skipping write to excel')
            node_pd_dict[node_name] = load_pd('../individual_nodes/' + f'{node_name}_2016.xlsx')

            continue

        for index, filename in enumerate(files):
            if (index + 1) % 20 == 0:
                log.info((f'On file {index+1} out of {number_files}'))

            df = load_pd(filename)

            if not infer_headers_flag:
                headers = list(df)
                infer_headers_flag = True

            try:
                df_name = df.loc[df[desired_key] == node_name]
                frames.append(df_name)

            except ValueError:
                raise ValueError(f"{node_name} doesn't exist!")

            except KeyError:
                raise KeyError(f"Can't find {desired_key} in {headers}")

        concatenated_df = pd.concat(frames)
        log.debug(concatenated_df)

        if write_to_excel:

            if f'{node_name}_2016.xlsx' in os.listdir('../individual_nodes'):
                log.debug(f'{node_name} excel already exists, so skipping write to excel')
                continue

            final_file_name = f'/{node_name}_2016.xlsx'

            file_path = '../individual_nodes' + final_file_name

            writer = pd.ExcelWriter(file_path)
            concatenated_df.to_excel(writer)
            writer.save()

            log.debug(f'Wrote {final_file_name} to excel in {file_path}')

        node_pd_dict[node_name] = concatenated_df

    log.debug('Done!, returning pd_dict')
    return node_pd_dict


@suspendlogging
def construct_desired_tables(inter_node_agent_pairings: np.array, use_pickle: bool = True, save_pickle: bool = False):
    # <wpp> <pnode_long_wpp> <pnode_lat_wpp> <pnode_name_wpp> <ngpp> <pnode_long_ngpp> <pnode_lat_ngpp> <pnode_name_ngpp>

    assert use_pickle != save_pickle, 'Ensure that use_pickle boolean is opposite of save_pickle'

    reshaped_wpp_node = inter_node_agent_pairings[:, 3].reshape(inter_node_agent_pairings[:, 3].size, 1)
    reshaped_ngpp_node = inter_node_agent_pairings[:, 7].reshape(inter_node_agent_pairings[:, 7].size, 1)

    criteria = 'Locational Marginal Price'

    desired_nodes = np.concatenate(
        (reshaped_wpp_node, reshaped_ngpp_node),
        axis=1
    )
    node_names = desired_nodes.flatten().tolist()

    if use_pickle:
        node_pd_dict = pickle.load(open('node_pd_dict_pickle.p', 'rb'))

    if save_pickle:
        node_pd_dict = construct_year_chart(node_names, write_to_excel=True)
        pickle.dump(node_pd_dict, open('node_pd_dict_pickle.p', 'wb'))

    log.debug(node_pd_dict)

    for pair in desired_nodes.tolist():
        node_1, node_2 = tuple(pair)

        node_1_df = node_pd_dict[node_1]
        node_2_df = node_pd_dict[node_2]

        node_1_np = np.array([node_1_df[criteria]]).T
        node_2_np = np.array([node_2_df[criteria]]).T

        if node_1_np.size == node_2_np.size:
            difference = node_1_np - node_2_np

            plt.figure(f'{pair}')
            plt.plot(difference)

        else: pass

        # note, we have node_1_df and node_2_df --> turn into numpy array and run std,


# -> MARK: Interpretation of the agents
def fractional_capacity():
    # compute the fractional capacity that could engaged
    # in a reliability contract
    return None


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger()

    # MARK: Setting up the workspace
    current_working_dir = os.getcwd()
    os.chdir(current_working_dir + "/Data")

    helper_main(des_wpps=np.arange(10), des_ngpps=np.arange(10), to_plot=False)

    plt.show()

    # construct_year_chart(['UN.FRNKLNSQ11.510CC'], write_to_excel=True)
