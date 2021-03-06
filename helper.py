import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, sys
import logging
import functools
import collections
import pickle
from scipy.spatial.distance import cdist, cosine
from geopy.distance import vincenty, distance
from mpl_toolkits.basemap import Basemap
from typing import List, Dict, Tuple


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

EXPECTED_NUMBER_OF_ENTRIES_FULL_YEAR = 8784  # Ensure full year hourly data set 24 * 366 = 8784

###############################################################################

# TODO: [x] Bundle the WPPs
# TODO: [] -> check above via coords
# TODO: [] Match those only with full datasets
# TODO: [] Add documentation for each function
# TODO: [] Delete any non-full year file

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


def write_to_excel(filename, df):
    try:
        filepath = os.getcwd() + '/' + filename
        writer = pd.ExcelWriter(filepath)
        df.to_excel(writer)
        writer.save()
    except:
        raise Exception('Not sure what went wrong . . .')


def closest_point(point, others):
    # return the arguments of the closest coord in others relative to point
    try:
        return cdist([point], others, lambda u, v: vincenty(u, v).kilometers).argmin()
    except KeyError:
        return None


def build_agents(save_to_excel=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    # Drop duplicates
    def filter_duplicates(dataframe, criteria=['Power Plant']) -> pd.DataFrame:
        return dataframe.drop_duplicates(subset=[*criteria])

    # Clean the data, i.e. remove na values, and drop duplicates
    p_nodes_f = p_nodes_f.dropna()
    ngpp_f = filter_tech_ngpp(ngpp_f)
    ngpp_f = filter_duplicates(ngpp_f)
    wpp_f = filter_duplicates(wpp_f)

    # Print to excel file
    if save_to_excel:
        write_to_excel('filtered_ngpp.xlsx', ngpp_f)
        write_to_excel('filtered_wpp.xlsx', wpp_f)
        write_to_excel('filtered_pnodes.xlsx', p_nodes_f)

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
def find_closest_pricing_node_to_agent(pnodesf=None, wppf=None, ngppf=None, to_excel=False):
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

    # < powerplant > < node_long > < node_lat > < node_name >
    inter_node_agent_pairings_df =  pd.DataFrame({'WPP': inter_node_agent_pairings[:,0],
                                                  'WPP PNode Longitude':inter_node_agent_pairings[:,1],
                                                  'WPP PNode Latitude': inter_node_agent_pairings[:,2],
                                                  'WPP PNode Name': inter_node_agent_pairings[:,3],
                                                  'NGPP': inter_node_agent_pairings[:,4],
                                                  'NGPP PNode Longitude': inter_node_agent_pairings[:,5],
                                                  'NGPP PNode Latitude': inter_node_agent_pairings[:,6],
                                                  'NGPP PNode Name': inter_node_agent_pairings[:,7]
                                                  })

    if to_excel:
        write_to_excel('Interagent Node Pairings.xlsx', inter_node_agent_pairings_df)

    return inter_node_agent_pairings


@suspendlogging
def helper_main(des_wpps: np.array=[], des_ngpps: np.array=[], to_plot=False, save_pickle=True, use_pickle=False):

    assert des_wpps.size, 'Provide indices of desired WPPs'
    assert des_ngpps.size, 'Provide indices of desired NGPPs'
    assert use_pickle != save_pickle, 'Ensure that argument use_pickle boolean is opposite of argument save_pickle'

    wpp_f, ngpp_f, p_nodes_f = build_agents(save_to_excel=False)

    print('Currently in', os.getcwd())

    # log.debug(f'WPP dataframe: {wpp_f}')
    # log.debug(f'NGPP dataframe: {ngpp_f}')
    # log.debug(f'P_Nodes dataframe: {p_nodes_f}')

    # Currently doesn't work, so do not use
    # if not (des_ngpps or des_ngpps):
    #     des_ngpps = np.arange(len(ngpp_f) - 1)
    #     des_wpps = np.arange(len(wpp_f) - 1)

    if save_pickle:
        empty_nodes_history: list = []

        for round_num in range(1,10):
            print(f'ROUND: {round_num} \n')

            inter_node_pairings = find_closest_pricing_node_to_agent(p_nodes_f, wppf=wpp_f, ngppf=ngpp_f)
            p_nodes_f, empty_nodes = construct_desired_tables(inter_node_pairings, p_nodes_f, wpp_f, ngpp_f, use_pickle=False, save_pickle=True)
            empty_nodes_history.append(empty_nodes)

            if round_num == 1:
                initial_inter_node_pairings = inter_node_pairings

            if len(empty_nodes) == 0:
                break

        to_package = {
                        'Initial inter_node_pairings': initial_inter_node_pairings,
                        'Final inter_node_pairings': inter_node_pairings,
                        'Empty_nodes_history': empty_nodes_history,
                        'p_nodes_f': p_nodes_f
                      }

        pickle.dump(to_package, open('helper_main_package.p', 'wb'))

    if use_pickle:
        helper_main_package = pickle.load(open('helper_main_package.p', 'rb'))

        helper_init_inter_node_pairings = helper_main_package['Initial inter_node_pairings']
        helper_final_inter_node_pairings = helper_main_package['Final inter_node_pairings']

        # _, _ = construct_desired_tables(helper_final_inter_node_pairings, p_nodes_f, wpp_f, ngpp_f, use_pickle=True, plot=True, save_images=True, log_output=True)

        # Compute the distance between previous pricing nodes and new pricing nodes
        distance_evolution = find_distances_between_pairings(helper_init_inter_node_pairings, helper_final_inter_node_pairings, wpp_f, ngpp_f)

        write_to_excel('distance_evolution.xlsx', distance_evolution)

    print('Done...')

    if to_plot:
        plot_agents(wpp_f=wpp_f, ngpp_f=ngpp_f, p_nodes_f=p_nodes_f, des_ngpps=des_ngpps, des_wpps=des_wpps)


@suspendlogging
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

            if filename == 'node_pd_dict_pickle.p': continue    # Avoid passing over the pickle file

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


@trackcalls
@suspendlogging
def construct_desired_tables(inter_node_agent_pairings: np.array, pnodesf: pd.DataFrame, wppf: pd.DataFrame, ngppf: pd.DataFrame,
                             use_pickle: bool = True, save_pickle: bool = False,
                             plot=False, save_images=False, log_output=False) -> Tuple[pd.DataFrame, np.array]:
    '''

    :param inter_node_agent_pairings: <wpp> <pnode_long_wpp> <pnode_lat_wpp> <pnode_name_wpp> <ngpp> <pnode_long_ngpp> <pnode_lat_ngpp> <pnode_name_ngpp>

    :param pnodesf:
    :param wppf:
    :param ngppf:
    :param use_pickle:
    :param save_pickle:
    :param plot:
    :param save_images:
    :param log_output:
    :return: <Tuple> (new pnodes_f with empty datasets removed with empty set, empty_dataset_nodes)
    '''

    assert use_pickle != save_pickle, 'Ensure that argument use_pickle boolean is opposite of argument save_pickle'

    log_file = None
    if log_output:
        log_file = open('statistical_info.log', 'w')

    reshaped_wpp = inter_node_agent_pairings[:, 0].reshape(inter_node_agent_pairings[:,0].size, 1)
    reshaped_wpp_node = inter_node_agent_pairings[:, 3].reshape(inter_node_agent_pairings[:, 3].size, 1)
    reshaped_ngpp = inter_node_agent_pairings[:, 4].reshape(inter_node_agent_pairings[:,4].size, 1)
    reshaped_ngpp_node = inter_node_agent_pairings[:, 7].reshape(inter_node_agent_pairings[:, 7].size, 1)

    criteria = 'Locational Marginal Price'

    desired_nodes = np.concatenate(
        (reshaped_wpp, reshaped_wpp_node, reshaped_ngpp, reshaped_ngpp_node),
        axis=1
    )

    node_names = np.concatenate(
        (reshaped_wpp_node, reshaped_ngpp_node),
        axis=1
    ).flatten().tolist()

    if use_pickle:
        node_pd_dict: Dict[str, pd.DataFrame] = pickle.load(open('node_pd_dict_pickle.p', 'rb'))

    if save_pickle:
        node_pd_dict: Dict[str, pd.DataFrame] = construct_year_chart(node_names, write_to_excel=True)
        pickle.dump(node_pd_dict, open('node_pd_dict_pickle.p', 'wb'))

    # All the pricing nodes with non full datasets
    empty_data_set_nodes: list = []

    for pair in desired_nodes.tolist():
        wpp, node_wpp, ngpp, node_ngpp = tuple(pair)

        node_wpp_df = node_pd_dict[node_wpp]
        node_ngpp_df = node_pd_dict[node_ngpp]

        node_wpp_np = np.array([node_wpp_df[criteria]]).T
        node_ngpp_np = np.array([node_ngpp_df[criteria]]).T

        # if node_1_np.size == node_2_np.size and node_1_np.size > 0:
        if node_wpp_np.size == node_ngpp_np.size == EXPECTED_NUMBER_OF_ENTRIES_FULL_YEAR:

            difference, mean, standard_dev, similarity, minimum, maximum = get_relevant_data(node_wpp_np, node_ngpp_np)

            statistical_info = (
                f"({node_wpp}, {node_ngpp}) ==> ({wpp}, {ngpp})\nMean: {mean}\nStandard Deviation: {standard_dev}\n"
                f"Similarity: {similarity}\nMin: {minimum}\nMax: {maximum}"
                )

            # Print the statistical information regarding the nodes
            if log_output:
                assert log_file is not None, FileNotFoundError
                sys.stdout = log_file
                print('\n' + statistical_info)
                sys.stdout = sys.__stdout__

            if plot:
                plt.figure(f'{pair}')
                plt.plot(difference)
                plt.figtext(0, 0.1, statistical_info, fontsize=8)

                if save_images:
                    plt.savefig(f'({node_wpp}, {node_ngpp})_({wpp}, {ngpp}).png')

        else:  # These nodes are not full datasets, collect and create table to drop
            log.debug('Do not match in size...')
            if node_wpp_np.size != EXPECTED_NUMBER_OF_ENTRIES_FULL_YEAR:
                empty_data_set_nodes.append([wpp, node_wpp])

            if node_ngpp_np.size != EXPECTED_NUMBER_OF_ENTRIES_FULL_YEAR:
                empty_data_set_nodes.append([ngpp, node_ngpp])

    empty_data_set_nodes = np.array(empty_data_set_nodes)

    if empty_data_set_nodes.size > 0:
        pnodesf = pnodesf[~pnodesf['LMP Name'].isin(empty_data_set_nodes[:, 1].tolist())]

    return pnodesf, empty_data_set_nodes


def get_relevant_data(node_1, node_2, modified = False) -> tuple:
    ''' Returns the mean, stdev, min and max of the difference between two arrays'''

    def compute(n_1: np.array, n_2: np.array) -> tuple:
        difference_array = n_1 - n_2

        return difference_array, np.mean(difference_array), np.std(difference_array), 1 - cosine(n_1, n_2), np.min(difference_array), np.max(difference_array)

    if modified:
        # node_1 and node_2 are of List[str]
        node_pd_dict: Dict[str, pd.DataFrame] = pickle.load(open('node_pd_dict_pickle.p', 'rb'))
        criteria = 'Locational Marginal Price'

        mean_list = []
        standard_dev_list = []
        similarity_list = []
        minimum_list = []
        maximum_list = []

        for wpp_node, ngpp_node in zip(node_1, node_2):
            node_wpp_np = np.array([node_pd_dict[wpp_node][criteria]]).T
            node_ngpp_np = np.array([node_pd_dict[ngpp_node][criteria]]).T

            _, mean, standard_dev, similarity, minimum, maximum = compute(node_wpp_np, node_ngpp_np)

            mean_list.append(mean)
            standard_dev_list.append(standard_dev)
            similarity_list.append(similarity)
            minimum_list.append(minimum)
            maximum_list.append(maximum)

        return _, mean_list, standard_dev_list, similarity_list, minimum_list, maximum_list

    else:
        return compute(node_1, node_2)


# -> MARK: Interpretation of the agents
def fractional_capacity():
    # compute the fractional capacity that could engaged
    # in a reliability contract
    return None


def find_distances_between_pairings(initial_pairing, final_pairing, wppf, ngppf):
    '''<wpp> <pnode_long_wpp> <pnode_lat_wpp> <pnode_name_wpp> <ngpp> <pnode_long_ngpp> <pnode_lat_ngpp> <pnode_name_ngpp>'''

    initial_wpp = initial_pairing[:, 0]
    initial_wpp_node = initial_pairing[:, 3]
    initial_wpp_coords = initial_pairing[:, 1:3]
    initial_ngpp = initial_pairing[:, 4]
    initial_ngpp_node = initial_pairing[:, 7]
    initial_ngpp_coords = initial_pairing[:, 5:7]

    final_wpp = final_pairing[:, 0]
    final_wpp_node = final_pairing[:, 3]
    final_wpp_coords = final_pairing[:, 1:3]
    final_ngpp = final_pairing[:, 4]
    final_ngpp_node = final_pairing[:, 7]
    final_ngpp_coords = final_pairing[:, 5:7]

    wpp_coordinates = np.concatenate((initial_wpp_coords, final_wpp_coords), axis=1)
    ngpp_coordinates = np.concatenate((initial_ngpp_coords, final_ngpp_coords), axis=1)

    wpp_coordinates = np.apply_along_axis(lambda u: distance((u[0], u[1]), (u[2], u[3])).km, axis=1, arr=wpp_coordinates)
    ngpp_coordinates = np.apply_along_axis(lambda u: distance((u[0], u[1]), (u[2], u[3])).km, axis=1, arr=ngpp_coordinates)

    final_wpp_node = final_wpp_node.tolist()
    final_ngpp_node = final_ngpp_node.tolist()

    _, mean_list, standard_dev_list, similarity_list, minimum_list, maximum_list = get_relevant_data(final_wpp_node, final_ngpp_node, modified=True)

    # Include the capacities
    final_wpp = final_wpp.tolist()
    final_ngpp = final_ngpp.tolist()
    wpp_capacities_list = []
    ngpp_capacities_list = []
    for wpp, ngpp in zip(final_wpp, final_ngpp):
        wpp_capacities_list.append(wppf[wppf["Power Plant"] == wpp]['Operating Capacity'].values[0])
        ngpp_capacities_list.append(ngppf[ngppf["Power Plant"] == ngpp]['Operating Capacity'].values[0])

    output = {
        'Final WPP': initial_wpp.tolist(),
        'Final WPP Capacity (MW)': wpp_capacities_list,
        'Initial NGPP': initial_ngpp.tolist(),
        'Final NGPP': final_ngpp,
        'Final NGPP Capacity (MW)': ngpp_capacities_list,
        'Initial WPP LMP': initial_wpp_node.tolist(),
        'Final WPP LMP': final_wpp_node,
        'Distance WPP (km)': wpp_coordinates.tolist(),
        'Initial NGPP LMP': initial_ngpp_node.tolist(),
        'Final NGPP LMP': final_ngpp_node,
        'Distance NGPP (km)': ngpp_coordinates.tolist(),
        'Mean Final': mean_list,
        'Standard Dev Final': standard_dev_list,
        'Similarity Final': similarity_list,
        'Minimum Final': minimum_list,
        'Maximum Final': maximum_list
    }

    return pd.DataFrame.from_dict(output)


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger()

    # MARK: Setting up the workspace
    current_working_dir = os.getcwd()
    os.chdir(current_working_dir + "/Data")

    helper_main(des_wpps=np.arange(10), des_ngpps=np.arange(10), to_plot=False, save_pickle=False, use_pickle=True)

    if plot_agents.has_been_called or construct_desired_tables.has_been_called:
        plt.show()
