# from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import agents

# MARK: Setting up the workspace
current_working_dir = os.getcwd()
os.chdir(current_working_dir+"/Data")


def load_pd(file_name, verbosity=False):
    data_frame = pd.read_excel(file_name, sheet_name=0)
    if verbosity: print(data_frame)

    return data_frame


def build_agents():
    '''
    As of now, we only want to look at the relevant agents,
    :return:
    '''
    wpp_f = load_pd('wind_power_producers.xls')
    ngpp_f = load_pd('natural_gas_power_producers.xls')

    return wpp_f, ngpp_f


def plot_agents():
    pass


if __name__ == '__main__':

    wpp_f, ngpp_f = build_agents()

