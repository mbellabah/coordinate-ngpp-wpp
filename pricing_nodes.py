import numpy as np
import scipy
import matplotlib.pyplot as plt
import requests
import pandas as pd
from requests.auth import HTTPBasicAuth

username = 'bellabah@mit.edu'
password = 'bella123'

auth = HTTPBasicAuth(username, password)

# TODO: Find the pricing nodes history between given date
# TODO: Fix up confusion regarding the LD, the labels used


class ISONEClient:
    NAME = 'ISONE'

    base_url = 'https://webservices.iso-ne.com/api/v1.1'

    locations = {
        'INTERNALHUB': 4000,
        'MAINE': 4001,
        'NEWHAMPSHIRE': 4002,
        'VERMONT': 4003,
        'CONNECTICUT': 4004,
        'RHODEISLAND': 4005,
        'SEMASS': 4006,
        'WCMASS': 4007,
        'NEMASSBOST': 4008,
    }

    def __init__(self, auth):
        try:
            self.auth = auth
        except KeyError:
            msg = 'Provide authentication'
            raise RuntimeError(msg)

    def fetch_data(self, endpoint):
        url = self.base_url + endpoint
        response = requests.get(url, auth=self.auth)

        if response:
            return response.json()
        else: return {}

    def parse_json_lmp_data(self, data):
        try:
            return data['HourlyLmps']['HourlyLmp']
        except (KeyError, TypeError):
            msg = f'Could not parse ISONE lmp data: {data}'
            raise ValueError(msg)

    def _parse_lmp(self, json):

        raise NotImplementedError

        dataframe = pd.DataFrame(json)
        return dataframe

    def get_lmp(self, node_id='INTERNALHUB', **kwargs):
        try:
            location_id = self.locations[node_id.upper()]
        except KeyError:
            msg = 'No LMP data available'
            raise ValueError(msg)

        raise NotImplementedError

    def compute_data_details(self, lmp_1_id: str, lmp_2_id: str) -> tuple:

        lmp_1 = self.get_lmp(node_id=lmp_1_id)
        lmp_2 = self.get_lmp(node_id=lmp_2_id)

        # convert into numpy array
        # get the difference between the two
        # compute statistical information
        # return correlation measure

        raise NotImplementedError


def client_main():
    Client = ISONEClient(auth)

    location_id = 'LD.COLBURN 13.8'
    # ext = '/location/%s' % location_id
    ext = ''
    base_endpoint = 'hourlylmp/da/final'
    endpoint = '/%s/current%s.json' % (base_endpoint, ext)

    # print(Client.parse_json_lmp_data(Client.fetch_data(endpoint)))
    print(Client.fetch_data(endpoint))





if __name__ == '__main__':
    client_main()

