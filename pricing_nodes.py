'''
Substantial credit for the following code goes to WattTime group
whose code I have adapted to fit the needs of the academic project.
https://github.com/WattTime/pyiso Version 2.17
'''

import sys
import requests
import numpy as np
import pandas as pd
import datetime
import pytz
from requests.auth import HTTPBasicAuth
from typing import Type
from collections import defaultdict


# TODO: Find the pricing nodes history between given date
# TODO: Fix up confusion regarding the LD, the labels used


class ISONEClient:
    NAME = 'ISONE'
    base_url = 'https://webservices.iso-ne.com/api/v1.1'
    TZ_NAME = 'America/New_York'

    def __init__(self, username, password):
        try:
            self.auth = HTTPBasicAuth(username, password)

        except KeyError:
            msg = 'Provide authentication'
            raise RuntimeError(msg)

        self.options = {}

    def fetch_data(self, endpoint):
        url = self.base_url + endpoint
        response = requests.get(url, auth=self.auth)

        if response:
            return response.json()
        else:
            return {}

    def parse_lmp_data(self, data: Type['json']):
        try:
            return data['FiveMinLmps']['FiveMinLmp']

        except (KeyError, TypeError):
            raise ValueError(f'Failed to parse {data}')

    def raw_lmp_data_to_pd(self, raw_data) -> pd.DataFrame:
        processed_dict = defaultdict(list)

        def extract_data(five_min_data: dict) -> dict:
            processed_dict['Date'].append(five_min_data['BeginDate'][:10])
            processed_dict['Minute Ending'].append(five_min_data['BeginDate'][14:16])
            processed_dict['Location ID'].append(five_min_data['Location']['@LocId'])
            processed_dict['Location Name'].append(five_min_data['Location']['$'])
            processed_dict['Locational Marginal Price'].append(five_min_data['LmpTotal'])
            processed_dict['Energy Component'].append(five_min_data['EnergyComponent'])
            processed_dict['Congestion Component'].append(five_min_data['CongestionComponent'])
            processed_dict['Marginal Loss Component'].append(five_min_data['LossComponent'])

        for day_dp in raw_data:
            for five_min in day_dp:
                extract_data(five_min)

        dataframe = pd.DataFrame.from_dict(processed_dict)
        return dataframe

    def get_lmp(self, node_id: int, **kwargs):
        self.options.update(kwargs)
        raw_data = []

        for endpoint in self.request_endpoints(node_id):
            data = self.fetch_data(endpoint)
            raw_data.append(self.parse_lmp_data(data))

        return self.raw_lmp_data_to_pd(raw_data)    # convert to pd df

    def request_endpoints(self, location_id=None):
        ext = ''
        if self.options['data'] == 'lmp' and location_id is not None:
            base_endpoint = 'fiveminutelmp'
            ext = f'/location/{location_id}'

        request_endpoints = []
        if self.options['latest']:
            request_endpoints.append(f'/{base_endpoint}/current{ext}.json')

        elif self.options['start_at'] and self.options['end_at']:
            for date in self.dates():
                date_str = date.strftime('%Y%m%d')
                request_endpoints.append(f'/{base_endpoint}/day/{date_str}{ext}.json')

        else:
            msg = 'Either latest or start_at and end_at must be both provided'
            raise ValueError(msg)

        return request_endpoints

    def dates(self):
        dates = []

        if self.options['start_at'] and self.options['end_at']:
            local_start = self.options['start_at'].astimezone(pytz.timezone(self.TZ_NAME))
            local_end = self.options['end_at'].astimezone(pytz.timezone(self.TZ_NAME))
            this_date = local_start.date()
            while this_date <= local_end.date():
                dates.append(this_date)
                this_date += datetime.timedelta(days=1)

        return dates



def isone_main():
    ISONE = ISONEClient(username=sys.argv[1], password=sys.argv[2])

    node_id = 355

    today = datetime.datetime.today()
    start_at = today - datetime.timedelta(days=2)
    end_at = today - datetime.timedelta(days=1)

    msg = ISONE.get_lmp(node_id=node_id, data='lmp', latest=False, start_at=start_at, end_at=end_at)

    print(msg)


if __name__ == '__main__':
    isone_main()

