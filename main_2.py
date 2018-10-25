import os
from geopy.geocoders import Nominatim
import pandas as pd


'''
TODO: Look for NGPP that are highly efficient, underutilized, competitive with clearing energy prices
      typically bids into energy reserve markets. 
TODO: Find the closest NGPP, particularly the combined cycle NGPPs -- make a claim that they are partial operation 
      operating in the reserves which isn't as a lucrative in New England. NGPP can't do frequency regulation - 
      can be dispatched in the 5-10 min range -- CCGT (Combined Cycle Gas Turbines)
'''

# MARK: Setting up the workspace
current_working_dir = os.getcwd()
os.chdir(current_working_dir+"/Data")


def load_pd(file_name):
    return pd.read_excel(file_name)


# wpp_f = load_pd('wind_power_producers.xls')


# MARK: find the relevant coords of each power producer
def find_coord(place: str, verbosity=False) -> str:
    geolocator = Nominatim(user_agent='_')
    location = geolocator.geocode(place)

    if location is not None:
        if verbosity: return location.raw
        return place, location.latitude, location.longitude
    else:
        return "Can't find", place


print(find_coord('UN.GULFISLD34.5GULF', True))

