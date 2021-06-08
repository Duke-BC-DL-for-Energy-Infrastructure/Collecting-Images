import os
import geopandas as gpd

from sklearn.cluster import DBSCAN, SpectralClustering
from pathlib import Path

REGION_NAMES = ['NE', 'EM', 'NW', 'SW']

# states abbreviations per region
NE_ab = ['PE', 'NY', 'NJ', 'DE', 'MD', 'CT', 'MA', 'VM', 'ME', 'NH']
EM_ab = ['MN', 'IA', 'MO', 'MI', 'WI', 'IL', 'IN', 'OH']
NW_ab = ['WA', 'ID', 'OR', 'MT', 'WY']
SW_ab = ['NM', 'TX', 'CA', 'AZ', 'UT', 'NV', 'CO']

# US state's border (polygons) from states_21basic dataset from
# https://hub.arcgis.com/datasets/

US_STATES_DIRECTORY = Path('input_data/states_21basic')
US_STATES_FILENAME = 'states.shp'
US_STATES = gpd.read_file(os.path.join(US_STATES_DIRECTORY, US_STATES_FILENAME))

# set of states per region
NE = ['Pennsylvania', 'New York', 'New Jersey', 'Delaware', 'Maryland',
      'Connecticut', 'Massachusetts', 'Vermont', 'Maine', 'New Hampshire']
EM = ['Minnesota', 'Iowa', 'Missouri', 'Michigan', 'Wisconsin', 'Illinois',
      'Indiana', 'Ohio']
NW = ['Washington', 'Idaho', 'Oregon', 'Montana', 'Wyoming']
SW = ['New Mexico', 'Texas', 'California', 'Arizona', 'Utah', 'Nevada',
      'Colorado']

# using the same region distribution
REGION_STATES = [NE, EM, NW, SW]

# setup path to file with wind turbine's coordinates
WIND_TURBINES_FILENAME = 'uswtdb_v3_3_20210114.csv'
WIND_TURBINES_FILEPATH = Path('input_data')/WIND_TURBINES_FILENAME
WIND_TURBINE_OUTPUT_DIR = Path('processed_data/wt')

# setup path to file with electric tower's coordinates
ELECTRIC_TOWERS_FILENAME = 'Transmission_Lines.shp'
ELECTRIC_TOWERS_FILEPATH = Path('input_data/'
                                'Electric_Power_Transmission_Lines-shp')/\
                                ELECTRIC_TOWERS_FILENAME

ELECTRIC_TOWERS_OUTPUT_DIR = Path('processed_data/et')

# Clustering functions and parameters
CLUSTERING = {
    'DBSCAN': {
    'function': DBSCAN,
    'params': {'eps':0.3, 'min_samples':10}
    },
    'SC': {
    'function':SpectralClustering,
    'params': {'n_clusters': 4, 'affinity':'nearest_neighbors'}
    }
}
