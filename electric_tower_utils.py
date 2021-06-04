'''
Set of functions to pre-process data downloaded from:
https://hifld-geoplatform.opendata.arcgis.com/datasets/geoplatform::electric-power-transmission-lines/about

Download all zipped folder Electric_Power_Transmission_Lines-shp.zip and put
in a directory. We are unzipping the folder in input_data folder.
'''

import os
import glob
import fiona
import pandas as pd
import geopandas as gpd

# maniputale geometric boundaries
from shapely import wkt
from shapely.geometry import Polygon, Point

# manage os path structures
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


DIR_PATH = Path('input_data/Electric_Power_Transmission_Lines-shp')
INPUT_FILENAME = 'Transmission_Lines.shp'

OUTPUT_PATH = Path('processed_data/et')

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
REGION_NAMES = ['NE', 'EM', 'NW', 'SW']
REGION_STATES = [NE, EM, NW, SW]

def convert_to_EPSG4326(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    '''
    Convert shapefile using pyproj and fiona to EPSG:4326 (lon, lat)
    '''
    gdf = gdf.to_crs({'proj':'longlat',
                      'ellps':'WGS84',
                      'datum':'WGS84'})

    print(f'geometry converted to CRS EPSG {gdf.crs.to_epsg()}')
    return gdf

def split_txlines_by_region(filepath: Path, output_dir: Path,
                            region_names: list,region_states: list) -> None:
    '''
    Given shapefile with tx_lines across the US, group them by states within
    a region (NE, EM, NW, SW)
    Arguments:
        filepath: path to input shapefile from HIFLD
        output_dir: folder to store output csv
        region_names: criteria used to subset dataset by region (NE, NW, SW, EM)
        region_states: US states that comprises each geographical region
    Returns:
        csv with tx lines coordinates grouped by region and US states.
    '''
    tmp_gdf = gpd.read_file(filepath)

    print(f'input shapefile\'s geometry CRS EPSG {tmp_gdf.crs.to_epsg()}')

    if not (tmp_gdf.crs.axis_info[0].abbrev == 'Lat') or \
           (tmp_gdf.crs.axis_info[1].abbrev == 'Lon'):

        tmp_gdf = convert_to_EPSG4326(tmp_gdf)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #create a log file for errors
    logf = open(os.path.join(output_dir, 'error.log'), 'w')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Logging split_txlines_by_region, current Time =", current_time)
    logf.write(f"Logging at {current_time}\n")

    # store partial tx_lines dataframes generated per region
    tx_lines = []

    for region_name, region_state in zip(region_names, region_states):
        print(f'\n***{region_name}***\n')
        logf.write(f'\n***{region_name}***\n')
        region_list = []
        for state in region_state:
            print(f'{state}')
            polygon = US_STATES[US_STATES.STATE_NAME == state]['geometry'].values[0]
            if (polygon.geom_type == 'Polygon' or polygon.geom_type == 'MultiPolygon'):
                valid_area = polygon
            else:
                raise ValueError('invalid polygon:', polygon.geom_type)
            temp_df = pd.DataFrame()
            for index, row in tmp_gdf.iterrows():
                line = row['geometry']
                if valid_area.contains(line):
                    temp_df = temp_df.append(row)
            temp_df['state'] = state
            logf.write(f'The state of {state} has {len(temp_df)} records\n')
            logf.flush()
            region_list.append(temp_df)
            region_df = pd.concat(region_list)
        region_df['region'] = region_name
        tx_lines.append(region_df)
    pd.concat(tx_lines).to_csv(os.path.join(output_dir,'tx_lines.csv'),
                               index=False)
    logf.close()
    return

def get_tower_coords(filename:Path) -> pd.DataFrame:
    '''
    From a tx lines csv file with geometry as LINESTRING get POINTS that
    correspond to each tower coordinate. Output file is stored within the same
    folder than input csv file.
    '''

    output_dir = os.path.dirname(filename)

    #create a log file for errors
    logf = open(os.path.join(output_dir, 'error.log'), 'wt')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Logging get_tower_coords, current Time =", current_time)
    logf.write(f"Logging at {current_time}\n")

    #region = (os.path.basename(filename)[:-13])
    #logf.write(f'\n***{region}***\n')

    df = pd.read_csv(filename)

    # convert dtype of geometry column in the csv file from str to geometry
    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # Get points from linestrings in a dataframe. Observe data of interest: ID, shape length, voltage, lon and lat
    temp_df = pd.DataFrame()

    with tqdm(desc='tower_coordinates', total=gdf.shape[0]) as pbar:

        for index, row in gdf.iterrows():
            pbar.update(1)
            if (row.geometry.geom_type == 'LineString'):
                x,y = row.geometry.coords.xy
                temp_df = temp_df.append(pd.DataFrame({
                                            'id':row['ID'],
                                            'tx_line_length':row['SHAPE_Leng'],
                                            'voltage':row['VOLTAGE'],
                                            'state':row['state'],
                                            'region':row['region'],
                                            'owner':row['OWNER'],
                                            'lon':x,'lat':y
                                            }))
            else:
                logf.write(f'{row.ID} with index {index} has the following '
                           f'geom type: {row.geometry.geom_type}\n')
                logf.flush()
                continue

    logf.close()
    temp_df.to_csv(os.path.join(output_dir, 'tower_coordinates.csv'),
                   index=False)
    return temp_df


def electric_voltage_subset(df:pd.DataFrame) -> pd.DataFrame:
    '''
    Given the tower coordinates dataframe, apply certain criteria tu subset
    the dataset and apply clustering methods

    '''
