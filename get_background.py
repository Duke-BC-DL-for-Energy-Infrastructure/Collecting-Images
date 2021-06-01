# Built-in
import os
import ee
import time
import pandas as pd
from pathlib import Path

# gee download package
from gee_download import bbox_from_point, downloadGStorage, get_NAIP_Task

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

ee.Initialize()

# setup path to file with wind farm coordinates
FILENAME = 'NE_train.csv'
INPUT_FILEPATH = Path('input_data/'+FILENAME)
OUTPUT_FILEPATH = Path('processed_data/')
REGIONS = ['NE', 'NW', 'EM', 'SW'] # Northeast, Northwest, Eastern Midwest

# Define name for the bucket ('directory' in gcp) and local directory to store images
BUCKET_NAME = 'naip_storage'
LOCAL_DST_DIRECTORY = 'test_download'

# Create local destination directory
!mkdir $LOCAL_DST_DIRECTORY

# function that takes one point as input (lon, lat), and returns a pair of
# coordinates at the SW and NE of the original point.

def add_adj_coords(input_filepath:Path = INPUT_FILEPATH,
                   output_filepath:Path = OUTPUT_FILEPATH,
                   distance = 2000):
                   """
                   Takes as input a csv file with column names lon, lat,
                   cluster:
                   Arguments:
                    input_filepath: path to input csv
                    output_filepath: path to store clustered data
                    distance: radial distance in meters from the input
                    (lon, lat) as center
                   Returns:
                    Dataframe with same input data and additional
                    columns for SW and NE coordinates (lon, lat).
                   """
                   temp_df = pd.read_csv(input_filepath)
                   #get file basename
                   base_name = (os.path.basename(input_filepath)[:-4])

                   for index, row in temp_df.iterrows():
                        # set latitude and longitude from row
                        lat = row['lat']
                        lon = row['lon']
                        # create a bounding box from coordinate
                        bbox = bbox_from_point(point = (lat, lon),
                                               dist = distance)
                        coord_names = ['SE_lat', 'SE_lon', 'NW_lat', 'NW_lon']
                        for c_name, coord in zip(coord_names, bbox):
                            temp_df.at[index, f'{c_name}'] = coord
                        # reset order of columns
                        temp_df = temp_df[['name', 'lon', 'lat', 'SE_lat',
                                           'SE_lon', 'NW_lat', 'NW_lon']]
                        #save to new file
                        temp_df.to_csv(os.path.join(OUTPUT_FILEPATH,
                                       f'{base_name}_background.csv'),
                                       index=False)


def get_background_img(input_filepath:Path, gcp_bucket: str, distance: float):
    """"
    Takes as input a csv file with column names lon, lat, SW_lon, SW_lat,
    NE_lon, NE_lat:
    Arguments:
        input_filepath: path to input csv
        gcp_bucket: bucket address in gcp to store images downloaded.
        filename: name of the output file.
        Returns:
        executable task to download an image from NAIP store it in the defined
        gcp bucket.
    """
    df_temp = pd.read_csv(input_filepath)

    for index, row in df_temp.loc[:,:].iterrows():
        #lat, lon = row['lat'], row['lon'];
        SE_lat, SE_lon = row['SE_lat'], row['SE_lon'];
        NW_lat, NW_lon = row['NW_lat'], row['NW_lon'];
        #bbox = bbox_from_point((lat, lon))
        bbox_SE = bbox_from_point(point = (SE_lat, SE_lon), dist=distance)
        bbox_NW = bbox_from_point(point = (NW_lat, NW_lon), dist=distance)

        for suffix, bb in zip(['_SE', '_NW'], [bbox_SE, bbox_NW]):
            task = get_NAIP_Task(bb, gcp_bucket, row['name']+suffix)
            if task == -1:
                print('error in the request')
                pass
            # Step 1: set task to download image from earth engine
            else:
                task.start()
                while (task.status()['state'] != 'COMPLETED'):
                    time.sleep(1)
                # Step 2: check the status of the request. It can be running or ready
            print(task.status())
