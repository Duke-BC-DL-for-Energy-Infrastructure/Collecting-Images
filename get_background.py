# Built-in
import os
import ee
import time

import pandas as pd
from pathlib import Path
from tqdm import tqdm

# gee download package
from gee_utils import bbox_from_point
from gee_local_download import convert_bbox_latlon_lonlat, download_NAIP_toLocal

import warnings
from argparse import ArgumentParser

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

ee.Initialize()


# function that takes one point as input (lon, lat), and returns a pair of
# coordinates at the SW and NE of the original point.

def add_adj_coords(input_filepath: str, output_dir: str,
                   distance: float, save_file = False):
    """
    Takes as input a csv file with column names lon, lat, cluster:
    Arguments:
        input_filepath: path to input csv
        output_dir: path to output directory
        distance: radial distance in meters from the input (lon, lat) used
            as new center of adjancent background images
        fname: column used to name files
        save_file: optional to download a csv file with output
   Returns:
        Dataframe with same input data and additional columns for SW and NE
        coordinates (lon, lat).
    """
    temp_df = pd.read_csv(input_filepath)
    #get file basename
    base_name = (os.path.basename(input_filepath)[:-4])
    for index, row in temp_df.iterrows():
        # set latitude and longitude from row
        lat = row['lat']
        lon = row['lon']
        # create a bounding box from coordinate
        bbox = bbox_from_point(point = (lat, lon), dist = distance)
        coord_names = ['SE_lat', 'SE_lon', 'NW_lat', 'NW_lon']
        for c_name, coord in zip(coord_names, bbox):
            temp_df.at[index, f'{c_name}'] = coord
    if save_file:
        temp_df.to_csv(os.path.join(output_dir, f'{base_name}_background.csv'),
                       index=False)
    return temp_df


def download_background_image(input_df:pd.DataFrame, output_dir: str,
                              errorlog: str, distance: float,
                              fname_col: str, id: str):
    """"
    Takes as input a df and returns raw images collected from surrounding areas
    Arguments:
        input_df: df with columns lon, lat, SW_lon, SW_lat, NE_lon, NE_lat
        output_dir: path to output directory
        distance: radial distance from input lon, lat as center
        fname: column used as filename for the output image record
        id: column that contains point id
    Returns:
        raw set of RGB-N images.
    """
    logf = open(errorlog, "w")
    for index, row in tqdm(input_df.iterrows()):
        #lat, lon = row['lat'], row['lon'];
        SE_lat, SE_lon = row['SE_lat'], row['SE_lon'];
        NW_lat, NW_lon = row['NW_lat'], row['NW_lon'];
        #bbox = bbox_from_point((lat, lon))
        bbox_SE = bbox_from_point(point = (SE_lat, SE_lon), dist=distance)
        bbox_SE = convert_bbox_latlon_lonlat(bbox_SE)
        bbox_NW = bbox_from_point(point = (NW_lat, NW_lon), dist=distance)
        bbox_NW = convert_bbox_latlon_lonlat(bbox_NW)

        for suffix, bb in zip(['_SE', '_NW'], [bbox_SE, bbox_NW]):
            fname = f"{output_dir}/{row[fname_col]}_id_{row[id]}_{index}"
            if os.path.exists(f'{fname+suffix}'): # or point['rand_point_id'] in error_points:
                continue
            try:
                download_NAIP_toLocal(bb, fname+suffix)
                os.remove(f'{fname+suffix}.zip')
            except Exception as e:
                logf.write(f"point id {row[id]}: {e}\n")
            pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='path to csv input file',
                        type=str)
    parser.add_argument('-ad', '--adjacent_distance',
                        help='radial distance from the input lon, lat used as '
                             'new centers to download background images',
                        default=3000,
                        type=float)
    parser.add_argument('-e', '--errorlog',
                        help='path to error log file',
                        default='images/error.log',
                        type=str)
    parser.add_argument('-o', '--output_dir',
                        help='path to output directory',
                        default='images/RGB_background',
                        type=str)
    parser.add_argument('-d', '--distance',
                        help='radial distance from the new adjacent centers',
                        default=1350,
                        type=float)
    parser.add_argument('-fn', '--fname_col',
                        help='name of column to use in the filename',
                        default='state',
                        type=str)
    parser.add_argument('-id', '--id_col',
                        help='name of the column that contains the point id',
                        default='id',
                        type=str)

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        pass

    adj_df = add_adj_coords(args.input, args.output_dir,
                            args.adjacent_distance)
    download_background_image(adj_df, args.output_dir, args.errorlog,
                              args.distance, args.fname_col, args.id_col)
