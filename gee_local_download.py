#code inspired by the

import ee
ee.Authenticate()
ee.Initialize()

import numpy as np
import pandas as pd

import os

from gee_cloud_download import bbox_from_point
from geetools import batch
from tqdm import tqdm

from argparse import ArgumentParser

def convert_bbox_latlon_lonlat(bbox):
    # convert bbox from lat lon to lon lat
    return bbox[1], bbox[0], bbox[3], bbox[2]


def download_NAIP_toLocal(bbox, name, scale=1):
    AOI = ee.Geometry.Rectangle(list(bbox),
                                'EPSG:4326',
                                False)

    collection = (ee.ImageCollection("USDA/NAIP/DOQQ")
                .filterDate('2010-01-01', '2019-01-01')
                .filterBounds(AOI)
                )

    image = ee.Image(collection.mosaic()).clip(AOI)
    batch.image.toLocal(image, name, scale=scale, region=AOI)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='path to csv input file',
                        type=str)
    parser.add_argument('-d', '--distance',
                        help='side length for area of interest',
                        default=650,
                        type=int)
    parser.add_argument('-e', '--errorlog',
                        help='path to error log file',
                        default='processed_data/wt/error.log',
                        type=str)
    parser.add_argument('-o', '--output_dir',
                        help='path to output directory',
                        default='data/',
                        type=str)
    parser.add_argument('-lat', '--lat_col',
                        help='name of the column that contains the latitude',
                        default='LAT',
                        type=str)
    parser.add_argument('-lon', '--lon_col',
                        help='name of the column that contains the longitude',
                        default='LON',
                        type=str)
    parser.add_argument('-id', '--id_col',
                        help='name of the column that contains the point id',
                        default='ID',
                        type=str)

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        pass

    points = pd.read_csv(args.input)

    logf = open(args.errorlog, "w")

    for state in tqdm(points.state.unique()):
        print(state)
        tmp = points[points.state == state]

        for i, point in tqdm(tmp.iterrows()):
            fname = f"{args.output_dir}/{point.state}_id_{point[args.id_col]}_{i}"
            if os.path.exists(f'{fname}'): # or point['rand_point_id'] in error_points:
                continue
            bbox = bbox_from_point((point[args.lat_col], point[args.lon_col]), args.distance)
            bbox = convert_bbox_latlon_lonlat(bbox)
            try:
                download_NAIP_toLocal(bbox, fname)
                os.remove(f'{fname}.zip')
            except Exception as e:
                logf.write(f"point id {point[args.id_col]}: {e}\n")
                pass
