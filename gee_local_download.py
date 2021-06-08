#code inspired by the
import os
import ee

import numpy as np
import pandas as pd

from gee_utils import bbox_from_point
from geetools import batch
from tqdm import tqdm

from argparse import ArgumentParser

ee.Initialize()

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
                        default='images/error.log',
                        type=str)
    parser.add_argument('-o', '--output_dir',
                        help='path to output directory',
                        default='images/RGB_N',
                        type=str)
    parser.add_argument('-lat', '--lat_col',
                        help='name of the column that contains the latitude',
                        default='lat',
                        type=str)
    parser.add_argument('-lon', '--lon_col',
                        help='name of the column that contains the longitude',
                        default='lon',
                        type=str)
    parser.add_argument('-st', '--strata_column',
                        help='name of column used to stratify the split in '
                             'train and test subsets',
                        default='cluster',
                        type=str)
    parser.add_argument('-fn', '--filename',
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

    points = pd.read_csv(args.input)

    logf = open(args.errorlog, "w")

    for item in tqdm(points[args.strata_column].unique()):
        print(item)
        tmp = points[points[args.strata_column] == item]

        for i, point in tqdm(tmp.iterrows()):
            fname = f"{args.output_dir}/"\
                    f"{point[args.filename]}_id_{point[args.id_col]}_{i}"
            if os.path.exists(f'{fname}'): # or point['rand_point_id'] in error_points:
                continue
            bbox = bbox_from_point((point[args.lat_col], point[args.lon_col]),
                                   args.distance)
            bbox = convert_bbox_latlon_lonlat(bbox)
            try:
                download_NAIP_toLocal(bbox, fname)
                os.remove(f'{fname}.zip')
            except Exception as e:
                logf.write(f"point id {point[args.id_col]}: {e}\n")
                pass
