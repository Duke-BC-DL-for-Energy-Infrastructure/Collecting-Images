"""
These approach to collect satellite imagery has been build upon the work
of previous teams at Duke University MIDS Capstone - Energy Infrastructure
Detection, url: https://github.com/jzisheng/mids_ei 

"""

# Import the Earth Engine API and initialize it
import ee # google earth engine (gee) api
import os # manage os
import time # set timer
import math # geopositional transformation
import folium # bridge between Python and Leaflet
import geehydro # add add_layers functionality to folium
import pandas as pd # dataframe manipulation

from ee import batch # manage batches of images from gee
from PIL import Image # manage image features
from pathlib import Path # manage directories between OS Mac & Windows
from typing import Tuple # add python data types annotation

ee.Initialize()

# function to create a bounding box around a point
def bbox_from_point(point: Tuple, dist: int = 1000):
    """
    Arguments:
        point: coordinate point (lat, long)
        dist: radial distance from the point as center
    Returns a bounding box (south, west, north, east) centered on that point
        with side length dist formula from:
        http://www.movable-type.co.uk/scripts/latlong.html#rhumblines
    """
    earth_radius = 6371000  # meters
    angular_distance = math.degrees(0.5 * (dist / earth_radius))

    lat, lon = point
    delta_lat = angular_distance
    delta_lon = angular_distance/math.cos(math.radians(lat))

    south, north = lat - delta_lat, lat + delta_lat
    west, east = lon - delta_lon, lon + delta_lon
    return south, west, north, east

# function to collect images from NAIP
def get_NAIP_Task(coords: Tuple, gcp_bucket: str, filename: str):
    """
    Takes as input coordinates of the boundary to export to google cloud bucket
    Argument:
        coords: coordinate point (south, west, north, east)
        gcp_bucket: bucket address in gcp. I am using my gcp account.
        filename: name of the output file.
    Returns:
        executable task to download an image from NAIP store it in the defined gcp bucket.
    """
    geom = ee.Geometry.Rectangle([coords[1],coords[0],coords[3],coords[2]])
    collection = ee.ImageCollection("USDA/NAIP/DOQQ") \
                .filter(ee.Filter.date('2017-01-01', '2018-12-31'));
    bands = ['R', 'G', 'B',]
    collection = collection.select(bands)
    trueColorVis = {
      min: 0.0,
      max: 255.0,
    }
    image = collection.sort('system:index', False).mosaic()
    image = image.clip(geom)
    image.projection()
    imageRGB = image.visualize(bands=bands,min=0,max=255)

    task = ee.batch.Export.image.toCloudStorage(image=imageRGB,
                                        region=image.geometry().bounds().\
                                        getInfo()['coordinates'],
                                        description='wind_farms',
                                        outputBucket=gcp_bucket,
                                        fileNamePrefix=filename,
                                        scale=1)
    return task

# function to download the data from gcp to local folder.
def downloadGStorage(buck_name, local_addr):
    os.system('gsutil cp -r '+ buck_name + ' ' + local_addr)
    pass
