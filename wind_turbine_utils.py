'''Set of functions that help to manipulate data preprocessing'''

#file management
import glob
import os
from pathlib import Path

#data wrangling
import pandas as pd
import numpy as np

#image resizing and formatting
import imagesize
import cv2
from PIL import Image

#track function's progress
import tqdm

from CONSTANTS import NE_ab, EM_ab, NW_ab, SW_ab, REGION_NAMES


def split_wind_turbines_by_region(input_file:Path, output_dir:Path) -> None:
    """
    Takes as input a csv file with wind turbines info downloaded from eGrid
    https://atlas.eia.gov/datasets/united-states-wind-turbine-database-uswtdb:
    Arguments:
        input_filepath: path to input csv
        output_filepath: path to store csv file
    Returns:
        input csv file with added region column that indicates region
        [NE, EM, NW, SW]
    """

    fields = ['t_state','p_name', 'xlong', 'ylat']
    eGrid_wt_df = pd.read_csv(input_file, usecols=fields)

    conditions = [
        eGrid_wt_df['t_state'].isin(NE_ab),
        eGrid_wt_df['t_state'].isin(EM_ab),
        eGrid_wt_df['t_state'].isin(NW_ab),
        eGrid_wt_df['t_state'].isin(SW_ab)
        ]
    outputs = REGION_NAMES
    eGrid_wt_df['region'] = np.select(conditions, outputs, 'Other')
    eGrid_wt_df.rename(columns={'t_state':'state', 'p_name':'name',
                                'xlong':'lon', 'ylat':'lat'}, inplace=True)
    eGrid_wt_df.to_csv(os.path.join(output_dir,
                       f'eGrid_wt_coords.csv'), index=False)
    return


def convert_to_jpg(input_filepath:Path):
    """
    Takes as input a path to image file (*.tif) and converts into jpg format:
    Arguments:
        filename: path to image file
    Returns:
        image in jpg format.
    """
    read = cv2.imread(input_filepath)
    outfile = input_filepath.split('.')[0] + '.jpg'
    cv2.imwrite(outfile,read,[int(cv2.IMWRITE_JPEG_QUALITY), 200])
    return


def check_img_size(dir_path:str, max_w: int=10000, max_h: int=10000):
    """
    Takes as input a directory path to image files (*.jpg or *.png) and returns
    max and mins of height and width in pixels:
    Arguments:
        dir_path: path to image files i.e r'/image_folder/*.jpg'
    Returns:
        prints out (max_width, min_width, max_height, min_height)
    """

    max_width = 0
    min_width = max_w
    max_height = 0
    min_height = max_h
    for image_file in tqdm.tqdm(glob.glob(dir_path)):
        shape = imagesize.get(image_file)
        if (shape[0] <= min_width):
            min_width = shape[0]
        elif (shape[0] >= max_width):
            max_width = shape[0]
        elif (shape[1] <= min_height):
            min_height = shape[1]
        elif (shape[1] >= max_height):
            max_height = shape[1]
    print(max_width, min_width, max_height, min_height)
    return


def resize_image(input_filepath:Path, output_filepath:Path, new_size:int):
    """
    Takes as input a path to image file (*.jpg) and crops it to square a
    resolution:
    Arguments:
        filename: path to image file
    Returns:
        crops an image to a new size square resolution.
    """
    new_width = new_size
    new_height = new_width
    #load image
    im = Image.open(input_filepath)
    #get dimensions
    width, height = im.size
    # crop the center of the image https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    im = im.crop((left, top, right, bottom))
    head, tail = os.path.split(input_filepath)
    im.save(os.path.join(output_filepath, tail))
    return
