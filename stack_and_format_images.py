import os
import pickle
from argparse import ArgumentParser
from glob import glob

import numpy as np
import rasterio

from PIL import Image
from skimage.io import imread, imsave
from tqdm import tqdm

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def stack_RGB_bands(directory):
    img = np.stack([imread('{}/download.R.tif'.format(directory)),
                    imread('{}/download.G.tif'.format(directory)),
                    imread('{}/download.B.tif'.format(directory))], -1)
    return img


def read_tiff_meta(geotiff_path):
    """Read geotiff, return reshaped image and metadata."""
    with rasterio.open(geotiff_path, 'r') as src:
        img_meta = src.meta
        pixel_size_x, pixel_size_y  = src.res
    return img_meta, pixel_size_x, pixel_size_y


def crop_image(img, new_squared_size: int):
    """crop the center of the image a defined squared size"""

    # get current dimensions
    width, height = img.size
    # crop image
    new_width = new_squared_size
    new_height = new_width
    # https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil.
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    img = img.crop((left, top, right, bottom))
    return img


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--in_dir',
                        help='path to data directory',
                        type=str)
    parser.add_argument('-o', '--out_dir',
                        help='path to processed data directory',
                        default='images/JPG',
                        type=str)
    parser.add_argument('-m', '--meta_file',
                        help='path to file containing meta information about the data',
                        default='images/meta.pkl',
                        type=str)
    parser.add_argument('-s', '--image_size',
                        help='squared image size in pixels',
                        default=1300,
                        type=int)

    args = parser.parse_args()

    meta_dict = {}

    img_dirs = glob("{}/*".format(args.in_dir))

    if not os.path.exists(args.out_dir):
        print('making output directory')
        os.makedirs(args.out_dir)
        pass

    # expected data directory structure is a directory for each image (as is the default output from the download pipeline)
    for directory in tqdm(img_dirs, desc="processing images"):
        img = stack_RGB_bands(directory) # dont stack infrared band
        img = Image.fromarray(img)
        img = crop_image(img, args.image_size)

        # the name of the directory has land cover type and point ID, we use it to name the file
        filename = os.path.split(directory)[-1]  # the split char could be different if

        img.save("{}/{}.jpg".format(args.out_dir, filename))

        # if infrared band exists save separately
        try:
            if not os.path.exists(args.out_dir+'/infrared'):
                print('making output directory')
                os.makedirs(args.out_dir+'/infrared')
                pass
            imsave("{}/{}_N.jpg".format(args.out_dir+'/infrared', filename), imread("{}/download.N.tif".format(directory)))
        except:
            pass

        # add meta information to the dictionary
        meta_dict[filename] = read_tiff_meta("{}/download.R.tif".format(directory))
        pass

    with open(args.meta_file,"wb") as f:
        pickle.dump(meta_dict,f)
        pass
    pass
