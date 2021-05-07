# Collecting-Images
<h1 align="center">Collecting images from GEE.</h1>

<p align="center">
  <a href="#about">About</a> •
  <a href="#modules">Modules</a> •
  <a href="#image_formatting">Image Formatting</a> •
  <a href="#installation">Installation</a>
</p>

## About

In this module, we implemented a series of functions to preprocess a geo-referenced dataset collected from the [United States Wind Turbine Database (USWTDB)](https://atlas.eia.gov/datasets/united-states-wind-turbine-database-uswtdb?geometry=-124.501%2C20.926%2C97.511%2C63.532).We collected both, images with a wind turbine infrastructure on it, and images without any wind turbines infrastructure, so that these can be used in the synthetic imagery generation process.

This dataset, supplements the [Object Detection Dataset for Overhead Images of Wind Turbines](https://figshare.com/projects/Object_Detection_Dataset_for_Overhead_Images_of_Wind_Turbines/86861) that was created by the [DataPlus 2020 team](https://dataplus-2020.github.io/).

## Modules

### Clustering
This module provides a series of functions that cluster and splits in training and testing the input dataset in 4 geographic regions used in the cross-domain experiments.

- North East (NE):  Pennsylvania, New York, New Jersey, Delaware, Maryland, Connecticut, Massachusetts, Vermont, Maine, New Hampshire
- East Midwestern (EM): Minnesota, Iowa, Missouri, Michigan, Wisconsin, Illinois, Indiana, Ohio
- North West (NW): Washington, Idaho, Oregon, Montana, Wyoming
- South West (SW): New Mexico, Texas, California, Arizona, Utah, Nevada, Colorado

clustering.py includes following functions:

- apply_clustering: applies DBScan clustering per region on the eGrid data. Other clustering methods can be added.
- plot_cluster: return a png file with the output of the clustering
- stratified_split: stratified random sampling from each cluster to construct our training and testing datasets   

### Gee Download
In order to connect to the Google Earth Engine (GEE) platform, the user needs to [sign up](https://signup.earthengine.google.com/#!/) with a **gmail** account. In order to store the data in the Google Cloud Platform (GCP) the user can use the same **gmail** account to create a project and use it.
Detailed instructions are provided in the following [quick guide](https://cloud.google.com/storage/docs/quickstart-console)  

gee_download.py connects to the GEE API, validates account access and provides following functions:

- bbox_from_point: generates a bounding box (bbox) using a radial distance parameter from an input coordinate located at the center of the bbox.
- get_NAIP_Task: uses the bounding box as input to download NAIP imagery from GEE platform into a bucket in Google's cloud storage service.
- downloadGStorage: connect to the Google storage site and downloads its content to a local folder. **Be aware that data downloaded from GEE is in tif format, which includes metadata about the features of the image**

### Background Imagery
Using input coordinates of latitude and longitude, this module generates a couple of new coordinates located at a distance in meters in the north east and south west directions. Then, we use those new coordinates as centers of the bounding boxes used to collect background imagery. We have to visually inspect that the images collected do not have ground truth wind turbines, so that these can be used to generate the new synthetic imagery dataset.  

get_background.py provides following functions:

- add_adj_coords: adds surrounding coordinates at a radial distance from a ground truth wind turbine infrastructure   
- get_background_img: uses the new surrounding coordinates as input to download NAIP imagery from the Google Earth Engine platform.

### Utils
We also have included a series of supportive functions that helped with initial region allocation, labelling formatting and image resizing and formatting.

- split_by_region: add to the USWTDB a column with the corresponding region [NE, EM, NW, SW]
- convert_to_jpg: convert images in tif format to jpg.
- check_img_size: gets all image from a directory and returns max_width, min_width, max_height, min_height across all images
- resize_image: crops an input image to a desired square output resolution, and stores the resulting image in a new output folder.   


## Image_Formatting  
As we used Yolov3 model for object detection, the training dataset required that input RGB images have a 608x608 resolution. NAIP image is approximately 1 m/pixel, we set a radial distance of 650 meters, so the output image could be cropped to 608x608.  

For the synthetic image generation, the input image resolution was 1300x1300. In this case we used around 1350 meters when setting the radial distance from the surrounding coordinates location.

As we mentioned before, images are downloaded in tif format, we used the convert_to_jpg function to change format. Then, we cropped the image the desired size.

## Installation

**Clone this repository and enter the Collecting-Images directory**

```bash
$git clone https://github.com/Duke-BC-DL-for-Energy-Infrastructure/Collecting-Images
$cd Collecting-Images
```
