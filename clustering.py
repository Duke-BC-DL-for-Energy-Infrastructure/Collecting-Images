# file management
import os
from pathlib import Path

# data wrangling
import pandas as pd
import numpy as np

# Create functions for each clustering methods
import sklearn.utils
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# mapbase functions
from mpl_toolkits.basemap import Basemap
from pylab import rcParams
#%matplotlib inline
rcParams['figure.figsize'] = (14,10)

# plotting package
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import math

from typing import Tuple


REGIONS = ['NE', 'NW', 'EM', 'SW'] # Northeast, Northwest, Eastern Midwest

CLUSTERING = {
    'DBSCAN': {
    'function': DBSCAN,
    'params': {'eps':0.4, 'min_samples':20}
    },
    'SC': {
    'function':SpectralClustering,
    'params': {'n_clusters': 4, 'affinity':'nearest_neighbors'}
    }
}


def apply_clustering(input_filepath:Path,
                     output_filepath:Path,
                     region: str='NE', method: str='DBSCAN') -> pd.DataFrame:
    """
    Takes as input a csv file with columns name, lon, lat and region:
    Arguments:
     input_filepath: path to input csv
     output_filepath: directory path to store clustered data
     region: geographical area must be one of [NE, NW, EM, SW] or None to apply
        clustering methods across all dataset points
     clustering_method: unsupervised ML method.
    Returns:
     csv file with clustered data, identified in the 'cluster' added column
    """
    #download csv file
    df = pd.read_csv(input_filepath)

    if region in ['NE', 'NW', 'EM', 'SW']:
        print(f'clustering {region} region')
        df = df.loc[df.region == region]

    elif region is None:
        print(f'clustering across all regions')

    else:
        print(f'{region} is an invalid region [NE, NW, EM, SW]')
        return

    df_lon_lat = df[["lon", "lat"]]
    # scaling coordinates before fitting the cluster method
    df_scaled = StandardScaler().fit_transform(df_lon_lat)

    print(f'Applying {CLUSTERING[method]}')

    cluster_function = CLUSTERING[method]['function']
    params = CLUSTERING[method]['params']

    cluster = cluster_function(**params).fit(df_scaled)
    labels = cluster.labels_
    df["cluster"]=labels

    # check # of clusters excluding noise (-1), if cluster method produce noise
    num_labels = set(labels)
    realClusterNum=len(num_labels) - (1 if -1 in labels else 0)
    clusterNum = len(num_labels)
    print(f'There are {realClusterNum} clusters, excluding noise')

    for clust_number in num_labels:
        clust_set = df[df.cluster == clust_number]
        print(f'cluster {clust_number} has {len(clust_set)} records')
        df.to_csv(os.path.join(output_filepath, f'{region}_clusters.csv'),
                  index=False)
    return df


def plot_clusters(input_filepath:Path,
                  output_filepath:Path, region: str='NE'):
    """
    Takes as input a csv file produced by the apply_clustering function:
    Arguments:
       output_filepath: path to store png file
       region: geographical area must be one of [NE, NW or EM]
    Returns:
       png file with plot of data clustered
    """

    FILENAME = f'{region}_clusters.csv'
    filepath = input_filepath/FILENAME
    df = pd.read_csv(filepath)
    # set area to plot
    print(df.describe().loc[['min', 'max']])

    stats = df.describe()
    llon = math.floor(stats.at['min', 'lon']-1)
    ulon = math.ceil(stats.at['max', 'lon']+1)
    llat = math.floor(stats.at['min', 'lat']-0.5)
    ulat = math.ceil(stats.at['max', 'lat']+0.5)

    map = Basemap(projection='merc',
                  resolution = 'l', area_thresh = 1000.0,
                  llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
                  urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

    #map.drawcoastlines()
    map.drawcountries()
    map.drawstates(color='black')
    map.drawlsmask(land_color='orange', ocean_color='skyblue')
    map.shadedrelief()
    map.bluemarble()

    # To create a color map
    labels = set(df.cluster)
    clusterNum = len(labels)
    colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))

    # collect data based on wind turbines
    xs,ys = map(np.asarray(df.lon), np.asarray(df.lat))
    df['xm']= xs.tolist()
    df['ym'] =ys.tolist()

    #Visualization2
    for clust_number in labels:
        c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
        clust_set = df[df.cluster == clust_number]
        print(f'cluster {clust_number} has {len(clust_set)} records')
        map.scatter(clust_set.xm, clust_set.ym, color=c,  marker='o',
                    s= 40, alpha = 0.65)
        if clust_number != -1:
            cenx=np.mean(clust_set.xm)
            ceny=np.mean(clust_set.ym)
            plt.text(cenx,ceny,str(clust_number), fontsize=30, color='red',
                     fontweight='bold')
            #print ("Cluster "+str(clust_number))
    plt.title(f'Clusters of collected images for wind turbines in the {region} '
              f'region', fontsize=14)
    plt.savefig(os.path.join(output_filepath,f'{region}_cluster.png'), dpi=300)
    return


def stratified_split(df:pd.DataFrame,
                     strata_column: str = 'cluster', n_splits: int = 5,
                     train_size: int = 100, test_size: int = 100,
                     random_state: int = 42) -> Tuple:

    """
    Takes as input a dataframe:
    Arguments:
     df: input dataframe
     strata_column: column used as reference for stratification
     n_splits, train_size, test_size and random_state are arguments of the
     StratifiedShuffleSplit function from scikit-learn.
    Returns:
     train and test dataframes stratified by user defined strata_column
    """

    # check for noise in strata_column
    labels = set(df[strata_column])

    if -1 in labels:
        #drop noise in clusters
        df = df[df[strata_column] != -1]

    sss = StratifiedShuffleSplit(n_splits = n_splits,
                                 train_size = train_size,
                                 test_size = test_size,
                                 random_state = random_state)

    try:
        for train_idx, test_idx in sss.split(df, df[strata_column]):
            train_set = df.iloc[train_idx]
            test_set = df.iloc[test_idx]
        pass

    except ValueError as e:
        print(f'Try stratifying on another column. \'{strata_column}\''
              f' has a class with only one member. {e}')
        return None, None

    print('input data cluster distribution:')
    print(df[strata_column].value_counts(normalize=True))

    print('training data cluster distribution:')
    print(train_set[strata_column].value_counts(normalize=True))

    print('test data cluster distribution:')
    print(test_set[strata_column].value_counts(normalize=True))

    return train_set, test_set
