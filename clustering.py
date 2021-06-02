#file management
import os
from pathlib import Path

#data wrangling
import pandas as pd
import numpy as np

#Create functions for each plot of the clustering methods
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings('ignore')

#mapbase functions
from mpl_toolkits.basemap import Basemap
from pylab import rcParams

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm

%matplotlib inline
rcParams['figure.figsize'] = (14,10)


from sklearn.cluster import DBSCAN, SpectralClustering
import sklearn.utils
from sklearn.preprocessing import StandardScaler

import math

# setup path to file with wind farm coordinates
FILENAME = 'eGrid_wt_coords.csv'
INPUT_FILEPATH = Path('processed_data/wt')/FILENAME
OUTPUT_FILEPATH = Path('processed_data/wt')

# perform clustering
REGIONS = ['NE', 'NW', 'EM', 'SW'] # Northeast, Northwest, Eastern Midwest

CLUSTERING = {
    'DBSCAN': {
    'function': DBSCAN,
    'params': {'eps':0.4, 'min_samples':20}
    }
}


def apply_clustering(input_filepath:Path = INPUT_FILEPATH,
                     output_filepath:Path = OUTPUT_FILEPATH,
                     region: str='NE', method: str='DBSCAN') -> pd.DataFrame:
                     """
                     Takes as input a csv file with column names name, lon, lat,
                     region:
                     Arguments:
                        input_filepath: path to input csv
                        output_filepath: path to store clustered data
                        region: geographical area must be one of [NE, NW, EM, SW]
                        clustering_method: unsupervised ML method.
                        eps, min_samples: parameters of DBScan clustering method
                     Returns:
                        csv file with same input data and additional column
                        named cluster with corresponding cluster.
                     """
                     #download csv file
                     wt_df = pd.read_csv(INPUT_FILEPATH)

                     if region in ['NE', 'NW', 'EM', 'SW']:
                         print(f'clustering {region} region')
                         wt_df = wt_df.loc[wt_df.region == region]

                     elif region is None:
                         print(f'clustering across all regions')

                     else:
                         print(f'{region} is an invalid region [NE, NW, EM, SW]')
                         return

                     wt_df_lon_lat = wt_df[["lon", "lat"]]
                     # scaling coordinates before fitting the cluster method
                     wt_scaled = StandardScaler().fit_transform(wt_df_lon_lat)

                     # fit clustering with eps and min_samples
                     #if CLUSTERING[cluster_method].lower() == 'dbscan':
                     print(f'Applying {CLUSTERING[method]}')

                     cluster_function = CLUSTERING[method]['function']
                     params = CLUSTERING[method]['params']

                     cluster = cluster_function(**params).fit(wt_scaled)
                     labels = cluster.labels_
                     wt_df["cluster"]=labels
                     num_labels = set(labels)
                     realClusterNum=len(num_labels) - (1 if -1 in labels else 0)
                     clusterNum = len(num_labels)
                     print(f'There are {realClusterNum} clusters, excluding noise')
                     for clust_number in num_labels:
                         clust_set = wt_df[wt_df.cluster == clust_number]
                         print(f'cluster {clust_number} has {len(clust_set)} records')
                     wt_df.to_csv(os.path.join(OUTPUT_FILEPATH,
                                  f'{region}_clusters.csv'), index=False)
                     return wt_df


def plot_clusters(region: str='NE',
                  input_filepath:Path = INPUT_FILEPATH,
                  output_filepath:Path = OUTPUT_FILEPATH):
    """
    Takes as input a csv file produced by the apply_clustering function:
    Arguments:
       output_filepath: path to store png file
       region: geographical area must be one of [NE, NW or EM]
    Returns:
       png file with plot of data clustered
    """

    FILENAME = f'{region}_clusters.csv'
    filepath = INPUT_FILEPATH/FILENAME
    wt_df = pd.read_csv(filepath)
    # set area to plot
    print(wt_df.describe().loc[['min', 'max']])

    stats = wt_df.describe()
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
    labels = set(wt_df.cluster)
    clusterNum = len(labels)
    colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))

    # collect data based on wind turbines
    xs,ys = map(np.asarray(wt_df.lon), np.asarray(wt_df.lat))
    wt_df['xm']= xs.tolist()
    wt_df['ym'] =ys.tolist()

    #Visualization2
    for clust_number in labels:
        c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
        clust_set = wt_df[wt_df.cluster == clust_number]
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
    plt.savefig(os.path.join(OUTPUT_FILEPATH,f'{region}_cluster.png'), dpi=300)
    return


def stratified_split(region: str='NE',
                     input_filepath:Path = INPUT_FILEPATH,
                     output_filepath:Path = OUTPUT_FILEPATH,
                     strata_column: str = 'cluster', n_splits: int = 5,
                     train_size: int = 100, test_size: int = 100,
                     random_state: int = 42) -> None:

                     """
                     Takes as input a csv file produced by the apply_clustering
                     function:
                     Arguments:
                        output_filepath: path to store png file
                        region: geographical area must be one of [NE, NW, EM, SW]
                        n_splits, train_size, test_size and random_state are
                        arguments of the StratifiedShuffleSplit function from
                        sklearn
                        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
                     Returns:
                        train and test csv files stratified by clusters
                     """

                     FILENAME = f'{region}_clusters.csv'
                     filepath = INPUT_FILEPATH/FILENAME)
                     data = pd.read_csv(filepath)

                     #drop noise in clusters
                     data = data[data[strata_column] != -1]

                     sss = StratifiedShuffleSplit(n_splits = n_splits,
                                                  train_size = train_size,
                                                  test_size = test_size,
                                                  random_state = random_state)

                     for train_idx, test_idx in sss.split(data,
                                                          data[strata_column]):
                         train_set = data.iloc[train_idx]
                         test_set = data.iloc[test_idx]

                     print('input data cluster distribution:')
                     print(data.cluster.value_counts(normalize=True))

                     print('training data cluster distribution:')
                     print(train_set.cluster.value_counts(normalize=True))

                     print('test data cluster distribution:')
                     print(test_set.cluster.value_counts(normalize=True))

                     train_set.to_csv(os.path.join(OUTPUT_FILEPATH,
                                  f'{region}_train.csv'), index=False)

                     test_set.to_csv(os.path.join(OUTPUT_FILEPATH,
                                 f'{region}_test.csv'), index=False)
                     return
