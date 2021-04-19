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
FILENAME = 'wt_coords.csv'
INPUT_FILEPATH = Path('data/'+FILENAME)
OUTPUT_FILEPATH = Path('data/')

# perform clustering
REGIONS = ['NE', 'NW', 'EM'] # Northeast, Northwest, Eastern Midwest
CLUSTERING_METHOD = 'DBSCAN'
DBS_EPS = 0.4
DBS_MIN_SAMPLES = 20

def apply_clustering(file_path:Path = INPUT_FILEPATH, region: str=None,
                     clustering_method: str = CLUSTERING_METHOD,
                     eps:float = DBS_EPS,
                     min_samples:int = DBS_MIN_SAMPLES) -> pd.DataFrame:
                     """ inputs a csv file with columns:
                     name: image filename,
                     lon, lat: image coordinates
                     region: geographical area must be one of [NE, NW or EM]

                     returns: a csv with same input data and additional column
                        named cluster with corresponding cluster.
                     """
                     #download csv file
                     wt_df = pd.read_csv(INPUT_FILEPATH)

                     if region in ['NE', 'NW', 'EM']:
                         print(f'clustering {region} region')
                         wt_df = wt_df.loc[wt_df.region == region]

                     elif region is None:
                         print(f'clustering across all regions')

                     else:
                         print(f'{region} is an invalid region [NE, NW, EM]')
                         return

                     wt_df = wt_df[["lon", "lat"]]
                     # scaling coordinates before fitting the cluster method
                     wt_scaled = StandardScaler().fit_transform(wt_df)

                     # fit clustering with eps and min_samples
                     if CLUSTERING_METHOD.lower() == 'dbscan':
                         print(f'Applying {CLUSTERING_METHOD}')
                         dbscan = DBSCAN(eps=DBS_EPS,
                                         min_samples=DBS_MIN_SAMPLES).fit(wt_scaled)
                         labels = dbscan.labels_
                         wt_df["cluster"]=labels
                     else:
                         print(f'{CLUSTERING_METHOD} is an invalid clustering'
                                'method')
                         return

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


def plot_clusters(region: str='NE'):
    FILENAME = f'{region}_clusters.csv'
    INPUT_FILEPATH = Path('data/'+FILENAME)
    wt_df = pd.read_csv(INPUT_FILEPATH)
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
              f'region: \n $ \epsilon = {DBS_EPS}$ and min_samples '
              f'= {DBS_MIN_SAMPLES}', fontsize=14)
    plt.savefig(os.path.join(OUTPUT_FILEPATH,f'{region}_cluster.png'), dpi=300)


def stratified_split(data: pd.DataFrame, n_splits: int = 5,
                     train_size: int = 100, test_size: int  = 100,
                     random_state: int = 42):

                     #drop noise in clusters
                     data = data[data.cluster != -1]

                     sss = StratifiedShuffleSplit(n_splits = n_splits,
                                                  train_size = train_size,
                                                  test_size = test_size,
                                                  random_state = random_state)

                     for train_idx, test_idx in sss.split(data, data.cluster):
                         train_set = data.iloc[train_idx]
                         test_set = data.iloc[test_idx]

                     

                     print('input data cluster distribution:')
                     print(data.cluster.value_counts(normalize=True))

                     print('training data cluster distribution:')
                     print(train_set.cluster.value_counts(normalize=True))

                     print('test data cluster distribution:')
                     print(test_set.cluster.value_counts(normalize=True))

                     return train_set, test_set

FILENAME = f'NE_clusters.csv'
INPUT_FILEPATH = Path('data/'+FILENAME)
wt_df = pd.read_csv(INPUT_FILEPATH)
stratified_split(wt_df)
