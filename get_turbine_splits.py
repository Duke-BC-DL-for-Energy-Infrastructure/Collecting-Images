import os
import pandas as pd

from pathlib import Path
from argparse import ArgumentParser

from wind_turbine_utils import REGION_NAMES
from clustering import apply_clustering, stratified_split, CLUSTERING

DIR_PATH = Path('processed_data/wt')
FILENAME = 'eGrid_wt_coords.csv'

input_file = os.path.join(DIR_PATH, FILENAME)

EPSILON = [0.4, 0.4, 0.4, 0.4]
MIN_SAMPLES = [20, 20, 50, 20]

clustered_turbines = []

for region, epsilon, min_samples in zip(REGION_NAMES, EPSILON, MIN_SAMPLES):

    CLUSTERING['DBSCAN']['params']['eps'] = epsilon
    CLUSTERING['DBSCAN']['params']['min_samples'] = min_samples

    clustered_region = apply_clustering(input_file, DIR_PATH, region)
    clustered_turbines.append(clustered_region)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-e', '--errorlog',
                        help='path to error log file',
                        default='processed_data/wt/error.log',
                        type=str)
    parser.add_argument('-o', '--output_dir',
                        help='path to output directory',
                        default=Path('processed_data/wt'),
                        type=Path)
    parser.add_argument('-st', '--strata_column',
                        help='name of column used to stratify the split in '
                             'train and test subsets',
                        default='cluster',
                        type=str)
    parser.add_argument('-tns', '--train_size',
                        help='size of training subset',
                        default=100,
                        type=int)
    parser.add_argument('-tts', '--test_size',
                        help='size of testing subset',
                        default=100,
                        type=int)

    args = parser.parse_args()

    df = pd.concat(clustered_turbines)

    train_subset = []; test_subset = []

    logf = open(args.errorlog, "w")

    for region in REGION_NAMES:

        tmp_df = df.loc[df.region == region]
        tmp_train, tmp_test = stratified_split(tmp_df,
                                               strata_column=args.strata_column,
                                               train_size=args.train_size,
                                               test_size=args.test_size)
        train_subset.append(tmp_train)
        test_subset.append(tmp_test)

        logf.write(f"{region} subset processed: \n {tmp_train.shape} train \n"
                   f"{tmp_test.shape} test \n")
        logf.flush()

    train_df = pd.concat(train_subset)
    test_df = pd.concat(test_subset)

    train_df.to_csv(os.path.join(args.output_dir, 'train_sample.csv'),
                    index=False)
    test_df.to_csv(os.path.join(args.output_dir, 'test_sample.csv'),
                   index=False)
