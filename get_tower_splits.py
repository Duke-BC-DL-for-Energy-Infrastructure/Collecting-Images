import os
import pandas as pd

from pathlib import Path
from argparse import ArgumentParser

from clustering import stratified_split
from electric_tower_utils import df_subset, REGION_NAMES


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='path to csv input file',
                        default=Path('processed_data/et/tower_coordinates.csv'),
                        type=str)
    parser.add_argument('-e', '--errorlog',
                        help='path to error log file',
                        default='processed_data/et/error.log',
                        type=str)
    parser.add_argument('-o', '--output_dir',
                        help='path to output directory',
                        default=Path('processed_data/et'),
                        type=Path)
    parser.add_argument('-s', '--subset_col',
                        help='name of column that is used to subset dataset',
                        default='voltage',
                        type=str)
    parser.add_argument('-min', '--min_value',
                        help='min value in column used to subset of dataset',
                        default=500,
                        type=float)
    parser.add_argument('-max', '--max_value',
                        help='max value in column used to subset of dataset',
                        default=float('Inf'),
                        type=float)
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

    df = pd.read_csv(args.input)

    df_subset = df_subset(df, args.subset_col, args.min_value, args.max_value)

    train_subset = []; test_subset = []

    logf = open(args.errorlog, "w")

    for region in REGION_NAMES:

        tmp_df = df_subset.loc[df_subset.region == region]
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
