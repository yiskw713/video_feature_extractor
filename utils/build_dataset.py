import argparse
import glob
import h5py
import os
import pandas as pd

from joblib import Parallel, delayed

from class_label_map import get_class_label_map


def get_arguments():
    """
    Parse all the arguments from Command Line Interface.
    Return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description='make csv files for training and validation')
    parser.add_argument(
        'dataset_dir', type=str, help='path of the dataset directory')
    parser.add_argument(
        'n_classes', type=int, help='the number of classes in kinetics dataset')
    parser.add_argument(
        'orig_csv', type=str, help='path to the original kinetics dataset csv')
    parser.add_argument(
        'split', type=str, help='train | val | test ')
    parser.add_argument(
        '--save_path', type=str, default='./dataset', help='path where you want to save csv files')

    return parser.parse_args()


def main():
    args = get_arguments()

    df = pd.read_csv(args.orig_csv)

    class_label_map = get_class_label_map(n_classes=args.n_classes)

    path = []
    cls_id = []

    for i in range(len(df)):
        path.append(
            df.iloc[i]['label'] + '/' + df.iloc[i]['youtube_id'] + '_'
            + str(df.iloc[i]['time_start']).zfill(6) + '_'
            + str(df.iloc[i]['time_end']).zfill(6)
        )
        cls_id.append(class_label_map[df.iloc[i]['label']])

    df['class_id'] = cls_id
    df['video'] = path

    # delete useless columns
    del df['youtube_id']
    del df['time_start']
    del df['time_end']
    del df['split']
    if 'is_cc' in df.columns:
        del df['is_cc']

    exists = []

    for i in range(len(df)):
        video_dir = os.path.join(args.dataset_dir, df.iloc[i]['video'])
        if os.path.exists(video_dir):
            exists.append(1)
        elif os.path.exists(video_dir + '.hdf5'):
            exists.append(1)
        else:
            exists.append(0)

    df['exists'] = exists
    df = df[df['exists'] == 1]

    df.to_csv(
        os.path.join(
            args.save_path,
            'kinetics_{}_{}.csv'.format(args.n_classes, args.split)
        ), index=None)

    print('Done!')


if __name__ == '__main__':
    main()
