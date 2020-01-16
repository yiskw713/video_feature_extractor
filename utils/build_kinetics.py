import argparse
import h5py
import glob
import os
import pandas as pd

from joblib import delayed, Parallel

from id_label_map import get_label2id_map


def get_arguments():
    """
    Parse all the arguments from Command Line Interface.
    Return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description='make csv files')
    parser.add_argument(
        'dataset', type=str, help='dataset name')
    parser.add_argument(
        'dataset_dir', type=str, help='path to the dataset directory')
    parser.add_argument(
        'orig_csv', type=str, help='path to the original kinetics dataset csv')
    parser.add_argument(
        '--file_format', type=str, default='hdf5', help='the video file format. [\'hdf5\', \'jpg\' or \'png\']')
    parser.add_argument(
        '--n_jobs', type=int, default=-1, help='the number of cores to load data')
    parser.add_argument(
        '--save_path', type=str, default='./csv', help='path where you want to save csv files')

    return parser.parse_args()


def check_n_frames(idx, video_path, dataset_dir, file_format):
    # idx is for sorting list in the same order as path
    if file_format == 'hdf5':
        path = os.path.join(dataset_dir, video_path)

        with h5py.File(path, 'r') as f:
            video_data = f['video']
            n_frames = len(video_data)
    else:
        imgs = glob.glob(os.path.join(
            dataset_dir, video_path, '*.{}'.format(file_format)))
        n_frames = len(imgs)

    return idx, n_frames


def main():
    args = get_arguments()

    df = pd.read_csv(args.orig_csv)

    label2id_map = get_label2id_map(args.dataset)

    paths = []
    cls_ids = []
    exists = []

    for i in range(len(df)):
        paths.append(
            df.iloc[i]['label'] + '/' + df.iloc[i]['youtube_id'] + '_'
            + str(df.iloc[i]['time_start']).zfill(6) + '_'
            + str(df.iloc[i]['time_end']).zfill(6)
        )
        cls_ids.append(label2id_map[df.iloc[i]['label']])

        video_dir = os.path.join(args.dataset_dir, paths[i])

        if args.file_format == 'hdf5':
            if os.path.exists(video_dir + '.hdf5'):
                paths[i] = paths[i] + '.hdf5'
                exists.append(1)
            else:
                exists.append(0)
        else:
            if os.path.exists(video_dir):
                exists.append(1)
            else:
                exists.append(0)

    df['cls_id'] = cls_ids
    df['video'] = paths
    df['exists'] = exists
    df = df[df['exists'] == 1]

    # delete useless columns
    del df['youtube_id']
    del df['time_start']
    del df['time_end']
    del df['split']
    del df['exists']
    if 'is_cc' in df.columns:
        del df['is_cc']

    n_frames = Parallel(n_jobs=args.n_jobs)([
        delayed(check_n_frames)(
            i, df.iloc[i]['video'], args.dataset_dir, args.file_format)
        for i in range(len(df))
    ])

    n_frames.sort(key=lambda x: x[0])
    n_frames = [x[1] for x in n_frames]

    df['n_frames'] = n_frames

    # remove videos where the number of frames is smaller than 16
    df = df[df['n_frames'] >= 16]

    if 'train' in args.orig_csv:
        split = 'train'
    elif 'val' in args.orig_csv:
        split = 'val'
    else:
        split = 'test'

    df.to_csv(
        os.path.join(
            args.save_path,
            '{}_{}.csv'.format(args.dataset, split)
        ), index=None)

    print('Done!')


if __name__ == '__main__':
    main()
