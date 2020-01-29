import argparse
import h5py
import os
import pandas as pd
import torch

from joblib import delayed, Parallel


def get_arguments():
    """
    Parse all the arguments from Command Line Interface.
    Return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description='make csv files')
    parser.add_argument(
        'csv_file', type=str, help='path to the csv file')
    parser.add_argument(
        'dataset_dir', type=str, help='path to the dataset directory')
    parser.add_argument(
        'feat_dir', type=str, help='path to the dataset directory')
    parser.add_argument(
        '--save_path', type=str, default='./csv', help='path where you want to save csv files')
    parser.add_argument(
        '--save_name', type=str, default='incorrect.csv', help='csv name ')
    parser.add_argument(
        '--n_jobs', type=int, default=-1, help='the number of cores to load data')

    return parser.parse_args()


def check_length(video_id, dataset_dir, feat_dir):
    # idx is for sorting list in the same order as path
    video_path = os.path.join(dataset_dir, video_id + '.hdf5')

    with h5py.File(video_path, 'r') as f:
        video_data = f['video']
        n_frames = len(video_data)

    feat_path = os.path.join(feat_dir, video_id + '.pth')
    feat = torch.load(feat_path)
    _, T = feat.shape

    return video_id, n_frames, T


def main():
    args = get_arguments()

    df = pd.read_csv(args.csv_file)

    video_feat_length = Parallel(n_jobs=args.n_jobs)([
        delayed(check_length)(
            df.iloc[i]['video'][:-5], args.dataset_dir, args.feat_dir)
        for i in range(len(df))
    ])

    paths = []
    n_frames = []
    feat_length = []
    for video_id, n, T in video_feat_length:
        if n == T:
            continue
        else:
            paths.append(video_id + ['.hdf5'])
            n_frames.append(n)
            feat_length.append(T)

    df = pd.DataFrame({
        "video": paths,
        "n_frames": n_frames,
        "feat_length": feat_length
    })

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    df.to_csv(
        os.path.join(args.save_path, args.save_name),
        index=None
    )

    print('Done!')


if __name__ == '__main__':
    main()
