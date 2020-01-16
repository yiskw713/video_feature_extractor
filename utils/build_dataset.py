import argparse
import h5py
import glob
import os
import pandas as pd

from joblib import delayed, Parallel


def get_arguments():
    """
    Parse all the arguments from Command Line Interface.
    Return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description='make csv files')
    parser.add_argument(
        'dataset', type=str, help='dataset name. [msrvtt, activitynet]')
    parser.add_argument(
        'dataset_dir', type=str, help='path to the dataset directory')
    parser.add_argument(
        '--split', type=str, default=None,
        help='if a dataset is divided into some splits, specify the split for which you want make csv files'
    )
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

    if args.file_format == 'hdf5':
        paths = glob.glob(os.path.join(args.dataset_dir, '*.hdf5'))
    else:
        paths = glob.glob(os.path.join(args.dataset_dir, '*'))

    n_frames = Parallel(n_jobs=args.n_jobs)([
        delayed(check_n_frames)(
            i, paths[i], args.dataset_dir, args.file_format)
        for i in range(len(paths))
    ])

    n_frames.sort(key=lambda x: x[0])
    n_frames = [x[1] for x in n_frames]

    df = pd.DataFrame({
        "video": paths,
        "n_frames": n_frames
    })

    # remove videos where the number of frames is smaller than 16
    df = df[df['n_frames'] >= 16]

    if args.split is not None:
        df.to_csv(
            os.path.join(
                args.save_path,
                '{}_{}.csv'.format(args.dataset, args.split)
            ), index=None)
    else:
        df.to_csv(
            os.path.join(
                args.save_path,
                '{}.csv'.format(args.dataset)
            ), index=None)

    print('Done!')


if __name__ == '__main__':
    main()
