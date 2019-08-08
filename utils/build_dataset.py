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
    parser.add_argument(
        '--th', type=int, default=128,
        help='threshold value which determines whether videos will be removed or not.')
    parser.add_argument(
        '--n_jobs', type=int, default=-1, help='the number of cores which load files')

    return parser.parse_args()


def count_n_frames(df, i, dataset_dir):
    """ count the number of frames for each video """

    video_dir = os.path.join(dataset_dir, df.iloc[i]['video'])
    if os.path.exists(video_dir):
        n_frames = len(glob.glob(os.path.join(video_dir, '*.jpg')))
    elif os.path.exists(video_dir + '.hdf5'):
        with h5py.File(video_dir + '.hdf5', 'r') as f:
            video = f['video']
            n_frames = len(video)
    else:
        n_frames = 0

    if i % 10000 == 0:
        print(i, flush=True)

    return n_frames, i


def main():
    args = get_arguments()

    df = pd.read_csv(args.orig_train_csv)

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

    print('Adding information about the number of frames...')

    # count the number of frames for each video
    processed_data = Parallel(n_jobs=args.n_jobs)(
        [delayed(count_n_frames)(df, i, args.dataset_dir) for i in range(len(df))])

    # sort the list with original index
    processed_data.sort(key=lambda x: x[1])
    processed_data = [p[0] for p in processed_data]

    # adding the information about the number of frames that each video has
    df['n_frames'] = processed_data

    # remove videos which have fewer frames
    df = df[df['n_frames'] >= args.th]

    df.to_csv(
        os.path.join(
            args.save_path,
            'kinetics_{}_{}.csv'.format(args.n_classes, args.split)
        ), index=None)

    print('Done!')


if __name__ == '__main__':
    main()
