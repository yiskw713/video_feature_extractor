import argparse
import h5py
import os
import pandas as pd
import tqdm


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='set csv files and parameters')
    parser.add_argument(
        'dataset_dir', type=str, help='path to dataset directory')
    parser.add_argument('train_csv', type=str, help='path to train.csv')
    parser.add_argument('val_csv', type=str, help='path to val.csv')
    parser.add_argument(
        '--remove',
        action='store_true',
        help='Add --remove option if you want to remove videos which have fewer frames.')
    parser.add_argument(
        '--th', type=int, default=128,
        help='threshold value which determines whether videos will be removed or not.')

    return parser.parse_args()


def main():
    args = get_arguments()

    df_train = pd.read_csv(args.train_csv)
    df_val = pd.read_csv(args.val_csv)

    for df in [df_train, df_val]:
        # add a new column for the number of frames
        df['n_frames'] = 0
        for i in tqdm.tqdm(range(len(df))):
            video_path = os.path.join(
                args.dataset_dir, df.iloc[i]['video'])

            with h5py.File(video_path + '.hdf5', 'r') as f:
                video = f['video']
                n_frames = len(video)
                df.iloc[i]['n_frames'] = n_frames

        if args.remove:
            df = df[df['n_frames'] >= args.th]

    df_train.to_csv(args.train_csv, index=None)
    df_val.to_csv(args.val_csv, index=None)

    print('done!')
