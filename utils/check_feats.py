import argparse
import os
import pandas as pd


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
        'feat_dir', type=str, help='path to the dataset directory')
    parser.add_argument(
        '--save_path', type=str, default='./csv', help='path where you want to save csv files')
    parser.add_argument(
        '--save_name', type=str, default='rest.csv', help='csv name ')

    return parser.parse_args()


def main():
    args = get_arguments()

    df = pd.read_csv(args.csv_file)
    exists = []
    for i in range(len(df)):
        video_id = df.iloc[i]['video']
        video_id = video_id[:-5]
        feat_name = video_id + '.pth'
        feat_path = os.path.join(args.feat_dir, feat_name)

        if os.path.exists(feat_path):
            exists.append(1)
        else:
            exists.append(0)

    df['exists'] = exists

    # keep id of video whose features are not extracted.
    df = df[df['exists'] == 0]

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    df.to_csv(
        os.path.join(args.save_path, args.save_name),
        index=None
    )

    print('Done!')


if __name__ == '__main__':
    main()
