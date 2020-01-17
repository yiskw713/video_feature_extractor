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
        'csv', type=str, help='the path to the csv file you want to split')
    parser.add_argument(
        '--n_splits', type=int, default=10, help='the number of splits')
    parser.add_argument(
        '--save_path', type=str, default='./csv', help='path where you want to save csv files')

    return parser.parse_args()


def main():
    args = get_arguments()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    df = pd.read_csv(args.csv)
    basename = os.path.splitext(os.path.basename(args.csv))[0]
    length = len(df) // args.n_splits

    for i in range(args.n_splits):
        if (i + 1) == args.n_splits:
            split_df = df[length * i:]
        else:
            split_df = df[length * i: length * (i + 1)]

        split_df.to_csv(
            os.path.join(
                args.save_path,
                '{}_{}.csv'.format(basename, i)
            ), index=None)

    print('Done!')


if __name__ == '__main__':
    main()
