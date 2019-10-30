import argparse
import json
import os
import subprocess
import tqdm


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='train a network for action recognition')
    parser.add_argument('video_dir', type=str, help='path to video_dir')
    parser.add_argument('json', type=str, help='path to json file')
    parser.add_argument('ids_json', type=str, help='path to ids json file')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    with open(args.json, 'r') as f:
        json_data = json.load(f)

    with open(args.ids_json, 'r') as f:
        ids_data = json.load(f)

    for idx in tqdm.tqdm(ids_data):
        video_path = os.path.join(args.video_dir, idx + '.mp4')

        if not os.path.exists(video_path):
            continue

        ffprobe_cmd = ['ffprobe', str(video_path)]
        p = subprocess.Popen(
            ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res = p.communicate()[1].decode('utf-8')

        fps = float([x for x in res.split(',') if 'fps' in x][0].rstrip('fps'))

        if idx in json_data.keys():
            json_data[idx]['fps'] = fps

    with open(args.json, 'w') as f:
        json.dump(json_data, f)
