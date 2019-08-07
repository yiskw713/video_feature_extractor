"""
Extract and save motion and semantic features from slowfast network or two-stream CNN
"""


import argparse
import os
import torch
import tqdm

from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, CenterCrop, Normalize

from utils.dataset import MSR_VTT
from utils.mean import get_mean, get_std
from models import slowfast


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='train a network for action recognition')
    parser.add_argument(
        'dataset_dir', type=str, help='path to dataset directory')
    parser.add_argument(
        'save_dir', type=str, help='path to the directory you want to save video features')
    parser.add_argument(
        'arch', type=str, help='model architecture. (slowfast152_nl)')
    parser.add_argument(
        'pretrained_weights', type=str, help='path to the pretrained model')
    parser.add_argument(
        '--n_classes', type=int, default=700, help='the number of output classes of the pretrained model')
    parser.add_argument(
        '--num_workers', type=int, default=4, help='the number of workes for data loding')
    parser.add_argument(
        '--batch_size', type=int, default=64, help='batch size')
    parser.add_argument(
        '--temp_downsamp_rate', type=int, default=2, help='temporal downsampling rate (default: 2)')
    parser.add_argument(
        '--image_file_format', type=str, default='hdf5', help=' jpg | png | hdf5 ')
    parser.add_argument(
        '--n_jobs', type=int, default=-1, help='the number of cores which save feats')

    return parser.parse_args()


def save_feats(semantic_feats, motion_feats, video_id, save_dir):
    # feats.shape => (C, T, H, W)
    torch.save(
        semantic_feats, os.path.join(save_dir, 'semantic', video_id + '.pth'))
    torch.save(
        motion_feats, os.path.join(save_dir, 'motion', video_id + '.pth'))


def extract(model, loader, save_dir, n_jobs, device):
    model.eval()

    for sample in tqdm.tqdm(loader, total=len(loader)):
        with torch.no_grad():
            x = sample['clip'].to(device)
            video_id = sample['video_id']

            batch_size = x.shape[0]

            _, semantic_feats, motion_feats = model(x)
            semantic_feats = semantic_feats.to('cpu')
            motion_feats = motion_feats.to('cpu')

            Parallel(n_jobs=n_jobs)(
                [delayed(save_feats)(
                    semantic_feats[i], motion_feats[i], video_id[i], save_dir
                ) for i in range(batch_size)]
            )


def main():
    args = get_arguments()

    # make directory
    if not os.path.exists(os.path.join(args.save_dir, 'semantic')):
        os.mkdir(os.path.join(args.save_dir, 'semantic'))
    if not os.path.exists(os.path.join(args.save_dir, 'motion')):
        os.mkdir(os.path.join(args.save_dir, 'motion'))

    # DataLoaders
    normalize = Normalize(mean=get_mean(), std=get_std())

    data = MSR_VTT(
        args.dataset_dir,
        args.temp_downsamp_rate,
        args.image_file_format,
        transform=Compose([
            CenterCrop((224, 224)),
            ToTensor(),
            normalize,
        ])
    )

    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # load model
    print('\n------------------------Loading Model------------------------\n')

    if args.arch == 'slowfast152_nl':
        print('ResNet50 will be used as a model.')
        model = slowfast.resnet152_NL(class_num=args.n_classes)
    else:
        print('There is no model appropriate to your choice. '
              'Instead, resnet50 will be used as a model.')
        model = slowfast.resnet152_NL(class_num=args.n_classes)

    # load pretrained model
    state_dict = torch.load(args.pretrained_weights)
    model.load_state_dict(state_dict)

    # send the model to cuda/cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)  # make parallel
        torch.backends.cudnn.benchmark = True

    # extract and save features
    print('\n------------------------Start extracting features------------------------\n')

    extract(model, loader, args.save_dir, args.n_jobs, device)

    print("Done!")


if __name__ == '__main__':
    main()
