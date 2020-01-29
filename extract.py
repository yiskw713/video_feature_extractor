import argparse
import os
import sys
import torch
import torch.nn.functional as F
import tqdm

from torch.utils.data import DataLoader

from libs.dataset import VideoDataset
from libs.mean_std import get_mean, get_std
from libs.models import generate_model, InceptionI3d, slowfast152_NL
from libs.spatial_transforms import Compose as SpatialCompose
from libs.spatial_transforms import (
    Normalize, Resize, CenterCrop, ToTensor)
from libs.temporal_transforms import Compose as TemporalCompose
from libs.temporal_transforms import TemporalSubsampling


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='feature extraction')
    parser.add_argument(
        'dataset_dir', type=str, help='path to dataset directory')
    parser.add_argument(
        'save_dir', type=str, help='path to the directory you want to save video features')
    parser.add_argument(
        'csv', type=str, help='path to the csv files which contains video path information')
    parser.add_argument(
        'arch', type=str, help='model architecture. (resnet50 | i3d | slowfast_nl)')
    parser.add_argument(
        'pretrained_weights', type=str, help='path to the pretrained model')
    parser.add_argument(
        '--sliding_window', action='store_true',
        help='Add --sliding_window option if you want to extract video features with sliding window')
    parser.add_argument(
        '--size', type=int, default=224, help='input image size')
    parser.add_argument(
        '--window_size', type=int, default=16, help='sliding window size')
    parser.add_argument(
        '--n_classes', type=int, default=700, help='the number of output classes of the pretrained model')
    parser.add_argument(
        '--num_workers', type=int, default=0, help='the number of workes for data loding')
    parser.add_argument(
        '--temp_downsamp_rate', type=int, default=1, help='temporal downsampling rate (default: 1)')
    parser.add_argument(
        '--file_format', type=str, default='hdf5', help=' jpg | png | hdf5 ')

    return parser.parse_args()


def extract(model, loader, save_dir, device):
    # when you extarct features without sliding window,
    # the features have spatial and temporal dimension
    model.eval()

    for sample in tqdm.tqdm(loader, total=len(loader)):
        with torch.no_grad():
            name = sample['name'][0]

            # if features already exist, the below process will be passes.
            if os.path.exists(os.path.join(save_dir, name + '.pth')):
                continue
            if os.path.exists(os.path.join(save_dir, 'motion', name + '.pth')):
                continue

            x = sample['clip'].to(device)
            feats = model.extract_features(x)

            # if features are extracted by slowfast
            if isinstance(feats, list):
                # squeeze batch dimension
                torch.save(
                    feats[0][0].to('cpu'), os.path.join(save_dir, 'semantic', name + '.pth'))
                torch.save(
                    feats[1][0].to('cpu'), os.path.join(save_dir, 'motion', name + '.pth'))
            else:
                torch.save(
                    feats.to('cpu'), os.path.join(save_dir, name + '.pth'))


def sliding_window_extract(model, loader, save_dir, window_size, device):
    # when you extarct features with sliding window,
    # the features do not keep spatial and temporal dimension
    # because they are too large
    model.eval()

    for sample in tqdm.tqdm(loader, total=len(loader)):
        with torch.no_grad():
            name = sample['name'][0]

            # if features already exist, the below process will be passes.
            if os.path.exists(os.path.join(save_dir, name + '.pth')):
                continue
            if os.path.exists(os.path.join(save_dir, 'motion', name + '.pth')):
                continue

            clip = sample['clip']
            _, _, t, h, w = clip.shape
            zeros = torch.zeros(1, 3, window_size - 1, h, w)
            clip = torch.cat([clip, zeros], dim=2)

            feats = []
            for i in range(t):
                x = clip[:, :, i:i + window_size].clone().detach().to(device)
                feat = model.extract_features(x)

                # if features are extracted by slowfast
                if isinstance(feat, list):
                    semantic = F.adaptive_avg_pool3d(feat[0], output_size=1)
                    semantic = semantic.squeeze()
                    motion = F.adaptive_avg_pool3d(feat[1], output_size=1)
                    motion = motion.squeeze()

                    # after squeeze, dim 0 is channel dimension
                    concated_feat = torch.cat([semantic, motion], dim=0)
                    feats.append(concated_feat.to('cpu'))
                else:
                    feat = F.adaptive_avg_pool3d(feat, output_size=1)
                    feat = feat.squeeze()
                    feats.append(feat.to('cpu'))

            # separate concated slowfast features into semantic and motion features
            if isinstance(feat, list):
                feats = torch.stack(feats, dim=1)
                c = len(semantic)
                semantics = feats[:c]
                motions = feats[c:]
                torch.save(
                    semantics, os.path.join(save_dir, 'semantic', name + '.pth'))
                torch.save(
                    motions, os.path.join(save_dir, 'motion', name + '.pth'))
            else:
                feats = torch.stack(feats, dim=1)
                torch.save(
                    feats, os.path.join(save_dir, name + '.pth'))


def main():
    args = get_arguments()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # spatial transformer
    spatial_transform = []
    spatial_transform.append(Resize(args.size))
    spatial_transform.append(CenterCrop(args.size))
    spatial_transform.append(ToTensor())
    spatial_transform.append(Normalize(mean=get_mean(), std=get_std()))

    # temporal transform
    temporal_transform = []
    if args.temp_downsamp_rate > 1:
        temporal_transform.append(
            TemporalSubsampling(args.temp_downsamp_rate))

    # dataset, dataloader
    data = VideoDataset(
        args.dataset_dir,
        csv_file=args.csv,
        spatial_transform=SpatialCompose(spatial_transform),
        temporal_transform=TemporalCompose(temporal_transform)
    )

    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # load model
    print('\n------------------------Loading Model------------------------\n')

    if args.arch == 'resnet50':
        print('ResNet50 will be used as a model.')
        model = generate_model(50, n_classes=args.n_classes)
    elif args.arch == 'i3d':
        print('I3D will be used as a model.')
        model = InceptionI3d(num_classes=args.n_classes)
    elif args.arch == 'slowfast_nl':
        print('SlowFast with Non Local Block will be used as a model.')
        model = slowfast152_NL(class_num=args.n_classes)

        # make directories for saving motion and semantic features
        if not os.path.exists(os.path.join(args.save_dir, 'semantic')):
            os.mkdir(os.path.join(args.save_dir, 'semantic'))
        if not os.path.exists(os.path.join(args.save_dir, 'motion')):
            os.mkdir(os.path.join(args.save_dir, 'motion'))
    else:
        print('There is no model appropriate to your choice.')
        sys.exit(1)

    # load pretrained model
    state_dict = torch.load(args.pretrained_weights)
    model.load_state_dict(state_dict)

    # send the model to cuda/cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    # extract and save features
    print('\n------------------------Start extracting features------------------------\n')

    if args.sliding_window:
        sliding_window_extract(
            model, loader, args.save_dir, args.window_size, device)
    else:
        extract(model, loader, args.save_dir, device)

    print("Done!")


if __name__ == '__main__':
    main()
