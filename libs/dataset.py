import glob
import h5py
import io
import numpy as np
import os
import pandas as pd
import sys
import torch

from PIL import Image
from torch.utils.data import Dataset

from .mean_std import get_mean, get_std


class ImageLoader(object):
    """
    Return sequential frames in video clips corresponding to frame_indices.
    self.temporal_transform changes frame_indices. See libs/temporal_transforms.py.
    """

    def __init__(self, temporal_transform=None):
        super().__init__()
        self.temporal_transform = temporal_transform
        self.mean = get_mean(norm_value=1)
        self.std = get_std(norm_value=1)

    def __call__(self, video_path):
        img_paths = []
        img_paths += glob.glob(os.path.join(video_path, '*.jpg'))
        img_paths += glob.glob(os.path.join(video_path, '*.png'))

        img_paths = sorted(img_paths)
        frame_indices = [i for i in range(len(img_paths))]

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        video = []
        for i in frame_indices:
            if i is not None:
                video.append(Image.open(img_paths[i]))
            else:
                # if index is None, insert noise instead of original frames
                w, h = video[0].size
                noise = np.random.normal(
                    self.mean, self.std, size=(h, w, 3)).astype(np.uint8)
                video.append(Image.fromarray(noise))

        return video


class HDF5Loader(object):
    """
    Return sequential frames in video clips corresponding to frame_indices.
    self.temporal_transform changes frame_indices. See libs/temporal_transforms.py.
    This loader supports only .hdf5 format.
    """

    def __init__(self, temporal_transform=None):
        super().__init__()
        self.temporal_transform = temporal_transform
        self.mean = get_mean(norm_value=1)
        self.std = get_std(norm_value=1)

    def __call__(self, video_path):
        with h5py.File(video_path, 'r') as f:
            video_data = f['video']
            frame_indices = [i for i in range(len(video_data))]
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)

            video = []
            for i in frame_indices:
                if i is not None:
                    video.append(Image.open(io.BytesIO(video_data[i])))
                else:
                    # if index is None, insert noise instead of original frames
                    w, h = video[0].size
                    noise = np.random.normal(
                        self.mean, self.std, size=(h, w, 3)).astype(np.uint8)
                    video.append(Image.fromarray(noise))

        return video


class VideoDataset(Dataset):
    """
    Dataset class for Video Datset
    """

    def __init__(
            self, dataset_dir, csv_file, spatial_transform=None,
            temporal_transform=None, file_format='hdf5'):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.df = pd.read_csv(csv_file)

        if file_format == 'hdf5':
            self.loader = HDF5Loader(temporal_transform)
        elif file_format == 'png' or 'jpg':
            self.loader = ImageLoader(temporal_transform)
        else:
            print(
                "you must choose \'hdf5\', \'jpg\' or \'png\' as file format in VideoDataset")
            sys.exit(1)

        self.spatial_transform = spatial_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['video']
        name = os.path.splitext(path)[0]
        video_path = os.path.join(self.dataset_dir, path)

        clip = self.loader(video_path)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        clip = torch.stack(clip, dim=0).permute(1, 0, 2, 3)

        sample = {
            'clip': clip,
            'name': name,
        }

        return sample
