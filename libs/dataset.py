import glob
import h5py
import io
import torch
import pandas as pd
import os
import sys

from PIL import Image
from torch.utils.data import Dataset


class ImageLoader(object):
    """
    Return sequential frames in video clips corresponding to frame_indices.
    self.temporal_transform changes frame_indices. See libs/temporal_transforms.py.
    """

    def __init__(self, temporal_transform=None):
        super().__init__()
        self.temporal_transform = temporal_transform

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
            video.append(Image.open(img_paths[i]))

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

    def __call__(self, video_path):
        with h5py.File(video_path, 'r') as f:
            video_data = f['video']
            frame_indices = [i for i in range(len(video_data))]
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)

            video = []
            for i in frame_indices:
                video.append(Image.open(io.BytesIO(video_data[i])))

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
        name = self.df.iloc[idx]['video']
        video_path = os.path.join(self.dataset_dir, name)

        if 'cls_id' in self.df.columns:
            cls_id = torch.tensor(int(self.df.iloc[idx]['cls_id'])).long()
        else:
            cls_id = None

        if 'label' in self.df.columns:
            label = self.df.iloc[idx]['label']
        else:
            label = None

        clip = self.loader(video_path)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        clip = torch.stack(clip, dim=0).permute(1, 0, 2, 3)

        sample = {
            'clip': clip,
            'cls_id': cls_id,
            'name': name,
            'label': label,
        }

        return sample
