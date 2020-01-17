# Video Feature Extractor
This repo is for extracting video features.

## Preparation
You just need make csv files which include video paths information.
Please run `python utils/build_dataset.py`. See `utils/build_dataset.py` for more details.


## Requirements
* python 3.x
* pytorch >= 1.0
* torchvision
* pandas
* numpy
* Pillow
* h5py
* tqdm
* PyYAML
* addict

## Pretrained Models
Supported models are 3DResNet, SlowFastNetwork with non local block, (I3D).  
You can download from [here](https://drive.google.com/drive/folders/1pBp4pkhRP-ucd4mRGiX0omDQ5hbg3c7a?usp=sharing)
Pretrained I3D model is not available yet.

## Extracting and Save Video Features
Please run
``` python extract.py [dataset_dir] [save_dir] [csv] [arch] [pretrained_weights] [--sliding_window] [--size] [--window_size] [--n_classes] [--num_workers] [--temp_downsamp_rate [--file_format]```.


## References
* [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch)
* [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d)
