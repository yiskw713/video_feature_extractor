# Video Feature Extractor
This repo is for extracting video features.
You can also finetune a pretrained model on your own setting.

## Dataset
Supported datasets are Kinetics400, Kinetics700, MSR-VTT.

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
* tensorboardX
* pyhd
* (accimage)

## Pretrained Models
You can download from [here](https://drive.google.com/drive/folders/1pBp4pkhRP-ucd4mRGiX0omDQ5hbg3c7a?usp=sharing)

## Extracting and Save Video Features
Coming soon.

## Finetunig on Your Own Setting
Make a directory in `./result/` and create your configuration file into it.
Then run `python finetuning.py ./result/****/config.yaml`
Follow the below example for a configuration file.

```
model: resnet50

class_weight: True    # if you use class weight to calculate cross entropy or not
writer_flag: True      # if you use tensorboardx or not

n_classes: 700
batch_size: 164
input_frames: 64
height: 224
width: 224
temp_downsamp_rate: 2
num_workers: 32
max_epoch: 50

optimizer: SGD
learning_rate: 0.0003
lr_patience: 10       # Patience of LR scheduler
momentum: 0.9         # momentum of SGD
dampening: 0.0        # dampening for momentum of SGD
weight_decay: 0.0001   # weight decay
nesterov: True        # enables Nesterov momentum
final_lr: 0.1         # final learning rate for AdaBound

image_file_format: hdf5
dataset_dir: /groups1/gaa50131/datasets/kinetics/videos_700_hdf5
train_csv: ./dataset/kinetics_700_train.csv
val_csv: ./dataset/kinetics_700_val.csv
pretrained_weights: ./weights/resnet50_kinetics700.pth
result_path: ./result/r50_k700_64f_dsr2

```

## References
* [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch)
* [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d)
