#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd
#$ -m abe
#$ -N sfnl

source /etc/profile.d/modules.sh
module purge
module load cuda/9.0/9.0.176.4
module load cudnn/7.3/7.3.1
module load nccl/2.1/2.1.15-1

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv global torch
cd /home/aab10820pu/video_feature_extractor

python extract.py /groups1/gaa50131/datasets/ActivityNet/hdf5 \
/groups1/gaa50131/datasets/ActivityNet/features/sf152nl_k700 \
./csv/activitynet_18.csv slowfast_nl ./weights/slowfast152_nl_kinetics700.pth --sliding_window