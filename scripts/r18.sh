#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd
#$ -m abe
#$ -N r18

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
cd /home/aab10820pu/anet2020_recog

python train.py ./result/k400/r18/config.yaml --resume
