#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g4.72h
#$ -ac d=nvcr-torch-1712,d_shm=10G

~/.pyenv/versions/anaconda3-5.2.0/envs/torch/bin/python train_model.py -c train_model_config.json -s -i 0
