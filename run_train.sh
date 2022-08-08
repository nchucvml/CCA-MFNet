#!/bin/bash
uname -a
#date
#env
date

DATASET_PATH=./dataset/gastric_dataset/
DATA_LIST=./dataset/list/gastric_dataset/train_224x224.lst
PRETRAIN_MODEL=./dataset/resnet101-imagenet.pth
LR=0.006
WD=0.0001
BS=8
RECUR_CCA=2
OHEM=1
OHEM_THRES=0.7
OHEM_KEEP=100000
GPU_IDS=0

# Training
CUDA_VISIBLE_DEVICES=${GPU_IDS} python train.py --data-dir ${DATASET_PATH} --data-list ${DATA_LIST} --restore-from ${PRETRAIN_MODEL} --gpu ${GPU_IDS} --learning-rate ${LR} --weight-decay ${WD} --batch-size ${BS} --recurrence ${RECUR_CCA} --ohem ${OHEM} --ohem-thres ${OHEM_THRES} --ohem-keep ${OHEM_KEEP} --model cca_mfnet
