#!/bin/bash
uname -a
#date
#env
date

DATASET_PATH=./dataset/gastric_dataset/
DATA_LIST=./dataset/list/gastric_dataset/test_224x224.lst
BS=1
CCA_MFNET_MODEL=snapshots/gastric_model.pth
RECUR_CCA=2
GPU_IDS=0

# Inference
CUDA_VISIBLE_DEVICES=${GPU_IDS} python evaluate.py --data-dir ${DATASET_PATH} --data-list ${DATA_LIST} --batch-size ${BS} --restore-from ${CCA_MFNET_MODEL} --gpu ${GPU_IDS} --model cca_mfnet
