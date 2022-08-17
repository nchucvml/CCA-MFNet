
# CCA-MFNet

# 
## Criss-Cross Attention based Multi-Level Fusion Network (CCA-MFNet) for Gastric Intestinal Metaplasia Segmentation 

The paper is accepted by MICCAI 2022 workshop. 

In this project, we demonstrate the performance of the CCA-MFNet for gastric intestinal metaplasia segmentation.

  

## Instructions for Code:
### Requirements

To install PyTorch>=1.4.0 or 1.8.2 LTS (recommended), please refer to https://github.com/pytorch/pytorch#installation.   
It is necessary that VRAM is greater than 11GB (_e.g._ RTX2080Ti)  
Python 3.7 
gcc (GCC) 4.8.5  
CUDA 9.0  

### Compiling

```bash
# Install **Pytorch**
$ conda install pytorch torchvision -c pytorch

# Install **Apex**
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Install **Inplace-ABN**
$ git clone https://github.com/mapillary/inplace_abn.git
$ cd inplace_abn
$ python setup.py install
```

### Training and Inference
Training script.
```bash
./run_train.sh
``` 

Inference script.
```bash
./run_inference.sh
``` 

### Model in Gastric Intestinal Metaplasia Dataset (MICCAI 2022 workshop paper)

| **Method** | **Backbone** | **mIOU** | **mDice** | **Recall** | **Precision** | **Accuracy** | **Link** |
|:-------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|Proposed| ResNet-50 | 68.92 | 78.47 | 74.94 | 83.45 | 96.13 | [Google Drive](https://drive.google.com/file/d/1PTkBTD-kttEK7HRqbeHHYjV7FZGp7rmb/view?usp=sharing) |


## Acknowledgments 
This work was supported in part by the National Science and Technology Council, Taiwan under Grant MOST 110-2634-F-006-022, 111-2327-B-006-007, and 111-2628-E-005-007-MY3. We would like to thank National Center for High-performance Computing (NCHC) for providing computational and storage resources.

## Particular Thanks
Department of Internal Medicine and Institute of Clinical Medicine, National Cheng Kung University Hospital, College of Medicine
