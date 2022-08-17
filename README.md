
# CCA-MFNet

# 
## Criss-Cross Attention based Multi-Level Fusion Network (CCA-MFNet) for Gastric Intestinal Metaplasia Segmentation 

The paper is accepted by MICCAI 2022 workshop. 

In this project, we demonstrate the performance of the CCA-MFNet for gastric intestinal metaplasia segmentation.

```    

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
