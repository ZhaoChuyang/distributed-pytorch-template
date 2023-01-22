# Distributed PyTorch Template

A concise and full-featured PyTorch project template for distributed training and evaluation. It shares similar structure with most of the popular modern frameworks like [detectron2](https://github.com/facebookresearch/detectron2/tree/main/detectron2) and [mmdetection](https://github.com/open-mmlab/mmdetection) but simplifies the complex designs and inherits the core componets. It also reduces the extra packages dependencies as much as possible. Hope this simple PyTorch template can help you get started on your project easily and built a scalable and high-performance deep learning project. 

## Installation

### Requirements
- Linux or macOS with Python ≥ 3.7
- PyTorch ≥ 1.8 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
- OpenCV

### Preparation
It is highly recommended that you rename the root package directory (originally is `distributed-pytorch-template/src`) to your package name, since `src` is a general and meaningless name for a package. Changing it to a meaningful name makes your project more explict and reduces potential package name conflicts.
After you rename the directory of the root package, it is recommended to change the package name in `setup.py` to the same name, so that pip can use this name for your package.  
Setup your datasets as introduced in [datasets/README.md](./datasets/README.md). Set the environment variable `DATASETS_ROOT` pointing to the directory containing all datasets in command line or shell config file for permanent usage:
```sh
export ALL_DATASETS=/path/to/datasets_root
```

### Build DeepLLE from source

In the root directory of the project, installing by:
```
python -m pip install -e .
```

## Getting Started

### Training
In `tools`, we provide a basic training script `train_net.py`. You can use it as a reference to write your own training scripts.  
To train with `tools/train_net.py`, you can run:
```
cd tools/
python train_net.py --num-gpus 4 --config ../configs/llie_base.json
```
The config is made to train with 4 GPUs. You can change the number of GPUs by modifying the `--num-gpus` option.

To specify the GPU devices, you can setting the environment variable `CUDA_VISIBLE_DEVICES`:
```
CUDA_VISIBLE_DEVICES=0,2 python train_net.py --num-gpus 2 --config ../configs/llie_base.json
```
The above config will use the GPU devices with id 0 and 2 for training.
