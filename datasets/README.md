## Setup Environment Variable
Before prepare the datasets, you should setup the environment variable `DATASETS_ROOT` pointing to the root directory of all datasets, so that python can found the directory where you saved the datasets. 

## CIFAR dataset
Download the (cifar-10)[https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz] and (cifar-100)[https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz] from the official website.
Untar the downloaded files and put the uncompressed files under the `DATASETS_ROOT`. Data directory should have the following structure:
```
DATASETS_ROOT/
    cifar-10-batches-py/
        batches.meta
        data_batch_{1..5}
        readme.html
        test_batch
    
```

## CIFAR-10-Images dataset
The official cifar-10 dataset is in pickle cached format. To illustrate how to work with image dataset, which is the case for most datasets, you can download the cifar-10 images dataset from [here](https://github.com/YoongiKim/CIFAR-10-images) and put it under the `DATASETS_ROOT`:
```
DATASETS_ROOT/
    CIFAR-10-images/
        test/
        train/
        README.md
```
