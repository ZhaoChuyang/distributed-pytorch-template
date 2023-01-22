# Created on Sat Jan 1 2023 by Chuyang Zhao
import os
from ..catalog import DATASET_CATALOG


def register_cifar_dataset():
    """
    Load the cifar dataset in a list of dataset dicts. Each dict contains:
        * image: original image without any preprocessing.
        * target: class label of the image.
    """
    pass