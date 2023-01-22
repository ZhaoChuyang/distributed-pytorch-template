# Created on Fri Oct 14 2022 by Chuyang Zhao
import os
"""
Hard-code registering the builtin datasets, so you can directly
use these builtin datasets in config by its name.

All builtin datasets are assumed to be store in 'DeepISP/datasets'.
If you want to change the root directoy of datasets, set the system
enviroment "ISP_DATASETS" by:
    export ISP_DATASETS='~/path/to/directory'

Don't modify this file to change the root directory of datasets.

This file is intended to register builtin datasets only, please
don't register your custom datasets in this file.
"""
from .cifar_img import register_cifar_image_dataset


def register_cifar_image_all(root):
    """
    Register all cifar image datasets.
    """
    register_cifar_image_dataset("cifar10_image_train", os.path.join(root, "CIFAR-10-images"), "train")
    register_cifar_image_dataset("cifar10_image_test", os.path.join(root, "CIFAR-10-images"), "test")


_root = os.path.expanduser(os.getenv("DATASETS_ROOT", "datasets"))
register_cifar_image_all(_root)
