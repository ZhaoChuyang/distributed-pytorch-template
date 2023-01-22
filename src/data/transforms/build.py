# Created on Thu Oct 13 2022 by Chuyang Zhao
from typing import List, Dict, Union
import torchvision.transforms as T


def build_transforms(cfg_transforms: List[Union[Dict, str]] = None):
    if cfg_transforms is None:
        transforms = T.Compose([
            T.ToTensor()
        ])
    else:
        transform_list = []
        for t in cfg_transforms:
            transform = None
            if isinstance(t, str):
                transform = getattr(T, t)()
            else:
                transform = getattr(T, t["name"])(**t["args"])
            transform_list.append(transform)
        
        transforms = T.Compose(transform_list)
    
    return transforms
