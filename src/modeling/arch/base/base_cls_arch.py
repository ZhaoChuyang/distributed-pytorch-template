# Created on Sat Jan 21 2023 by Chuyang Zhao
import logging
from typing import List, Dict, Tuple

import torch
from torch import Tensor
from .base_arch import BaseModel


logger = logging.getLogger(__name__)


class BaseClsModel(BaseModel):
    """
    Base class for all classification model.
    """
    def __init__(self, testing=False):
        super().__init__(testing=testing)

    def preprocess_inputs(self, batched_inputs: List[Dict[str, Tensor]]) -> Tuple[Tensor, Tensor]:
        """
        Preprocess inputs before feeding them into the model. In classification
        task, we assume the input image and its class label are saved with keys
        'image' and 'target' in the dataset dict respectively.
        We expect the shape of the input images in a batch are of the same shape,
        so we directly concatenate them into a single batched tensor.

        NOTE: in inference mode (self.testing is True), 'target' is not expected
        to be in the dataset dict.

        Args:
            batched_inputs (list): a list of dataset dicts. Each dict should contains
                'image' and 'target' in training mode, and only 'image' in inference mode.
        
        Returns:
            Tensor: a batched tensor of input images.
            Tensor: a batched tensor of class labels.
        """
        images = torch.stack([x['image'] for x in batched_inputs], dim=0)
        images = self._move_to_current_device(images)
        labels = None
        if not self.testing:
            try:
                labels = torch.tensor([x['target'] for x in batched_inputs])
                labels = self._move_to_current_device(labels)
            except KeyError:
                logger.warning(
                    "'target' does not exist in dataset dict, assuming in inference mode, setting `self.testing` to True.")
                self.testing = True
        
        return images, labels
