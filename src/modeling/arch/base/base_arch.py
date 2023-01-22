# Created on Tue Oct 11 2022 by Chuyang Zhao
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import torch
from torch import Tensor
from torch import nn

from ...processing import pad_collate_images, remove_padding


class BaseModel(nn.Module, ABC):
    """
    Base class for all models.
    """
    def __init__(self, testing=False):
        """
        Args:
            testing (bool): Set to False in inference mode. In inference mode
                the target is not provided. The model should return the outputs
                directly without computing the losses and metrics. Although it
                can be implictly inferred from the input data, we recommend
                explicitly setting this flag to False when in inference mode.
                NOTE: in training/validation/evaluation mode where target is
                provided, this flag should be set to False. Only in inference
                mode where target is not required, this flag should be set to True.
        
        """
        super().__init__()
        self.testing = testing

        # register a dummy tensor, which will be used to infer the device of the current model
        self.register_buffer("dummy", torch.empty(0), False)
    
    @property
    def device(self):
        return self.dummy.device

    @property
    def size_divisibility(self):
        """
        Some networks require the size of the input image to be divisible
        by some factor, which is often used in encoder-decoder style networks.

        If the network you implemented needs some size divisibility, you can
        override this property, otherwise 1 will be returned which will turn
        off this feature when padding images.
        """
        return 1

    def _move_to_current_device(self, x):
        return x.to(self.device)

    @abstractmethod
    def losses(self, inputs: Tensor, targets: Tensor):
        raise NotImplementedError

    @abstractmethod
    def metrics(self, inputs: Tensor, targets: Tensor):
        raise NotImplementedError

    @abstractmethod
    def forward(self, batched_inputs: List[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: batched outputs got from data loader.

        Returns:
            * in training stage, returns loss_dict and metric_dict.
            * in testing stage, returns the outputs of the model.
        """
        raise NotImplementedError

    def preprocess_inputs(self, batched_inputs: List[Dict[str, Tensor]]):
        """
        Preprocess inputs before feeding them into the model. Typically, this
        includes padding images to the same size and then collate them into a
        batched tensor.
        For tasks like detection and segmentation, you many need to process the
        bounding boxes and segmentation masks as well. Implement your own 
        preprocess function according to your task.
        """
        raise NotImplementedError

    def process_outputs(self, outputs: Tensor):
        """
        Process the outputs of the model. For example, remove padding from the
        images for image to image tasks.
        """
        raise NotImplementedError
        