# Created on Sat Jan 21 2023 by Chuyang Zhao
from typing import List, Dict, Tuple
import torch
from torch import nn, Tensor
from torchvision import models
from ..build import MODEL_REGISTRY
from .base import BaseClsModel
from ..metrics import accuracy, f1, precision, recall
from ...utils.config import configurable


@MODEL_REGISTRY.register()
class ResNet(BaseClsModel):
    """
    ResNet model for classification task.
    """
    @configurable
    def __init__(self, resnet: nn.Module, num_features: int, num_classes: int, testing=False):
        super().__init__(testing=testing)
        self.resnet = resnet
        self.fc = nn.Linear(num_features, num_classes)
        
        self.ce_loss = nn.CrossEntropyLoss()
    
    @classmethod
    def from_config(cls, cfg):
        cfg_resnet = cfg["model"]["args"]["resnet"]
        resnet = getattr(models, cfg_resnet.name)(**cfg_resnet["args"])
        num_features = resnet.fc.in_features
        resnet.fc = nn.Identity()
        return {
            "resnet": resnet,
            "num_features": num_features,
            "num_classes": cfg["model"]["args"]["num_classes"],
            "testing": cfg["model"]["args"]["testing"]
        }

    def metrics(self, inputs, targets):
        metric_dict = {}
        metric_dict["accuracy"] = accuracy(inputs, targets)
        metric_dict["f1"] = f1(inputs, targets)
        metric_dict["precision"] = precision(inputs, targets)
        metric_dict["recall"] = recall(inputs, targets)
        return metric_dict

    def losses(self, inputs, targets):
        loss_dict = {}
        loss_dict["ce_loss"] = self.ce_loss(inputs, targets)
        return loss_dict

    def forward(self, batched_inputs: List[Dict[str, Tensor]]) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the model.

        Args:
            batched_inputs (list): a list of dataset dicts. Each dict should contains
                'image' and 'target' in training mode, and only 'image' in inference mode.
        
        Returns:
            Tensor: a batched tensor of logits.
            Tensor: a batched tensor of class labels.
        """
        images, labels = self.preprocess_inputs(batched_inputs)
        x = self.resnet(images)
        logits = self.fc(x)
        if self.testing:
            return logits
        loss_dict = self.losses(logits, labels)
        metric_dict = self.metrics(logits, labels)
        return loss_dict, metric_dict
