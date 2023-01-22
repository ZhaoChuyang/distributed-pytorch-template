from typing import Union
import torch
import numpy as np
from sklearn.metrics import accuracy_score, top_k_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .build import METRIC_REGISTRY


def convert_tensor_to_ndarray(*args):
    """
    Convert torch.Tensor to numpy.ndarray.

    Args:
        args (tuple): tuple of torch.Tensor or numpy.ndarray.
    """
    return [arg.detach().cpu().numpy() if isinstance(arg, torch.Tensor) else arg for arg in args]


def get_pred_and_target(input: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]):
    """
    Get prediction and target in numpy.ndarray.

    Args:
        input (np.array or torch.Tensor): input array of outputs logits, shape is (n_samples, n_classes).
        target (np.array or torch.Tensor): target array, shape is (n_samples,).

    Returns:
        np.array: prediction array of shape (n_samples,).
        np.array: target array of shape (n_samples,).
    """
    input, target = convert_tensor_to_ndarray(input, target)
    assert len(input) == len(target), (
        f'Length of input and target are different: {len(input)}, {len(target)}.')
    pred = np.argmax(input, axis=1)
    return pred, target


@METRIC_REGISTRY.register()
def accuracy(input: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]):
    """
    Calculate accuracy.

    Args:
        input (np.array or torch.Tensor): input array of outputs logits.
        target (np.array or torch.Tensor): target array.
    """
    pred, target = get_pred_and_target(input, target)
    return accuracy_score(target, pred)


@METRIC_REGISTRY.register()
def top_k_accuracy(input: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor], k: int = 5):
    """
    Calculate top-k accuracy.

    Args:
        input (np.array or torch.Tensor): input array of outputs logits.
        target (np.array or torch.Tensor): target array.
        k (int): top-k accuracy.
    """
    input, target = convert_tensor_to_ndarray(input, target)
    assert len(input) == len(target), (
        f'Length of input and target are different: {len(input)}, {len(target)}.')
    return top_k_accuracy_score(target, input, k=k)


@METRIC_REGISTRY.register()
def precision(input: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor], average: str = 'macro'):
    """
    Calculate precision.

    Args:
        input (np.array or torch.Tensor): input array of outputs logits.
        target (np.array or torch.Tensor): target array.
        average (str): average method, can be 'macro', 'micro', 'weighted', 'samples'.
    """
    pred, target = get_pred_and_target(input, target)
    return precision_score(target, pred, average=average, zero_division=0)


@METRIC_REGISTRY.register()
def recall(input: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor], average: str = 'macro'):
    """
    Calculate recall.

    Args:
        input (np.array or torch.Tensor): input array of outputs logits.
        target (np.array or torch.Tensor): target array.
        average (str): average method, can be 'macro', 'micro', 'weighted', 'samples'.
    """
    pred, target = get_pred_and_target(input, target)
    return recall_score(target, pred, average=average, zero_division=0)


@METRIC_REGISTRY.register()
def f1(input: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor], average: str = 'macro'):
    """
    Calculate f1 score.

    Args:
        input (np.array or torch.Tensor): input array of outputs logits.
        target (np.array or torch.Tensor): target array.
        average (str): average method, can be 'macro', 'micro', 'weighted', 'samples'.
    """
    pred, target = get_pred_and_target(input, target)
    return f1_score(target, pred, average=average, zero_division=0)


@METRIC_REGISTRY.register()
def roc_auc(input: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor], average: str = 'macro'):
    """
    Calculate roc auc score.

    Args:
        input (np.array or torch.Tensor): input array of outputs logits.
        target (np.array or torch.Tensor): target array.
        average (str): average method, can be 'macro', 'micro', 'weighted', 'samples'.
    """
    input, target = convert_tensor_to_ndarray(input, target)
    assert len(input) == len(target), (
        f'Length of input and target are different: {len(input)}, {len(target)}.')
    return roc_auc_score(target, input, average=average)
