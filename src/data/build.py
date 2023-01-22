# Created on Mon Oct 10 2022 by Chuyang Zhao
from typing import List, Optional, Union
import torch.utils.data as torchdata
import itertools
import logging
from ..data.common import ToIterableDataset, CommClsDataset
from ..data.samplers import TrainingSampler, BalancedSampler, InferenceSampler
from ..data.catalog import DATASET_CATALOG
from ..utils.config import configurable, ConfigDict
from ..data.transforms.build import build_transforms


__all__ = ['build_batch_data_loader', 'build_train_loader', 'build_test_loader']


logger = logging.getLogger(__name__)


def build_batch_data_loader(
    dataset,
    sampler,
    batch_size,
    *,
    num_workers=0,
    collate_fn=None
):
    """
    Build a batched dataloader.
    """
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        dataset = ToIterableDataset(dataset, sampler)
    
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn
    )


def get_dataset_dicts(
    names: Union[str, List[str]],
):
    """
    Args:
        names (str or list[str]): dataset name or a list of dataset names.

    Returns:
        merged list of all dataset dicts.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names
    dataset_dicts = [DATASET_CATALOG.get(name) for name in names]

    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset: {} is empty!".format(dataset_name)

    # combine dataset dicts
    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    return dataset_dicts


def get_dataset_sizes(
    names
):
    """
    Get the sizes of all datasets specified by names.

    Args:
        names (str or list[str]): dataset name or a list of dataset names.

    Returns:
        list of the sizes of all datasets specified by names.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names

    dataset_sizes = [len(DATASET_CATALOG.get(name)) for name in names]

    return dataset_sizes


def _train_loader_from_config(cfg: ConfigDict, *, dataset = None):
    """
    Args:
        cfg (ConfigDict): cfg is the config dict of the train data factory: `cfg.data_factory.train`.
        dataset (optional): provide dataset if you don't want to construct dataset using dataset names
            specified in `cfg.names`.
    """
    if dataset is not None:
        names = None
    else:
        names = cfg.names

    transforms = build_transforms(cfg.transforms)

    return {
        "names": names,
        "dataset": dataset,
        "batch_size": cfg.batch_size,
        "transforms": transforms,
        "sampler": cfg.sampler,
        "num_workers": cfg.num_workers,
    }


@configurable(from_config=_train_loader_from_config)
def build_train_loader(
    names=None,
    dataset=None,
    shuffle=True,
    batch_size=1,
    transforms=None,
    sampler=None,
    num_workers=0,
    collate_fn=None,
):
    """
    Build a train loader.

    Args:
        names (str or list[str]): dataset name or a list of dataset names, you must provide either `names` or `dataset`.
        dataset (torchdata.Dataset or torchdata.IterableDataset): instantiated dataset, you must provide either `names` or `dataset`.
        shuffle (bool): whether to shuffle the dataset.
        batch_size (int): total batch size, the batch size each GPU got is batch_size // num_gpus.
        transforms (torchvision.Transforms): transforms applied to dataset.
        sampler (str): specify this argument if you use map-style dataset, by default use the TrainingSampler. You should
            not provide this if you use iterable-style dataset.
        num_worker (int): num_worker of the dataloader.
        collate_fn (callable): use trivial_collate_fn by default.

    NOTE: For dataset dicts compatible with CommClsDataset, you can register it in `data/datasets` and 
    load the dataset by its names with argument `names`. For dataset that is not compatible with 
    CommClsDataset, you can directly read and process the dataset and pass it with the argument `dataset`.
    """
    if dataset is None:
        dataset_dicts = get_dataset_dicts(names)
        dataset_sizes = get_dataset_sizes(names)
        # TODO: show datasets information

        # if you want to build dataset of other type, change it here.
        dataset = CommClsDataset(dataset_dicts, transforms)

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler == "TrainingSampler":
            sampler = TrainingSampler(sum(dataset_sizes), shuffle=shuffle)
        elif sampler == "BalancedSampler":
            sampler = BalancedSampler(dataset_sizes, shuffle=shuffle)
        else:
            raise NotImplementedError(f"sampler: {sampler} is not implemented.")
    
    return build_batch_data_loader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


@configurable(from_config=_train_loader_from_config)
def build_test_loader(
    names = None,
    dataset = None,
    transforms = None,
    batch_size: int = 1,
    sampler=None,
    num_workers: int = 0,
    collate_fn = None,
):
    """
    Build a test loader.
    """
    if dataset is None:
        dataset_dicts = get_dataset_dicts(names)
        dataset = CommClsDataset(dataset_dicts, transforms)
    else:
        assert transforms is None, "You should not provide transforms when you use dataset to construct the dataloader."

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None or sampler == "InferenceSampler":
            sampler = InferenceSampler(len(dataset))
        else:
            logger.error(f"sampler: {sampler} is not implemented.")
            raise NotImplementedError(f"sampler: {sampler} is not implemented.")
    
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn
    )


def trivial_batch_collator(batch):
    """
    A batch collator that do not do collation on the
    batched data but directly returns it as a list.
    """
    return batch
