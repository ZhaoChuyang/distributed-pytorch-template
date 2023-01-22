# Created on Mon Oct 10 2022 by Chuyang Zhao
from torch.nn.parallel import DistributedDataParallel
from ..utils import Registry
from ..utils import comm
import torch


MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for the whole model.

The registry object will be called with `obj(**cfg_model)`,
and expected to return a nn.Module.
"""


def build_model(cfg):
    """
    Build the whole model, arguments for building the model
    should be defined in `cfg_model`. 
    """
    name = cfg["model"]["name"]
    args = cfg["model"]["args"]
    
    assert "testing" in args, (
        f"'testing' needs to be explicitly specified in config, " \
        "but was not found in model's args: {args}" \
        "set it to False if you are training model, True if you are evaluating or tracing model"
    )
    
    # model = MODEL_REGISTRY.get(name)(**args)
    model = MODEL_REGISTRY.get(name)(cfg)
    model.to(torch.device(cfg["model"]["device"]))
    return model


def create_ddp_model(model, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module.
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """
    if comm.get_world_size() == 1:
        return model
    
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
    
    ddp = DistributedDataParallel(model, **kwargs)
    return ddp
