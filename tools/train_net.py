#!/usr/bin/env python
# Created on Thu Jan 06 2023 by Chuyang Zhao
import os
import sys
import json
import argparse
# NOTE: change `src.` to the root module name of your project
from src import PROJECT_NAME
from src.engine.launch import launch
from src.engine.trainer import SimpleTrainer
from src.utils.logger import setup_logger
from src.utils import comm
from src.utils import mkdirs, get_ip_address, init_config
from src.utils.nn_utils import get_model_info
from src.modeling import build_model, create_ddp_model
from src.solver import build_optimizer, build_lr_scheduler
from src.data.transforms.build import build_transforms
from src.data import build_train_loader, build_test_loader


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog or (
            f"Examples:\nRun on single machine:\n"
            f"\t$ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml\n"
            f"Change some config options:\n"
            f"\t$ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "--opts",
        help='modify config using the command line, e.g. --opts model.name "ResNet50" data.batch_size=30',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


class DefaultTrainer(SimpleTrainer):
    def __init__(self, config):
        model = self.build_model(config)

        cfg_train_factory = config["data_factory"]["train"]
        cfg_valid_factory = config["data_factory"]["valid"]
        train_loader = self.build_train_loader(cfg_train_factory)
        valid_loader = self.build_valid_loader(cfg_valid_factory)

        cfg_solver = config["solver"]
        optimizer = self.build_optimizer(cfg_solver, model)
        lr_scheduler = self.build_lr_scheduler(cfg_solver, optimizer)
        
        super().__init__(model, train_loader, optimizer, config, valid_loader, lr_scheduler)

    @classmethod
    def build_model(cls, cfg):
        # turn off testing when in training mode
        cfg["model"]["args"]["testing"] = False

        model = build_model(cfg)
        model = create_ddp_model(model, broadcast_buffers=False)
        
        return model

    @classmethod
    def build_train_loader(cls, cfg_train_factory):
        dataloader = build_train_loader(cfg_train_factory)
        return dataloader

    @classmethod
    def build_valid_loader(cls, cfg_valid_factory):
        dataloader = build_test_loader(cfg_valid_factory)
        return dataloader

    @classmethod
    def build_optimizer(cls, cfg_solver, model):
        name = cfg_solver["optimizer"]["name"]
        args = cfg_solver["optimizer"]["args"]
        optimizer = build_optimizer(model, name, **args)
        return optimizer

    @classmethod
    def build_lr_scheduler(cls, cfg_solver, optimizer):
        cfg_lr_scheduler = cfg_solver["lr_scheduler"]
        lr_scheduler = build_lr_scheduler(
            optimizer, cfg_lr_scheduler["name"], **cfg_lr_scheduler["args"]) if cfg_lr_scheduler else None
        return lr_scheduler


def default_setup(config):
    # create the log dir and checkpoints saving dir if not exist
    mkdirs(config["trainer"]["ckp_dir"])
    mkdirs(config["trainer"]["log_dir"])

    rank = comm.get_rank()

    logger = setup_logger(config["trainer"]["log_dir"], rank, name=PROJECT_NAME)

    logger.info("Configuration:")
    logger.info(json.dumps(config, indent=4))

    if config["trainer"]["tensorboard"]:
        logger.info(
            f"Tensorboard is enabled, you can start tensorboard by:\n"
            f"\t$ tensorboard --logdir={config['trainer']['log_dir']} --port=8080 --host=0.0.0.0\n"
        )


def main(args):
    config = init_config(args)
    default_setup(config)
    
    trainer = DefaultTrainer(config)
    trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
