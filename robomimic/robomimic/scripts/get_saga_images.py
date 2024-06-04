"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes    
"""

import argparse
import json
import os
import shutil
import socket
import sys
import time
import traceback
from collections import OrderedDict

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

import diffusion_policy.model.vision.crop_randomizer as dmvc
import robomimic.models.base_nets as rmbn
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from diffusion_policy.common.pytorch_util import replace_submodules
from robomimic.algo import RolloutPolicy, algo_factory
from robomimic.config import config_factory
from robomimic.utils.log_utils import DataLogger, PrintLogger
from robosaga.saliency_guided_augmentation import SaliencyGuidedAugmentation


def train(device):
    """
    Train a model using the algorithm.
    """
    ckpt = getattr(args, "ckpt", None)
    # first set seeds

    if ckpt is not None:
        model, _, config = FileUtils.resume_from_checkpoint(
            ckpt_path=ckpt,
            device=device,
        )
    else:
        exit("No checkpoint provided to resume from! Exiting...")

    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    # log_dir, ckpt_dir, video_dir, saliency_dir = TrainUtils.get_exp_dir(config)

    # if config.experiment.logging.terminal_output_to_txt:
    #     # log stdout and stderr to a text file
    #     logger = PrintLogger(os.path.join(log_dir, "log.txt"))
    #     sys.stdout = logger
    #     sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data, all_obs_keys=config.all_obs_keys, verbose=True
    )

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environment
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        # create environments for validation runs
        env_names = [env_meta["env_name"]]

        if config.experiment.additional_envs is not None:
            for name in config.experiment.additional_envs:
                env_names.append(name)

        for env_name in env_names:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
            )
            envs[env.name] = env
            print(envs[env.name])

    print("")

    replace_submodules(
        root_module=model.nets["policy"].nets["encoder"].nets["obs"],
        predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
        func=lambda x: dmvc.CropRandomizer(
            input_shape=x.input_shape,
            crop_height=x.crop_height,
            crop_width=x.crop_width,
            num_crops=x.num_crops,
            pos_enc=x.pos_enc,
        ),
    )

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"]
    )
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
    )
    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps

    saga = None
    if "saliency" in config and config.saliency.enabled:
        config.unlock()
        config.saliency.buffer_depth = len(trainset)
        config.saliency.warmup_epochs = 0
        config.saliency.disable_buffer = True
        config.saliency.update_ratio = 1
        config.saliency.aug_ratio = 1
        config.saliency.vis_out_dir = f"../saliency_images/{args.task}"
        saga = SaliencyGuidedAugmentation(model.nets["policy"], **config.saliency)
        saga.unregister_hooks()

    for epoch in range(1, config.train.num_epochs + 1):  # epoch numbers start at 1
        model.nets["policy"].disable_low_noise = False
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            saga=saga,
        )


def main(args):

    # get torch device
    device = TorchUtils.get_torch_device("cuda")

    try:
        train(device=device)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="(optional) path to a model checkpoint to resume training from",
    )
    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="set this flag to run a quick training run for debugging purposes",
    )

    args = parser.parse_args()
    main(args)
