import torch 
import hydra
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
import dill

import zarr
import numpy

from diffusion_policy.dataset.real_image_dataset import real_data_to_replay_buffer

# # load checkpoint
# ckpt_path = "/home/x_zhzhu/RoboSaGA/experiments/robosaga/archive/green_pnp_real/diffusion_policy/baseline/checkpoints/epoch=0550-train_loss=0.004.ckpt"
# payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
# cfg = payload['cfg']
# cls = hydra.utils.get_class(cfg._target_)
# workspace = cls(cfg)
# workspace: BaseWorkspace
# workspace.load_payload(payload, exclude_keys=None, include_keys=None)

# # hacks for method-specific setup.
# action_offset = 0
# delta_action = False
# if 'diffusion' in cfg.name:
#     # diffusion model
#     policy: BaseImagePolicy
#     policy = workspace.model
#     if cfg.training.use_ema:
#         policy = workspace.ema_model

#     device = torch.device('cuda')
#     policy.eval().to(device)

#     # set inference params
#     policy.num_inference_steps = 16 # DDIM inference iterations
#     policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

# 
data_path = 'data/green_pnp/replay_buffer.zarr'
store = zarr.DirectoryStore(data_path)
src_root = zarr.group(store)
#
data = src_root['data']
meta = src_root['meta']

print(data.tree())