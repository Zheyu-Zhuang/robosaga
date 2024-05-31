import torch 
import hydra
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
import dill

import zarr
import numpy

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
dataset = zarr.open(data_path, mode='w')

data = dataset.create_group('data')
meta = dataset.create_group('meta')

#
print(meta.keys())

episode_ends = meta['episode_ends'][:]


###
channel = 'ee_rgb_camera'
print(f"Creating {channel} array")
data[channel] = data.create_dataset(
    name=channel,
    shape=(episode_ends[-1],) + obs_dict[channel].shape[1:],
    dtype=obs_dict[channel].numpy().dtype,
    # Chunk the data by episode
    chunks=(int(episode_ends[-1] / len(episode_ends)),) + obs_dict[channel].shape[1:]
)

channel = 'front_rgb_camera'
print(f"Creating {channel} array")
data[channel] = data.create_dataset(
    name=channel,
    shape=(episode_ends[-1],) + obs_dict[channel].shape[1:],
    dtype=obs_dict[channel].numpy().dtype,
    # Chunk the data by episode
    chunks=(int(episode_ends[-1] / len(episode_ends)),) + obs_dict[channel].shape[1:]
)

channel = 'robot_state'
print(f"Creating {channel} array")
data[channel] = data.create_dataset(
    name=channel,
    shape=(episode_ends[-1],) + obs_dict[channel].shape[1:],
    dtype=obs_dict[channel].numpy().dtype,
    # Chunk the data by episode
    chunks=(int(episode_ends[-1] / len(episode_ends)),) + obs_dict[channel].shape[1:]
)
    
# Action
channel = 'ee_vel'
data['action'] = data.create_dataset(
    name='action',
    shape=(episode_ends[-1], 7), # Add 7 for the gripper
    dtype=action_dict[channel].numpy().dtype,
    chunks=(int(episode_ends[-1] / len(episode_ends)),) + action_dict[channel].shape[1:]
)
    
# Create the meta arrays
meta['episode_ends'] = meta.create_dataset(
    name='episode_ends',
    shape=(len(episode_ends),),
    dtype=np.array(episode_ends).dtype,
    chunks=(1,),
)


