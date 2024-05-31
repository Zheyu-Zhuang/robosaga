import torch 
import hydra
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
import dill

import zarr
import numpy as np

from diffusion_policy.dataset.real_image_dataset import real_data_to_replay_buffer

# load checkpoint
ckpt_path = "/home/x_zhzhu/RoboSaGA/experiments/robosaga/archive/green_pnp_real/diffusion_policy/baseline/checkpoints/epoch=0550-train_loss=0.004.ckpt"
payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
cfg = payload['cfg']
cls = hydra.utils.get_class(cfg._target_)
workspace = cls(cfg)
workspace: BaseWorkspace
workspace.load_payload(payload, exclude_keys=None, include_keys=None)

# hacks for method-specific setup.
action_offset = 0
delta_action = False
if 'diffusion' in cfg.name:
    # diffusion model
    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device('cuda')
    policy.eval().to(device)

    # set inference params
    policy.num_inference_steps = 32 # DDIM inference iterations
    # print( policy.horizon, policy.n_obs_steps)
    # policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

# 
data_path = 'data/green_pnp.zarr'
store = zarr.DirectoryStore(data_path)
src_root = zarr.group(store=store, overwrite=False)
#
data = src_root['data']
meta = src_root['meta']

# #
# data
#  ├── action (24703, 7) float32
#  ├── ee_rgb_camera (24703, 84, 84, 3) float32
#  ├── front_rgb_camera (24703, 3, 84, 84) float32
#  └── robot_state (24703, 16) float32

ee_rgb_camera = data['ee_rgb_camera']
front_rgb_camera = data['front_rgb_camera']
robot_state = data['robot_state']
action = data['action']

#
episode_slices = meta['episode_ends']

def get_episode(idx):
    if idx == 0:
        start = 0
    else:
        start = episode_slices[idx-1]
    end = episode_slices[idx]
    ee_rgb_camera_ = np.array(ee_rgb_camera[start:end]).transpose(0, 3, 1, 2)/255.0
    front_rgb_camera_ = np.array(front_rgb_camera[start:end])/255.0
    robot_state_ = np.array(robot_state[start:end])
    action_ = np.array(action[start:end])
    return ee_rgb_camera_, front_rgb_camera_, robot_state_, action_


def get_data(idx, obs_len = 2, pred_len = 8):
    i_th_episode = get_episode(idx)
    ee_rgb_camera_, front_rgb_camera_, robot_state_, action_ = i_th_episode
    rand_start = np.random.randint(0, len(ee_rgb_camera_)-obs_len-pred_len)
    
    ground_truth = torch.tensor(action_[rand_start+obs_len:rand_start+obs_len+pred_len]).unsqueeze(0).to('cuda')
    obs_dict =   {
        'ee_rgb_camera': torch.tensor(ee_rgb_camera_[rand_start:rand_start+obs_len]).unsqueeze(0).to('cuda'),
        'front_rgb_camera': torch.tensor(front_rgb_camera_[rand_start:rand_start+obs_len]).unsqueeze(0).to('cuda'),
        'robot_state': torch.tensor(robot_state_[rand_start:rand_start+obs_len]).unsqueeze(0).to('cuda')
    }
    return obs_dict, ground_truth
        
        
episode_idx = 0

obs_dict, ground_truth = get_data(episode_idx)
pred = policy.predict_action(obs_dict)['action']

pred = pred.squeeze(0).cpu().detach().numpy()


ee_vel_pred = pred[:, :, :6]
gripper_width_pred = pred[:, :, 6]

ee_vel_gt = ground_truth[:, :, :6]
gripper_width_gt = ground_truth[:, :, 6]

import matplotlib.pyplot as plt

for i in range(6):
    plt.plot(ee_vel_pred[:, :, i].cpu().numpy(), label=f'ee_vel_pred_{i}')
    plt.plot(ee_vel_gt[:, :, i].cpu().numpy(), label=f'ee_vel_gt_{i}')

# plot the actions
plt.plot(lin_x.cpu().numpy(), label='linear x')
plt.plot(lin_y.cpu().numpy(), label='linear y')
plt.plot(lin_z.cpu().numpy(), label='linear z')

plt.plot(gripper.cpu().numpy(), label='gripper')

# add legend
plt.legend()