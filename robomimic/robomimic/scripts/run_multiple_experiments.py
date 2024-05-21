import os

import numpy as np

experiment_path = "../experiments/robosaga/robomimic/bc_rnn/bc_rnn_square_ph_saga_mixup_0.5_aug_0.5_clip/20240520204616/models"

checkpoints = {
    "model_epoch_80_NutAssemblySquare_success_0.86.pth": 0.86,
    "model_epoch_280.pth": 0.84,
    "model_epoch_540.pth": 0.84,
}

checkpoint_paths = [os.path.join(experiment_path, ckpt) for ckpt in checkpoints.keys()]


def get_average_success_states(checkpoints):
    success_rate = [checkpoints[ckpt] for ckpt in checkpoints]
    avg = np.mean(success_rate)
    std = np.std(success_rate)
    print(f"Average success rate: {avg:.2f} +/- {std:.2f}")


get_average_success_states(checkpoints)

n_rollouts = 50
texture = "textile"

commands = []
for ckpt_path in checkpoint_paths:
    commands.append(
        f"python robomimic/scripts/eval_trained_agent.py --n_rollout 50 --agent {ckpt_path} --n_rollout {n_rollouts}  --texture_category {texture}"
    )

terminal_commands = " & ".join(commands)

os.system(terminal_commands)
