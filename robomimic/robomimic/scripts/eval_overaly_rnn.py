import os

import numpy as np

experiment_path = (
    "../experiments/robosaga/robomimic/bc_rnn_square_ph_image_simple_overlay/20240516183622/models"
)
print(experiment_path)


checkpoints = {
    "100": 0.82,
    "400": 0.82,
    "260": 0.74,
}


def find_checkpoint_by_name(checkpoints, experiment_path):
    checkpoints_ = []
    all_files = os.listdir(experiment_path)
    for epoch in checkpoints.keys():
        for file in all_files:
            if f"epoch_{epoch}" in file:
                checkpoints_.append(file)
                break

    return checkpoints_


checkpoint_paths = find_checkpoint_by_name(checkpoints, experiment_path)
checkpoint_paths = [os.path.join(experiment_path, ckpt) for ckpt in checkpoint_paths]


def get_average_success_states(checkpoints):
    success_rate = [checkpoints[ckpt] for ckpt in checkpoints]
    avg = np.mean(success_rate)
    std = np.std(success_rate)
    print(f"Average success rate: {avg:.2f} +/- {std:.2f}")


get_average_success_states(checkpoints)

n_rollouts = 50
texture = "null"
distractors = "bottle lemon milk can"

commands = []
for ckpt_path in checkpoint_paths:
    commands.append(
        f"python robomimic/scripts/eval_trained_agent.py --n_rollout 50 --agent {ckpt_path} --n_rollout {n_rollouts}  --distractors {distractors} --video_path ../{ckpt_path.split('/')[-1].split('.')[0]}.mp4"
    )

terminal_commands = " & ".join(commands)

os.system(terminal_commands)
