import argparse
import os

import numpy as np


def find_checkpoints(exp_path, top_n=3):
    checkpoint_path = os.path.join(exp_path, "checkpoints")
    all_files = os.listdir(checkpoint_path)
    assert len(all_files) > 0, f"No checkpoints found in {checkpoint_path}"
    assert (
        len(all_files) >= top_n
    ), f"Only {len(all_files)} checkpoints found, but {top_n} requested"
    if "latest.ckpt" in all_files:
        all_files.remove("latest.ckpt")
    rollout_scores = [x.split("=")[-1].replace(".ckpt", "") for x in all_files]
    rollout_scores = [float(x) for x in rollout_scores]
    print(all_files)
    print(np.argsort(rollout_scores)[-top_n:])
    top_n_indices = np.argsort(rollout_scores)[-top_n:]
    top_n_files = [all_files[i] for i in top_n_indices]
    top_n_scores = [rollout_scores[i] for i in top_n_indices]
    return [os.path.join(checkpoint_path, f) for f in top_n_files], top_n_scores


def get_average_success_states(success_rate):
    avg = np.mean(success_rate)
    std = np.std(success_rate)
    print(f"Average success rate: {avg:.2f} +/- {std:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_path", type=str, default=None, required=True)
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--texture", type=str)
    parser.add_argument("--n_rollouts", type=int, default=50)
    parser.add_argument(
        "--distractors", type=str, nargs="+", default=["bottle", "lemon", "milk", "can"]
    )
    parser.add_argument(
        "-m", "--mode", type=str, default="texture", choices=["texture", "distractors"]
    )
    args = parser.parse_args()

    checkpoint_paths, top_n_scores = find_checkpoints(args.exp_path)

    get_average_success_states(top_n_scores)

    n_rollouts = args.n_rollouts
    texture = args.texture
    distractors = args.distractors
    commands = []
    if args.mode == "texture":
        print(f"Running evaluation for texture {texture}")
        for ckpt_path in checkpoint_paths:
            ckpt_name = os.path.basename(ckpt_path).replace(".ckpt", "")
            eval_dir = os.path.join(args.exp_path, "eval", texture, ckpt_name)
            commands.append(
                f"python eval.py --checkpoint {ckpt_path}  --output_dir {eval_dir} --rand_texture {texture}"
            )
    elif args.mode == "distractors":
        print(f"Running evaluation for distractors {distractors}")
        for ckpt_path in checkpoint_paths:
            ckpt_name = os.path.basename(ckpt_path).replace(".ckpt", "")
            distractor_str = "_".join(distractors)
            eval_dir = os.path.join(args.exp_path, "eval", distractor_str, ckpt_name)
            command_ = f"python eval.py --checkpoint {ckpt_path}  --output_dir {eval_dir}"
            for distractor in distractors:
                command_ += f" --distractors {distractor}"
            commands.append(command_)
    terminal_commands = " && ".join(commands)
    os.system(terminal_commands)
