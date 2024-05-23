import concurrent.futures
import os
import re
import subprocess

import numpy as np

from robomimic.utils.eval_utils import get_top_n_experiments


def run_script(script_name, script_args):
    command = ["python", script_name] + script_args
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, errors = process.communicate()
    return script_name, script_args, output, errors


def run_scripts_in_parallel(scripts_with_args, output_file):

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_script, script_name, script_args)
            for script_name, script_args in scripts_with_args
        ]

    with open(output_file, "a") as f:
        for future in concurrent.futures.as_completed(futures):
            script_name, script_args, output, errors = future.result()
            f.write(
                f"Output from {script_name} with arguments {' '.join(script_args)}:\n{output}\n"
            )
            print(f"Output from {script_name} with arguments {' '.join(script_args)}:\n{output}\n")


def extract_success_rates(file_path):
    pattern = r'"Success_Rate": ([\d\.]+)'
    success_rates = []
    with open(file_path, "r") as file:
        content = file.read()
        matches = re.findall(pattern, content)
        for match in matches:
            success_rates.append(float(match))
    return success_rates


def get_results_string(output_file, top_n_success_rate, offdomain_type):
    success_rates = extract_success_rates(output_file)
    indomain_ssr_mean = np.around(np.mean(top_n_success_rate), 2)
    indomain_ssr_std = np.around(np.std(top_n_success_rate), 2)
    offdomain_ssr_mean = np.around(np.mean(success_rates), 2)
    offdomain_ssr_std = np.around(np.std(success_rates), 2)

    results_string = (
        f"\n===== Results for {args.exp_path} =====\n"
        f"Indomain: {top_n_success_rate}, {indomain_ssr_mean} +/- {indomain_ssr_std}\n"
        f"  data: {top_n_success_rate}\n"
        f"Off-domain for {offdomain_type}: {offdomain_ssr_mean} +/- {offdomain_ssr_std}\n"
        f"   data: {success_rates}\n"
    )

    return results_string


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_path", type=str, default=None, required=True)
    parser.add_argument("-m", "--mode", help="type of offdomain evaluation", required=True)
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--n_rollouts", type=int, default=50)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    assert args.mode in ["indoor", "outdoot", "textile", "distractors"], "Invalid mode"

    distractors = ["bottle", "lemon", "milk", "can"]

    log_file_path = os.path.join(args.exp_path, "logs/log.txt")
    eval_dir = os.path.join(args.exp_path, "eval")
    video_dir = os.path.join(eval_dir, "videos")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    top_n_checkpoints, top_n_success_rate = get_top_n_experiments(log_file_path, n=3)

    py_script = "robomimic/scripts/eval_trained_agent.py"
    scripts_with_args = []

    print("\n=====================")
    print(f"Running evaluation for {args.mode} with top {args.top_n} checkpoints")

    for i, ckpt_path in enumerate(top_n_checkpoints):
        ckpt_name = os.path.basename(ckpt_path).replace(".pth", "")
        video_name = f"{args.mode}_ckpt_{ckpt_name}.mp4"
        video_path = os.path.join(video_dir, video_name)
        video_command = ["--video_path", video_path] if args.video else []

        if args.mode in ["indoor", "outdoor", "textile"]:
            scripts_with_args.append(
                (
                    py_script,
                    [
                        "--agent",
                        ckpt_path,
                        "--n_rollouts",
                        str(args.n_rollouts),
                        "--texture_category",
                        args.mode,
                        "--env_id",
                        str(i),
                    ]
                    + video_command,
                )
            )
        else:
            scripts_with_args.append(
                (
                    py_script,
                    [
                        "--agent",
                        ckpt_path,
                        "--n_rollouts",
                        str(args.n_rollouts),
                        "--distractors",
                    ]
                    + distractors
                    + video_command,
                )
            )

    output_file = os.path.join(eval_dir, f"{args.mode}_stats.txt")

    # Execute each script with its arguments and save the output
    if (
        os.path.exists(output_file)
        and input(f"Output file {output_file} already exists. Overwrite? (y/n): ").lower() != "y"
    ):
        print("Computing Stats from existing output file")
    else:
        os.remove(output_file) if os.path.exists(output_file) else None
        run_scripts_in_parallel(scripts_with_args, output_file)

    stats = get_results_string(output_file, top_n_success_rate, args.mode)
    print(stats)

    with open(output_file, "w") as f:
        f.write(stats)
