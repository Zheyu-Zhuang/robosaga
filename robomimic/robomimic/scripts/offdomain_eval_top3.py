import os
import re
import subprocess

import numpy as np

from robomimic.utils.eval_utils import get_top_n_experiments


def run_script(script_name, script_args, output_file):
    """
    Runs a python script with arguments in argparse format and writes its output to a specified file.
    """
    with open(output_file, "a") as f:  # Append mode to add outputs sequentially
        # Create the command with script and its arguments
        command = ["python", script_name] + script_args

        # Execute the script and capture the output
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        # Get the output and errors
        output, errors = process.communicate()

        # Write the outputs to the file
        f.write(f"Output from {script_name} with arguments {' '.join(script_args)}:\n{output}\n")
        # if errors:
        #     f.write(
        #         f"Errors from {script_name} with arguments {' '.join(script_args)}:\n{errors}\n"
        #     )

        # Print the output and errors to the terminal
        print(f"Output from {script_name} with arguments {' '.join(script_args)}:\n{output}\n")


def extract_success_rates(file_path):
    # Pattern to match JSON blocks and extract Success_Rate
    pattern = r'"Success_Rate": ([\d\.]+)'

    success_rates = []

    with open(file_path, "r") as file:
        content = file.read()
        # Find all occurrences of the pattern
        matches = re.findall(pattern, content)

        # Collect all success rates
        for match in matches:
            success_rates.append(float(match))

    return success_rates


if __name__ == "__main__":

    import argparse

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

    log_file_path = os.path.join(args.exp_path, "logs/log.txt")
    top_n_checkpoints, top_n_success_rate = get_top_n_experiments(log_file_path, n=3)

    n_rollouts = args.n_rollouts
    texture = args.texture
    distractors = args.distractors
    commands = []
    py_script = "robomimic/scripts/eval_trained_agent.py"
    scripts_with_args = []
    if args.mode == "texture":
        print(f"Running evaluation for texture {texture}")
        for i, ckpt_path in enumerate(top_n_checkpoints):
            scripts_with_args.append(
                (
                    py_script,
                    [
                        "--agent",
                        ckpt_path,
                        "--n_rollouts",
                        str(n_rollouts),
                        "--texture",
                        texture,
                        '--env_id',
                        str(i),
                    ],
                )
            )

    elif args.mode == "distractors":
        print(f"Running evaluation for distractors {distractors}")
        for ckpt_path in top_n_checkpoints:
            ckpt_name = os.path.basename(ckpt_path).replace(".ckpt", "")
            distractor_str = " ".join(distractors)
            scripts_with_args.append(
                (
                    py_script,
                    [
                        "--agent",
                        ckpt_path,
                        "--n_rollouts",
                        str(n_rollouts),
                        "--distractors",
                    ]
                    + distractors,
                )
            )

    if args.mode == "texture":
        output_file = os.path.join(args.exp_path, f"logs/eval_results_{texture}.txt")
    else:
        output_file = os.path.join(args.exp_path, f"logs/eval_results_{distractors}.txt")

    # Execute each script with its arguments and save the output
    if os.path.exists(output_file):
        overwrite = input(f"Output file {output_file} already exists. Overwrite? (y/n): ")
        if overwrite.lower() == "n":
            print("Computing Stats from existing output file")
        else:
            print("Rerun Evaluation")
            os.remove(output_file)
            for script_name, script_args in scripts_with_args:
                print(f"Running script {script_name} with arguments {script_args}")
                run_script(script_name, script_args, output_file)
    else:
        for script_name, script_args in scripts_with_args:
            print(f"Running script {script_name} with arguments {script_args}")
            run_script(script_name, script_args, output_file)

    # Extract success rates from the output file
    success_rates = extract_success_rates(output_file)

    print(f"\n===== Results for {args.exp_path} =====\n")
    indomain_ssr_mean = np.around(np.mean(top_n_success_rate), 2)
    indomain_ssr_std = np.around(np.std(top_n_success_rate), 2)
    print(f"Indomain: {top_n_success_rate}, {indomain_ssr_mean} +/- {indomain_ssr_std}")
    offdomain_ssr_mean = np.around(np.mean(success_rates), 2)
    offdomain_ssr_std = np.around(np.std(success_rates), 2)
    if args.mode == "texture":
        print(
            f"Off-domain for {texture}: {success_rates}, {offdomain_ssr_mean} +/- {offdomain_ssr_std}"
        )
    else:
        print(
            f"Off-domain for {distractors}: {success_rates}, {offdomain_ssr_mean} +/- {offdomain_ssr_std}"
        )
