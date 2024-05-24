import concurrent.futures
import os
import re
import subprocess

import numpy as np

import shutil

import json 

def get_top_n_experiments(exp_path, top_n=3):
    checkpoint_path = os.path.join(exp_path, "checkpoints")
    all_files = os.listdir(checkpoint_path)
    assert len(all_files) > 0, f"No checkpoints found in {checkpoint_path}"
    assert (
        len(all_files) >= top_n
    ), f"Only {len(all_files)} checkpoints found, but {top_n} requested"
    if "latest.ckpt" in all_files:
        all_files.remove("latest.ckpt")
        
    epoch_idx = [int(re.findall(r'\d+', x)[0]) for x in all_files]
    rollout_scores = [x.split("=")[-1].replace(".ckpt", "") for x in all_files]
    rollout_scores = [float(x) for x in rollout_scores]
    # Sort the files based on the rollout scores, get the later epoch if the scores are the same
    # Pair up the epoch indices and rollout scores
    paired = list(zip(epoch_idx, rollout_scores, all_files))

    # Sort the pairs: first by rollout score (descending), then by epoch index (descending)
    sorted_pairs = sorted(paired, key=lambda x: (x[1], x[0]), reverse=True)

    # Extract the sorted files
    sorted_files = [x[2] for x in sorted_pairs]
    top_n_files = sorted_files[:top_n]
    top_n_scores = [x[1] for x in sorted_pairs[:top_n]]
    
    return [os.path.join(checkpoint_path, f) for f in top_n_files], top_n_scores


def get_average_success_states(success_rate):
    avg = np.mean(success_rate)
    std = np.std(success_rate)
    print(f"Average success rate: {avg:.2f} +/- {std:.2f}")
    
    
def run_script(script_name, script_args):
    command = ["python", script_name] + script_args
    # print(f"Running {' '.join(command)}")
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
        # if errors:
        #     f.write(f"Errors: {errors}\n")
        #     print(f"Errors: {errors}\n")

def run_scripts_sequentially(scripts_with_args, output_file):
    with open(output_file, "a") as f:
        for script_name, script_args in scripts_with_args:
            output, errors = run_script(script_name, script_args)
            f.write(
                f"Output from {script_name} with arguments {' '.join(script_args)}:\n{output}\n"
            )
            print(f"Output from {script_name} with arguments {' '.join(script_args)}:\n{output}\n")
            # if errors:
            #     f.write(f"Errors: {errors}\n")
            #     print(f"Errors: {errors}\n")

def extract_success_rates(file_path):

    train_mean_scores = []
    test_mean_scores = []

    with open(file_path, 'r') as file:
        for line in file:
            if 'mean_score' in line and 'mean_socre=' not in line:
                try:
                    # Convert the string to a dictionary
                    dict_line = json.loads(line.replace("'", '"'))

                    # Extract the mean_score
                    train_mean_score = dict_line.get('train/mean_score', None)
                    test_mean_score = dict_line.get('test/mean_score', None)

                    # Append the scores to the lists
                    if train_mean_score is not None:
                        train_mean_scores.append(train_mean_score)
                    if test_mean_score is not None:
                        test_mean_scores.append(test_mean_score)
                except json.JSONDecodeError:
                    continue
    return test_mean_scores

            
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
    args = parser.parse_args()
    assert args.mode in ["indoor", "outdoor", "textile", "distractors"], "Invalid mode"

    distractors = ["bottle", "lemon", "milk", "can"]
    distractor_command = []
    for distractor in distractors:
        distractor_command += [f"--distractors", distractor]
    # log_file_path = os.path.join(args.exp_path, "logs/log.txt")
    eval_dir = os.path.join(args.exp_path, "eval")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    top_n_checkpoints, top_n_success_rate = get_top_n_experiments(args.exp_path, top_n=3)
    
    py_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), "eval.py")
    scripts_with_args = []

    print("\n=====================")
    print(f"Running evaluation for {args.mode} with top {args.top_n} checkpoints")

    for i, ckpt_path in enumerate(top_n_checkpoints):
        ckpt_name = os.path.basename(ckpt_path).replace(".pth", "")
        eval_dir_ckpt = os.path.join(args.exp_path, "eval", args.mode, ckpt_name)
        if args.mode in ["indoor", "outdoor", "textile"]:
            scripts_with_args.append(
                (
                    py_script,
                    [
                        "--checkpoint",
                        ckpt_path,
                        "--rand_texture",
                        args.mode,
                        "--output_dir",
                        eval_dir_ckpt,
                    ],
                )
            )
        else:
            scripts_with_args.append(
                (
                    py_script,
                    [
                        "--checkpoint",
                        ckpt_path,
                        "--output_dir",
                        eval_dir_ckpt,
                    ]
                    + distractor_command
                )
            )

    output_file = os.path.join(eval_dir, f"{args.mode}_stats.txt")
    # Execute each script with its arguments and save the output
    if os.path.exists(output_file):
        archive_folder = os.path.join(eval_dir, "archive")
        if not os.path.exists(archive_folder):
            os.makedirs(archive_folder)
        n_old = len(os.listdir(archive_folder))
        shutil.move(output_file, os.path.join(archive_folder, f"{args.mode}_stats_{n_old}.txt"))
        print('WARNING: output file already exists, moving to archive folder')
    run_scripts_in_parallel(scripts_with_args, output_file)


    stats = get_results_string(output_file, top_n_success_rate, args.mode)
    print(stats)

    with open(output_file, "a") as f:
        f.write(stats)
    