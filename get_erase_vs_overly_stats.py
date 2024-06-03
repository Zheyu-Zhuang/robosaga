import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import pyperclip


def parse_stats_from_text(text, file_path=None):
    """
    Parses the text to extract mean, standard deviation, and raw data for "Indomain" and various "Off-domain"   policies.
    Data for each category is spread across multiple lines.

    Parameters:
    - text (str): Multi-line string containing the data.

    Returns:
    - dict: A dictionary containing the parsed data for each category.
    """
    stats = {}
    current_domain = None

    # Splitting the input text into lines
    lines = text.split("\n")

    # Regex for matching "Indomain" with its complex multiline format
    indomain_regex = re.compile(r"^Indomain:\s*\[(.*?)\],\s*(\d+\.\d+)\s*\+/\-\s*(\d+\.\d+)")
    # Regex for matching "Off-domain" with mean and std in the same line
    off_domain_regex = re.compile(r"^Off-domain for ([\w\s]+):\s*(\d+\.\d+)\s*\+/\-\s*(\d+\.\d+)")
    # Regex for matching the raw data line
    raw_data_regex = re.compile(r"^\s*data:\s*\[(.*?)\]")

    for line in lines:
        # Check for "Indomain" summary statistics
        indomain_match = indomain_regex.match(line.strip())
        if indomain_match:
            current_domain = "indomain"
            try:
                raw_data = [float(x) for x in indomain_match.group(1).split(",")]
            except ValueError:
                print(f"Error parsing raw data from {file_path}")
                raw_data = []
            mean = float(indomain_match.group(2))
            std = float(indomain_match.group(3))
            stats[current_domain] = {"mean": mean, "std": std, "raw_data": raw_data}

        # Check for "Off-domain" summary statistics
        off_domain_match = off_domain_regex.match(line.strip())
        if off_domain_match:
            current_domain = "offdomain"
            offdomain_type = off_domain_match.group(1)
            mean = float(off_domain_match.group(2))
            std = float(off_domain_match.group(3))
            stats[current_domain] = {
                "mean": mean,
                "std": std,
                "raw_data": [],
                "type": offdomain_type,
            }

        # Check for raw data
        data_match = raw_data_regex.match(line.strip())
        if data_match and current_domain:
            # Convert string data to floats and store in the current domain's data
            try:
                raw_data = [float(x) for x in data_match.group(1).split(",")]
            except ValueError:
                print(f"Error parsing raw data from {file_path}")
                raw_data = []
            if len(raw_data) <= 2:
                print(f"Warning: Raw data has less than 3 elements in {file_path}")
            stats[current_domain]["raw_data"] = raw_data

    return stats


def format_mean_std(
    mean,
    std,
    include_std=True,
    bold=False,
    precision=2,
):
    """
    Formats mean and standard deviation into a single string.
    """
    if mean is None:
        return ""

    line = f"{mean:.{precision}f}"
    if bold:
        line = "\\mathbf{" + line + "}"
    if include_std and std is not None:
        line += f"_{{\\pm {std:.{precision}f}}}"

    return "$" + line + "$"


def process_all_experiments(base_path):
    tasks = ["can", "lift", "square"]
    task_dict = {}
    for task in tasks:
        task_path = os.path.join(base_path, f"{task}_image")
        if os.path.exists(task_path):
            results = process_directory(task_path)
            stats = further_process_results(results)
            task_dict[task] = stats
    return task_dict


def process_directory(base_path):
    policies = ["bc"]
    aug_methods = ["overlay", "saga", "saga_default", "erase_0.5"]
    experiments = ["shuffle_env", "distractors", "vanilla"]
    results = defaultdict(dict)
    for policy in policies:
        for aug_method in aug_methods:
            sub_path = os.path.join(base_path, policy, aug_method)
            if os.path.exists(sub_path):
                for root, dirs, files in os.walk(sub_path):
                    if "eval" in dirs:
                        eval_path = os.path.join(root, "eval")
                        eval_results = defaultdict(dict)
                        for experiment in experiments:
                            matched_files = [
                                f_
                                for f_ in os.listdir(eval_path)
                                if experiment.lower() in f_.lower() and f_.endswith(".txt")
                            ]
                            for f_ in matched_files:
                                file_path = os.path.join(eval_path, f_)
                                with open(file_path, "r") as f:
                                    text = f.read()
                                    data = parse_stats_from_text(text, file_path)
                                    if "indomain" in data:
                                        eval_results["indomain"] = data["indomain"]["raw_data"]
                                    if "offdomain" in data:
                                        eval_results[f"{data['offdomain']['type']}"] = data[
                                            "offdomain"
                                        ]["raw_data"]
                        aug_method = aug_method.replace("_default", "")
                        if aug_method == "baseline":
                            aug_method = "no_aug"
                        if "vanilla" in eval_results and eval_results["vanilla"] != {}:
                            eval_results["indomain"] = eval_results["vanilla"]
                            eval_results.pop("vanilla")
                        results[policy][aug_method] = eval_results
    return results


def further_process_results(exp_data):
    exp_stats = defaultdict(lambda: defaultdict(dict))
    for policy in exp_data.keys():
        highscores = defaultdict(float)
        for aug_method, results in exp_data[policy].items():
            indomain = []
            average = []
            for experiment in results.keys():
                if experiment == "indomain":
                    indomain.append(results[experiment])
                    continue
                else:
                    average += results[experiment]

                exp_mean = np.around(np.mean(results[experiment]), 2)
                if exp_mean > highscores[experiment]:
                    highscores[experiment] = exp_mean
                if (
                    results[experiment] == "No data"
                    or len(results[experiment]) == 0
                    or results[experiment] is None
                ):
                    exp_stats[policy][aug_method][experiment] = [None, None]
                    continue
                exp_std = np.around(np.std(results[experiment]), 2)
                exp_stats[policy][aug_method][experiment] = [exp_mean, exp_std]
            if len(indomain) == 0 or "No data" in indomain:
                exp_stats[policy][aug_method]["indomain"] = [None, None]
            else:
                exp_stats[policy][aug_method]["indomain"] = [
                    np.around(np.mean(indomain), 2),
                    np.around(np.std(indomain), 2),
                ]
            if len(average) == 0 or "No data" in average:
                exp_stats[policy][aug_method]["average"] = [None, None]
            else:
                exp_stats[policy][aug_method]["average"] = [
                    np.around(np.mean(average), 2),
                    np.around(np.std(average), 2),
                ]
            if (
                exp_stats[policy][aug_method]["indomain"][0] is not None
                and exp_stats[policy][aug_method]["indomain"][0] > highscores["indomain"]
            ):
                highscores["indomain"] = exp_stats[policy][aug_method]["indomain"][0]
            if (
                exp_stats[policy][aug_method]["average"][0] is not None
                and exp_stats[policy][aug_method]["average"][0] > highscores["average"]
            ):
                highscores["average"] = exp_stats[policy][aug_method]["average"][0]
        exp_stats[policy]["highscores"] = highscores

    return exp_stats


def print_latex_tables(stats_dict, print_std=False):
    output = []

    block_start = """\\begin{table}[ht]
\\scriptsize
    \\centering
    \\setlength{\\tabcolsep}{3pt} % Adjust column spacing
    \\centering
    \\begin{tabular}{l ccc ccc ccc}
    \\toprule
        & \\multicolumn{3}{c}{\\textbf{Lift}} & \\multicolumn{3}{c}{\\textbf{Can}} & \\multicolumn{3}{c}{\\textbf{Square}} \\\\
        \\cmidrule(lr){2-4}  \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}
        & \\textit{Sa-Erase} & \\textit{Over.} & \\textit{SaGA} & \\textit{Sa-Erase} & \\textit{Over.} & \\textit{SaGA} & \\textit{Sa-Erase} & \\textit{Over} & \\textit{SaGA} \\\\
    \\midrule
"""

    output.append(block_start)

    tasks = ["lift", "can", "square"]
    block_title = {
        "In-domain": "indomain",
        "Texture": "shuffle_env",
        "Distractor": "distractors",
        "mean": "average",
    }
    evaluation_contexts = ["In-domain", "Texture", "Distractor", "mean"]
    methods = ["erase_0.5", "overlay", "saga"]

    for domain in evaluation_contexts:
        line = []
        line.append("\\textbf{" + domain + "} & ")
        for task in tasks:
            for method in methods:
                try:
                    mean, std = stats_dict[task]["bc"][method][block_title[domain]]
                except KeyError:
                    mean, std = None, None
                bold = mean == stats_dict[task]["bc"]["highscores"][block_title[domain]]
                line.append(format_mean_std(mean, std, include_std=print_std, bold=bold))
                line.append(" & ")
        line.pop(-1)
        if domain == "In-domain":
            line.append(" \\\\ \\midrule")
        else:
            line.append(" \\\\")
        output.append("".join(line))
    output.append("\\bottomrule")
    output.append("\\end{tabular}")
    output.append("\\end{table}")

    return "\n".join(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_path", type=str, default=None, required=True)
    parser.add_argument("--format", type=str, default="latex", choices=["markdown", "latex"])

    args = parser.parse_args()
    assert os.path.exists(args.task_path), "Task path does not exist."

    stats = process_all_experiments(args.task_path)

    # print(stats)
    if args.format == "latex":
        # Call the function and store its output
        output = print_latex_tables(stats, print_std=True)
        # Copy the output to the clipboard
        pyperclip.copy(output)
