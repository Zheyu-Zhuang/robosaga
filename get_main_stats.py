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
            if len(raw_data) != 3:
                print(
                    f"Warning: Raw data needs three data points. Found {len(raw_data)} in {file_path}"
                )
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
    tasks = ["can", "lift", "square", "transport"]
    task_dict = {}
    methods = ["no_aug", "overlay", "soda", "saga"]
    method_dict = {method: [] for method in methods}
    for task in tasks:
        task_path = os.path.join(base_path, f"{task}_image")
        if os.path.exists(task_path):
            results = process_directory(task_path)
            stats, raw_data = further_process_results(results)
            task_dict[task] = stats
            for method in methods:
                method_dict[method] += raw_data[method]
    for key, value in method_dict.items():
        method_dict[key] = f"{np.around(np.mean(value), 3)} +/- {np.around(np.std(value), 3)}"
    print(method_dict)
    return task_dict


def process_directory(base_path):
    policies = ["bc", "bc_rnn", "diffusion_policy"]
    aug_methods = ["baseline", "overlay", "soda", "soda_default", "saga", "saga_default"]
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
            raw_data = []
            for experiment in results.keys():
                if experiment == "indomain":
                    indomain.append(results[experiment])
                    continue
                else:
                    raw_data += results[experiment]

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
            exp_stats[policy][aug_method]["raw_data"] = raw_data
            if len(raw_data) == 0 or "No data" in raw_data:
                exp_stats[policy][aug_method]["average"] = [None, None]
            else:
                exp_stats[policy][aug_method]["average"] = [
                    np.around(np.mean(raw_data), 2),
                    np.around(np.std(raw_data), 2),
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

    methods = ["no_aug", "overlay", "soda", "saga"]
    raw_success = {method: [] for method in methods}
    for policy in exp_stats.keys():
        for method in methods:
            try:
                raw_success[method] += exp_stats[policy][method]["raw_data"]
            except KeyError:
                print(f"Error: {policy} {method}")
    return exp_stats, raw_success


def print_latex_tables(stats_dict, policies, print_std=True):
    output = []
    policy_titles = {"bc": "BC", "bc_rnn": "BC-RNN", "diffusion_policy": "Diffusion Policy"}
    methods = ["no_aug", "overlay", "soda", "saga"]

    # Dynamically generate the table header
    header = "\\toprule\n"
    header += (
        "\\textit{Policy} & & "
        + " & ".join(
            ["\\multicolumn{4}{c}{\\textbf{" + policy_titles[policy] + "}}" for policy in policies]
        )
        + " \\\\\n"
    )
    header += " ".join(
        [
            "\\cmidrule(lr){" + str(3 + 4 * i) + "-" + str(6 + 4 * i) + "}"
            for i in range(len(policies))
        ]
    )
    header += (
        "\\textit{Method} & & "
        + " & ".join(
            [
                "\\textit{None} & \\textit{Overlay} & \\textit{SODA} & \\textit{SaGA}"
                for _ in policies
            ]
        )
        + " \\\\\n"
    )
    header += "\\midrule\n"
    header += (
        "\\textit{Task} & \\textit{Domain} " + " ".join(["&&&&" for _ in policies]) + " \\\\\n"
    )
    header += "\\midrule\n"

    block_start = (
        """\\begin{table}[ht]
    \\scriptsize
    \\centering
    \\setlength{\\tabcolsep}{4pt} % Adjust column spacing
        \\centering
        \\begin{tabular}{cl """
        + " ".join(["cccc" for _ in policies])
        + """}
    """
    )

    output.append(block_start)
    output.append(header)

    tasks = ["lift", "can", "square", "transport"]
    block_title = {
        "In-domain": "indomain",
        "Texture": "shuffle_env",
        "Distractor": "distractors",
        "mean": "average",
    }
    evaluation_contexts = ["In-domain", "Texture", "Distractor", "mean"]

    def print_task_block(task_stats, task):
        is_first_line = True
        for title in evaluation_contexts:
            if is_first_line:
                line = f"\\multirow{{4}}{{*}}{{\\rotatebox{{90}}{{{task.capitalize()}}}}}"
                is_first_line = False
            else:
                line = ""
            if title == "mean":
                line += f" & \multicolumn{{1}}{{r}}{{\\textit{{{title}}}}}"
            else:
                line += f" & \\textit{{{title}}} "
            for policy in policies:
                for method in methods:
                    try:
                        mean = task_stats[policy][method][block_title[title]][0]
                        std = task_stats[policy][method][block_title[title]][1]
                        std = std if print_std else None
                        bold = mean == task_stats[policy]["highscores"][block_title[title]]
                        line += " & " + format_mean_std(
                            mean, std, include_std=print_std, bold=bold
                        )

                    except KeyError:
                        line += " & "
            output.append(line + " \\\\")
            if title == "In-domain":
                output.append("\\cmidrule{2-" + str(2 + 4 * len(policies)) + "}")

    for i, task in enumerate(tasks):
        print_task_block(stats_dict[task], task)
        if i < len(tasks) - 1:
            output.append("\\midrule")

    output.append("\\bottomrule")
    output.append("\\end{tabular}")
    caption = """\\vspace{1mm}
\\caption{\\textbf{Off-domain performance of Random Overlay, SODA and RoboSaGA.} Each is evaluated with BC-MLP, BC-RNN, Diffusion Policy across four simulated tasks against distractors and background variations.\\textit{\\textcolor{red}{Full table with standard deviations are included in the Appendix.}}}"""
    output.append(caption)
    output.append("\\end{table}")
    output.append("\\label{tab:main_results}")
    return "\n".join(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_path", type=str, default=None, required=True)
    parser.add_argument("--format", type=str, default="latex", choices=["markdown", "latex"])

    args = parser.parse_args()
    assert os.path.exists(args.task_path), "Task path does not exist."

    stats = process_all_experiments(args.task_path)

    if args.format == "latex":
        # Call the function and store its output
        output = print_latex_tables(stats, ["bc", "bc_rnn", "diffusion_policy"], print_std=False)
        # output = print_latex_tables(stats, ["diffusion_policy"], print_std=True)
        # Copy the output to the clipboard
        pyperclip.copy(output)
