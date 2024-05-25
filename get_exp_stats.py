import re
import pandas as pd
import os
import numpy as np
from collections import defaultdict

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
    current_data = None

    # Splitting the input text into lines
    lines = text.split('\n')

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
                raw_data = [float(x) for x in indomain_match.group(1).split(',')]
            except ValueError:
                print(f"Error parsing raw data from {file_path}")
                raw_data = []
            mean = float(indomain_match.group(2))
            std = float(indomain_match.group(3))
            stats[current_domain] = {'mean': mean, 'std': std, 'raw_data': raw_data}

        # Check for "Off-domain" summary statistics
        off_domain_match = off_domain_regex.match(line.strip())
        if off_domain_match:
            current_domain = "offdomain"
            offdomain_type = off_domain_match.group(1)
            mean = float(off_domain_match.group(2))
            std = float(off_domain_match.group(3))
            stats[current_domain] = {'mean': mean, 'std': std, 'raw_data': [], 'type': offdomain_type}

        # Check for raw data
        data_match = raw_data_regex.match(line.strip())
        if data_match and current_domain:
            # Convert string data to floats and store in the current domain's data
            try:
                raw_data = [float(x) for x in data_match.group(1).split(',')]
            except ValueError:
                print(f"Error parsing raw data from {file_path}")
                raw_data = []
            if len(raw_data) <=2:
                print(f"Warning: Raw data has less than 3 elements in {file_path}")
            stats[current_domain]['raw_data'] = raw_data

    return stats


def format_mean_std(mean, std):
    """
    Formats mean and standard deviation into a single string.
    """
    mean = f"{mean:.2f}" if mean is not None else "N/A"
    std = f"{std:.2f}" if std is not None else "N/A"
    return f"{mean}"


def process_directory(base_path):
    policies = ['bc', 'bc_rnn', 'diffusion_policy']
    aug_methods = ['baseline', 'overlay', 'soda', 'soda_default', 'saga', 'saga_default']
    experiments = ['outdoor', 'indoor', 'textile', 'distractors']
    results = defaultdict(dict)
    for policy in policies:
        for aug_method in aug_methods:
            sub_path = os.path.join(base_path, policy, aug_method)
            if os.path.exists(sub_path):
                for root, dirs, files in os.walk(sub_path):
                    if 'eval' in dirs:
                        eval_path = os.path.join(root, 'eval')
                        eval_results = {'indomain': 'No data'}
                        for experiment in experiments:
                            matched_files = [f_ for f_ in os.listdir(eval_path) if experiment.lower() in f_.lower() and f_.endswith('.txt')]
                            for f_ in matched_files:
                                file_path = os.path.join(eval_path, f_)
                                with open(file_path, 'r') as f:
                                    text = f.read()
                                    data = parse_stats_from_text(text, file_path)
                                    if 'indomain' in data:
                                        eval_results['indomain'] = data['indomain']['raw_data']
                                    if 'offdomain' in data:
                                        eval_results[f"{data['offdomain']['type']}"] = data['offdomain']['raw_data']
                        aug_method = aug_method.replace('_default', '')
                        if aug_method == 'baseline':
                            aug_method = 'no_aug'
                        results[policy][aug_method] = eval_results
    return results
                        

def further_process_results(exp_data):
    exp_stats = defaultdict(lambda: defaultdict(dict))
    for policy in exp_data.keys():
        highscores = defaultdict(float)
        for aug_method, results in exp_data[policy].items():
            indomain= []
            average = []
            for experiment in results.keys():
                if experiment == 'indomain':
                    indomain.append(results[experiment])
                    continue
                else:
                    average += results[experiment]
                exp_mean = np.around(np.mean(results[experiment]), 2)
                if exp_mean > highscores[experiment]:
                    highscores[experiment] = exp_mean
                if results[experiment] == 'No data' or len(results[experiment]) == 0 or results[experiment] is None:
                    exp_stats[policy][aug_method][experiment] = [None, None]
                    continue
                exp_std = np.around(np.std(results[experiment]), 2)
                exp_stats[policy][aug_method][experiment] = [exp_mean, exp_std]
            if len(indomain) == 0 or 'No data' in indomain:
                exp_stats[policy][aug_method]['indomain'] = [None, None]
            else:
                exp_stats[policy][aug_method]['indomain'] = [np.around(np.mean(indomain), 2), np.around(np.std(indomain), 2)]
            if len(average) == 0 or 'No data' in average:   
                exp_stats[policy][aug_method]['average'] = [None, None]
            else:
                exp_stats[policy][aug_method]['average'] = [np.around(np.mean(average), 2), np.around(np.std(average), 2)]
            if exp_stats[policy][aug_method]['indomain'][0] is not None and exp_stats[policy][aug_method]['indomain'][0] > highscores['indomain']:
                highscores['indomain'] = exp_stats[policy][aug_method]['indomain'][0]
            if exp_stats[policy][aug_method]['average'][0] is not None and exp_stats[policy][aug_method]['average'][0] > highscores['average']:
                highscores['average'] = exp_stats[policy][aug_method]['average'][0]
        exp_stats[policy]['highscores'] = highscores
            
    return exp_stats
        


# def print_markdown_tables(df):
#     """
#     Prints a single Markdown table from the DataFrame,
#     highlighting the maximum mean values for each column. Improved formatting for correct alignment.
#     """
#     policies = df['Policy'].unique()  # Get all unique policies

#     # Print header for the table
#     print("| Policy | Method | Indomain | Outdoor | Indoor | Textile | Distractors |")
#     print("|--------|--------|----------|---------|--------|---------|-------------|")

#     for policy in policies:
#         df_policy = df[df['Policy'] == policy]  # Filter the DataFrame for the current policy
        
#         # Extract means and determine the max for each column
#         means_df = df_policy.applymap(extract_mean)  # Apply the extract_mean function
#         max_values = means_df.max()  # Get max values for each column to highlight

#         # Iterate over rows to format each line
#         for _, row in df_policy.iterrows():
#             title = policy if row.name == df_policy.index[0] else ""  # Print policy name only once
#             line = f"| {title} | {row['Method']} |"
#             for col in ['Indomain', 'Outdoor', 'Indoor', 'Textile', 'Distractors']:
#                 value = f'{row[col]}'
#                 mean_value = extract_mean(value)
#                 # Highlight the max value in bold
#                 if mean_value == max_values[col]:
#                     value = f"**{value}**"
#                 line += f" {value} |"
#             print(line)
#     print("\n")  # Newline for spacing between tables
        
def print_latex_tables(stats_dict, print_std=False):
    """
    Prints a LaTeX table from the DataFrame,
    highlighting the maximum mean values for each column including the average.
    """
    policies = stats_dict.keys()  # Get all unique policies
    titles = {'bc': '\\textbf{BC}', 'bc_rnn': '\\textbf{BC-RNN}', 'diffusion_policy': '\\textbf{Diffusion Policy}'}
    aug_methods_pretty = {'no_aug': '\\textsc{None}', 'overlay': '\\textsc{Overlay}', 'soda': '\\textsc{SODA}', 'saga': '\\textsc{SaGA}'}
    
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Summary of Performance Metrics}")
    print("\\label{tab:performance_metrics}")
    print("\\setlength{\\tabcolsep}{4.5pt}") 
    print("\\begin{tabular}{c|c|cccc|c}")
    print("\\toprule")
    print(" & \\textbf{Indomain} & \\textbf{Outdoor} & \\textbf{Indoor} & \\textbf{Textile} & \\textbf{Distractors} & \\textbf{Mean}\\\\")
    print("\\midrule")
    
    for i, policy in enumerate(policies):
        policy_title = titles[policy]
        highscores = stats_dict[policy]['highscores']
        aug_methods = [k for k in stats_dict[policy].keys() if k != 'highscores']
        print(f"\\multicolumn{{7}}{{c}}{{{policy_title}}}\\\\")
        print("\\midrule")
        for aug_method in aug_methods:
            line = f"{aug_methods_pretty[aug_method]} &"
            for experiment in ['indomain', 'outdoor', 'indoor', 'textile', 'distractors', 'average']:
                if experiment not in stats_dict[policy][aug_method]:
                    mean, std = [None, None]
                else:
                    mean, std = stats_dict[policy][aug_method][experiment]
                if print_std:
                    if mean is None:
                        str_ = "N/A"
                    else:
                        str_ = f"{mean:.2f} $\pm$ {std:.2f}"
                else:
                    if mean is None:
                        str_ = "N/A"
                    else:
                        str_ = f"{mean:.2f}"
                if mean == highscores[experiment]:
                    str_ = f"\\textbf{{{str_}}}"
                line += f" {str_} &"
            print(line[:-2] + "\\\\")
            if i != len(policies) - 1:  # Check if it's the last method
                print("\\midrule")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--task_path", type=str, default=None, required=True)
    parser.add_argument("--format", type=str, default="markdown", choices=["markdown", "latex"])
    
    args  = parser.parse_args()
    
    raw_data = process_directory(args.task_path)

    stats = further_process_results(raw_data)
    if args.format == "latex":
        print_latex_tables(stats, print_std=True)
    # else:
    #     print_markdown_tables(stats)