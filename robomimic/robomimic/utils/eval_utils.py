import os
import re


def parse_log_file(filepath):
    experiments = []
    with open(filepath, "r") as file:
        content = file.read()
    pattern = (
        r"(Epoch \d+ Rollouts.*?"
        r"save checkpoint to .*?\.pth\s*"
        r"Epoch \d+ Memory Usage: \d+ MB)"
    )
    matches = re.findall(pattern, content, re.DOTALL)
    for match in matches:
        experiments.append(match)
        
    return experiments


def extract_details(text_block): 
    text_block = text_block.split("\n")
    for line in text_block:
        if "Epoch" in line:
            epoch_num = int(line.split(" ")[1])
        if "Success_Rate" in line:
            numbers = re.findall(r'\d+\.?\d*', line)
            success_rate = numbers[0]
        if "save checkpoint to" in line:
            checkpoint_path = line.split(" ")[-1]
    
    return epoch_num, success_rate, checkpoint_path


def get_top_n_experiments(log_file_path, n=3):
    all_checkpoint_logs = parse_log_file(log_file_path)
    exp_details = []
    for experiment in all_checkpoint_logs:
        epoch_num, success_rate, checkpoint_path = extract_details(experiment)
        exp_details.append(
            {
                "epoch_num": epoch_num,
                "success_rate": success_rate,
                "checkpoint_path": checkpoint_path,
            }
        )
    assert len(exp_details) > 0, "No experiments found in the log file"
    assert n <= len(exp_details), f"Only {len(exp_details)} experiments found"
    sorted_experiments = sorted(
        exp_details, key=lambda x: (x["success_rate"], x["epoch_num"]), reverse=True
    )
    selected_exps = sorted_experiments[:n]
    for i, exp in enumerate(selected_exps, 1):
        print(f"Checkpoint {i}:")
        print(f"Epoch Number: {exp['epoch_num']}")
        print(f"Success Rate: {exp['success_rate']}")
        print(f"Checkpoint Path: {exp['checkpoint_path']}\n")
    return [exp["checkpoint_path"] for exp in selected_exps], [
        float(exp["success_rate"]) for exp in selected_exps
    ]


def main(exp_path):
    # Specify the path to the log file
    log_file_path = os.path.join(exp_path, "logs/log.txt")
    # Get the parsed experiments data
    experiments = get_top_n_experiments(log_file_path, n=3)
    print(experiments)


if __name__ == "__main__":
    # Specify the path to the experiment directory
    exp_path = "experiments/robosaga/lift_image/bc/overlay/20240524001358/"
    main(exp_path)    

