import os

from robomimic.utils.eval_utils import get_top_n_experiments


def main(exp_path):
    # Specify the path to the log file
    log_file_path = os.path.join(exp_path, "logs/log.txt")
    top_n_checkpoints = get_top_n_experiments(log_file_path, n=3)
