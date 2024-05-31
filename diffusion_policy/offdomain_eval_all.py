import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp_path", type=str, default=None, required=True)

args = parser.parse_args()


offdomain_types = ['shuffle_env', 'distractors']

commands = []


scripts_dir = os.path.dirname(os.path.realpath(__file__))

commands.append(f"python {scripts_dir}/offdomain_eval_top3.py -e {args.exp_path} --distractors")
commands.append(f"python {scripts_dir}/offdomain_eval_top3.py -e {args.exp_path} --shuffle_env")

commands = " && ".join(commands)

os.system(commands)
