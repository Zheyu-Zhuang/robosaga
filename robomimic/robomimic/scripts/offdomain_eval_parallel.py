import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp_path", type=str, default=None, required=True)

args = parser.parse_args()


offdomain_types = ["indoor", "outdoor", "textile", "distractors"]

commands = []


scripts_dir = os.path.dirname(os.path.realpath(__file__))

for m in offdomain_types:
    commands.append(
        f"python {scripts_dir}/offdomain_eval_top3.py -m {m} --video -e {args.exp_path}"
    )
commands = " & ".join(commands)

os.system(commands)

# TODO: remove all temp files
