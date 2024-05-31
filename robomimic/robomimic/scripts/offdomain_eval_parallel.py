import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp_path", type=str, default=None, required=True)

args = parser.parse_args()


commands = []


scripts_dir = os.path.dirname(os.path.realpath(__file__))


commands.append(f"python {scripts_dir}/offdomain_eval_top3.py --video -e {args.exp_path}")

commands.append(
    f"python {scripts_dir}/offdomain_eval_top3.py --shuffle_env --video -e {args.exp_path}"
)

commands.append(
    f"python {scripts_dir}/offdomain_eval_top3.py --distractors --video -e {args.exp_path}"
)

commands = " && ".join(commands)

os.system(commands)

# TODO: remove all temp files
