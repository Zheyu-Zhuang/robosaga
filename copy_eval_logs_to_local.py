import os
import shutil


def copy_txt_files(src_dir, dst_dir):
    for dirpath, dirnames, filenames in os.walk(src_dir):
        if "eval" in dirpath:
            structure = os.path.join(dst_dir, os.path.relpath(dirpath, src_dir))
            if not os.path.isdir(structure):
                os.makedirs(structure)
            for filename in filenames:
                if filename.endswith(".txt"):
                    src_file = os.path.join(dirpath, filename)
                    dst_file = os.path.join(structure, filename)
                    shutil.copy2(src_file, dst_file)


# Usage
copy_txt_files("experiments/robosaga", "evaluation_logs/robosaga")
