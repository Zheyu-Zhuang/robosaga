import copy
import json
import os
from collections import OrderedDict

background_path = "../data/coco_5k_84x84/"


saga_default_configs = {
    "saliency": {
        "enabled": True,
        "update_ratio": 0.1,
        "aug_ratio": 0.5,
        "debug_vis": False,
        "debug_save": True,
        "buffer_shape": [84, 84],
        "save_dir": "",
        "save_debug_im_every_n_batches": 5,
        "background_path": background_path,
        "aug_strategy": "saga_mixup",
        "aug_obs_pairs": False,
    }
}

# HACK: overload the saliency configs for soda for only the background path
soda_default_configs = {"saliency": {"enabled": True, "background_path": background_path}}


overlay_default_configs = {
    "saliency": {
        "aug_strategy": "simple_overlay",
        "aug_ratio": 0.5,
        "debug_vis": False,
        "debug_save": True,
        "save_dir": "",
        "save_debug_im_every_n_batches": 5,
        "background_path": background_path,
    }
}

saga_exp_configs = {
    "saga": saga_default_configs,
    "soda": soda_default_configs,
    "overlay": overlay_default_configs,
    "baseline": {},
}


def generate_saga_configs(config_dir, task):
    input_dir = os.path.join(config_dir, f"{task}")
    for net_type in ["bc", "bc_rnn"]:
        input_file_path = os.path.join(input_dir, f"{net_type}.json")
        assert os.path.exists(input_file_path), f"File {input_file_path} does not exist"
        with open(input_file_path, "r") as f:
            print(f"Reading config file: {input_file_path}")
            task_configs = json.load(f)
            for key, value in saga_exp_configs.items():
                config_temp = copy.deepcopy(task_configs)
                if value != {}:
                    config_temp["saliency"] = value["saliency"]
                config_temp["train"]["data"] = f"../data/robomimic/{task}/ph/image.hdf5"
                config_temp["train"][
                    "output_dir"
                ] = f"../experiments/robosaga/{task}_image/{net_type}"
                config_temp["experiment"]["name"] = f"{key}"
                if net_type == "bc":
                    config_temp["train"]["batch_size"] = 64
                if task == "lift":
                    config_temp["train"]["num_epochs"] = 200
                config_temp["train"]["num_data_workers"] = 8

                out_dir = os.path.join(input_dir, "saga")
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                output_file_path = os.path.join(out_dir, f"{net_type}_{key}.json")
                with open(output_file_path, "w") as f:
                    ordered_config = OrderedDict()
                    if "saliency" in config_temp:
                        ordered_config["saliency"] = config_temp.pop("saliency")
                    for k, v in config_temp.items():
                        ordered_config[k] = v
                    json.dump(ordered_config, f, indent=4)
                print(f"Generated config file: {output_file_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the default configs",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=["lift", "square", "transport", "can"]
    )
    parser.add_argument("--config_dir", type=str, help="Directory to save the default configs")

    args = parser.parse_args()

    for task in args.tasks:
        generate_saga_configs(args.config_dir, task)
