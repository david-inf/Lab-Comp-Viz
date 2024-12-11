import torch
from types import SimpleNamespace
import argparse
import yaml

if __name__ == "__main__":

    ## mio personale
    # project = "comp-viz"
    # entity = "david-inf-team"

    ## gruppo lab2
    # project = "cv-lab2"
    # entity = "alessiochen98-university-of-florence"

    def merge_configs(global_config, task_config):
        merged = global_config.copy()
        merged.update(task_config)
        return merged

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML COnfiguration file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        # get hyper-parameters
        hp = yaml.load(f, Loader=yaml.SafeLoader)  # dict

    device = "cuda" if torch.cuda.is_available() else "cpu"

    global_hp = hp["global"]
    global_hp["device"] = device
    pre_hp = hp["tasks"]["pretraining"]
    down_hp = hp["tasks"]["downstream"]
    pre_config = merge_configs(global_hp, pre_hp)  # dict
    down_config = merge_configs(global_hp, down_hp)  # dict

    pre_opts = SimpleNamespace(**pre_config)
    down_opts = SimpleNamespace(**down_config)
    print(pre_opts)
    print(down_opts)
