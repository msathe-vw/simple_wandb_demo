import os, argparse, json
from turtle import dot
from addict import Dict

def merge(d1, d2):
    for k in d2:
        if k in d1 and isinstance(d1[k], dict) and isinstance(d2[k], dict):
            merge(d1[k], d2[k])
        else:
            d1[k] = d2[k]


def combined_parser():
    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("-n", "--name", default=None, help="Name of the run")
    parser.add_argument(
        "-c",
        "--config",
        default="default_config.json",
        help="Path to config file",
    )
    namespace, _ = parser.parse_known_args()
    command_line_args = {k: v for k, v in vars(namespace).items() if v is not None}

    if command_line_args["config"] is not None:
        with open(command_line_args["config"]) as f:
            json_dict = json.load(f)
    else:
        json_dict = {}

    merge(json_dict, command_line_args)
    merge(json_dict, json.loads(os.environ.get("SM_HPS", "{}")))
    dot_dict = Dict(json_dict)
    return dot_dict