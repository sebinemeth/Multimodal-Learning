import argparse
import yaml
import os
import sys
import git
import datetime
from collections import defaultdict

from utils.log_maker import set_log_dir_path, write_log
from utils.discord import DiscordBot


def str2bool(v):
    if v.lower() in ['none']:
        return None
    elif isinstance(v, bool):
        return v
    elif v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2intlist(v):
    if v.lower() in ['none']:
        return []
    else:
        v_list = v.strip("'\"").split(',')
        map_object = map(int, v_list)
        return list(map_object)


def str2strlist(v):
    if v.lower() in ['none']:
        return []
    else:
        v_list = v.strip("'\"").split(',')
        map_object = map(str, v_list)
        return list(map_object)


def str2int(v):
    if v.lower() in ['none']:
        return None
    else:
        return int(v)


def str2float(v):
    if v.lower() in ['none']:
        return None
    else:
        return float(v)


def str2str(v):
    if v.lower() in ['none']:
        return None
    else:
        return str(v)


def get_config_dict():
    parser = argparse.ArgumentParser()
    req = parser.add_argument_group("required arguments")
    opt = parser.add_argument_group("optional arguments")

    req.add_argument("--name", default="test", required=True, help="name of the training")
    opt.add_argument("--config_yaml", default="./config.yaml", type=str2str, help="path to the config yaml file")

    opt.add_argument("--print_summary", default=False, type=str2bool, help="training only with gestures")
    opt.add_argument("--write_feature_map", default=False, type=str2bool, help="training only with gestures")

    # Training parameters
    opt.add_argument("--epoch", default=None, type=str2int, help="number of epochs")
    opt.add_argument("--learning_rate", default=None, type=str2float, help="starting value of learning rate")
    opt.add_argument("--only_with_gesture", default=None, type=str2bool, help="training only with gestures")
    opt.add_argument("--train_batch_size", default=None, type=str2int, help="batch size during training")
    opt.add_argument("--val_batch_size", default=None, type=str2int, help="batch size during validation")

    # Other parameters
    opt.add_argument("--img_x", default=None, type=str2int, help="")
    opt.add_argument("--img_y", default=None, type=str2int, help="")
    opt.add_argument("--depth_x", default=None, type=str2int, help="")
    opt.add_argument("--depth_y", default=None, type=str2int, help="")
    opt.add_argument("--sample_duration", default=None, type=str2int, help="")
    opt.add_argument("--frame_jump", default=None, type=str2int, help="")

    # Paths for training
    opt.add_argument("--dataset_path", default=None, type=str2str, help="path to dataset folder")
    opt.add_argument("--train_annotation_path", default=None, type=str2str, help="")
    opt.add_argument("--val_annotation_path", default=None, type=str2str, help="")
    opt.add_argument("--rgb_ckp_model_path", default=None, type=str2str,
                     help="path to rgb model in case of continue training")
    opt.add_argument("--depth_ckp_model_path", default=None, type=str2str,
                     help="path to depth model in case of continue training")

    args = parser.parse_args()

    # make base folder path
    output_date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
    base_dir_path = os.path.join("./training_outputs", args.name, output_date)

    # make log folder
    log_dir_path = os.path.join(base_dir_path, "log")
    set_log_dir_path(log_dir_path)

    # write git info in the log
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    branch_name = "detached" if repo.head.is_detached else repo.active_branch.name

    write_log("init", sha, title="commit hash")
    write_log("init", branch_name, title="branch name")

    if args.config_yaml is None:
        config_dict = dict()
        config_dict.update(vars(args))
    else:
        with open(args.config_yaml, 'r') as yaml_file:
            config_dict = yaml.safe_load(yaml_file)

        for key, item in vars(args).items():
            if item is not None:
                config_dict[key] = item

    if len(config_dict["used_classes"]) > 0:
        config_dict["num_of_classes"] = len(config_dict["used_classes"])
    else:
        config_dict["num_of_classes"] = 25

    config_dict["base_dir_path"] = base_dir_path
    config_dict["log_dir_path"] = log_dir_path
    print_dict(config_dict)

    write_log("init", " ".join(sys.argv), title="command")
    write_log("init", config_dict, title="config dict")

    with open(os.path.join(log_dir_path, 'config.yaml'), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)

    config_dict = defaultdict(lambda: None, config_dict)
    return config_dict


def refresh_config(original_config):
    with open(original_config["config_yaml"], 'r') as yaml_file:
        new_config_dict = yaml.safe_load(yaml_file)

    for key, item in new_config_dict.items():
        if item is not None:
            original_config[key] = item

    return original_config


def print_dict(dictionary: dict):
    for key, value in dictionary.items():
        print(str(key) + ': ' + str(value))


def print_to_discord(discord: DiscordBot, dictionary: dict):
    fields = list()

    for key, value in dictionary.items():
        if key == "device":
            value = str(value)
        fields.append({"name": key, "value": value, "inline": True})

    discord.send_message(title="Training started with parameters:", fields=fields)


