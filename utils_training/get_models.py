import torch
from torch.nn import Module
from torch.optim import Optimizer
from typing import Dict, Tuple

from models.i3dpt import Inception3D
from models.resnet import generate_resnet_model
from utils_datasets.nv_gesture.nv_utils import ModalityType, NetworkType
from utils.log_maker import write_log

path_map_dict = {ModalityType.RGB: "rgb_ckp_model_path",
                 ModalityType.DEPTH: "depth_ckp_model_path"}


def get_model(config_dict: dict, modality: ModalityType) -> Module:
    if config_dict["network"] == NetworkType.DETECTOR:
        model = generate_resnet_model(model_depth=10, config_dict=config_dict)
    elif config_dict["network"] == NetworkType.CLASSIFIER:
        model = Inception3D(num_classes=config_dict["num_of_classes"],
                            modality=modality,
                            dropout_prob=config_dict["dropout_prob"],
                            name='inception').to(config_dict["device"])
    else:
        raise ValueError("network type error: {}".format(config_dict["network"]))
    return model


def get_models(config_dict) -> Tuple[Dict[ModalityType, Module], Dict[ModalityType, Optimizer]]:
    model_dict = dict()
    optimizer_dict = dict()
    for modality in config_dict["modalities"]:
        model = get_model(config_dict, modality)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config_dict["learning_rate"],
                                     weight_decay=config_dict["weight_decay"])
        if config_dict[path_map_dict[modality]] is not None:
            try:
                checkpoint = torch.load(config_dict[path_map_dict[modality]])
                model.load_state_dict(checkpoint['model_state_dict'])
                model.dropout.p = config_dict["dropout_prob"]
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                write_log("init",
                          "{} model weights could not be loaded with path: {}\n{}".format(
                              modality.name, config_dict[path_map_dict[modality]], e),
                          title="load model", print_out=True, color="red")
            else:
                write_log("init",
                          "{} model weights are loaded with path: {}".format(
                              modality.name, config_dict[path_map_dict[modality]]),
                          title="load model", print_out=True, color="green")

        model_dict[modality] = model
        optimizer_dict[modality] = optimizer

    return model_dict, optimizer_dict


