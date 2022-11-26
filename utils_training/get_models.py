import torch
from torch.nn import Module
from torch.optim import Optimizer
from typing import Dict, Tuple

from models.i3dpt import Inception3D
from utils_datasets.nv_gesture.nv_utils import ModalityType
from utils.log_maker import write_log

path_map_dict = {ModalityType.RGB: "rgb_ckp_model_path",
                 ModalityType.DEPTH: "depth_ckp_model_path"}


def get_models(config_dict) -> Tuple[Dict[ModalityType, Module], Dict[ModalityType, Optimizer]]:
    model_dict = dict()
    optimizer_dict = dict()
    for modality in config_dict["modalities"]:
        model = Inception3D(num_classes=config_dict["num_of_classes"],
                            modality=modality,
                            dropout_prob=config_dict["dropout_prob"],
                            name='inception').to(config_dict["device"])

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config_dict["learning_rate"],
                                     weight_decay=config_dict["weight_decay"])

        if config_dict[path_map_dict[modality]] is not None:
            try:
                checkpoint = torch.load(config_dict[path_map_dict[modality]])
                model.load_state_dict(checkpoint['model_state_dict'])
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
        optimizer_dict[modality] = optimizer_dict

    return model_dict, optimizer_dict

    # rgb_cnn = Inception3D(num_classes=config_dict["num_of_classes"],
    #                       modality=ModalityType.RGB,
    #                       dropout_prob=config_dict["dropout_prob"],
    #                       name='inception').to(config_dict["device"])
    #
    # rgb_optimizer = torch.optim.Adam(rgb_cnn.parameters(),
    #                                  lr=config_dict["learning_rate"],
    #                                  weight_decay=config_dict["weight_decay"])
    #
    # if config_dict["rgb_ckp_model_path"] is not None:
    #     try:
    #         checkpoint = torch.load(config_dict["rgb_ckp_model_path"])
    #         rgb_cnn.load_state_dict(checkpoint['model_state_dict'])
    #         rgb_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     except Exception as e:
    #         write_log("init",
    #                   "rgb model weights could not be loaded with path: {}\n{}".format(
    #                       config_dict["rgb_ckp_model_path"], e),
    #                   title="load model", print_out=True, color="red")
    #     else:
    #         write_log("init",
    #                   "rgb model weights are loaded with path: {}".format(config_dict["rgb_ckp_model_path"]),
    #                   title="load model", print_out=True, color="green")
    #
    # if only_rgb:
    #     return rgb_cnn, rgb_optimizer
    #
    # depth_cnn = Inception3D(num_classes=config_dict["num_of_classes"],
    #                         modality=ModalityType.DEPTH,
    #                         dropout_prob=config_dict["dropout_prob"],
    #                         name='inception').to(config_dict["device"])
    #
    # depth_optimizer = torch.optim.Adam(depth_cnn.parameters(), lr=config_dict["learning_rate"])
    #
    # if config_dict["depth_ckp_model_path"] is not None:
    #     try:
    #         checkpoint = torch.load(config_dict["depth_ckp_model_path"])
    #         depth_cnn.load_state_dict(checkpoint['model_state_dict'])
    #         depth_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     except Exception as e:
    #         write_log("init",
    #                   "depth model weights could not be loaded with path: {}\n{}".format(
    #                       config_dict["depth_ckp_model_path"], e),
    #                   title="load model", print_out=True, color="red")
    #     else:
    #         write_log("init",
    #                   "depth model weights are loaded with path: {}".format(config_dict["depth_ckp_model_path"]),
    #                   title="load model", print_out=True, color="green")
    #
    # return rgb_cnn, rgb_optimizer, depth_cnn, depth_optimizer
