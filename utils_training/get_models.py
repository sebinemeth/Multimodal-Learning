import torch

from models.i3dpt import Inception3D
from utils_datasets.nv_gesture.nv_utils import ModalityType
from utils.log_maker import write_log


def get_models(config_dict):
    rgb_cnn = Inception3D(num_classes=config_dict["num_of_classes"],
                          modality=ModalityType.RGB,
                          dropout_prob=config_dict["dropout_prob"],
                          name='inception').to(config_dict["device"])

    depth_cnn = Inception3D(num_classes=config_dict["num_of_classes"],
                            modality=ModalityType.DEPTH,
                            dropout_prob=config_dict["dropout_prob"],
                            name='inception').to(config_dict["device"])

    if config_dict["rgb_ckp_model_path"] is not None:
        try:
            rgb_cnn.load_state_dict(torch.load(config_dict["rgb_ckp_model_path"]))
        except Exception as e:
            write_log("init",
                      "rgb model weights could not be loaded with path: {}\n{}".format(
                          config_dict["rgb_ckp_model_path"], e),
                      title="load model", print_out=True, color="red")
        else:
            write_log("init",
                      "rgb model weights are loaded with path: {}".format(config_dict["rgb_ckp_model_path"]),
                      title="load model", print_out=True, color="green")

    if config_dict["depth_ckp_model_path"] is not None:
        try:
            depth_cnn.load_state_dict(torch.load(config_dict["depth_ckp_model_path"]))
        except Exception as e:
            write_log("init",
                      "depth model weights could not be loaded with path: {}\n{}".format(
                          config_dict["depth_ckp_model_path"], e),
                      title="load model", print_out=True, color="red")
        else:
            write_log("init",
                      "depth model weights are loaded with path: {}".format(config_dict["depth_ckp_model_path"]),
                      title="load model", print_out=True, color="green")

    return rgb_cnn, depth_cnn

