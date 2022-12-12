import yaml
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from utils_training.get_models import get_models
from utils_datasets.nv_gesture.nv_utils import SubsetType, ModalityType, MetricType, NetworkType

def convert_modalities_and_network(config_dict: dict):
    if config_dict["network"] == "DETECTOR":
        config_dict["network"] = NetworkType.DETECTOR
    elif config_dict["network"] == "CLASSIFIER":
        config_dict["network"] = NetworkType.CLASSIFIER
    else:
        raise ValueError("unknown modality: {}".format(config_dict["network"]))

    modalities = list()
    for modality in config_dict["modalities"]:
        if modality == "RGB":
            modalities.append(ModalityType.RGB)
        elif modality == "DEPTH":
            modalities.append(ModalityType.DEPTH)
        else:
            raise ValueError("unknown modality: {}".format(modality))

    config_dict["modalities"] = modalities

config_yaml_path = "./training_outputs/rgb_detector_binary_2/2022-12-08T18:54/log/config.yaml"
with open(config_yaml_path, 'r') as yaml_file:
    config_dict = yaml.safe_load(yaml_file)

convert_modalities_and_network(config_dict)
device = torch.device("cpu")
config_dict["device"] = device
config_dict["rgb_ckp_model_path"] = "./training_outputs/rgb_detector_binary_2/2022-12-08T18:54/model/RGB_best.pt"

model_dict, optimizer_dict = get_models(config_dict)
rgb_cnn = model_dict[ModalityType.RGB]
rgb_cnn.eval()
example = torch.rand(1, 3, 32, 224, 224)  # (batch_size, chanel, duration, width, height)
traced_module = torch.jit.trace(rgb_cnn, example)
traced_module.save("detector_rgb_cnn.pt")

traced_script_module_optimized = optimize_for_mobile(traced_module)
traced_script_module_optimized._save_for_lite_interpreter("detector_rgb_cnn.ptl")


