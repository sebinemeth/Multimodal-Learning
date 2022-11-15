import yaml
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from utils_training.get_models import get_models


config_yaml_path = "./training_outputs/unimodal_rgb_5_2/2022-11-15T11:47/log/config.yaml"
with open(config_yaml_path, 'r') as yaml_file:
    config_dict = yaml.safe_load(yaml_file)

config_dict["rgb_ckp_model_path"] = "./training_outputs/unimodal_rgb_5_2/2022-11-15T11:47/model/model_rgb_18.pt"

rgb_cnn, _ = get_models(config_dict, only_rgb=True)
rgb_cnn.eval()
example = torch.rand(1, 3, 32, 224, 224)  # (batch_size, chanel, duration, width, height)
traced_module = torch.jit.trace(rgb_cnn, example)
traced_module.save("rgb_cnn.pt")

traced_script_module_optimized = optimize_for_mobile(traced_module)
traced_script_module_optimized._save_for_lite_interpreter("rgb_cnn.ptl")


