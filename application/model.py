import numpy as np
import torch

from utils_datasets.nv_gesture.nv_utils import NetworkType, ModalityType
from utils_training.get_models import get_models
from utils_transforms.spatial_transforms import Normalize, ToTensor
import torch.nn.functional as F


class Model(object):
    def __init__(self):
        config_dict = {
            "rgb_ckp_model_path": "./models/rgb_cnn.pt",
            "network": NetworkType.CLASSIFICATOR,
            "modalities": [ModalityType.RGB],
            "img_x": 224,
            "img_y": 224,
            "depth_x": 224,
            "depth_y": 224,
            "used_classes": [10, 14, 20, 22, 24],
            "num_of_classes": 5,
            "dropout_prob": 0,
            "learning_rate": 0,
            "weight_decay": 0,
        }

        use_cuda = torch.cuda.is_available()  # check if GPU exists
        self.device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

        print(self.device)

        config_dict["device"] = self.device

        # model_dict, _ = get_models(config_dict)

        # self.model = model_dict[ModalityType.RGB].to(self.device)
        self.model = torch.jit.load(config_dict["rgb_ckp_model_path"]).to(self.device)
        self.model.eval()
        self.norm_method = Normalize([0, 0, 0], [1, 1, 1])
        self.to_tensor = ToTensor()

    def __call__(self, frames):
        frame_list = list()
        for frame_idx in range(frames.shape[0]):
            frame = self.norm_method(self.to_tensor(frames[frame_idx, :, :, :]))
            frame_list.append(frame)

        frames = torch.stack(frame_list, dim=1)
        frames = torch.unsqueeze(frames, dim=0)
        frames = frames.to(self.device)

        with torch.no_grad():
            output, _ = self.model(frames)

        print(F.softmax(output, dim=1))
