import time

import numpy as np
import torch

from utils_datasets.nv_gesture.nv_utils import NetworkType, ModalityType
from utils_training.get_models import get_models
from utils_transforms.spatial_transforms import Normalize, ToTensor
import torch.nn.functional as F


class Model(object):
    def __init__(self, network_type):
        if network_type == NetworkType.CLASSIFIER:
            model_path = "/Users/sebinemeth/Multimodal-Learning/models/rgb_cnn.pt"
        else:
            model_path = "/Users/sebinemeth/Multimodal-Learning/models/detector_rgb_cnn.pt"

        self.config_dict = {
            "rgb_ckp_model_path": model_path,
            "network": network_type,
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

        self.config_dict["device"] = self.device

        # model_dict, _ = get_models(config_dict)

        # self.model = model_dict[ModalityType.RGB].to(self.device)
        self.model = torch.jit.load(self.config_dict["rgb_ckp_model_path"]).to(self.device)
        self.model.eval()
        self.norm_method = Normalize([0, 0, 0], [1, 1, 1])
        self.to_tensor = ToTensor()

    def __call__(self, frames):
        frame_list = list()
        num_frames = frames.shape[0] if self.config_dict["network"] == NetworkType.CLASSIFIER else 8

        for frame_idx in range(frames.shape[0] - num_frames, frames.shape[0]):
            frame = self.norm_method(self.to_tensor(frames[frame_idx, :, :, :]))
            frame_list.append(frame)

        frames = torch.stack(frame_list, dim=1)
        frames = torch.unsqueeze(frames, dim=0)
        frames = frames.to(self.device)

        with torch.no_grad():
            output, _ = self.model(frames)

        if self.config_dict["network"] == NetworkType.CLASSIFIER:
            return F.softmax(output, dim=1).detach().cpu().numpy().squeeze()
        return torch.sigmoid(output).detach().cpu().numpy().squeeze()


def model_processor(frame_queue, result_queue, stop_event):
    classifier = Model(NetworkType.CLASSIFIER)
    detector = Model(NetworkType.DETECTOR)
    print(" - started process")

    while True:
        # time.sleep(0.01)
        if stop_event.is_set():
            break
        if not frame_queue.empty():
            frames = frame_queue.get_nowait()
            det = detector(frames)
            result = None
            # print(det)
            if det > 0.6:
                # print(" - classifying")
                result = classifier(frames)
            result_queue.put((det, result))

    print(" - ended process")
