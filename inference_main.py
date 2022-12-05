import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

from utils_training.get_models import get_models
from utils_datasets.nv_gesture.nv_dataset import NV
from utils_transforms.spatial_transforms import Compose, ToTensor, Normalize, Scale
from utils_datasets.nv_gesture.nv_utils import SubsetType, ModalityType, MetricType, NetworkType, convert_to_tqdm_dict

# to supress pytorch nn.MaxPool3d warning
warnings.filterwarnings("ignore", category=UserWarning)

config_dict = {
    "dataset_path": "./datasets/nvGesture",
    "val_annotation_path": "./datasets/nvGesture/nvgesture_test_correct_cvpr2016_v2.lst",
    "rgb_ckp_model_path": "",
    "network": NetworkType.CLASSIFICATOR,
    "modalities": [ModalityType.RGB],
    "img_x": 224,
    "img_y": 224,
    "depth_x": 224,
    "depth_y": 224,
    "sample_duration": 64,
    "frame_jump": 2,
    "cover_ratio": 0.5,
    "used_classes": [10, 14, 20, 22, 24],
    "val_batch_size": 32,
    "val_shuffle_data": False,
    "val_num_of_workers": 10,
}

use_cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

norm_method = Normalize([0, 0, 0], [1, 1, 1])
spatial_transform = Compose([Scale((config_dict["img_x"], config_dict["img_y"])), ToTensor(), norm_method])

valid_data = NV(subset_type=SubsetType.VAL,
                config_dict=config_dict,
                spatial_transform=spatial_transform,
                temporal_transform=None)

valid_loader = DataLoader(valid_data,
                          batch_size=config_dict["val_batch_size"],
                          shuffle=config_dict["val_shuffle_data"],
                          num_workers=config_dict["val_num_of_workers"],
                          pin_memory=True)

model_dict, _ = get_models(config_dict)
modalities = config_dict["modalities"]

with torch.no_grad():
    predictions_dict = dict()
    correct_dict = dict()
    for modality in modalities:
        model_dict[modality].eval()
        predictions_dict[modality] = list()
        correct_dict[modality] = 0

    y_test = list()
    total = 0
    tqdm_dict = dict()

    tq = tqdm(total=(len(valid_loader)))
    tq.set_description('Inference')
    for batch_idx, (data_dict, y) in enumerate(valid_loader):
        y_test.append(y.numpy().copy())
        total += y.size(0)
        y = y.to(device)

        for modality in modalities:
            data_dict[modality] = data_dict[modality].to(device)
            output, _ = model_dict[modality](data_dict[modality])

            _, predicted = output.max(1)
            correct_dict[modality] += predicted.eq(y).sum().item()
            predictions_dict[modality].append(predicted.cpu().numpy())
            tqdm_dict[SubsetType.VAL, modality, MetricType.ACC] = correct_dict[modality] / total

        tq.update(1)
        tq.set_postfix(**convert_to_tqdm_dict(tqdm_dict))

    tq.close()

predictions = predictions_dict[ModalityType.RGB]
