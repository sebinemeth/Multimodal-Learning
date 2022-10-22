from torch.utils.data import DataLoader

from utils_datasets.nv_gesture.nv_dataset import NV
from utils_datasets.nv_gesture.nv_utils import SubsetType, ModalityType
from spatial_transforms import Compose, ToTensor, Normalize, Scale


def get_loaders(config_dict):
    norm_method = Normalize([0, 0, 0], [1, 1, 1])
    spatial_transform = Compose([Scale((config_dict["img_x"], config_dict["img_y"])), ToTensor(), norm_method])

    training_data = NV(subset_type=SubsetType.TRAIN,
                       modality=ModalityType.RGB_DEPTH,
                       config_dict=config_dict,
                       spatial_transform=spatial_transform,
                       temporal_transform=None)

    train_loader = DataLoader(training_data,
                              batch_size=config_dict["train_batch_size"],
                              shuffle=config_dict["train_shuffle_data"],
                              num_workers=config_dict["val_num_of_workers"],
                              pin_memory=True)

    valid_data = NV(subset_type=SubsetType.VALIDATION,
                    modality=ModalityType.RGB_DEPTH,
                    config_dict=config_dict,
                    spatial_transform=spatial_transform,
                    temporal_transform=None)

    valid_loader = DataLoader(valid_data,
                              batch_size=config_dict["val_batch_size"],
                              shuffle=config_dict["val_shuffle_data"],
                              num_workers=config_dict["val_num_of_workers"],
                              pin_memory=True)

    return train_loader, valid_loader
