import torch
import torch.utils.data as data
from typing import Tuple, Dict

from utils_datasets.nv_gesture.read_data_info import get_data_info_list
from utils_datasets.nv_gesture.nv_utils import SubsetType, ModalityType
from utils_datasets.nv_gesture.image_loader import image_list_loader


class NV(data.Dataset):
    """
    Args:
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
    """

    def __init__(self,
                 subset_type: SubsetType,
                 config_dict: dict,
                 spatial_transform=None,
                 temporal_transform=None):

        self.data_info_list = get_data_info_list(subset_type, config_dict)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.modalities = config_dict["modalities"]
        self.img_size = config_dict["img_x"], config_dict["img_y"]

    def __getitem__(self, index: int) -> Tuple[Dict[ModalityType, torch.Tensor], torch.Tensor]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path = self.data_info_list[index]['video_folder']
        frame_indices = self.data_info_list[index]['frame_indices']
        image_list_dict = image_list_loader(path, frame_indices, self.modalities, self.img_size)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            for modality in image_list_dict.keys():
                image_list_dict[modality] = [self.spatial_transform(image) for image in image_list_dict[modality]]

        for modality in image_list_dict.keys():
            # shape: (chanel, duration, width, height)
            image_list_dict[modality] = torch.stack(image_list_dict[modality], dim=1)

        # shape: (1)
        target = self.data_info_list[index]["label"]
        return image_list_dict, target

    def __len__(self) -> int:
        return len(self.data_info_list)


if __name__ == "__main__":
    _video_path = "../nvGesture"
    _train_annotation_path = "../nvGesture/nvgesture_train_correct_cvpr2016_v2.lst"
    _subset = "train"

    training_data = NV(
        _video_path,
        _train_annotation_path,
        _subset,
        # spatial_transform=spatial_transform,
        # temporal_transform=temporal_transform,
        # target_transform=target_transform,
        # sample_duration=sample_duration,
        modality="RGB-D")

    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=100,
        shuffle=True,
        num_workers=2,
        pin_memory=True)

    for k, (inputs, targets) in enumerate(train_loader):
        print(inputs)
        print(targets)
        input()
