import torch
import torch.utils.data as data

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
                 modality: ModalityType,
                 config_dict: dict,
                 spatial_transform=None,
                 temporal_transform=None):

        self.data_info_list = get_data_info_list(subset_type, config_dict)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.modality = modality
        self.sample_duration = config_dict["sample_duration"]
        self.img_size = config_dict["resized_img_x"], config_dict["resized_img_x"]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path = self.data_info_list[index]['video_folder']

        frame_indices = self.data_info_list[index]['frame_indices']
        image_depth_list = image_list_loader(path, frame_indices, self.modality, self.img_size)
        rgb_images = [rgb_depth[0] for rgb_depth in image_depth_list]
        depth_images = [rgb_depth[1] for rgb_depth in image_depth_list]

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            rgb_images = [self.spatial_transform(rgb_image) for rgb_image in rgb_images]
            depth_images = [self.spatial_transform(depth_image) for depth_image in depth_images]

        rgb = torch.stack(rgb_images, dim=1)
        depth = torch.stack(depth_images, dim=1)

        target = self.data_info_list[index]["label"]
        return rgb, depth, target

    def __len__(self):
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
