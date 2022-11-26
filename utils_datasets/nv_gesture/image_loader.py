import os
import numpy as np
from typing import Tuple, List, Dict

from PIL import Image

from utils.log_maker import write_log
from utils_datasets.nv_gesture.nv_utils import ModalityType

print_out = False


def pil_loader(path: str, modality: ModalityType):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if modality == ModalityType.RGB:
                return img.convert('RGB')
            elif modality == ModalityType.DEPTH:
                # 8-bit pixels, black and white check from https://pillow.readthedocs.io/en/3.0.x/handbook/concepts.html
                return img.convert('L')


def image_list_loader(video_dir_path: str,
                      frame_indices,  #: list,
                      modalities,  #: List[ModalityType],
                      img_size: Tuple[int, int],
                      frame_idx_offset: int = 0):  # -> Dict[ModalityType: list]:

    video_dir_path = os.path.join(video_dir_path, "sk_color_all")
    image_list_dict = dict()

    for modality in modalities:
        image_list_dict[modality] = list()
        if modality == ModalityType.RGB:
            for frame_idx in frame_indices:
                image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(frame_idx + frame_idx_offset))
                if os.path.exists(image_path):
                    image_list_dict[modality].append(pil_loader(image_path, modality))
                else:
                    image_list_dict[modality].append(np.zeros((img_size[0], img_size[1], 3)))
                    write_log("dataloader", image_path, title="image is not found", print_out=print_out, color="red")

        elif modality == ModalityType.DEPTH:
            for frame_idx in frame_indices:
                image_path = os.path.join(video_dir_path.replace('color', 'depth'),
                                          '{:05d}.jpg'.format(frame_idx + frame_idx_offset))
                if os.path.exists(image_path):
                    image_list_dict[modality].append(pil_loader(image_path, modality))
                else:
                    image_list_dict[modality].append(np.zeros((img_size[0], img_size[1], 1)))
                    write_log("dataloader", image_path, title="image is not found", print_out=print_out, color="red")

        else:
            raise ValueError("unknown modality: {}".format(modality))

    return image_list_dict
