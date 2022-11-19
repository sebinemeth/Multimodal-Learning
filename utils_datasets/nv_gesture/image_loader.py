import os
import numpy as np

from PIL import Image

from utils.log_maker import write_log
from utils_datasets.nv_gesture.nv_utils import ModalityType

print_out = False


def pil_loader(path, modality):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if modality == ModalityType.RGB:
                return img.convert('RGB')
            elif modality == ModalityType.DEPTH:
                # 8-bit pixels, black and white check from https://pillow.readthedocs.io/en/3.0.x/handbook/concepts.html
                return img.convert('L')


def image_list_loader(video_dir_path, frame_indices, modality, img_size, frame_idx_offset=0):
    video_dir_path = os.path.join(video_dir_path, "sk_color_all")

    rgb_image_list = list()
    depth_image_list = list()
    if modality == ModalityType.RGB:
        for frame_idx in frame_indices:
            image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(frame_idx + frame_idx_offset))
            if os.path.exists(image_path):
                rgb_image_list.append(pil_loader(image_path, modality))
            else:
                rgb_image_list.append(np.zeros((img_size[0], img_size[1], 3)))
                write_log("dataloader", image_path, title="image is not found", print_out=print_out, color="red")

    elif modality == ModalityType.DEPTH:
        for frame_idx in frame_indices:
            image_path = os.path.join(video_dir_path.replace('color', 'depth'),
                                      '{:05d}.jpg'.format(frame_idx + frame_idx_offset))
            if os.path.exists(image_path):
                depth_image_list.append(pil_loader(image_path, modality))
            else:
                depth_image_list.append(np.zeros((img_size[0], img_size[1], 1)))
                write_log("dataloader", image_path, title="image is not found", print_out=print_out, color="red")

    elif modality == ModalityType.RGB_DEPTH:
        for frame_idx in frame_indices:  # index 35 is used to change img to flow
            if frame_idx < 0:
                rgb_image_list.append(np.zeros((img_size[0], img_size[1], 3)))
                depth_image_list.append(np.zeros((img_size[0], img_size[1], 1)))
            else:
                rgb_image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(frame_idx + frame_idx_offset))
                if os.path.exists(rgb_image_path):
                    rgb_image_list.append(pil_loader(rgb_image_path, ModalityType.RGB))
                else:
                    rgb_image_list.append(np.zeros((img_size[0], img_size[1], 3)))
                    write_log("dataloader", rgb_image_path, title="image is not found", print_out=print_out,
                              color="red")

                depth_image_path = os.path.join(video_dir_path.replace('color', 'depth'),
                                                '{:05d}.jpg'.format(frame_idx + frame_idx_offset))
                if os.path.exists(depth_image_path):
                    depth_image_list.append(pil_loader(depth_image_path, ModalityType.DEPTH))
                else:
                    depth_image_list.append(np.zeros((img_size[0], img_size[1], 1)))
                    write_log("dataloader", depth_image_path, title="image is not found", print_out=print_out,
                              color="red")

    return rgb_image_list, depth_image_list
