import torch
import torch.utils.data as data
from PIL import Image
# from spatial_transforms import *
import os
import math
import functools
import json
import copy
from tqdm import tqdm
from glob import glob


# from numpy.random import randint
import numpy as np
# import random
#
# from utils import load_value_file
# import pdb


def pil_loader(path, modality):
    # print(path, modality)
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        # print(path)
        with Image.open(f) as img:
            if modality == 'RGB':
                return img.convert('RGB')
            elif modality == 'Depth':
                return img.convert(
                    'L')  # 8-bit pixels, black and white check from https://pillow.readthedocs.io/en/3.0.x/handbook/concepts.html


def accimage_loader(path, modality):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, modality, img_size, image_loader, frame_idx_offset=0):
    video_dir_path = os.path.join(video_dir_path, "sk_color_all")

    video = []
    if modality == 'RGB':
        for frame_idx in frame_indices:
            image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(frame_idx + frame_idx_offset))
            if os.path.exists(image_path):

                video.append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality == 'Depth':
        for frame_idx in frame_indices:
            image_path = os.path.join(video_dir_path.replace('color', 'depth'),
                                      '{:05d}.jpg'.format(frame_idx + frame_idx_offset))
            if os.path.exists(image_path):
                video.append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality == 'RGB-D':
        for frame_idx in frame_indices:  # index 35 is used to change img to flow
            if frame_idx < 0:
                video.append((np.zeros((img_size[0], img_size[1], 3)), np.zeros((img_size[0], img_size[1]))))
            else:

                image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(frame_idx + frame_idx_offset))

                image_path_depth = os.path.join(video_dir_path.replace('color', 'depth'),
                                                '{:05d}.jpg'.format(frame_idx + frame_idx_offset))

                image = image_loader(image_path, 'RGB')
                image_depth = image_loader(image_path_depth, 'Depth')

                if os.path.exists(image_path):
                    video.append((image, image_depth))
                    # video.append(image_depth)
                else:
                    print(image_path, "------- Does not exist")
                    return video
    return video


def get_default_video_loader(frame_idx_offset):
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader, frame_idx_offset=frame_idx_offset)


def get_lst_data(data_file_path, root_path):
    video_names = list()
    annotations = list()

    with open(data_file_path, 'r') as data_file:
        for line in data_file.readlines():
            if len(line) == 0:
                break

            line = line.strip()
            path, depth, color, duo_left, label = line.split(' ')
            path = path.split(":")[-1]
            max_index_color = int(list(sorted(glob(os.path.join(root_path, path, "sk_color_all/*.jpg"))))[-1].split('/')[-1].split('.')[0])
            max_index_depth = int(list(sorted(glob(os.path.join(root_path, path, "sk_depth_all/*.jpg"))))[-1].split('/')[-1].split('.')[0])
            max_frame_idx = min(max_index_color, max_index_depth)

            # with open(os.path.join(root_path, path, "sk_color_log.txt"), 'r') as sk_color_log:
            #     last_line_color = sk_color_log.readlines()[-1]
            #     if len(last_line_color) == 0:
            #         last_line_color = sk_color_log.readlines()[-1]
            #
            #     assert len(last_line_color) != 0
            #
            # with open(os.path.join(root_path, path, "sk_depth_log.txt"), 'r') as sk_depth_log:
            #     last_line_depth = sk_depth_log.readlines()[-1]
            #     if len(last_line_depth) == 0:
            #         last_line_depth = sk_depth_log.readlines()[-1]
            #
            #     assert len(last_line_depth) != 0
            #
            # # indexing starts from 0
            # max_frame_idx = min(int(last_line_color[3:8]), int(last_line_depth[3:8]))

            annotation = {"start_frame": depth.split(":")[-2],
                          "end_frame": depth.split(":")[-1],
                          "label": int(label.split(":")[-1]) - 1,
                          "max_frame_idx": max_frame_idx}

            video_names.append(path)
            annotations.append(annotation)

    return video_names, annotations


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            # video_names.append('{}/{}'.format(label, key))
            video_names.append(key.split('^')[0])
            annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, sample_duration, frame_jump, num_classes, only_with_gesture):
    video_names, annotations = get_lst_data(annotation_path, root_path)

    # data = load_annotation_data(annotation_path)
    # video_names, annotations = get_video_names_and_annotations(data, subset)
    # class_to_idx = get_class_labels(data)
    # idx_to_class = {}
    # for name, label in class_to_idx.items():
    #     idx_to_class[label] = name

    dataset = list()
    print("[INFO]: NV Dataset - " + subset + " is loading...")
    for i in tqdm(range(len(video_names))):
        # if i % 100 == 0:
        #     print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])

        if not os.path.exists(video_path):
            continue

        max_frame_idx = annotations[i]['max_frame_idx']
        begin_t = int(annotations[i]['start_frame'])
        end_t = int(annotations[i]['end_frame'])

        if only_with_gesture:
            for last_frame_idx in range(end_t, begin_t, -1):
                frame_indices = sorted([last_frame_idx - (i * frame_jump) for i in range(sample_duration)])
                assert len(frame_indices) == sample_duration, (len(frame_indices), sample_duration)

                label = int(annotations[i]['label'])

                sample = {
                    'video': video_path,
                    # 'segment': [begin_t, end_t],
                    # 'n_frames': n_frames,
                    'frame_indices': frame_indices,
                    'label': label
                }

                dataset.append(sample)
        else:
            for start_frame_idx in range(max_frame_idx + 1):
                last_frame_idx = start_frame_idx + (sample_duration * frame_jump)
                if last_frame_idx > max_frame_idx:
                    break

                frame_indices = sorted(list(range(start_frame_idx, last_frame_idx, frame_jump)))
                assert len(frame_indices) == sample_duration, (len(frame_indices), sample_duration)

                if begin_t < last_frame_idx < end_t:
                    # last frame is a gesture
                    label = int(annotations[i]['label'])
                else:
                    # last frame is not a gesture, last label belongs to invalid
                    label = num_classes - 1

                # n_frames = end_t - begin_t + 1

                sample = {
                    'video': video_path,
                    # 'segment': [begin_t, end_t],
                    # 'n_frames': n_frames,
                    'frame_indices': frame_indices,
                    'label': label
                }

                dataset.append(sample)
        # if len(annotations) != 0:
        #     # sample['label'] = class_to_idx[annotations[i]['label']]
        #     sample['label'] = annotations[i]['label']
        # else:
        #     sample['label'] = -1
        #
        # if n_samples_for_each_video == 1:
        #     sample['frame_indices'] = list(range(begin_t, end_t + 1))
        #     dataset.append(sample)
        # else:
        #     if n_samples_for_each_video > 1:
        #         step = max(1, math.ceil((n_frames - 1 - sample_duration) / (n_samples_for_each_video - 1)))
        #     else:
        #         step = sample_duration
        #     for j in range(1, n_frames, step):
        #         sample_j = copy.deepcopy(sample)
        #         sample_j['frame_indices'] = list(range(j, min(n_frames + 1, j + sample_duration)))
        #         dataset.append(sample_j)

    return dataset


class NV(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 num_classes,
                 frame_jump=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 modality='RGB',
                 only_with_gesture=False,
                 img_size=None,
                 get_loader=get_default_video_loader):
        self.data = make_dataset(root_path, annotation_path, subset, sample_duration, frame_jump, num_classes,
                                 only_with_gesture)

        # self.class_names = None

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.modality = modality
        self.sample_duration = sample_duration
        self.img_size = img_size
        self.loader = get_loader(frame_idx_offset=1)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        image_depth_list = self.loader(path, frame_indices, self.modality, self.img_size)
        rgb_images = [rgb_depth[0] for rgb_depth in image_depth_list]
        depth_images = [rgb_depth[1] for rgb_depth in image_depth_list]

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            rgb_images = [self.spatial_transform(rgb_image) for rgb_image in rgb_images]
            depth_images = [self.spatial_transform(depth_image) for depth_image in depth_images]

        #rgb = torch.cat(rgb_images, 0)
        #depth = torch.cat(depth_images, 0)

        rgb = torch.stack(rgb_images, dim=1)
        depth = torch.stack(depth_images, dim=1)

        target = self.data[index]["label"]
        #print(rgb.shape)
        #print(depth.shape)

        return rgb, depth, target

    def getitem2(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices, self.modality, self.sample_duration)
        oversample_clip = []
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        im_dim = clip[0].size()[-2:]
        clip = torch.cat(clip, 0).view((self.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip[::2], clip[1::2], target

    def __len__(self):
        return len(self.data)


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
