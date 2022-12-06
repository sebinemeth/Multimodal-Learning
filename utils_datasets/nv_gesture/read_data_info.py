import os
import numpy as np
from tqdm import tqdm
from glob import glob
from statistics import mean
from typing import Tuple, List

from utils.log_maker import write_log
from utils_datasets.nv_gesture.nv_utils import SubsetType, NetworkType


def print_stat(data_stat_dict: dict, num_of_all_samples: int):
    log_lines = list()
    for label in sorted(data_stat_dict.keys()):
        num_of_samples = data_stat_dict[label]["num_of_samples"]
        percentage = num_of_samples / num_of_all_samples * 100

        num_of_zeros = data_stat_dict[label]["num_of_zeros"]
        zero_percentage = num_of_zeros / data_stat_dict[label]["num_of_frames"] * 100

        len_of_gesture = mean(data_stat_dict[label]["len_of_gesture"])
        line = "label: {}, avg lo gesture: {:.1f}, no. samples: {} ({:.2f}%)," \
               " no. zeros: {} ({:.1f}%)".format(label,
                                                 len_of_gesture,
                                                 num_of_samples,
                                                 percentage,
                                                 num_of_zeros,
                                                 zero_percentage)
        log_lines.append(line)

    write_log("dataloader", "\n".join(log_lines), title="Data statistics", print_out=True, color="blue")


def get_annot_and_video_paths(annotation_file_path: str, root_path: str) -> Tuple[List[str], List[dict]]:
    """
    one line:
    path:./Video_data/class_01/subject13_r0 ...
    depth:sk_depth:138:218 ...
    color:sk_color:138:218 ...
    duo_left:duo_left:161:254 ...
    label:1
    """
    video_folders = list()
    annotations = list()

    with open(annotation_file_path, 'r') as data_file:
        for line in data_file.readlines():
            if len(line) == 0:
                break

            line = line.strip()
            path, depth, color, duo_left, label = line.split(' ')
            path = path.split(":")[-1]
            max_index_color = int(
                sorted(glob(os.path.join(root_path, path, "sk_color_all/*.jpg")))[-1].split('/')[-1].split('.')[0])
            max_index_depth = int(
                sorted(glob(os.path.join(root_path, path, "sk_depth_all/*.jpg")))[-1].split('/')[-1].split('.')[0])
            max_frame_idx = min(max_index_color, max_index_depth)

            annotation = {"start_frame": depth.split(":")[-2],
                          "end_frame": depth.split(":")[-1],
                          "label": int(label.split(":")[-1]) - 1,
                          "max_frame_idx": max_frame_idx}

            video_folders.append(path)
            annotations.append(annotation)

    return video_folders, annotations


def get_data_info_list(subset_type: SubsetType, config_dict: dict) -> List[dict]:
    """

    """
    if subset_type == SubsetType.TRAIN:
        annotation_file_path = config_dict["train_annotation_path"]
    elif subset_type == SubsetType.VAL:
        annotation_file_path = config_dict["val_annotation_path"]
    else:
        raise ValueError

    video_folders, annotations = get_annot_and_video_paths(annotation_file_path, root_path=config_dict["dataset_path"])

    data_info_list = list()
    data_stat_dict = dict()
    for i in tqdm(range(len(video_folders)), desc="NV Dataset - {}".format(subset_type.name)):
        video_folder = os.path.join(config_dict["dataset_path"], video_folders[i])

        if not os.path.exists(video_folder):
            write_log("dataloader", video_folder, title="Video folder is not found", print_out=True, color="red")
            continue

        max_frame_idx = annotations[i]['max_frame_idx']
        begin_t = int(annotations[i]['start_frame'])
        end_t = int(annotations[i]['end_frame'])

        frame_jump = config_dict["frame_jump"]
        sample_duration = config_dict["sample_duration"]

        if config_dict["network"] == NetworkType.CLASSIFIER:
            label = int(annotations[i]['label'])

            if len(config_dict["used_classes"]) > 0 and label not in config_dict["used_classes"]:
                # if config_dict["used_classes"] is empty, each class will be used
                continue

            if len(config_dict["used_classes"]) > 0:
                # map labels from 0 to num of classes - 1
                label = config_dict["used_classes"].index(label)

            if label not in data_stat_dict:
                data_stat_dict[label] = {"num_of_samples": 0, "num_of_zeros": 0, "num_of_frames": 0,
                                         "len_of_gesture": list()}
            data_stat_dict[label]["len_of_gesture"].append(min(end_t, max_frame_idx) - begin_t)

            for last_frame_idx in range(min(end_t, max_frame_idx), begin_t, -1):
                frame_indices = sorted([last_frame_idx - i for i in range(sample_duration)])
                assert len(frame_indices) == sample_duration, (len(frame_indices), sample_duration)

                # the cut is provided before the end_t
                # the cover ratio is the ratio of frames after begin_t
                cover_ratio = np.sum(np.array(frame_indices) > begin_t) / len(frame_indices)

                if cover_ratio < config_dict["cover_ratio"]:
                    continue

                frame_indices = frame_indices[::frame_jump]

                sample = {
                    'video_folder': video_folder,
                    'frame_indices': frame_indices,
                    'label': label
                }

                data_info_list.append(sample)
                data_stat_dict[label]["num_of_samples"] += 1
                data_stat_dict[label]["num_of_zeros"] += len([idx for idx in frame_indices if idx < 0])
                data_stat_dict[label]["num_of_frames"] += len(frame_indices)
        elif config_dict["network"] == NetworkType.DETECTOR:
            for last_frame_idx in range(1, max_frame_idx + 1):
                frame_indices = sorted([last_frame_idx - i for i in range(sample_duration)])
                assert len(frame_indices) == sample_duration, (len(frame_indices), sample_duration)

                # the cut is provided before the end_t
                # the cover ratio is the ratio of frames after begin_t
                cover_ratio = np.sum(np.logical_and(np.array(frame_indices) > begin_t,
                                                    np.array(frame_indices) < end_t)) / len(frame_indices)

                if cover_ratio < config_dict["cover_ratio"]:
                    label = 0
                else:
                    label = 1

                frame_indices = frame_indices[::frame_jump]

                sample = {
                    'video_folder': video_folder,
                    'frame_indices': frame_indices,
                    'label': label
                }
                if label not in data_stat_dict:
                    data_stat_dict[label] = {"num_of_samples": 0, "num_of_zeros": 0, "num_of_frames": 0,
                                             "len_of_gesture": list()}
                data_stat_dict[label]["len_of_gesture"].append(min(end_t, max_frame_idx) - begin_t)

                data_info_list.append(sample)
                data_stat_dict[label]["num_of_samples"] += 1
                data_stat_dict[label]["num_of_zeros"] += len([idx for idx in frame_indices if idx < 0])
                data_stat_dict[label]["num_of_frames"] += len(frame_indices)
        else:
            raise ValueError("network type error: {}".format(config_dict["network"]))
    print_stat(data_stat_dict, len(data_info_list))
    return data_info_list
