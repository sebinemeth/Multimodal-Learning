import os
import numpy as np
from tqdm import tqdm
from glob import glob

from utils.log_maker import write_log
from utils_datasets.nv_gesture.nv_utils import SubsetType


def get_annot_and_video_paths(annotation_file_path, root_path):
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
            max_index_color = int(list(sorted(glob(os.path.join(root_path, path,
                                                                "sk_color_all/*.jpg"))))[-1].split('/')[-1].split('.')[0])
            max_index_depth = int(list(sorted(glob(os.path.join(root_path, path,
                                                                "sk_depth_all/*.jpg"))))[-1].split('/')[-1].split('.')[0])
            max_frame_idx = min(max_index_color, max_index_depth)

            annotation = {"start_frame": depth.split(":")[-2],
                          "end_frame": depth.split(":")[-1],
                          "label": int(label.split(":")[-1]) - 1,
                          "max_frame_idx": max_frame_idx}

            video_folders.append(path)
            annotations.append(annotation)

    return video_folders, annotations


def get_data_info_list(subset_type: SubsetType, config_dict: dict):
    """

    """
    if subset_type == SubsetType.TRAIN:
        annotation_file_path = config_dict["train_annotation_path"]
    elif subset_type == SubsetType.VALIDATION:
        annotation_file_path = config_dict["val_annotation_path"]
    else:
        raise ValueError

    video_folders, annotations = get_annot_and_video_paths(annotation_file_path, root_path=config_dict["dataset_path"])

    data_info_list = list()
    for i in tqdm(range(len(video_folders)), desc="NV Dataset - {}".format(subset_type.name)):
        # if i % 100 == 0:
        #     print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_folder = os.path.join(config_dict["dataset_path"], video_folders[i])

        if not os.path.exists(video_folder):
            write_log("dataloader", video_folder, title="Video folder is not found", print_out=True, color="red")
            continue

        max_frame_idx = annotations[i]['max_frame_idx']
        begin_t = int(annotations[i]['start_frame'])
        end_t = int(annotations[i]['end_frame'])

        frame_jump = config_dict["frame_jump"]
        sample_duration = config_dict["sample_duration"]
        num_classes = config_dict["num_classes"]

        if config_dict["only_with_gesture"]:
            # TODO: end_t + 1 does not work
            for last_frame_idx in range(min(end_t, max_frame_idx), begin_t, -1):
                frame_indices = sorted([last_frame_idx - i for i in range(sample_duration)])
                # frame_indices = sorted([last_frame_idx - (i * frame_jump) for i in range(sample_duration)])
                assert len(frame_indices) == sample_duration, (len(frame_indices), sample_duration)

                # the cut is provided before the end_t
                # the cover ratio is the ratio of frames after begin_t
                cover_ratio = np.sum(np.array(frame_indices) > begin_t) / len(frame_indices)
                if cover_ratio < config_dict["cover_ratio"]:
                    continue

                frame_indices = frame_indices[::frame_jump]
                label = int(annotations[i]['label'])

                sample = {
                    'video_folder': video_folder,
                    'frame_indices': frame_indices,
                    'label': label
                }

                data_info_list.append(sample)
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
                    'video_folder': video_folder,
                    'frame_indices': frame_indices,
                    'label': label
                }

                data_info_list.append(sample)

    return data_info_list

