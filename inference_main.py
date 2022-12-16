import warnings
import json
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import sigmoid
from tqdm import tqdm
import numpy as np

from utils_training.get_models import get_models
from utils_datasets.nv_gesture.nv_dataset import NV
from utils_transforms.spatial_transforms import Compose, ToTensor, Normalize, Scale
from utils_datasets.nv_gesture.nv_utils import SubsetType, ModalityType, MetricType, NetworkType, convert_to_tqdm_dict

if __name__ == '__main__':
    # to supress pytorch nn.MaxPool3d warning
    warnings.filterwarnings("ignore", category=UserWarning)

    config_dict = {
        # "dataset_path": "/Users/sebinemeth/Nextcloud/nvGesture_v1.7z",
        # "val_annotation_path": "/Users/sebinemeth/Nextcloud/nvGesture_v1.7z/nvgesture_test_correct_cvpr2016_v2.lst",
        # "rgb_ckp_model_path": "/Users/sebinemeth/Multimodal-Learning/models/rgb_cnn.pt",
        "dataset_path": "./datasets/nvGesture",
        "val_annotation_path": "./datasets/nvGesture/nvgesture_test_correct_cvpr2016_v2.lst",
        #"rgb_ckp_model_path": "./training_outputs/multimodal_after_unimod_1/2022-12-05T12:01/model/RGB_end.pt",
        "rgb_ckp_model_path": "./training_outputs/multimodal_detector_16/2022-12-13T19:35/model/RGB_best.pt",
        "network": NetworkType.DETECTOR,
        "modalities": [ModalityType.RGB],
        "img_x": 224,
        "img_y": 224,
        "depth_x": 224,
        "depth_y": 224,
        "sample_duration": 64,
        "frame_jump": 2,
        "cover_ratio": 0.5,
        "used_classes": [10, 14, 20, 22, 24],
        "val_batch_size": 8,
        "val_shuffle_data": False,
        "val_num_of_workers": 10,
        "num_of_classes": 5,
        "dropout_prob": 0,
        "learning_rate": 0,
        "weight_decay": 0,
    }

    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    print("device: {}".format(str(device)))

    config_dict["device"] = device

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
        probability_dict = dict()
        correct_dict = dict()
        for modality in modalities:
            model_dict[modality].eval()
            predictions_dict[modality] = list()
            probability_dict[modality] = list()
            correct_dict[modality] = 0

        y_test = list()
        frame_idx_list = list()
        path_list = list()
        total = 0
        tqdm_dict = dict()

        tq = tqdm(total=(len(valid_loader)))
        tq.set_description('Inference')
        for batch_idx, (data_dict, y, data_info) in enumerate(valid_loader):
            y_test.append(y.numpy().copy())
            frame_idx_list.append(data_info[0].numpy().copy())
            path_list.append(data_info[1])
            total += y.size(0)
            y = y.to(device)

            for modality in modalities:
                data_dict[modality] = data_dict[modality].to(device)
                output, _ = model_dict[modality](data_dict[modality])

                if config_dict["network"] == NetworkType.DETECTOR:
                    probability = sigmoid(output)
                    predicted = torch.round(probability)
                elif config_dict["network"] == NetworkType.CLASSIFIER:
                    probability, predicted = output.max(1)
                else:
                    raise ValueError("unknown modality: {}".format(config_dict["network"]))

                correct_dict[modality] += predicted.eq(y).sum().item()
                predictions_dict[modality].append(predicted.cpu().numpy())
                probability_dict[modality].append(probability.cpu().numpy())
                tqdm_dict[SubsetType.VAL, modality, MetricType.ACC] = correct_dict[modality] / total

            tq.update(1)
            tq.set_postfix(**convert_to_tqdm_dict(tqdm_dict))

        tq.close()

    assert len(modalities) == 1
    predictions = np.concatenate(predictions_dict[modalities[0]], axis=0)
    probabilities = np.concatenate(probability_dict[modalities[0]], axis=0)
    y_test = np.concatenate(y_test, axis=0)
    frame_indices = np.concatenate(frame_idx_list, axis=0)
    path_list = sum(path_list, [])

    data = {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "y_test": y_test.tolist(),
        "frame_indices": frame_indices.tolist(),
        "path_list": path_list
    }

    with open('./infer_data_{}.json'.format(config_dict["network"].name), 'w') as f:
        json.dump(data, f)

