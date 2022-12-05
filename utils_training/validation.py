import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

from utils.confusion_matrix import plot_confusion_matrix
from utils.history import History
from utils_datasets.nv_gesture.nv_utils import SubsetType, ModalityType, MetricType, convert_to_tqdm_dict


def validation_step(model_dict: Dict[ModalityType, Module],
                    criterion: Module,
                    epoch: int,
                    valid_loader: DataLoader,
                    config_dict: dict,
                    history: History):
    modalities = config_dict["modalities"]
    device = config_dict["device"]

    with torch.no_grad():
        predictions_dict = dict()
        correct_dict = dict()
        loss_dict = dict()
        for modality in modalities:
            model_dict[modality].eval()
            predictions_dict[modality] = list()
            correct_dict[modality] = 0
            loss_dict[modality] = list()

        y_test = list()
        total = 0
        tqdm_dict = dict()

        tq = tqdm(total=(len(valid_loader)))
        tq.set_description('Validation')
        for batch_idx, (data_dict, y) in enumerate(valid_loader):
            y_test.append(y.numpy().copy())
            total += y.size(0)
            y = y.to(device)

            for modality in modalities:
                data_dict[modality] = data_dict[modality].to(device)
                output, _ = model_dict[modality](data_dict[modality])
                loss_dict[modality].append(criterion(output, y).item())
                tqdm_dict[SubsetType.VAL, modality, MetricType.LOSS] = loss_dict[modality][-1]

                _, predicted = output.max(1)
                correct_dict[modality] += predicted.eq(y).sum().item()
                predictions_dict[modality].append(predicted.cpu().numpy())
                tqdm_dict[SubsetType.VAL, modality, MetricType.ACC] = correct_dict[modality] / total

            tq.update(1)
            tq.set_postfix(**convert_to_tqdm_dict(tqdm_dict))

        tq.close()
        history.end_of_epoch_val(tqdm_dict, loss_dict)
        plot_confusion_matrix(y_test, predictions_dict, epoch, SubsetType.VAL, config_dict)

