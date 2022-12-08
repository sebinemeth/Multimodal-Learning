import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from typing import Dict, List

from utils_datasets.nv_gesture.nv_utils import SubsetType, ModalityType, NetworkType


def plot_confusion_matrix(_y_test: List[np.ndarray],
                          predictions_dict: Dict[ModalityType, List[np.ndarray]],
                          epoch: int,
                          subset_type: SubsetType,
                          config_dict: dict):

    _y_test = np.concatenate(_y_test, axis=0)
    for modality, predictions in predictions_dict.items():
        predictions = np.concatenate(predictions, axis=0)
        cm = confusion_matrix(_y_test, predictions, normalize="pred")
        if config_dict["network"] == NetworkType.CLASSIFIER:
            display_labels = config_dict["used_classes"]
        elif config_dict["network"] == NetworkType.DETECTOR:
            display_labels = [0, 1]
        else:
            raise ValueError("network type {} is not supported". format(config_dict["network"]))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        fig, ax = plt.subplots(figsize=(15, 15))
        disp.plot(ax=ax, values_format='.2f')
        cm_path = os.path.join(config_dict["log_dir_path"], "cm_{}_{}_{}.png".format(modality.name,
                                                                                     subset_type.name,
                                                                                     epoch))
        plt.savefig(cm_path, dpi=300)
        plt.close(fig)

        if "last_cm_paths" not in config_dict:
            config_dict["last_cm_paths"] = dict()

        config_dict["last_cm_paths"]["last_cm_path_{}_{}".format(modality.name, subset_type.name)] = cm_path


