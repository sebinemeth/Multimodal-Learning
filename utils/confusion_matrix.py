import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from typing import Dict

from utils.log_maker import write_log
from utils_datasets.nv_gesture.nv_utils import SubsetType


def plot_confusion_matrix(_y_test: list,
                          predictions_dict: Dict[list],
                          epoch: int,
                          subset_type: SubsetType,
                          config_dict: dict):
    try:
        for modality, predictions in predictions_dict.items():
            _y_test = np.concatenate(_y_test, axis=0)
            predictions = np.concatenate(predictions, axis=0)

            cm = confusion_matrix(_y_test, predictions, normalize="pred")
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config_dict["used_classes"])
            fig, ax = plt.subplots(figsize=(15, 15))
            disp.plot(ax=ax, values_format='.2f')
            cm_path = os.path.join(config_dict["log_dir_path"], "cm_{}_{}_{}.png".format(modality, subset_type, epoch))
            plt.savefig(cm_path, dpi=300)

            if "last_cm_paths" not in config_dict:
                config_dict["last_cm_paths"] = dict()

            config_dict["last_cm_paths"]["last_cm_path_{}_{}".format(modality, subset_type)] = cm_path
    except ValueError as e:
        write_log("training", "error during plot confusion matrix: {}".format(e), print_out=True, color="red",
                  title="plot cm")

