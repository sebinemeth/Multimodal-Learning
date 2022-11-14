import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from utils.log_maker import write_log


def plot_confusion_matrix(_y_test, predictions, epoch, config_dict, post_fix=""):
    try:
        cm = confusion_matrix(_y_test, predictions, normalize="pred")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(_y_test))
        fig, ax = plt.subplots(figsize=(15, 15))
        disp.plot(ax=ax, values_format='.2f')
        cm_path = os.path.join(config_dict["log_dir_path"], "cm_{}_{}.png".format(post_fix, epoch))
        plt.savefig(cm_path, dpi=300)
        return cm_path
    except ValueError as e:
        write_log("training", "error during plot confusion matrix: {}".format(e), print_out=True, color="red",
                  title="plot cm")
        return ""


