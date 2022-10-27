import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(_y_test, predictions, epoch,  config_dict):
    cm = confusion_matrix(_y_test, predictions, normalize="pred")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(_y_test.argmax(axis=1)))
    disp.figure_.savefig(os.path.join(config_dict["log_dir_path"], "cm_{}.png".format(epoch)), dpi=300)


