import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def plot_confusion_matrix(_y_test, predictions, epoch, config_dict, post_fix=""):
    cm = confusion_matrix(_y_test, predictions, normalize="pred")
    print(np.unique(_y_test))
    print(np.unique(predictions))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(_y_test))
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.savefig(os.path.join(config_dict["log_dir_path"], "cm_{}_{}.png".format(post_fix, epoch)), dpi=300)


