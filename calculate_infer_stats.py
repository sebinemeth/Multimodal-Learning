import json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pylab as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class C(object):
    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_CYAN = '\033[96m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def get(v, v_min=0.0, v_max=1.0):
        if v_min + v < (v_max - v_min) * 0.25:
            return C.FAIL
        if v_min + v < (v_max - v_min) * 0.5:
            return C.WARNING
        if v_min + v < (v_max - v_min) * 0.75:
            return C.OK_BLUE
        return C.OK_GREEN


def read_infer_json(json_path):
    with open(json_path) as f:
        json_data = json.load(f)

    predictions = np.array(json_data["predictions"])
    probabilities = np.array(json_data["probabilities"])
    y_test = np.array(json_data["y_test"])
    frame_indices = json_data["frame_indices"]
    path_list = json_data["path_list"]

    data = dict()

    for _prob, _pred, _y, idx, _path in zip(probabilities, predictions, y_test, frame_indices, path_list):
        class_number = _path.split('/')[-2]
        if class_number not in data:
            data[class_number] = [np.array([]), np.array([]), np.array([]), np.array([])]
        data[class_number][0] = np.append(data[class_number][0], _prob)
        data[class_number][1] = np.append(data[class_number][1], _pred)
        data[class_number][2] = np.append(data[class_number][2], _y)
        data[class_number][3] = np.append(data[class_number][3], idx)
    return data


def read_infer_json_2(json_path):
    with open(json_path) as f:
        json_data = json.load(f)

    predictions = np.array(json_data["predictions"])
    probabilities = np.array(json_data["probabilities"])
    y_test = np.array(json_data["y_test"])
    frame_indices = np.array(json_data["frame_indices"])
    path_list = json_data["path_list"]

    data = dict()

    for _prob, _pred, _y, idx, _path in zip(probabilities, predictions, y_test, frame_indices, path_list):
        if path not in data:
            data[_path] = dict()
        if isinstance(_pred, list):
            data[_path][idx] = (_prob[0], _pred[0], _y)
        else:
            data[_path][idx] = (_prob, _pred, _y)
    return data


def get_accuracy(_data, hit_rate_threshold, disp=False):
    hits = np.array([(_data[p][0] == _data[p][1]).sum() / len(_data[p][0]) > hit_rate_threshold for p in _data])
    if disp:
        for p in sorted(_data.keys()):
            val = (_data[p][0] == _data[p][1]).sum() / len(_data[p][0])
            print(f"{C.get(val)}{p}\t\t{val:.2f}{C.ENDC}\t{len(_data[p][0])}\t{_data[p][1].mean()}")

    return hits.sum() / len(_data.keys())


classifier_data = read_infer_json("./infer_data_CLASSIFIER.json")
detector_data = read_infer_json("./infer_data_DETECTOR.json")

plot = False

# DETECTOR investigation

accuracy_results = dict()
all_pred = dict()
all_y = list()
for class_number, infer_data in detector_data.items():
    accuracy_results[class_number] = list()
    y = infer_data[2].astype(bool)
    all_y.append(y)
    for threshold in range(101):
        pred = (infer_data[0] > (threshold / 100)).astype(bool)
        accuracy = (pred == y).sum() / len(infer_data[0])
        accuracy_results[class_number].append(accuracy)

        if threshold not in all_pred:
            all_pred[threshold] = list()
        all_pred[threshold].append(pred)

all_y = np.concatenate(all_y, axis=0)
accuracy_result_all = list()
for threshold in sorted(all_pred.keys()):
    all_pred[threshold] = np.concatenate(all_pred[threshold], axis=0)
    accuracy = (all_pred[threshold] == all_y).sum() / len(all_pred[threshold])
    accuracy_result_all.append(accuracy)

accuracy_result_all = np.array(accuracy_result_all)
max_acc = np.max(accuracy_result_all)
max_th = np.argmax(accuracy_result_all)
ratio_of_true = all_y.sum() / len(all_y)

if plot:
    plt.plot(sorted(all_pred.keys()), accuracy_result_all)
    plt.vlines(max_th, ymin=0, ymax=1, colors="red", linestyles="dashed")
    plt.title('max acc: {:.2f} at th: {}, ratio of true: {:.2f}'.format(max_acc, max_th, ratio_of_true))
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=len(accuracy_results.keys()), figsize=(36, 6))
    for plot_idx, (class_number, acc_result) in enumerate(accuracy_results.items()):
        y = detector_data[class_number][2].astype(bool)
        ratio_of_true = y.sum() / len(y)

        acc_result = np.array(acc_result)
        max_acc = np.max(acc_result)
        max_th = np.argmax(acc_result)

        axes[plot_idx].plot(range(101), acc_result)
        axes[plot_idx].vlines(max_th, ymin=0, ymax=1, colors="red", linestyles="dashed")
        axes[plot_idx].set_title(
            'max acc: {:.2f} at th: {}, ratio of true: {:.2f}'.format(max_acc, max_th, ratio_of_true))
        axes[plot_idx].set_xlabel("Threshold")
        axes[plot_idx].set_ylabel("Accuracy")

    plt.show()

# with time window
plot = False
if False:
    max_acc_list = list()
    max_th_list = list()
    for window_size in range(1, 10):
        print(window_size)
        score_results = dict()
        all_pred = dict()
        all_y = list()
        for class_number, infer_data in detector_data.items():
            score_results[class_number] = list()
            y = infer_data[2].astype(bool)[window_size - 1:]
            all_y.append(y)
            for threshold in range(101):
                pred = (infer_data[0] > (threshold / 100))
                window_pred = np.round_(sliding_window_view(pred, window_size).mean(axis=-1)).astype(bool)
                # window_prob = sliding_window_view(infer_data[0], window_size).mean(axis=-1)
                # window_pred = (window_prob > (threshold / 100))
                # accuracy = (window_pred == y).sum() / len(y)

                # score_results[class_number].append([accuracy_score(y, window_pred),
                #                                     precision_score(y, window_pred),
                #                                     recall_score(y, window_pred),
                #                                     f1_score(y, window_pred)])

                if threshold not in all_pred:
                    all_pred[threshold] = list()
                all_pred[threshold].append(window_pred)

        all_y = np.concatenate(all_y, axis=0)
        score_results_all = list()
        for threshold in sorted(all_pred.keys()):
            all_pred[threshold] = np.concatenate(all_pred[threshold], axis=0)
            assert len(all_pred[threshold]) == len(all_y), (len(all_pred[threshold]), len(all_y))
            # accuracy = (all_pred[threshold] == all_y).sum() / len(all_pred[threshold])
            score_results_all.append([round(accuracy_score(all_y, all_pred[threshold]), 2),
                                      round(precision_score(all_y, all_pred[threshold]), 2),
                                      round(recall_score(all_y, all_pred[threshold]), 2),
                                      round(f1_score(all_y, all_pred[threshold]), 2)])

        score_results_all = np.array(score_results_all)
        # max_acc = np.max(score_results_all, axis=0)
        max_th = np.argmax(score_results_all, axis=0)
        max_acc = score_results_all[max_th[-1]]
        ratio_of_true = all_y.sum() / len(all_y)
        max_acc_list.append(max_acc)
        max_th_list.append(max_th)

    print(max_acc_list)
    print(max_th_list)

if plot:
    plt.plot(sorted(all_pred.keys()), accuracy_result_all)
    plt.vlines(max_th, ymin=0, ymax=1, colors="red", linestyles="dashed")
    plt.title('max acc: {:.2f} at th: {}, ratio of true: {:.2f}'.format(max_acc, max_th, ratio_of_true))
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=len(accuracy_results.keys()), figsize=(36, 6))
    for plot_idx, (class_number, acc_result) in enumerate(accuracy_results.items()):
        y = detector_data[class_number][2].astype(bool)
        ratio_of_true = y.sum() / len(y)

        acc_result = np.array(acc_result)
        max_acc = np.max(acc_result)
        max_th = np.argmax(acc_result)

        axes[plot_idx].plot(range(101), acc_result)
        axes[plot_idx].vlines(max_th, ymin=0, ymax=1, colors="red", linestyles="dashed")
        axes[plot_idx].set_title(
            'max acc: {:.2f} at th: {}, ratio of true: {:.2f}'.format(max_acc, max_th, ratio_of_true))
        axes[plot_idx].set_xlabel("Threshold")
        axes[plot_idx].set_ylabel("Accuracy")

    plt.show()


# CLASSIFIER investigation

max_acc_list = list()
max_th_list = list()
for window_size in range(1, 30):
    print(window_size)
    score_results = dict()
    all_pred = list()
    all_y = list()
    for class_number, infer_data in classifier_data.items():
        score_results[class_number] = list()
        y = infer_data[2][window_size - 1:]
        all_y.append(y)

        window_pred = sliding_window_view(infer_data[1], window_size)
        window_prob = sliding_window_view(infer_data[0], window_size)

        values = np.unique(window_pred, axis=None)
        value_prob_list = list()
        for value in range(5):
            value_prob = window_prob.copy()
            value_prob[window_pred != value] = 0
            value_prob_list.append(np.mean(value_prob, axis=-1))

        value_probs_prob = np.array(value_prob_list).T
        window_pred = value_probs_prob.argmax(axis=-1)
        # breakpoint()
        # window_prob = sliding_window_view(infer_data[0], window_size).mean(axis=-1)
        # window_pred = (window_prob > (threshold / 100))
        # accuracy = (window_pred == y).sum() / len(y)

        score_results[class_number].append([accuracy_score(y, window_pred),
                                            precision_score(y, window_pred, average=None),
                                            recall_score(y, window_pred, average=None),
                                            f1_score(y, window_pred, average=None)])
        all_pred.append(window_pred)

    all_y = np.concatenate(all_y, axis=0)
    score_results_all = list()

    all_pred = np.concatenate(all_pred, axis=0)
    assert len(all_pred) == len(all_y), (len(all_pred), len(all_y))
    # accuracy = (all_pred[threshold] == all_y).sum() / len(all_pred[threshold])
    score_results_all.append([accuracy_score(all_y, all_pred),
                              precision_score(all_y, all_pred, average=None),
                              recall_score(all_y, all_pred, average=None),
                              f1_score(all_y, all_pred, average=None)])

    print(score_results_all)
    # score_results_all = np.array(score_results_all)
    # max_acc = np.max(score_results_all, axis=0)
#     max_th = np.argmax(score_results_all, axis=0)
#     max_acc = score_results_all[max_th[-1]]
#     ratio_of_true = all_y.sum() / len(all_y)
#     max_acc_list.append(max_acc)
#     max_th_list.append(max_th)
#
# print(max_acc_list)
# print(max_th_list)


