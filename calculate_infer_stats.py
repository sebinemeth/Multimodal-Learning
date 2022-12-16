import json
import numpy as np


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


def read_infer_json_old(json_path):
    with open(json_path) as f:
        json_data = json.load(f)

    predictions = np.array(json_data["predictions"])
    y_test = np.array(json_data["y_test"])
    frame_indices = json_data["frame_indices"]
    path_list = json_data["path_list"]

    data = dict()

    for pred, y, idx, path in zip(predictions, y_test, frame_indices, path_list):
        # path = "/".join(path.split("/")[:-1])
        if path not in data:
            data[path] = [np.array([]), np.array([])]
        data[path][0] = np.append(data[path][0], pred)
        data[path][1] = np.append(data[path][1], y)
    return data


def read_infer_json(json_path):
    with open(json_path) as f:
        json_data = json.load(f)

    predictions = json_data["predictions"]
    y_test = json_data["y_test"]
    frame_indices = json_data["frame_indices"]
    path_list = json_data["path_list"]

    data = dict()

    for pred, y, idx, path in zip(predictions, y_test, frame_indices, path_list):
        if path not in data:
            data[path] = dict()
        data[path][idx] = (pred, y)
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

# for path, (pred, y) in detector_data:
#     pass


print(detector_data)
exit()


print(get_accuracy(data, 0.2, True))
print(get_accuracy(data, 0.5))
print(get_accuracy(data, 0.75))

print()
print("all", (predictions == y_test).sum() / len(predictions))
