import json

import numpy as np

with open("./infer_data.json") as f:
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


class C:
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


def get_accuracy(d, hit_rate_threshold, disp=False):
    hits = np.array([(d[p][0] == d[p][1]).sum() / len(d[p][0]) > hit_rate_threshold for p in d])
    if disp:
        for p in sorted(data.keys()):
            val = (data[p][0] == data[p][1]).sum() / len(data[p][0])
            print(f"{C.get(val)}{p}\t\t{val:.2f}{C.ENDC}\t{len(data[p][0])}\t{data[p][1].mean()}")

    return hits.sum() / len(d.keys())


print(get_accuracy(data, 0.2, True))
print(get_accuracy(data, 0.5))
print(get_accuracy(data, 0.75))

print()
print("all", (predictions == y_test).sum() / len(predictions))
