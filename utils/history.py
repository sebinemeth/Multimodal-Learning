from typing import List, Tuple
from utils_datasets.nv_gesture.nv_utils import SubsetType, ModalityType, MetricType


class History(object):
    def __init__(self, keys: List[Tuple[SubsetType, ModalityType, MetricType]]):
        self.history_dict = {key: list() for key in keys}

    def add_items(self, items_dict: dict):
        for k, v in items_dict.items():
            assert k in self.history_dict, "key {} is not in history".format(k)
            self.history_dict[k].append(v)

    def get_last(self, key: Tuple[SubsetType, ModalityType, MetricType]):
        assert len(self.history_dict[key]) > 0, "list in history with key {} is empty".format(key)
        return self.history_dict[key][-1]

    def is_best_last(self, key: Tuple[SubsetType, ModalityType, MetricType]):
        if key[2] == MetricType.LOSS:
            value = min(self.history_dict[key])
        else:
            value = max(self.history_dict[key])
        return self.history_dict[key].index(value) == (len(self.history_dict[key]) - 1)



