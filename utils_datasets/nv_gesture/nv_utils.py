from enum import Enum
from typing import Tuple


class SubsetType(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class ModalityType(Enum):
    RGB = 1
    DEPTH = 2
    RGB_DEPTH = 3  # for regularization loss


class MetricType(Enum):
    LOSS = 1
    ACC = 2


def keys_to_str(keys: Tuple[SubsetType, ModalityType, MetricType]) -> str:
    return '_'.join([key.name for key in keys])


def convert_to_tqdm_dict(input_dict: dict):
    result_dict = dict()
    for key, item in input_dict.items():
        if key[2] == MetricType.LOSS:
            result_dict[keys_to_str(key)] = "{:.2f}".format(item)
        else:
            result_dict[keys_to_str(key)] = "{:.1f}%".format(item * 100)
    return result_dict
