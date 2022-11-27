from enum import Enum
from typing import Tuple


class SubsetType(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class ModalityType(Enum):
    RGB = 1
    DEPTH = 2
    RGB_DEPTH = 3  # for multi-modal items


class MetricType(Enum):
    ACC = 1
    LOSS = 2
    REG_LOSS = 3


def keys_to_str(keys: Tuple[SubsetType, ModalityType, MetricType]) -> str:
    return '_'.join([key.name for key in keys])


def convert_to_tqdm_dict(input_dict: dict) -> dict:
    result_dict = dict()
    for key, item in input_dict.items():
        if key[2] == MetricType.ACC:
            result_dict[keys_to_str(key)] = "{:.1f}%".format(item * 100)
        else:
            result_dict[keys_to_str(key)] = "{:.2f}".format(item)
    return result_dict
