from enum import Enum
from typing import Tuple


class SubsetType(Enum):
    TRAIN = 1
    VAL = 2
    Test = 3


class ModalityType(Enum):
    RGB = 1
    DEPTH = 2
    RGB_DEPTH = 3


class MetricType(Enum):
    LOSS = 1
    ACC = 2


def keys_to_str(keys: Tuple[SubsetType, ModalityType, MetricType]) -> str:
    return '_'.join([key.name for key in keys])

