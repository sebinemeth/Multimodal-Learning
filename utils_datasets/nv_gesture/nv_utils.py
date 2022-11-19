from enum import Enum


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

