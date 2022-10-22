from enum import Enum


class SubsetType(Enum):
    TRAIN = 1
    VALIDATION = 2


class ModalityType(Enum):
    RGB = 1
    DEPTH = 2
    RGB_DEPTH = 3

