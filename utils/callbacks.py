import os
import torch
from abc import ABC, abstractmethod
from typing import List, Tuple

from utils.history import History
from utils.log_maker import write_log
from utils_datasets.nv_gesture.nv_utils import SubsetType, ModalityType, MetricType


class EarlyStopException(Exception):
    pass


class Callback(ABC):
    @abstractmethod
    def on_epoch_end(self, epoch: int):
        pass

    def on_training_end(self):
        pass


class EarlyStopping(Callback):
    def __init__(self, history: History, key: Tuple[SubsetType, ModalityType, MetricType], patience: int, delta: float):
        """
        Parameters
        ----------
        history: History object with loss and accuracy values
        key: key to the value in history, which is observed
        patience: number of epoch without improvement
        delta: minimum improvement
        """
        self.history = history
        self.key = key
        self.patience = patience
        assert delta >= 0, "in EarlyStopping delta must be non-negative"
        self.delta = delta
        self.best_epoch = None
        self.best_value = None
        self.stop = False

        if self.key[2] == MetricType.LOSS:
            self.multiplier = -1
        else:
            self.multiplier = 1

    def on_epoch_end(self, epoch: int):
        current_value = self.history.get_last(self.key)

        if epoch == 0 or self.multiplier * (current_value - self.best_value) > self.delta:
            self.best_epoch = epoch
            self.best_value = current_value
        else:
            if epoch - self.best_epoch > self.patience:
                raise EarlyStopException


class SaveModel(Callback):
    def __init__(self, history: History, model: torch.nn.Module, modality: ModalityType, config_dict: dict,
                 only_best_key: str = None):
        """
        Parameters
        ----------
        history: History object with loss and accuracy values
        model: torch model to save
        modality: used modality
        config_dict: dictionary with all the parameters
        only_best_key: model is saved only if the value for this key in the history is the best
        """
        self.history = history
        self.model = model
        self.modality = modality
        self.config_dict = config_dict
        self.only_best_key = only_best_key

    def on_epoch_end(self, epoch: int):
        if self.only_best_key is None:
            self.save_model("{}_{}".format(self.modality.name, epoch))
        elif self.history.is_best_last(self.only_best_key):
            self.save_model("{}_{}".format(self.modality.name, "best"))

    def on_training_end(self):
        self.save_model("{}_{}".format(self.modality.name, "end"))

    def save_model(self, model_name: str):
        write_log("training", "{} model are saved".format(self.modality.name), title="save model")
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.model.state_dict()},
                   os.path.join(self.config_dict["model_save_dir"], "{}.pt".format(model_name)))


class CallbackRunner(object):
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks

    def on_epoch_end(self, epoch: int):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)

    def on_training_end(self):
        for callback in self.callbacks:
            callback.on_training_end()
