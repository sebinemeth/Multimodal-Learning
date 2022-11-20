import os
import torch
from abc import ABC
from typing import List, Tuple
from torch.utils.tensorboard import SummaryWriter

from utils.history import History
from utils.log_maker import write_log
from utils_datasets.nv_gesture.nv_utils import SubsetType, ModalityType, MetricType, keys_to_str


class EarlyStopException(Exception):
    pass


class Callback(ABC):
    def on_batch_end(self, batch_idx: int):
        pass

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
        current_value = self.history.get_epoch_last_item(self.key)

        if epoch == 0 or self.multiplier * (current_value - self.best_value) > self.delta:
            self.best_epoch = epoch
            self.best_value = current_value
        else:
            if epoch - self.best_epoch > self.patience:
                raise EarlyStopException


class SaveModel(Callback):
    def __init__(self, history: History, model: torch.nn.Module, modality: ModalityType, config_dict: dict,
                 only_best_key: Tuple[SubsetType, ModalityType, MetricType] = None):
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
        elif self.history.is_epoch_best_last(self.only_best_key):
            self.save_model("{}_{}".format(self.modality.name, "best"))

    def on_training_end(self):
        self.save_model("{}_{}".format(self.modality.name, "end"))

    def save_model(self, model_name: str):
        write_log("training", "{} model are saved".format(self.modality.name), title="save model")
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.model.state_dict()},
                   os.path.join(self.config_dict["model_save_dir"], "{}.pt".format(model_name)))


class Tensorboard(Callback):
    def __init__(self, history: History,
                 config_dict: dict,
                 batch_end_keys: List[Tuple[SubsetType, ModalityType, MetricType]],
                 epoch_end_keys: List[Tuple[SubsetType, ModalityType, MetricType]]):
        self.history = history
        self.config_dict = config_dict
        tb_log_path = os.path.join(config_dict["base_dir_path"], "tensorboard_logs")
        os.makedirs(tb_log_path, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=tb_log_path)

        self.batch_end_keys = batch_end_keys
        self.epoch_end_keys = epoch_end_keys

        self.on_batch_end_step = 0

    def on_batch_end(self, batch_idx: int):
        if batch_idx % self.config_dict["tb_batch_freq"] == 0:
            for key in self.batch_end_keys:
                if key[2] == MetricType.LOSS:
                    value = self.history.get_batch_mean(key)
                else:
                    value = self.history.get_batch_last(key)

                self.tb_writer.add_scalar(tag=keys_to_str(key),
                                          scalar_value=value,
                                          global_step=self.on_batch_end_step)

        self.on_batch_end_step += 1

    def on_epoch_end(self, epoch: int):
        for key in self.epoch_end_keys:
            value = self.history.get_epoch_last_item(key)
            self.tb_writer.add_scalar(tag=keys_to_str(key),
                                      scalar_value=value,
                                      global_step=epoch)


class CallbackRunner(object):
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks

    def on_batch_end(self, batch_idx: int):
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx)

    def on_epoch_end(self, epoch: int):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)

    def on_training_end(self):
        for callback in self.callbacks:
            callback.on_training_end()

