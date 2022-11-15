from abc import ABC, abstractmethod
from typing import List

from utils.history import History


class EarlyStopException(Exception):
    pass


class Callback(ABC):
    @abstractmethod
    def on_epoch_end(self, epoch: int, history: History):
        pass


class EarlyStopping(Callback):
    def __init__(self, key: str, patience: int, delta: float):
        """
        Parameters
        ----------
        key: key to the value in history, which is observed
        patience: number of epoch without improvement
        delta: minimum improvement
        """
        self.key = key
        self.patience = patience
        assert delta >= 0, "in EarlyStopping delta must be non-negative"
        self.delta = delta
        self.best_epoch = None
        self.best_value = None
        self.stop = False

        if key.find("loss") != -1:
            self.multiplier = -1
        else:
            self.multiplier = 1

    def on_epoch_end(self, epoch: int, history: History):
        current_value = history.get_last(self.key)

        if epoch == 0 or self.multiplier * (current_value - self.best_value) > self.delta:
            self.best_epoch = epoch
            self.best_value = current_value
        else:
            if epoch - self.best_epoch > self.patience:
                raise EarlyStopException


class CallbackRunner(object):
    def __init__(self, callbacks: List[Callback], history: History):
        self.callbacks = callbacks
        self.history = history

    def on_epoch_end(self, epoch: int):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, self.history)









