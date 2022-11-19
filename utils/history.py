from typing import List, Tuple
from statistics import mean

from utils_datasets.nv_gesture.nv_utils import SubsetType, ModalityType, MetricType, keys_to_str
from utils.log_maker import write_log
from utils.discord import DiscordBot


class History(object):
    def __init__(self, config_dict: dict, epoch_keys: List[Tuple[SubsetType, ModalityType, MetricType]],
                 batch_keys: List[Tuple[SubsetType, ModalityType, MetricType]], discord: DiscordBot = None):
        self.config_dict = config_dict
        self.epoch_keys = epoch_keys
        self.epoch_history_dict = {key: list() for key in epoch_keys}
        self.batch_keys = batch_keys
        self.batch_history_dict = {key: list() for key in batch_keys}

        self.discord = discord

    def add_epoch_items(self, items_dict: dict, reset_batch=False):
        for k, v in items_dict.items():
            assert k in self.epoch_history_dict, "key {} is not in history".format(k)
            self.epoch_history_dict[k].append(v)

        if reset_batch:
            self.batch_history_dict = {key: list() for key in self.batch_keys}

    def add_batch_items(self, items_dict: dict):
        for k, v in items_dict.items():
            assert k in self.batch_history_dict, "key {} is not in history".format(k)
            self.batch_history_dict[k].append(v)

    def get_epoch_last_item(self, key: Tuple[SubsetType, ModalityType, MetricType]) -> float:
        assert len(self.epoch_history_dict[key]) > 0, "list in history with key {} is empty".format(key)
        return self.epoch_history_dict[key][-1]

    def get_batch_mean(self, key: Tuple[SubsetType, ModalityType, MetricType]) -> float:
        assert len(self.batch_history_dict[key]) > 0, "list in history with key {} is empty".format(key)
        return mean(self.batch_history_dict[key])

    def get_batch_last(self, key: Tuple[SubsetType, ModalityType, MetricType]) -> float:
        assert len(self.batch_history_dict[key]) > 0, "list in history with key {} is empty".format(key)
        return self.batch_history_dict[key][-1]

    def is_epoch_best_last(self, key: Tuple[SubsetType, ModalityType, MetricType]) -> bool:
        if key[2] == MetricType.LOSS:
            value = min(self.epoch_history_dict[key])
        else:
            value = max(self.epoch_history_dict[key])
        return self.epoch_history_dict[key].index(value) == (len(self.epoch_history_dict[key]) - 1)

    def print_epoch_values(self, epoch: int):
        name_value = [("epoch", epoch)]
        for key in self.epoch_keys:
            value = self.get_epoch_last_item(key)
            name_value.append((keys_to_str(key), "{:.2f}".format(value)))

        write_log("training", ' '.join(["{}: {}".format(name, value) for name, value in name_value]), title="metrics")

        if self.discord is not None:
            self.discord.send_message(fields=[{"name": name,
                                               "value": str(value),
                                               "inline": True} for name, value in name_value],
                                      file_names=[self.config_dict["last_cm_path_rgb_train"],
                                                  self.config_dict["last_cm_path_rgb_val"]])



