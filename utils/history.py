from typing import List


class History(object):
    def __init__(self, keys: List[str]):
        self.history_dict = {key: list() for key in keys}

    def add_items(self, items_dict: dict):
        for k, v in items_dict.items():
            assert k in self.history_dict, "key {} is not in history".format(k)
            self.history_dict[k].append(v)

    def get_last(self, key: str):
        assert len(self.history_dict[key]) > 0, "list in history with key {} is empty".format(key)
        return self.history_dict[key][-1]



