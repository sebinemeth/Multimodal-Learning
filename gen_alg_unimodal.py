import threading
import yaml
import traceback
import numpy as np
from geneticalgorithm import geneticalgorithm as ga

from unimodal_train_main import main
from utils.discord import DiscordBot
from utils_datasets.nv_gesture.nv_utils import SubsetType, ModalityType, MetricType

base_config_yaml_path = "./ori_config.yaml"
ga_config_yaml_path = "./config.yaml"


def fitness(x):
    with open(base_config_yaml_path, 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)

    config_dict["weight_decay"] = float(10 ** (-x[0]))
    config_dict["dropout_prob"] = float(x[1])
    config_dict["cover_ratio"] = float(x[2])
    config_dict["frame_jump"] = int(x[3])

    config_dict["train_batch_size"] = int(5 * 50 / config_dict["sample_duration"] * min(config_dict["frame_jump"], 2))
    config_dict["val_batch_size"] = int(5 * 50 / config_dict["sample_duration"] * min(config_dict["frame_jump"], 2))

    with open(ga_config_yaml_path, 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)

    history = main()
    return history.get_epoch_last_item((SubsetType.VAL, ModalityType.RGB, MetricType.LOSS))


var_bounds = np.array([[3, 6],      # weight_decay (exp)
                       [0, 0.5],    # dropout_prob
                       [0.1, 0.6],  # cover_ratio
                       [1, 4]])     # frame_jump

var_types = np.array([['int'],
                      ['real'],
                      ['real'],
                      ['int']])

algorithm_param = {'max_num_iteration': None,
                   'population_size': 5,
                   'mutation_probability': 0.1,
                   'elit_ratio': 0.2,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type': 'uniform',
                   'max_iteration_without_improv': 5}

model = ga(function=fitness,
           dimension=len(var_bounds),
           variable_type_mixed=var_types,
           variable_boundaries=var_bounds,
           function_timeout=threading.TIMEOUT_MAX)

discord = DiscordBot()

try:
    model.run()
    print(model.report)
except Exception:
    discord.send_message(fields=[{"name": "Error",
                                  "value": "Parameter search is stopped with error: {}".format(traceback.format_exc()),
                                  "inline": True}])


