import threading
import yaml
import numpy as np
from geneticalgorithm import geneticalgorithm as ga

from unimodal_train_main import main

base_config_yaml_path = "./ori_config.yaml"
ga_config_yaml_path = "./config.yaml"


def fitness(x):
    with open(base_config_yaml_path, 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)

    config_dict["learning_rate"] = float(10 ** (-x[0]))
    config_dict["dropout_prob"] = float(x[1])
    config_dict["cover_ratio"] = float(x[2])
    config_dict["sample_duration"] = int(x[3])
    config_dict["frame_jump"] = int(x[4])

    config_dict["train_batch_size"] = int(6 * 64 / config_dict["sample_duration"] * config_dict["frame_jump"])
    config_dict["val_batch_size"] = int(6 * 64 / config_dict["sample_duration"] * config_dict["frame_jump"])

    with open(ga_config_yaml_path, 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)

    history = main()
    return history.get_last("val_rgb_loss")


var_bounds = np.array([[3, 4],      # learning_rate
                       [0, 0.5],    # dropout_prob
                       [0.1, 1],    # cover_ratio
                       [16, 84],    # sample_duration
                       [1, 4]])     # frame_jump

var_types = np.array([['int'],
                      ['real'],
                      ['real'],
                      ['int'],
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

model.run()
print(model.report)
