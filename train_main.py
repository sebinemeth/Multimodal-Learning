import os
import traceback
import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from utils.log_maker import start_log_maker, write_log
from utils.arg_parser import get_config_dict
from utils_training.get_loaders import get_loaders
from utils_training.train_loop import TrainLoop
from utils_training.get_models import get_models

start_log_maker()
config_dict = get_config_dict()

# Detect devices
use_cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
write_log("init", str(device), title="used device")
config_dict["device"] = device

# Initialize Tensorboard
tb_log_path = os.path.join(config_dict["base_dir_path"], "tensorboard_logs")
os.makedirs(tb_log_path, exist_ok=True)
tb_writer = SummaryWriter(log_dir=tb_log_path)

# Make save folder for the model
model_save_dir = os.path.join(config_dict["base_dir_path"], "model")
os.makedirs(model_save_dir, exist_ok=True)
config_dict["model_save_dir"] = model_save_dir

train_loader, valid_loader = get_loaders(config_dict)
rgb_cnn, depth_cnn = get_models(config_dict)

if config_dict["print_summary"]:
    summary(rgb_cnn, input_size=(3, 32, 224, 224))  # (batch size, chanel, duration, width, height)
    summary(depth_cnn, input_size=(1, 32, 224, 224))  # (batch size, chanel, duration, width, height)

# optimize all cnn parameters
rgb_optimizer = torch.optim.Adam(rgb_cnn.parameters(), lr=config_dict["learning_rate"])
depth_optimizer = torch.optim.Adam(depth_cnn.parameters(), lr=config_dict["learning_rate"])
criterion = torch.nn.CrossEntropyLoss()

train_loop = TrainLoop(config_dict,
                       rgb_cnn,
                       depth_cnn,
                       rgb_optimizer,
                       depth_optimizer,
                       criterion,
                       train_loader,
                       valid_loader,
                       tb_writer)
try:
    train_loop.run_loop()
except KeyboardInterrupt:
    write_log("training", "training is stopped by keyboard interrupt", title="error", print_out=True, color="red")
except Exception:
    write_log("training", "training is stopped with error:\n{}".format(traceback.format_exc()), title="error",
              print_out=True, color="red")
finally:
    train_loop.save_models("end")
