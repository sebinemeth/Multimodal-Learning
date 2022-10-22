import torch
import os
from torch.utils.tensorboard import SummaryWriter

from utils.log_maker import start_log_maker, write_log
from utils.arg_parser import get_config_dict
from utils_training.get_loaders import get_loaders
from utils_training.train_loop import TrainLoop
from utils_datasets.nv_gesture.nv_utils import ModalityType

from models.i3dpt import Inception3D

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

rgb_cnn = Inception3D(num_classes=config_dict["num_of_classes"],
                      modality=ModalityType.RGB,
                      dropout_prob=0,
                      name='inception').to(device)

depth_cnn = Inception3D(num_classes=config_dict["num_of_classes"],
                        modality=ModalityType.DEPTH,
                        dropout_prob=0,
                        name='inception').to(device)

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
train_loop.run_loop()
