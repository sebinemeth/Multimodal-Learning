import torch
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.log_maker import start_log_maker, write_log
from utils.arg_parser import get_config_dict
from utils_training.get_loaders import get_loaders

# from util import *
# from i3dpt import *
# from validation import *

start_log_maker()
config_dict = get_config_dict()

# Detect devices
use_cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
write_log("init", device, title="used device")
config_dict["device"] = device

# Initialize Tensorboard
tb_log_path = os.path.join(config_dict["base_dir_path"], "tensorboard_logs")
os.makedirs(tb_log_path, exist_ok=True)
tb_writer = SummaryWriter(log_dir=tb_log_path)

# Make save folder for the model
model_save_dir = os.path.join(config_dict["base_dir_path"], "model")
os.makedirs(model_save_dir, exist_ok=True)


train_loader, valid_loader = get_loaders(config_dict)

for k, (rgb, depth, targets) in enumerate(train_loader):
    print(rgb)
    print(depth)
    print(targets)
    input()


# model_rgb_cnn = I3D(num_classes=num_classes,
#                     modality='rgb',
#                     dropout_prob=0,
#                     name='inception').to(device)
#
#
# model_depth_cnn = I3D(num_classes=num_classes,
#                       modality='depth',
#                       dropout_prob=0,
#                       name='inception').to(device)
#
# optimizer_rgb = torch.optim.Adam(model_rgb_cnn.parameters(), lr=config_dict["base_dir_path"])  # optimize all cnn parameters
# optimizer_depth = torch.optim.Adam(model_depth_cnn.parameters(), lr=config_dict["base_dir_path"])  # optimize all cnn parameters
# criterion = torch.nn.CrossEntropyLoss()
#
#
# def regularizer(loss1, loss2, beta=2):
#     if loss1 - loss2 > 0:
#         return (beta * math.exp(loss1 - loss2)) - 1
#     return 0.0
#
#
# for epoch in range(n_epoch):
#     tq = tqdm(total=(len(train_loader)))
#     tq.set_description('ep {}, {}'.format(epoch, lr))
#     """
#     {"loss_rgb": mean_rgb, "loss_reg_rgb": mean_reg_rgb, "loss_depth": mean_depth,
#         "loss_reg_depth": mean_reg_depth}
#     """
#     train(args=args,
#           model_rgb=model_rgb_cnn,
#           model_depth=model_depth_cnn,
#           optimizer_rgb=optimizer_rgb,
#           optimizer_depth=optimizer_depth,
#           train_loader=train_loader,
#           valid_loader=valid_loader,
#           criterion=criterion,
#           regularizer=regularizer,
#           epoch=epoch,
#           tb_writer=tb_writer,
#           tq=tq)

