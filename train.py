import os
import torchvision
import argparse
import numpy as np
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
from torch.optim import Adam
from tqdm import tqdm
import math
import random

from model import CNN3D
from dataset import Senz3dDataset
from util import *
from i3dpt import *
from validation import *
import torch.nn.functional as F

data_path = "./datasets/senz3d_dataset/acquisitions"
# train_path = "./datasets/senz3d_dataset"
# test_path = "./datasets/senz3d_dataset"
# train_path = "/home/sagar/data/senz3d_dataset/dataset/train/"
# test_path = "/home/sagar/data/senz3d_dataset/dataset/test/"
img_x, img_y = 256, 256  # resize video 2d frame size
depth_x, depth_y = 320, 240
# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 8, 1
n_epoch = 10
num_classes = 11
lr = 1e-4
_lambda = 0.05  # 50 x 10^-3


def train(args,
          model_rgb,
          model_depth,
          optimizer_rgb,
          optimizer_depth,
          train_loader,
          valid_loader,
          criterion,
          regularizer,
          epoch,
          tb_writer,
          tq):
    device = args.device
    model_rgb.train()
    model_depth.train()
    rgb_losses = []
    depth_losses = []
    rgb_regularized_losses = []
    depth_regularized_losses = []
    train_result = {}
    valid_result = {}

    for batch_idx, (rgb, depth, y) in enumerate(train_loader):
        # distribute data to device
        rgb, depth, y = rgb.to(device), depth.to(device), y.to(device)

        optimizer_rgb.zero_grad()
        optimizer_depth.zero_grad()

        rgb_out, rgb_feature_map = model_rgb(rgb)
        depth_out, depth_feature_map = model_depth(depth)

        # rgb_feature_map_T = torch.transpose(rgb_feature_map, 1, 2)
        # depth_feature_map_T = torch.transpose(depth_feature_map, 1, 2)
        # # print("RGB fmap shape :: {}".format(rgb_feature_map.shape))
        # # print("depth fmap shape :: {}".format(depth_feature_map.shape))
        # # torch.save(rgb_feature_map_T, "rgbFeatureMapT.pt")
        #
        # rgb_sq_ft_map = rgb_feature_map_T.squeeze()
        # rgb_avg_sq_ft_map = torch.mean(rgb_sq_ft_map, 0)
        # depth_sq_ft_map = depth_feature_map_T.squeeze()
        # depth_avg_sq_ft_map = torch.mean(depth_sq_ft_map, 0)
        #
        # rgb_corr = torch.mul(rgb_feature_map, rgb_feature_map_T)
        # depth_corr = torch.mul(depth_feature_map, depth_feature_map_T)
        # # print("RGB correlation ::  {}".format(rgb_corr.shape))
        # # print("depth correlation :: {}".format(depth_corr.shape))
        #
        loss_rgb = criterion(rgb_out, torch.max(y, 1)[1])  # index of the max log-probability
        loss_depth = criterion(depth_out, torch.max(y, 1)[1])
        # # print("RGB loss :: {}".format(loss_rgb))
        # # print("depth loss :: {}".format(loss_depth))
        #
        # focal_reg_param = regularizer(loss_rgb, loss_depth)
        #
        # """
        # norm || x ||
        #     Take the difference element wise
        #     Square all the values
        #     Add them all together
        #     Take the square root
        #     Multiply it with rho
        # """
        # corr_diff_rgb = torch.sqrt(torch.sum(torch.sub(rgb_corr, depth_corr) ** 2))
        # corr_diff_depth = torch.sqrt(torch.sum(torch.sub(depth_corr, rgb_corr) ** 2))
        #
        # # loss (m,n)
        # ssa_loss_rgb = focal_reg_param * corr_diff_rgb
        # ssa_loss_depth = focal_reg_param * corr_diff_depth
        #
        # # total loss
        reg_loss_rgb = loss_rgb  #+ (_lambda * ssa_loss_rgb)
        reg_loss_depth = loss_depth #+ (_lambda * ssa_loss_depth)
        #
        reg_loss_rgb.backward(retain_graph=True)
        reg_loss_depth.backward()

        optimizer_rgb.step()
        optimizer_depth.step()

        rgb_losses.append(loss_rgb.item())
        depth_losses.append(loss_depth.item())
        rgb_regularized_losses.append(reg_loss_rgb.item())
        depth_regularized_losses.append(reg_loss_depth.item())
        # tq.update(1)
        # if batch_idx == 0:
        #     train_result.update({"rgb_ft_map": rgb_avg_sq_ft_map, "depth_ft_map": depth_avg_sq_ft_map})

    valid_result = validation(model_rgb=model_rgb, model_depth=model_depth, criterion=criterion,
                              valid_loader=valid_loader, num_classes=num_classes)
    mean_rgb = np.mean(rgb_losses)
    mean_reg_rgb = np.mean(rgb_regularized_losses)
    mean_depth = np.mean(depth_losses)
    mean_reg_depth = np.mean(depth_regularized_losses)
    train_result.update({"loss_rgb": mean_rgb, "loss_reg_rgb": mean_reg_rgb, "loss_depth": mean_depth,
                         "loss_reg_depth": mean_reg_depth})
    tq.set_postfix(RGB_loss='{:.5f}'.format(train_result["loss_rgb"]),
                   regularized_rgb_loss='{:.5f}'.format(train_result["loss_reg_rgb"]))
    update_tensorboard(tb_writer=tb_writer, epoch=epoch, train_dict=train_result, valid_dict=valid_result)
    update_tensorboard_image(tb_writer, epoch, train_result)


def main():
    # Detect devices
    # print(torch.__version__)
    args = parse()
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    # Initialize Tensorboard

    tb_writer = initialize_tensorboard(
        log_dir="./tensorboard_logs/",
        common_name="experiment{}".format(args.save_as))

    train_videos_path = []
    test_videos_path = []

    test_idx_list = [9, 10, 11]

    for s_idx in range(1, 5):
        for g_idx in range(1, 12):
            path = os.path.join(data_path, f"S{s_idx}", f"G{g_idx}")
            if g_idx in test_idx_list:
                test_videos_path.append(path)
            else:
                train_videos_path.append(path)

    # for folder in os.listdir(train_path):
    #     train_videos_path.append(train_path + folder)
    #
    # for folder in os.listdir(test_path):
    #     test_videos_path.append(test_path + folder)

    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

    train_rgb_set = Senz3dDataset(train_videos_path, selected_frames, to_augment=False, mode='train')
    test_rgb_set = Senz3dDataset(test_videos_path, selected_frames, to_augment=False, mode='test')

    train_loader = data.DataLoader(train_rgb_set, pin_memory=True, batch_size=1)
    valid_loader = data.DataLoader(test_rgb_set, pin_memory=True, batch_size=1)

    # model_rgb_cnn = CNN3D(t_dim=len(selected_frames), img_x=img_x, img_y=img_y, num_classes=11).to(device)
    model_rgb_cnn = I3D(num_classes=num_classes,
                        modality='rgb',
                        dropout_prob=0,
                        name='inception').to(device)
    # model_depth_cnn = CNN3D(t_dim=len(selected_frames), img_x=depth_x, img_y=depth_y, num_classes=11).to(device)
    model_depth_cnn = I3D(num_classes=num_classes,
                          modality='rgb',
                          dropout_prob=0,
                          name='inception').to(device)
    optimizer_rgb = torch.optim.Adam(model_rgb_cnn.parameters(), lr=lr)  # optimize all cnn parameters
    optimizer_depth = torch.optim.Adam(model_depth_cnn.parameters(), lr=lr)  # optimize all cnn parameters
    criterion = torch.nn.CrossEntropyLoss()
    args = parse()
    args.device = device

    def regularizer(loss1, loss2):
        beta = 2.0
        if loss1 - loss2 > 0:
            return (beta * math.exp(loss1 - loss2)) - 1
        return 0.0

    # print(model_rgb_cnn)

    for epoch in range(n_epoch):
        tq = tqdm(total=(len(train_loader)))
        tq.set_description('ep {}, {}'.format(epoch, lr))
        """
        {"loss_rgb": mean_rgb, "loss_reg_rgb": mean_reg_rgb, "loss_depth": mean_depth,
            "loss_reg_depth": mean_reg_depth}
        """
        train(args=args,
              model_rgb=model_rgb_cnn,
              model_depth=model_depth_cnn,
              optimizer_rgb=optimizer_rgb,
              optimizer_depth=optimizer_depth,
              train_loader=train_loader,
              valid_loader=valid_loader,
              criterion=criterion,
              regularizer=regularizer,
              epoch=epoch,
              tb_writer=tb_writer,
              tq=tq)


if __name__ == "__main__":
    main()
