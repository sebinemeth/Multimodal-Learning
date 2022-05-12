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

from spatial_transforms import Compose, ToTensor, Normalize, Scale
from temporal_transforms import TemporalRandomCrop

from model import CNN3D
from dataset import Senz3dDataset
from nv_dataset import NV
from util import *
from i3dpt import *
from validation import *
import torch.nn.functional as F
from torchsummary import summary

video_path = "./datasets/nvGesture"
train_annotation_path = "./datasets/nvGesture/nvgesture_train_correct_cvpr2016_v2.lst"
valid_annotation_path = "./datasets/nvGesture/nvgesture_test_correct_cvpr2016_v2.lst"
model_save_dir = "./saved_models/model"
# train_path = "./datasets/senz3d_dataset"
# test_path = "./datasets/senz3d_dataset"
# train_path = "/home/sagar/data/senz3d_dataset/dataset/train/"
# test_path = "/home/sagar/data/senz3d_dataset/dataset/test/"

# TODO
# img_x, img_y = 320, 240  # resize video 2d frame size
img_x, img_y = 224, 224  # resize video 2d frame size
depth_x, depth_y = 224, 224
sample_duration = 64
# depth_x, depth_y = 320, 240
# Select which frame to begin & end in videos
# begin_frame, end_frame, skip_frame = 1, 8, 1
n_epoch = 10
only_with_gesture = True
num_classes = 25 if only_with_gesture else 25 + 1
lr = 1e-4
_lambda = 0.05  # 50 x 10^-3

tb_step = 0


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

    rgb_correct = 0
    depth_correct = 0
    total = 0

    tb_batch_freq = 20
    global tb_step

    for batch_idx, (rgb, depth, y) in enumerate(train_loader):
        # distribute data to device
        rgb, depth, y = rgb.to(device), depth.to(device), y.to(device)  # F.one_hot(y).to(device)

        optimizer_rgb.zero_grad()
        optimizer_depth.zero_grad()

        rgb_out, rgb_feature_map = model_rgb(rgb)
        depth_out, depth_feature_map = model_depth(depth)

        rgb_feature_map = rgb_feature_map.view(rgb_feature_map.shape[0], rgb_feature_map.shape[1], -1)
        rgb_feature_map_T = torch.transpose(rgb_feature_map, 1, 2)

        depth_feature_map = depth_feature_map.view(depth_feature_map.shape[0], depth_feature_map.shape[1], -1)
        depth_feature_map_T = torch.transpose(depth_feature_map, 1, 2)
        # print("RGB fmap shape :: {}".format(rgb_feature_map.shape))
        # print("RGB fmap shape T :: {}".format(rgb_feature_map_T.shape))
        # print("depth fmap shape :: {}".format(depth_feature_map.shape))
        # print("depth fmap shape T:: {}".format(depth_feature_map_T.shape))
        # torch.save(rgb_feature_map_T, "rgbFeatureMapT.pt")

        rgb_sq_ft_map = rgb_feature_map_T.squeeze()
        rgb_avg_sq_ft_map = torch.mean(rgb_sq_ft_map, 0)
        depth_sq_ft_map = depth_feature_map_T.squeeze()
        depth_avg_sq_ft_map = torch.mean(depth_sq_ft_map, 0)

        rgb_corr = torch.bmm(rgb_feature_map_T, rgb_feature_map)
        depth_corr = torch.bmm(depth_feature_map_T, depth_feature_map)
        # print("RGB correlation ::  {}".format(rgb_corr.shape))
        # print("depth correlation :: {}".format(depth_corr.shape))

        # print("RGB  ::  {}".format(rgb_out.shape))
        # print("y :: {}".format(y))

        # loss_rgb = criterion(rgb_out, torch.max(y, 1)[1])  # index of the max log-probability
        # loss_depth = criterion(depth_out, torch.max(y, 1)[1])
        loss_rgb = criterion(rgb_out, y)  # index of the max log-probability
        loss_depth = criterion(depth_out, y)
        # print("RGB loss :: {}".format(loss_rgb))
        # print("depth loss :: {}".format(loss_depth))

        focal_reg_param = regularizer(loss_rgb, loss_depth)

        """
        norm || x ||
            Take the difference element wise
            Square all the values
            Add them all together
            Take the square root
            Multiply it with rho
        """
        corr_diff_rgb = torch.sqrt(torch.sum(torch.sub(rgb_corr, depth_corr) ** 2))
        corr_diff_depth = torch.sqrt(torch.sum(torch.sub(depth_corr, rgb_corr) ** 2))

        # loss (m,n)
        ssa_loss_rgb = focal_reg_param * corr_diff_rgb
        ssa_loss_depth = focal_reg_param * corr_diff_depth

        # total loss
        reg_loss_rgb = loss_rgb + (_lambda * ssa_loss_rgb)
        reg_loss_depth = loss_depth + (_lambda * ssa_loss_depth)

        reg_loss_rgb.backward(retain_graph=True)
        reg_loss_depth.backward()

        optimizer_rgb.step()
        optimizer_depth.step()

        rgb_losses.append(loss_rgb.item())
        depth_losses.append(loss_depth.item())
        rgb_regularized_losses.append(reg_loss_rgb.item())
        depth_regularized_losses.append(reg_loss_depth.item())

        total += y.size(0)

        _, rgb_predicted = rgb_out.max(1)
        rgb_correct += rgb_predicted.eq(y).sum().item()

        _, depth_predicted = depth_out.max(1)
        depth_correct += depth_predicted.eq(y).sum().item()

        acc_rgb = rgb_correct / total
        acc_depth = depth_correct / total

        tq.update(1)
        if batch_idx == 0:
            train_result.update({"rgb_ft_map": rgb_avg_sq_ft_map, "depth_ft_map": depth_avg_sq_ft_map})
        tq.set_postfix(RGB_loss='{:.2f}'.format(rgb_losses[-1]),
                       regularized_rgb_loss='{:.2f}'.format(rgb_regularized_losses[-1]),
                       acc_rgb='{:.1f}%'.format(acc_rgb * 100),
                       acc_depth='{:.1f}%'.format(acc_depth * 100))

        if batch_idx % tb_batch_freq == 0:
            mean_rgb = np.mean(rgb_losses)
            mean_reg_rgb = np.mean(rgb_regularized_losses)
            mean_depth = np.mean(depth_losses)
            mean_reg_depth = np.mean(depth_regularized_losses)
            train_result.update({"loss_rgb": mean_rgb, "loss_reg_rgb": mean_reg_rgb, "acc_rgb": acc_rgb,
                                 "loss_depth": mean_depth, "loss_reg_depth": mean_reg_depth, "acc_depth": acc_depth})
            update_tensorboard_train(tb_writer=tb_writer, epoch=tb_step, train_dict=train_result)
            update_tensorboard_image(tb_writer, tb_step, train_result)

            tb_step += 1

            rgb_losses = []
            depth_losses = []
            rgb_regularized_losses = []
            depth_regularized_losses = []

    valid_result = validation(model_rgb=model_rgb, model_depth=model_depth, criterion=criterion,
                              valid_loader=valid_loader, num_classes=num_classes)
    update_tensorboard_val(tb_writer=tb_writer, epoch=epoch, valid_dict=valid_result)

    torch.save(model_rgb.state_dict(), os.path.join(model_save_dir, "model_rgb_{}".format(epoch)))
    torch.save(model_depth.state_dict(), os.path.join(model_save_dir, "model_depth_{}".format(epoch)))
    # mean_rgb = np.mean(rgb_losses)
    # mean_reg_rgb = np.mean(rgb_regularized_losses)
    # mean_depth = np.mean(depth_losses)
    # mean_reg_depth = np.mean(depth_regularized_losses)
    # train_result.update({"loss_rgb": mean_rgb, "loss_reg_rgb": mean_reg_rgb, "loss_depth": mean_depth,
    #                      "loss_reg_depth": mean_reg_depth})
    # tq.set_postfix(RGB_loss='{:.5f}'.format(train_result["loss_rgb"]),
    #                regularized_rgb_loss='{:.5f}'.format(train_result["loss_reg_rgb"]))
    # update_tensorboard(tb_writer=tb_writer, epoch=epoch, train_dict=train_result, valid_dict=valid_result)
    # update_tensorboard_image(tb_writer, epoch, train_result)
    # tb_writer.flush()


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

    global model_save_dir
    model_save_dir += args.save_as

    train_videos_path = []
    test_videos_path = []

    # test_idx_list = [9, 10, 11]

    # for s_idx in range(1, 5):
    #     for g_idx in range(1, 12):
    #         path = os.path.join(data_path, f"S{s_idx}", f"G{g_idx}")
    #         if g_idx in test_idx_list:
    #             test_videos_path.append(path)
    #         else:
    #             train_videos_path.append(path)

    # for folder in os.listdir(train_path):
    #     train_videos_path.append(train_path + folder)
    #
    # for folder in os.listdir(test_path):
    #     test_videos_path.append(test_path + folder)

    # TODO Scale(img_x)
    # selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()
    # print("len(selected_frames): {}".format(len(selected_frames)))

    norm_method = Normalize([0, 0, 0], [1, 1, 1])

    training_data = NV(
        video_path,
        train_annotation_path,
        "train",
        num_classes,
        frame_jump=1,
        spatial_transform=Compose([Scale((img_x, img_y)), ToTensor(), norm_method]),
        # temporal_transform=TemporalRandomCrop(sample_duration, downsample=1),
        # target_transform=target_transform,
        sample_duration=sample_duration,
        modality="RGB-D",
        only_with_gesture=only_with_gesture,
        img_size=(img_x, img_y)
    )

    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=2,
        shuffle=True,
        num_workers=10,
        pin_memory=True)

    valid_data = NV(
        video_path,
        valid_annotation_path,
        "valid",
        num_classes,
        frame_jump=1,
        spatial_transform=Compose([Scale((img_x, img_y)), ToTensor(), norm_method]),
        # temporal_transform=TemporalRandomCrop(sample_duration, downsample=1),
        # target_transform=target_transform,
        sample_duration=sample_duration,
        modality="RGB-D",
        only_with_gesture=only_with_gesture,
        img_size=(img_x, img_y)
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=2,
        shuffle=False,
        num_workers=10,
        pin_memory=True)

    # train_rgb_set = Senz3dDataset(train_videos_path, selected_frames, to_augment=False, mode='train')
    # test_rgb_set = Senz3dDataset(test_videos_path, selected_frames, to_augment=False, mode='test')

    # train_loader = data.DataLoader(train_rgb_set, pin_memory=True, batch_size=1)
    # valid_loader = data.DataLoader(test_rgb_set, pin_memory=True, batch_size=1)

    # model_rgb_cnn = CNN3D(t_dim=sample_duration, ch_in=3, img_x=img_x, img_y=img_y, num_classes=num_classes).to(device)
    # summary(model_rgb_cnn, input_size=(sample_duration * 3, img_y, img_x))
    model_rgb_cnn = I3D(num_classes=num_classes,
                        modality='rgb',
                        dropout_prob=0,
                        name='inception').to(device)

    # model_depth_cnn = CNN3D(t_dim=sample_duration, ch_in=1, img_x=depth_x, img_y=depth_y, num_classes=num_classes).to(device)
    # summary(model_depth_cnn, input_size=(sample_duration, img_y, img_x))
    model_depth_cnn = I3D(num_classes=num_classes,
                          modality='depth',
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
