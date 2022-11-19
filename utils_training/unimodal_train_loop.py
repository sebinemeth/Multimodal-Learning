import torch
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.confusion_matrix import plot_confusion_matrix
from utils.tensorboard_utils import update_tensorboard_train, update_tensorboard_val
from utils_training.validation import unimodal_validation_step
from utils.log_maker import write_log
from utils.history import History
from utils.callbacks import CallbackRunner
from utils.discord import DiscordBot
from utils_datasets.nv_gesture.nv_utils import SubsetType, ModalityType, MetricType


class UniModalTrainLoop(object):
    def __init__(self,
                 config_dict: dict,
                 rgb_cnn: torch.nn.Module,
                 rgb_optimizer,
                 criterion,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 history: History,
                 callback_runner: CallbackRunner,
                 tb_writer: SummaryWriter,
                 discord: DiscordBot):
        self.config_dict = config_dict
        self.rgb_cnn = rgb_cnn
        self.rgb_optimizer = rgb_optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.history = history
        self.callback_runner = callback_runner
        self.tb_writer = tb_writer
        self.discord = discord

    def run_loop(self):
        for epoch in range(self.config_dict["epoch"]):
            self.rgb_cnn.train()

            rgb_losses = [[]]
            train_result = dict()

            y_train = list()
            predictions = list()

            rgb_correct = 0
            total = 0
            tb_step = 0

            tq = tqdm(total=(len(self.train_loader)))
            tq.set_description('ep {}'.format(epoch))
            for batch_idx, (rgb, _, y) in enumerate(self.train_loader):
                # distribute data to device
                rgb = rgb.to(self.config_dict["device"])
                y = y.to(self.config_dict["device"])

                self.rgb_optimizer.zero_grad()

                rgb_out, _ = self.rgb_cnn(rgb)
                loss_rgb = self.criterion(rgb_out, y)  # index of the max log-probability
                loss_rgb.backward()

                self.rgb_optimizer.step()

                rgb_losses[-1].append(loss_rgb.item())

                total += y.size(0)

                _, rgb_predicted = rgb_out.max(1)
                rgb_correct += rgb_predicted.eq(y).sum().item()

                y_train.append(y.cpu().numpy())
                predictions.append(rgb_predicted.cpu().numpy())

                acc_rgb = rgb_correct / total

                tq.update(1)
                tq.set_postfix(RGB_loss='{:.2f}'.format(rgb_losses[-1][-1]),
                               RGB_acc='{:.1f}%'.format(acc_rgb * 100))

                if batch_idx % self.config_dict["tb_batch_freq"] == 0:
                    mean_rgb = np.mean(rgb_losses[-1])
                    train_result.update({"loss_rgb": mean_rgb, "acc_rgb": acc_rgb})
                    update_tensorboard_train(tb_writer=self.tb_writer, global_step=tb_step, train_dict=train_result,
                                             only_rgb=True)
                    tb_step += 1
                    rgb_losses.append([])

            plot_confusion_matrix(np.concatenate(y_train, axis=0), np.concatenate(predictions, axis=0),
                                  epoch, self.config_dict, post_fix="rgb_train")
            valid_result = unimodal_validation_step(model_rgb=self.rgb_cnn, criterion=self.criterion,
                                                    epoch=epoch, valid_loader=self.valid_loader,
                                                    config_dict=self.config_dict)
            update_tensorboard_val(tb_writer=self.tb_writer, global_step=epoch, valid_dict=valid_result, only_rgb=True)
            self.history.add_items({(SubsetType.TRAIN, ModalityType.RGB, MetricType.LOSS): np.mean(sum(rgb_losses, [])),
                                    (SubsetType.TRAIN, ModalityType.RGB, MetricType.ACC): acc_rgb,
                                    (SubsetType.VAL, ModalityType.RGB, MetricType.LOSS): valid_result["valid_rgb_loss"],
                                    (SubsetType.VAL, ModalityType.RGB, MetricType.ACC): valid_result["valid_rgb_acc"]})
            write_log("training", "epoch: {},"
                                  " RGB_loss: {:.2f},"
                                  " RGB_acc: {:.1f}%,"
                                  " val_RGB_loss: {:.2f},"
                                  " val_RGB_acc: {:.1f}%".format(epoch,
                                                                 np.mean(sum(rgb_losses, [])),
                                                                 acc_rgb * 100,
                                                                 valid_result["valid_rgb_loss"],
                                                                 valid_result["valid_rgb_acc"] * 100),
                      title="metrics")
            self.discord.send_message(fields=[{"name": "Epoch", "value": "{}".format(epoch), "inline": True},
                                              {"name": "RGB_loss",
                                               "value": "{:.2f}".format(np.mean(sum(rgb_losses, []))),
                                               "inline": True},
                                              {"name": "RGB_acc",
                                               "value": "{:.1f}%".format(acc_rgb * 100),
                                               "inline": True},
                                              {"name": "val_RGB_loss",
                                               "value": "{:.2f}".format(valid_result["valid_rgb_loss"]),
                                               "inline": True},
                                              {"name": "val_RGB_acc",
                                               "value": "{:.1f}%".format(valid_result["valid_rgb_acc"] * 100),
                                               "inline": True}],
                                      file_names=[self.config_dict["last_cm_path_rgb_train"],
                                                  self.config_dict["last_cm_path_rgb_val"]]
                                      )
            self.callback_runner.on_epoch_end(epoch)

