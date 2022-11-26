import math
from tqdm import tqdm
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict

from utils.confusion_matrix import plot_confusion_matrix
from utils_training.validation import validation_step
from utils.history import History
from utils.callbacks import CallbackRunner
from utils.discord import DiscordBot
from utils_datasets.nv_gesture.nv_utils import SubsetType, ModalityType, MetricType


class UniModalTrainLoop(object):
    def __init__(self,
                 config_dict: dict,
                 model_dict: Dict[ModalityType, Module],
                 optimizer_dict: Dict[ModalityType, Optimizer],
                 criterion: Module,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 history: History,
                 callback_runner: CallbackRunner,
                 discord: DiscordBot):
        self.config_dict = config_dict
        self.modalities = self.config_dict["modalities"]
        self.device = self.config_dict["device"]
        self._lambda = self.config_dict["lambda"]
        self.model_dict = model_dict
        self.optimizer_dict = optimizer_dict
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.history = history
        self.callback_runner = callback_runner
        self.discord = discord

    def run_loop(self):
        for epoch in range(self.config_dict["epoch"]):
            predictions_dict = dict()
            correct_dict = dict()
            for modality in self.modalities:
                self.model_dict[modality].train()
                predictions_dict[modality] = list()
                correct_dict[modality] = 0

            y_train = list()
            total = 0

            tq = tqdm(total=(len(self.train_loader)))
            tq.set_description('ep {}'.format(epoch))
            for batch_idx, (data_dict, y) in enumerate(self.train_loader):
                y_train.append(y.numpy())
                y = y.to(self.device)
                total += y.size(0)

                model_output_dict = dict()
                feature_map_dict = dict()
                loss_dict = dict()
                for modality in self.modalities:
                    data_dict[modality] = data_dict[modality].to(self.device)
                    self.optimizer_dict[modality].zero_grad()

                    output, feature_map = self.model_dict[modality](data_dict[modality])
                    model_output_dict[modality] = output
                    feature_map_dict[modality] = feature_map
                    loss_dict[modality] = self.criterion(output, y)

                # only in multi-modal case
                if len(self.modalities) > 1:
                    correlation_dict = dict()
                    corr_diff_dict = dict()
                    focal_reg_dict = dict()
                    for modality in self.modalities:
                        feature_map = feature_map_dict[modality].view(feature_map_dict[modality].shape[0],
                                                                      feature_map_dict[modality].shape[1],
                                                                      -1)
                        correlation_dict[modality] = torch.bmm(torch.transpose(feature_map, 1, 2), feature_map)

                    corr_diff_dict[ModalityType.RGB] = torch.sqrt(
                        torch.sum(torch.sub(correlation_dict[ModalityType.RGB],
                                            correlation_dict[ModalityType.DEPTH]) ** 2))
                    corr_diff_dict[ModalityType.DEPTH] = torch.sqrt(
                        torch.sum(torch.sub(correlation_dict[ModalityType.DEPTH],
                                            correlation_dict[ModalityType.RGB]) ** 2))

                    focal_reg_dict[ModalityType.RGB] = self.regularizer(loss_dict[ModalityType.RGB],
                                                                        loss_dict[ModalityType.DEPTH])
                    focal_reg_dict[ModalityType.DEPTH] = self.regularizer(loss_dict[ModalityType.DEPTH],
                                                                          loss_dict[ModalityType.RGB])

                    for modality in self.modalities:
                        # loss (m,n)
                        ssa_loss = self._lambda * focal_reg_dict[modality] * corr_diff_dict[modality]
                        # total loss
                        reg_loss = loss_dict[modality] + ssa_loss
                        reg_loss.backward(retain_graph=modality == ModalityType.RGB)
                        self.history.add_batch_items({(SubsetType.TRAIN,
                                                       ModalityType.RGB_DEPTH,
                                                       MetricType.LOSS): ssa_loss.item()})
                else:
                    # uni-modal case
                    for modality in self.modalities:
                        loss_dict[modality].backward()

                for modality in self.modalities:
                    self.optimizer_dict[modality].step()
                    _, predicted = model_output_dict[modality].max(1)
                    correct_dict[modality] += predicted.eq(y).sum().item()
                    predictions_dict[modality].append(predicted.cpu().numpy())
                    acc = correct_dict[modality] / total

                    self.history.add_batch_items({(SubsetType.TRAIN,
                                                   modality,
                                                   MetricType.LOSS): loss_dict[modality].item(),
                                                  (SubsetType.TRAIN,
                                                   modality,
                                                   MetricType.ACC): acc})

                tq.update(1)
                tq.set_postfix(**self.history.get_batch_tqdm_dict())
                self.callback_runner.on_batch_end(batch_idx)

            # epoch end
            plot_confusion_matrix(y_train, predictions_dict, epoch, SubsetType.TRAIN, self.config_dict)
            validation_step(model_dict=self.model_dict, criterion=self.criterion, epoch=epoch,
                            valid_loader=self.valid_loader, config_dict=self.config_dict, history=self.history)

            epoch_end_dict = dict()
            for key in self.history.epoch_keys:
                if key[2] == MetricType.LOSS:
                    epoch_end_dict[key] = self.history.get_batch_mean(key)
                else:
                    epoch_end_dict[key] = self.history.get_batch_last(key)

            self.history.add_epoch_items(epoch_end_dict, reset_batch=True)
            self.history.print_epoch_values(epoch)
            self.callback_runner.on_epoch_end(epoch)

    @staticmethod
    def regularizer(loss1: torch.Tensor, loss2: torch.Tensor, beta: float = 2) -> float:
        if loss1 > loss2:
            return (beta * math.exp(loss1 - loss2)) - 1
        return 0.0

#
# class TrainLoop(object):
#     def __init__(self,
#                  config_dict: dict,
#                  rgb_cnn: torch.nn.Module,
#                  depth_cnn: torch.nn.Module,
#                  rgb_optimizer,
#                  depth_optimizer,
#                  criterion,
#                  train_loader,
#                  valid_loader,
#                  tb_writer,
#                  discord):
#         self.config_dict = config_dict
#         self.rgb_cnn = rgb_cnn
#         self.depth_cnn = depth_cnn
#         self.rgb_optimizer = rgb_optimizer
#         self.depth_optimizer = depth_optimizer
#         self.criterion = criterion
#         self.train_loader = train_loader
#         self.valid_loader = valid_loader
#         self.tb_writer = tb_writer
#         self.discord = discord
#
#     def run_loop(self):
#         for epoch in range(self.config_dict["epoch"]):
#             self.rgb_cnn.train()
#             self.depth_cnn.train()
#
#             rgb_losses = [[]]
#             depth_losses = [[]]
#             rgb_regularized_losses = [[]]
#             depth_regularized_losses = [[]]
#             train_result = dict()
#
#             rgb_correct = 0
#             depth_correct = 0
#             total = 0
#             tb_step = 0
#
#             tq = tqdm(total=(len(self.train_loader)))
#             tq.set_description('ep {}'.format(epoch))
#             for batch_idx, (rgb, depth, y) in enumerate(self.train_loader):
#                 # distribute data to device
#                 rgb, depth = rgb.to(self.config_dict["device"]), depth.to(self.config_dict["device"])
#                 y = y.to(self.config_dict["device"])
#
#                 self.rgb_optimizer.zero_grad()
#                 self.depth_optimizer.zero_grad()
#
#                 rgb_out, rgb_feature_map = self.rgb_cnn(rgb)
#                 depth_out, depth_feature_map = self.depth_cnn(depth)
#
#                 rgb_feature_map = rgb_feature_map.view(rgb_feature_map.shape[0], rgb_feature_map.shape[1], -1)
#                 rgb_feature_map_T = torch.transpose(rgb_feature_map, 1, 2)
#
#                 depth_feature_map = depth_feature_map.view(depth_feature_map.shape[0], depth_feature_map.shape[1], -1)
#                 depth_feature_map_T = torch.transpose(depth_feature_map, 1, 2)
#
#                 rgb_corr = torch.bmm(rgb_feature_map_T, rgb_feature_map)
#                 depth_corr = torch.bmm(depth_feature_map_T, depth_feature_map)
#
#                 loss_rgb = self.criterion(rgb_out, y)  # index of the max log-probability
#                 loss_depth = self.criterion(depth_out, y)
#
#                 rgb_focal_reg_param = self.regularizer(loss_rgb, loss_depth)
#                 depth_focal_reg_param = self.regularizer(loss_depth, loss_rgb)
#
#                 """
#                 norm || x ||
#                     Take the difference element wise
#                     Square all the values
#                     Add them all together
#                     Take the square root
#                     Multiply it with rho
#                 """
#                 corr_diff_rgb = torch.sqrt(torch.sum(torch.sub(rgb_corr, depth_corr) ** 2))
#                 corr_diff_depth = torch.sqrt(torch.sum(torch.sub(depth_corr, rgb_corr) ** 2))
#
#                 # loss (m,n)
#                 ssa_loss_rgb = self.config_dict["lambda"] * rgb_focal_reg_param * corr_diff_rgb
#                 ssa_loss_depth = self.config_dict["lambda"] * depth_focal_reg_param * corr_diff_depth
#
#                 # total loss
#                 reg_loss_rgb = loss_rgb + ssa_loss_rgb
#                 reg_loss_depth = loss_depth + ssa_loss_depth
#
#                 reg_loss_rgb.backward(retain_graph=True)
#                 reg_loss_depth.backward()
#
#                 self.rgb_optimizer.step()
#                 self.depth_optimizer.step()
#
#                 rgb_losses[-1].append(loss_rgb.item())
#                 depth_losses[-1].append(loss_depth.item())
#                 rgb_regularized_losses[-1].append(ssa_loss_rgb.item())
#                 depth_regularized_losses[-1].append(ssa_loss_depth.item())
#
#                 total += y.size(0)
#
#                 _, rgb_predicted = rgb_out.max(1)
#                 rgb_correct += rgb_predicted.eq(y).sum().item()
#
#                 _, depth_predicted = depth_out.max(1)
#                 depth_correct += depth_predicted.eq(y).sum().item()
#
#                 acc_rgb = rgb_correct / total
#                 acc_depth = depth_correct / total
#
#                 tq.update(1)
#                 tq.set_postfix(RGB_loss='{:.2f}'.format(rgb_losses[-1][-1]),
#                                DEPTH_loss='{:.2f}'.format(depth_losses[-1][-1]),
#                                RGB_reg_loss='{:.2f}'.format(rgb_regularized_losses[-1][-1]),
#                                DEPTH_reg_loss='{:.2f}'.format(depth_regularized_losses[-1][-1]),
#                                RGB_acc='{:.1f}%'.format(acc_rgb * 100),
#                                DEPTH_acc='{:.1f}%'.format(acc_depth * 100))
#
#                 if self.config_dict["write_feature_map"] and batch_idx == 0:
#                     rgb_sq_ft_map = rgb_feature_map_T.squeeze()
#                     rgb_avg_sq_ft_map = torch.mean(rgb_sq_ft_map, 0)
#                     depth_sq_ft_map = depth_feature_map_T.squeeze()
#                     depth_avg_sq_ft_map = torch.mean(depth_sq_ft_map, 0)
#                     train_result.update({"rgb_ft_map": rgb_avg_sq_ft_map, "depth_ft_map": depth_avg_sq_ft_map})
#                     update_tensorboard_image(self.tb_writer, tb_step, train_result)
#
#                 if batch_idx % self.config_dict["tb_batch_freq"] == 0:
#                     mean_rgb = np.mean(rgb_losses[-1])
#                     mean_reg_rgb = np.mean(rgb_regularized_losses[-1])
#                     mean_depth = np.mean(depth_losses[-1])
#                     mean_reg_depth = np.mean(depth_regularized_losses[-1])
#                     train_result.update({"loss_rgb": mean_rgb, "loss_reg_rgb": mean_reg_rgb, "acc_rgb": acc_rgb,
#                                          "loss_depth": mean_depth, "loss_reg_depth": mean_reg_depth,
#                                          "acc_depth": acc_depth})
#                     update_tensorboard_train(tb_writer=self.tb_writer, global_step=tb_step, train_dict=train_result)
#
#                     tb_step += 1
#                     rgb_losses.append([])
#                     depth_losses.append([])
#                     rgb_regularized_losses.append([])
#                     depth_regularized_losses.append([])
#
#             valid_result = validation_step(model_rgb=self.rgb_cnn, model_depth=self.depth_cnn, criterion=self.criterion,
#                                            valid_loader=self.valid_loader, config_dict=self.config_dict, epoch=epoch)
#             update_tensorboard_val(tb_writer=self.tb_writer, global_step=epoch, valid_dict=valid_result)
#             write_log("training", "epoch: {}, "
#                                   "RGB_loss: {:.2f}, "
#                                   "RGB_reg_loss: {:.2f}, "
#                                   "RGB_acc: {:.1f}%, "
#                                   "DEPTH_loss: {:.2f}, "
#                                   "DEPTH_reg_loss: {:.2f}, "
#                                   "DEPTH_acc: {:.1f}%, "
#                                   "val_RGB_loss: {:.2f}, "
#                                   "val_RGB_acc: {:.1f}%, "
#                                   "val_DEPTH_loss: {:.2f}, "
#                                   "val_DEPTH_acc: {:.1f}%".format(epoch,
#                                                                   np.mean(rgb_losses[-2]),
#                                                                   np.mean(rgb_regularized_losses[-2]),
#                                                                   acc_rgb * 100,
#                                                                   np.mean(depth_losses[-2]),
#                                                                   np.mean(depth_regularized_losses[-2]),
#                                                                   acc_depth * 100,
#                                                                   valid_result["valid_rgb_loss"],
#                                                                   valid_result["valid_rgb_acc"] * 100,
#                                                                   valid_result["valid_depth_loss"],
#                                                                   valid_result["valid_depth_acc"] * 100
#                                                                   ), title="metrics")
#             self.discord.send_message(fields=[{"name": "Epoch", "value": "{}".format(epoch), "inline": True},
#                                               {"name": "RGB_loss",
#                                                "value": "{:.2f}".format(np.mean(rgb_losses[-2])),
#                                                "inline": True},
#                                               {"name": "RGB_reg_loss",
#                                                "value": "{:.2f}".format(np.mean(rgb_regularized_losses[-2])),
#                                                "inline": True},
#                                               {"name": "RGB_acc",
#                                                "value": "{:.1f}%".format(acc_rgb * 100),
#                                                "inline": True},
#                                               {"name": "val_RGB_loss",
#                                                "value": "{:.2f}".format(valid_result["valid_rgb_loss"]),
#                                                "inline": True},
#                                               {"name": "val_RGB_acc",
#                                                "value": "{:.1f}%".format(valid_result["valid_rgb_acc"] * 100),
#                                                "inline": True},
#                                               {"name": "DEPTH_loss",
#                                                "value": "{:.2f}".format(np.mean(depth_losses[-2])),
#                                                "inline": True},
#                                               {"name": "DEPTH_reg_loss",
#                                                "value": "{:.2f}".format(np.mean(depth_regularized_losses[-2])),
#                                                "inline": True},
#                                               {"name": "DEPTH_acc",
#                                                "value": "{:.1f}%".format(acc_depth * 100),
#                                                "inline": True},
#                                               {"name": "val_DEPTH_loss",
#                                                "value": "{:.2f}".format(valid_result["valid_depth_loss"]),
#                                                "inline": True},
#                                               {"name": "val_DEPTH_acc",
#                                                "value": "{:.1f}%".format(valid_result["valid_depth_acc"] * 100),
#                                                "inline": True}])
#             self.save_models(epoch)
#
#     def save_models(self, epoch):
#         write_log("training", "models are saved", title="save models")
#         torch.save({'epoch': epoch,
#                     'model_state_dict': self.rgb_cnn.state_dict(),
#                     'optimizer_state_dict': self.rgb_optimizer.state_dict()},
#                    os.path.join(self.config_dict["model_save_dir"], "model_rgb_{}.pt".format(epoch)))
#
#         torch.save({'epoch': epoch,
#                     'model_state_dict': self.depth_cnn.state_dict(),
#                     'optimizer_state_dict': self.depth_optimizer.state_dict()},
#                    os.path.join(self.config_dict["model_save_dir"], "model_depth_{}.pt".format(epoch)))
#
#     @staticmethod
#     def regularizer(loss1, loss2, beta=2):
#         if loss1 > loss2:
#             return (beta * math.exp(loss1 - loss2)) - 1
#         return 0.0
