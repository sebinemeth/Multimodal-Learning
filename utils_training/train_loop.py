import math
from tqdm import tqdm
import torch
from torch.nn import Module
from torch.nn.functional import sigmoid
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict

from utils.confusion_matrix import plot_confusion_matrix
from utils_training.validation import validation_step
from utils.history import History
from utils.callbacks import CallbackRunner
from utils.discord import DiscordBot
from utils_datasets.nv_gesture.nv_utils import SubsetType, ModalityType, MetricType, NetworkType


class TrainLoop(object):
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
            for batch_idx, (data_dict, y, _) in enumerate(self.train_loader):
                y_train.append(y.numpy().copy())
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
                    if self.config_dict["network"] == NetworkType.DETECTOR:
                        loss_dict[modality] = self.criterion(output.float(), torch.unsqueeze(y, 1).float())
                    elif self.config_dict["network"] == NetworkType.CLASSIFIER:
                        loss_dict[modality] = self.criterion(output, y)
                    else:
                        raise ValueError("unknown modality: {}".format(self.config_dict["network"]))

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
                                                       modality,
                                                       MetricType.REG_LOSS): ssa_loss.item()})
                else:
                    # uni-modal case
                    for modality in self.modalities:
                        loss_dict[modality].backward()

                for modality in self.modalities:
                    self.optimizer_dict[modality].step()
                    if self.config_dict["network"] == NetworkType.DETECTOR:
                        predicted = torch.round(sigmoid(model_output_dict[modality])).detach()
                    elif self.config_dict["network"] == NetworkType.CLASSIFIER:
                        _, predicted = model_output_dict[modality].max(1)
                    else:
                        raise ValueError("unknown modality: {}".format(self.config_dict["network"]))

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
            tq.close()
            plot_confusion_matrix(y_train, predictions_dict, epoch, SubsetType.TRAIN, self.config_dict)
            validation_step(model_dict=self.model_dict, criterion=self.criterion, epoch=epoch,
                            valid_loader=self.valid_loader, config_dict=self.config_dict, history=self.history)

            self.history.end_of_epoch_train(reset_batch=True)
            self.history.print_epoch_values(epoch)
            self.callback_runner.on_epoch_end(epoch)

    @staticmethod
    def regularizer(loss1: torch.Tensor, loss2: torch.Tensor, beta: float = 2) -> float:
        if loss1 > loss2:
            return (beta * math.exp(loss1 - loss2)) - 1
        return 0.0
