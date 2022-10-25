import torch
import os
import numpy as np
from tqdm import tqdm

from utils.tensorboard_utils import update_tensorboard_train, update_tensorboard_val
from utils_training.validation import unimodal_validation_step
from utils.log_maker import write_log


class UniModalTrainLoop(object):
    def __init__(self,
                 config_dict: dict,
                 rgb_cnn: torch.nn.Module,
                 rgb_optimizer,
                 criterion,
                 train_loader,
                 valid_loader,
                 tb_writer):
        self.config_dict = config_dict
        self.rgb_cnn = rgb_cnn
        self.rgb_optimizer = rgb_optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.tb_writer = tb_writer

    def run_loop(self):
        for epoch in range(self.config_dict["epoch"]):
            self.rgb_cnn.train()

            rgb_losses = list()
            rgb_regularized_losses = list()
            train_result = dict()

            rgb_correct = 0
            total = 0
            tb_step = 0

            tq = tqdm(total=(len(self.train_loader)))
            tq.set_description('ep {}, {}'.format(epoch, self.config_dict["learning_rate"]))
            for batch_idx, (rgb, _, y) in enumerate(self.train_loader):
                # distribute data to device
                rgb = rgb.to(self.config_dict["device"])
                y = y.to(self.config_dict["device"])

                self.rgb_optimizer.zero_grad()

                rgb_out, _ = self.rgb_cnn(rgb)
                loss_rgb = self.criterion(rgb_out, y)  # index of the max log-probability
                loss_rgb.backward()

                self.rgb_optimizer.step()

                rgb_losses.append(loss_rgb.item())
                rgb_regularized_losses.append(loss_rgb.item())

                total += y.size(0)

                _, rgb_predicted = rgb_out.max(1)
                rgb_correct += rgb_predicted.eq(y).sum().item()

                acc_rgb = rgb_correct / total

                tq.update(1)
                tq.set_postfix(RGB_loss='{:.2f}'.format(rgb_losses[-1]),
                               RGB_reg_loss='{:.2f}'.format(rgb_regularized_losses[-1]),
                               RGB_acc='{:.1f}%'.format(acc_rgb * 100))

                if batch_idx % self.config_dict["tb_batch_freq"] == 0:
                    mean_rgb = np.mean(rgb_losses)
                    mean_reg_rgb = np.mean(rgb_regularized_losses)
                    train_result.update({"loss_rgb": mean_rgb, "loss_reg_rgb": mean_reg_rgb, "acc_rgb": acc_rgb})
                    update_tensorboard_train(tb_writer=self.tb_writer, global_step=tb_step, train_dict=train_result,
                                             only_rgb=True)

                    tb_step += 1

                    rgb_losses = list()
                    rgb_regularized_losses = list()

            valid_result = unimodal_validation_step(model_rgb=self.rgb_cnn, criterion=self.criterion,
                                                    valid_loader=self.valid_loader)
            update_tensorboard_val(tb_writer=self.tb_writer, global_step=tb_step, valid_dict=valid_result,
                                   only_rgb=True)
            self.save_model(epoch)

    def save_model(self, epoch):
        write_log("training", "models are saved", title="save models")
        torch.save(self.rgb_cnn.state_dict(),
                   os.path.join(self.config_dict["model_save_dir"], "model_rgb_{}.pt".format(epoch)))