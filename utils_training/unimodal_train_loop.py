from tqdm import tqdm
from torch.nn import Module
from torch.utils.data import DataLoader

from utils.confusion_matrix import plot_confusion_matrix
from utils_training.validation import unimodal_validation_step
from utils.history import History
from utils.callbacks import CallbackRunner
from utils.discord import DiscordBot
from utils_datasets.nv_gesture.nv_utils import SubsetType, ModalityType, MetricType


class UniModalTrainLoop(object):
    def __init__(self,
                 config_dict: dict,
                 rgb_cnn: Module,
                 rgb_optimizer,
                 criterion,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 history: History,
                 callback_runner: CallbackRunner,
                 discord: DiscordBot):
        self.config_dict = config_dict
        self.rgb_cnn = rgb_cnn
        self.rgb_optimizer = rgb_optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.history = history
        self.callback_runner = callback_runner
        self.discord = discord

    def run_loop(self):
        for epoch in range(self.config_dict["epoch"]):
            self.rgb_cnn.train()

            y_train = list()
            predictions = list()

            rgb_correct = 0
            total = 0

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

                total += y.size(0)

                _, rgb_predicted = rgb_out.max(1)
                rgb_correct += rgb_predicted.eq(y).sum().item()

                y_train.append(y.cpu().numpy())
                predictions.append(rgb_predicted.cpu().numpy())

                acc_rgb = rgb_correct / total

                tq.update(1)
                tq.set_postfix(RGB_loss='{:.2f}'.format(loss_rgb.item()),
                               RGB_acc='{:.1f}%'.format(acc_rgb * 100))

                self.history.add_batch_items({(SubsetType.TRAIN, ModalityType.RGB, MetricType.LOSS): loss_rgb.item(),
                                              (SubsetType.TRAIN, ModalityType.RGB, MetricType.ACC): acc_rgb})

                self.callback_runner.on_batch_end(batch_idx)

            plot_confusion_matrix(y_train, predictions, epoch, self.config_dict, post_fix="rgb_train")
            unimodal_validation_step(model_rgb=self.rgb_cnn, criterion=self.criterion,
                                     epoch=epoch, valid_loader=self.valid_loader,
                                     config_dict=self.config_dict, history=self.history)

            self.history.add_epoch_items(
                {(SubsetType.TRAIN, ModalityType.RGB, MetricType.LOSS): self.history.get_batch_mean((SubsetType.TRAIN,
                                                                                                     ModalityType.RGB,
                                                                                                     MetricType.LOSS)),
                 (SubsetType.TRAIN, ModalityType.RGB, MetricType.ACC): self.history.get_batch_last((SubsetType.TRAIN,
                                                                                                    ModalityType.RGB,
                                                                                                    MetricType.ACC))},
                reset_batch=True)
            self.history.print_epoch_values(epoch)
            self.callback_runner.on_epoch_end(epoch)
