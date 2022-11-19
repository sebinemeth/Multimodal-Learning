import torch
from torch import nn
from statistics import mean
from tqdm import tqdm

from utils.confusion_matrix import plot_confusion_matrix
from utils.history import History
from utils_datasets.nv_gesture.nv_utils import SubsetType, ModalityType, MetricType


def validation_step(model_rgb: nn.Module, model_depth: nn.Module, criterion, valid_loader, epoch, config_dict):
    with torch.no_grad():
        model_rgb.eval()
        model_depth.eval()
        rgb_loss = list()
        depth_loss = list()
        y_test = list()
        rgb_predictions = list()
        depth_predictions = list()
        rgb_correct = 0
        depth_correct = 0
        total = 0

        use_cuda = torch.cuda.is_available()  # check if GPU exists
        device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

        tq = tqdm(total=(len(valid_loader)))
        tq.set_description('Validation')
        for batch_idx, (rgb, depth, y) in enumerate(valid_loader):
            rgb, depth, y = rgb.to(device), depth.to(device), y.to(device)
            rgb_out, _ = model_rgb(rgb)
            depth_out, _ = model_depth(depth)
            # loss_rgb = criterion(rgb_out, torch.max(y, 1)[1])  # index of the max log-probability
            # loss_depth = criterion(depth_out, torch.max(y, 1)[1])
            loss_rgb = criterion(rgb_out, y)  # index of the max log-probability
            loss_depth = criterion(depth_out, y)
            rgb_loss.append(loss_rgb.item())
            depth_loss.append(loss_depth.item())

            total += y.size(0)

            _, rgb_predicted = rgb_out.max(1)
            rgb_correct += rgb_predicted.eq(y).sum().item()

            _, depth_predicted = depth_out.max(1)
            depth_correct += depth_predicted.eq(y).sum().item()

            y_test.append(y.cpu().numpy())
            rgb_predictions.append(rgb_predicted.cpu().numpy())
            depth_predictions.append(depth_predicted.cpu().numpy())

            acc_rgb = rgb_correct / total
            acc_depth = depth_correct / total

            tq.update(1)
            tq.set_postfix(RGB_loss='{:.2f}'.format(mean(rgb_loss)),
                           DEPTH_loss='{:.2f}'.format(mean(depth_loss)),
                           RGB_acc='{:.1f}%'.format(acc_rgb * 100),
                           DEPTH_acc='{:.1f}%'.format(acc_depth * 100))

        valid_rgb_acc = rgb_correct / total
        valid_depth_acc = depth_correct / total
        valid_rgb_loss = mean(rgb_loss)  # type: float
        valid_depth_loss = mean(depth_loss)  # type: float

        plot_confusion_matrix(y_test, rgb_predictions, epoch, config_dict, post_fix="rgb")
        plot_confusion_matrix(y_test, depth_predictions, epoch, config_dict, post_fix="depth")

        return {'valid_rgb_loss': valid_rgb_loss, 'valid_depth_loss': valid_depth_loss,
                'valid_rgb_acc': valid_rgb_acc, 'valid_depth_acc': valid_depth_acc}


def unimodal_validation_step(model_rgb: nn.Module, criterion, valid_loader, epoch: int, config_dict: dict,
                             history: History):
    with torch.no_grad():
        model_rgb.eval()
        y_test = list()
        predictions = list()

        rgb_loss_list = list()
        rgb_correct = 0
        total = 0

        use_cuda = torch.cuda.is_available()  # check if GPU exists
        device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

        tq = tqdm(total=(len(valid_loader)))
        tq.set_description('Validation')
        for batch_idx, (rgb, _, y) in enumerate(valid_loader):
            rgb, y = rgb.to(device), y.to(device)
            rgb_out, _ = model_rgb(rgb)

            loss_rgb = criterion(rgb_out, y)  # index of the max log-probability
            rgb_loss_list.append(loss_rgb.item())

            total += y.size(0)

            _, rgb_predicted = rgb_out.max(1)
            rgb_correct += rgb_predicted.eq(y).sum().item()

            y_test.append(y.cpu().numpy())
            predictions.append(rgb_predicted.cpu().numpy())

            acc_rgb = rgb_correct / total

            tq.update(1)
            tq.set_postfix(RGB_loss='{:.2f}'.format(rgb_loss_list[-1]),
                           RGB_acc='{:.1f}%'.format(acc_rgb * 100))

        history.add_epoch_items({
            (SubsetType.VAL, ModalityType.RGB, MetricType.LOSS): rgb_correct / total,
            (SubsetType.VAL, ModalityType.RGB, MetricType.ACC): mean(rgb_loss_list)},
            reset_batch=False)
        plot_confusion_matrix(y_test, predictions, epoch, config_dict, post_fix="rgb_val")
