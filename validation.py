from torch import nn
import torch
import numpy as np


def validation(model_rgb: nn.Module, model_depth: nn.Module, criterion, valid_loader, num_classes):
    with torch.no_grad():
        model_rgb.eval()
        model_depth.eval()
        rgb_loss = []
        depth_loss = []
        rgb_correct = 0
        depth_correct = 0
        total = 0

        use_cuda = torch.cuda.is_available()  # check if GPU exists
        device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
        for batch_idx, (rgb, depth, y) in enumerate(valid_loader):
            rgb, depth, y = rgb.to(device), depth.to(device), y.to(device)
            rgb_out, rgb_feature_map = model_rgb(rgb)
            depth_out, depth_feature_map = model_depth(depth)
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

        valid_rgb_acc = rgb_correct / total
        valid_depth_acc = depth_correct / total
        valid_rgb_loss = np.mean(rgb_loss)  # type: float
        valid_depth_loss = np.mean(depth_loss)  # type: float
        return {'valid_rgb_loss': valid_rgb_loss, 'valid_depth_loss': valid_depth_loss,
                'valid_rgb_acc': valid_rgb_acc, 'valid_depth_acc': valid_depth_acc}
