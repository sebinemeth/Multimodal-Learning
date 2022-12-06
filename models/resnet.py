from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Type

from utils_datasets.nv_gesture.nv_utils import ModalityType


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block: Union[Type[BasicBlock], Type[Bottleneck]],
                 layers: list,
                 config_dict: dict,
                 conv1_t_size: int = 7,
                 conv1_t_stride: int = 1,
                 no_max_pool: bool = False,
                 shortcut_type: str = 'B',
                 widen_factor: float = 1.0
                 ):
        super().__init__()
        block_inplanes = [int(x * widen_factor) for x in [64, 128, 256, 512]]
        if config_dict["modality"] == ModalityType.RGB:
            n_input_channels = 3
        elif config_dict["modality"] == ModalityType.DEPTH:
            n_input_channels = 1
        else:
            raise ValueError("modality type is not supported: {}".format(config_dict["modality"]))

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, config_dict["num_of_classes"])

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _down_sample_basic_block(x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        down_sample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                down_sample = partial(self._downsample_basic_block,
                                      planes=planes * block.expansion,
                                      stride=stride)
            else:
                down_sample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = list()
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  down_sample=down_sample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feature_map = x
        #####################################################################################

        variance, sample_mean = torch.var_mean(feature_map)
        sub_map = torch.sub(feature_map, sample_mean)
        correlation_matrix = torch.div(sub_map, variance)

        #####################################################################################

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, correlation_matrix


def generate_resnet_model(model_depth: int, config_dict: dict) -> nn.Module:
    supported_depth_list = (10, 18, 34, 50, 101, 152, 200)

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], config_dict)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], config_dict)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], config_dict)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], config_dict)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], config_dict)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], config_dict)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], config_dict)
    else:
        raise ValueError("given model depth {} is  not supported {}".format(model_depth, supported_depth_list))

    return model
