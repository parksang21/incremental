import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

__all__ = [
    'resnet34', 'resnet50', 'resnet18'
]


class DistResNet(ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(DistResNet, self).__init__(block, layers, num_classes, zero_init_residual,
                                         groups, width_per_group, replace_stride_with_dilation,
                                         norm_layer)

        self.centers = nn.Parameter(torch.randn(num_classes, 512 * block.expansion))
        self.R = nn.Parameter(torch.randn(num_classes, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xag = self.avgpool(x4)
        x = torch.flatten(xag, 1)
        # x = self.fc(x)

        return x, (x1, x2, x3, x4, xag)


def _dst_resnet(block, layers, num_classes=10, **kwargs):
    model = DistResNet(block, layers=layers, num_classes=num_classes, **kwargs)

    return model


def resnet18():
    return _dst_resnet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    return _dst_resnet(BasicBlock, [3, 4, 6, 3])


def resnet50():
    return _dst_resnet(Bottleneck, [3, 4, 6, 3])
