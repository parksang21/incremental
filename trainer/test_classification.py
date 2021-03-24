from trainer import BaseTrainer
from data.cifar10 import general_transform
from data.cifar10 import SplitCifar10

import torch
import torch.nn as nn
import torchvision


class TestClassification(BaseTrainer):
    def __init__(self, config):
        super(TestClassification, self).__init__(config)

        # self.trainset = torchvision.datasets.CIFAR10(self.config.data_dir, True,
        #                                              general_transform['train'])
        #
        # self.testset = torchvision.datasets.CIFAR10(self.config.data_dir,
        #                                             train=False,
        #                                             transform=general_transform['test'])
        self.trainset = SplitCifar10(self.config.data_dir,
                                     train=True,
                                     transform=general_transform['train'])

        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.config.emb_dim)

        print(self.model)