from trainer import BaseTrainer
from data.cifar10 import general_transform
from data.cifar10 import SplitCifar10

import torch
import torch.nn as nn
import torchvision
import shutil


class Classification(BaseTrainer):
    def __init__(self, config):
        super(Classification, self).__init__(config)

        if not self.config.debug:
            if self.config.log_dir is not None:
                shutil.copy(__file__, self.config.log_dir)
                print(f"{__file__} has been saved")

        self.trainset = SplitCifar10(self.config.data_dir,
                                     train=True,
                                     transform=general_transform['train'])

        self.testset = SplitCifar10(self.config.data_dir,
                                    train=True,
                                    transform=general_transform['test'])

        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

        self.model = nn.Sequential(
            *list(torchvision.models.resnet18(pretrained=False).children())[:-1]
        )

        if config.debug:
            print(self.model)

    def train(self):
        print(f"TRAIN")

