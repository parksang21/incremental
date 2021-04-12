from trainer import BaseTrainer
from data.cifar10 import general_transform
from data.cifar10 import SplitCifar10
from data.cifar10 import CIFAR10
from model import resnet50
from cutils import print_log

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.utils.data import DataLoader
import torchvision
import shutil
from tqdm import tqdm


class Classification(BaseTrainer):
    def __init__(self, config):
        super(Classification, self).__init__(config)

        if not self.config.debug:
            if self.config.log_dir is not None:
                shutil.copy(__file__, self.config.log_dir)
                print(f"{__file__} has been saved")

        self.model = resnet50()

        self.model = self.model.to(self.device)

        if config.debug:
            print(self.model)

    def train(self):
        train_set = CIFAR10(self.config.data_dir,
                            train=True,
                            transform=general_transform['train'],
                            download=True)
        loader = DataLoader(train_set, self.config.batch_size,
                            shuffle=True, num_workers=self.config.num_workers)

        optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr,
                              momentum=self.config.momentum,
                              weight_decay=self.config.weight_decay)
        criterion = nn.CrossEntropyLoss()

        classes = torch.Tensor([i for i in range(10)]).float().to(self.device)

        for epoch in range(1, self.config.epoch + 1):

            log_dict = {
                'batch_loss': .0,
                'acc': .0,
                'lr': .0
            }

            min_loss = 100
            max_acc = 0

            total = 0
            correct = 0
            for inputs, targets in tqdm(loader):
                inputs, targets = self.to_device(inputs, targets)

                output, _ = self.model(inputs) # output : batch x 512 x 1 x 1

                # calculate l2 distance and use squared distance.
                dist = torch.cdist(self.model.centers, output.squeeze()) ** 2
                dist = dist.T # output shape must be Batch x Class

                r_target = targets.repeat(self.config.num_classes).view(self.config.num_classes, -1)
                mask = (r_target == classes.unsqueeze(1)).T.float()

                print(targets.shape)
                print(r_target.shape)
                print(mask.shape)
                exit()

                inclass_loss = dist * mask


                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                #
                # # _, pred = center_sim.T.max(1)
                #
                # total += targets.size(0)
                # correct += pred.eq(targets).sum()
                # log_dict['batch_loss'] += loss.item()

            log_dict['acc'] = correct / total
            log_dict['batch_loss'] /= len(loader)
            print_log(f"{epoch} / {self.config.epoch}", log_dict)

    def test(self):
        print(f"TEST")


def normalize(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def cosine_sim(x):
    x = normalize(x)
    return torch.mm(x, x.T)


def center_cosine(c, x):
    # input shape c: n x d, x: b x d
    c = F.normalize(c, p=2, dim=-1, eps=1e-8).unsqueeze(1)
    x = F.normalize(x, p=2, dim=-1, eps=1e-8)
    return torch.sum(c * x, dim=-1)


def L2_dist(c, x):
    return torch.cdist(c, x) ** 2



