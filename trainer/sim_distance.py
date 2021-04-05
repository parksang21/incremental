from trainer import BaseTrainer
from model.simclr import SimModel
from model.cnn import encoder32
from data.cifar10 import SplitSimCIFAR10, simclr_transform
from loss.contrastive import Contrastive
from cutils import print_log
from loss.distance import CosineSim

from torch.utils.data import DataLoader
from torch import optim
import torch
import torch.nn as nn

import shutil
from tqdm import tqdm


class SimDistanceModel(BaseTrainer):
    def __init__(self, config):
        super(SimDistanceModel, self).__init__(config)
        if not self.config.debug:
            if self.config.log_dir is not None:
                shutil.copy(__file__, self.config.log_dir)
                print(f"{__file__} has been saved")

        self.backbone = encoder32(latent_size=config.emb_dim).to(self.device)
        self.model = SimModel(self.backbone, config.emb_dim, config.emb_dim, 10).to(self.device)

    def train(self):
        print("train phase start")

        dataset = SplitSimCIFAR10(self.config.data_dir, True,
                                       simclr_transform['train'])
        dataset.set_split(self.config.split)

        loader = DataLoader(dataset, self.config.batch_size,
                            shuffle=True, num_workers=2)

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr,
                              weight_decay=self.config.weight_decay)

        similarity = Contrastive().to(self.device)
        criterion = nn.CrossEntropyLoss().to(self.device)

        min_loss = 10

        for epoch in range(self.config.epoch):

            log_info = {
                'batch_loss': .0,
            }

            batch_loss = .0
            for pos_1, pos_2, targets in tqdm(loader):
                pos_1, pos_2, targets = pos_1.to(self.device), pos_2.to(self.device), targets.to(self.device)

                outputs, logit = self.model(torch.cat([pos_1, pos_2], dim=0))

                sim, logit = similarity(outputs)

                loss = criterion(sim, logit)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()

            batch_loss /= len(loader)
            log_info['batch_loss'] = batch_loss
            print_log(f"{epoch} / {self.config.epoch}", log_info)
            if min_loss > batch_loss:
                min_loss = batch_loss
                self.save()


    def test(self):
        pass

    def save(self):
        if not self.config.debug:
            state_dict = self.model.state_dict()
            torch.save(state_dict, self.config.log_dir+"/model.pth")
            print(f"model has been saved")
