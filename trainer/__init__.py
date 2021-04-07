import torch
import torch.nn as nn

import os
import shutil
from typing import Any
from abc import ABCMeta, abstractmethod


def get_trainer(config):
    if config.trainer == 'classification':
        from trainer.classification import Classification
        return Classification(config)

    if config.trainer == 'sim_distance':
        from trainer.sim_distance import SimDistanceModel
        return SimDistanceModel(config)

class BaseTrainer(metaclass=ABCMeta):
    def __init__(self,
                 config: Any
                 ) -> None:
        """
        :param config: Config type configuration object
        """
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass
