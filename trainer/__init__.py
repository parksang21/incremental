import torch
import torch.nn as nn

import os
import shutil
from typing import Any


def get_trainer(config):
    if config.trainer == 'test_classification':
        from trainer.test_classification import TestClassification
        return TestClassification(config)


class BaseTrainer():
    def __init__(self,
                 config: Any
                 ) -> None:
        """
        :param config: Config type configuration object
        """
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
