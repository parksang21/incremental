import torch
import torch.nn as nn


class BaseTrainer():
    def __init__(self, config):
        self.config = config
        import os

        print(os.path.abspath("."))