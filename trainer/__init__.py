import torch
import torch.nn as nn

import os
import shutil


class BaseTrainer():
    def __init__(self, config):
        self.config = config

        # logging 예외처리
        try:
            shutil.copy(__file__, config.dir)
        except Exception:
            print(f"No logging dir is defined")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
