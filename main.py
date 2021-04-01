from cutils import dict_str, get_yaml, update_config, set_log_dir
from trainer import get_trainer
from config import Config

import argparse
import os
import sys

import numpy as np
import random
import shutil
import torch


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help='configuration file path', required=True)

parser.add_argument("-n", "--network", help='choose model')
parser.add_argument("-t", "--tensorboard", help="use tensorboard", action="store_true")
parser.add_argument("--trainer", help="choose trainer")
parser.add_argument("-e", "--epoch", type=int)
parser.add_argument("--batch-size", type=int)
parser.add_argument("--lr", help="learning rate", type=float)
parser.add_argument("--weight-decay", help="if necessary", type=float)
parser.add_argument("--momentum", help="optim momentum", type=float)
parser.add_argument("--emb-dim", type=int)

parser.add_argument("--debug", default=False, action='store_true')
parser.add_argument("--train", default=False, action='store_true')
parser.add_argument("--test", default=False, action='store_true')
parser.add_argument("--weight", help='weight path')
# parser.add_argument("--map", action='store_true', default=False)
argv = parser.parse_args()
config_argv = argv.__dict__


config_yaml = get_yaml(argv.config)
update_config(config_yaml, argv)

msg = (
    f'Start Program....{__file__}\n'
    f'{"=" * 30} arguments {"=" * 30}\n'
    f'{dict_str(config_yaml, indent=1)}\n'
)
print(msg)

if __name__ == '__main__':

    config = Config(config_yaml)

    # log dir is ./log under current directory
    if not config.debug:
        message = input(f"trial name: ")
        config.log_dir = os.path.join(os.path.abspath("."), 'log', config.trainer ,message)
        config.log_dir = set_log_dir(config.log_dir)
        print(config.log_dir)
        config.save(config.log_dir + "/config.yml")

    trainer = get_trainer(config)

    if config.train:
        trainer.train()

    if config.test:
        trainer.test()

    print(f"program successfully ended")
