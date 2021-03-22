
from trainer import BaseTrainer


class BasicIncrementalTrainer(BaseTrainer):
    def __init__(self, config):
        super(BasicIncrementalTrainer, self).__init__(config)

    def train(self):
        print("training phase start")

    def test(self):
        print("testing phase start")
