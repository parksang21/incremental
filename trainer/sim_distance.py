from trainer import BaseTrainer
from model.simclr import SimModel
from model.cnn import encoder32
import shutil


class SimDistanceModel(BaseTrainer):
    def __init__(self, config):
        super(SimDistanceModel, self).__init__(config)
        if not self.config.debug:
            if self.config.log_dir is not None:
                shutil.copy(__file__, self.config.log_dir)
                print(f"{__file__} has been saved")

        self.backbone = encoder32(latent_size=config.emb_dim)
        self.model = SimModel(self.backbone, config.emb_dim, config.emb_dim, 10)

    def train(self):
        print(se)

    def test(self):
        pass