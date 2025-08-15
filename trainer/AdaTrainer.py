import hydra
import torch
from omegaconf import OmegaConf
from utils import misc
from DDPTrainer import DDPTrainer

class AdaTrainer(DDPTrainer):
    def __init__(self, config: OmegaConf):
        super().__init__(config)

    def setup_schedulers(self):
        def lr_lambda(epoch):
            return max(0.9 ** ((epoch - 0) / 21), 0.02) if epoch >= 0 else max(epoch / 0, 0.001)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, last_epoch=-1)
        bn_scheduler = misc.BNMomentumScheduler(self.model.module, lr_lambda)
        return [lr_scheduler, bn_scheduler]

    def extract_input_data(self, train_data):
        _, _, partial_point_cloud, _ = train_data
        return partial_point_cloud

    def extract_gt_data(self, train_data):
        _, gt_point_cloud, _, _ = train_data
        return gt_point_cloud


def main():
    config = OmegaConf.load("config.yaml")
    trainer = AdaTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()