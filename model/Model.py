import torch.nn as nn
from abc import ABC, abstractmethod

class Model(ABC, nn.Module):
    """
    Base class for all models.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_func = self.setup_loss_func()

    @abstractmethod
    def setup_loss_func(self):
        """
        Setup the loss function.
        Returns:
            - A loss function.
        """
        pass

    @abstractmethod
    def get_loss(self, pred, gt):
        """
        Args:
            - pred: The predicted result.
            - gt: The ground truth.
        Returns:
            - A dictionary of loss values. The key is the name of the loss, the value is the loss value.
        """
        pass

    @abstractmethod
    def forward(self, xyz):
        pass