import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils

def jitter_points(pc, std=0.01, clip=0.05):
    bsize = pc.size()[0]
    for i in range(bsize):
        jittered_data = pc.new(pc.size(1), 3).normal_(
            mean=0.0, std=std
        ).clamp_(-clip, clip)
        pc[i, :, 0:3] += jittered_data
    return pc

def random_sample(data, number):
    '''
        data B N 3
        number int
    '''
    assert data.size(1) > number
    assert len(data.shape) == 3
    ind = torch.multinomial(torch.rand(data.size()[:2]).float(), number).to(data.device)
    data = torch.gather(data, 1, ind.unsqueeze(-1).expand(-1, -1, data.size(-1)))
    return data

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data


def set_bn_momentum_default(bn_momentum):
    """
    Creates a function that sets the momentum parameter for all BatchNorm layers.
    
    Args:
        bn_momentum (float): The momentum value to set for BatchNorm layers
        
    Returns:
        function: A function that can be applied to a model to set BatchNorm momentum
    """
    def set_momentum_for_batch_norm_layers(module):
        """Set momentum for all BatchNorm layers in the module."""
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.momentum = bn_momentum
    return set_momentum_for_batch_norm_layers


class BNMomentumScheduler:
    """
    Scheduler for dynamically adjusting BatchNorm momentum during training.
    
    This scheduler allows you to change the momentum parameter of all BatchNorm layers
    in a model based on the current training epoch. This is useful because:
    - Higher momentum early in training allows faster adaptation to data distribution
    - Lower momentum later in training provides more stable statistics for better generalization
    
    The momentum controls how much the running statistics are updated:
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        running_var = (1 - momentum) * running_var + momentum * batch_var
    """
    
    def __init__(self, model, momentum_lambda, last_epoch=-1, setter=None):
        """
        Initialize the BatchNorm momentum scheduler.
        
        Args:
            model (nn.Module): The PyTorch model containing BatchNorm layers
            momentum_lambda (callable): Function that takes epoch and returns momentum value
            last_epoch (int): The index of the last epoch (default: -1)
            setter (callable, optional): Function to set momentum. If None, uses default setter
        """
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                f"Model must be a PyTorch nn.Module, got {type(model).__name__}"
            )
        
        self.model = model
        self.momentum_lambda = momentum_lambda
        self.last_epoch = last_epoch
        self.setter = setter if setter is not None else set_bn_momentum_default
        
        # Initialize momentum for the first epoch
        self.step(last_epoch + 1)
    
    def step(self, epoch=None):
        """
        Update the BatchNorm momentum for the specified epoch.
        
        Args:
            epoch (int, optional): The current epoch. If None, uses last_epoch + 1
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        
        # Calculate new momentum value based on current epoch
        current_momentum = self.momentum_lambda(epoch)
        
        # Apply the momentum update to all BatchNorm layers in the model
        self.model.apply(self.setter(current_momentum))
    
    def get_momentum(self, epoch=None):
        """
        Get the momentum value for a specific epoch.
        
        Args:
            epoch (int, optional): The epoch to get momentum for. If None, uses last_epoch + 1
            
        Returns:
            float: The momentum value for the specified epoch
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.momentum_lambda(epoch)

