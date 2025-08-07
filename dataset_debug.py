import torch
from torch.utils.data import DataLoader, DistributedSampler

from dataset.dataset import ABCDataset

SEED = 42

def worker_init_fn(worker_id):
    """Initialize worker function for reproducible data loading"""
    import random
    import numpy as np
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dataset_builder(args, config):
    """Build dataset and dataloader with support for distributed training"""
    dataset = ABCDataset(root=config.data_root, mode=config.mode)
    shuffle = config.mode == 'train'
    
    if args.distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=config.batch_size if shuffle else 1,
            num_workers=int(args.num_workers),
            drop_last=config.mode == 'train',
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2  # Prefetch batches in workers
        )
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=config.batch_size if shuffle else 1,
            shuffle=shuffle, 
            drop_last=config.mode == 'train',
            num_workers=int(args.num_workers),
            worker_init_fn=worker_init_fn,
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2  # Prefetch batches in workers
        )
    return sampler, dataloader


if __name__ == "__main__":
    # Simple debug version (non-distributed)
    class Args:
        def __init__(self):
            self.distributed = False
            self.num_workers = 4
    
    class Config:
        def __init__(self):
            self.data_root = "/mnt/d/data/test"
            self.mode = "train"
            self.batch_size = 1
    
    args = Args()
    config = Config()
    
    # Use the dataset_builder function
    sampler, dataloader = dataset_builder(args, config)
    
    # Debug loop
    for data in dataloader:
        print(data[0].shape)
        print(data[1].shape)
        print(data[2].shape)
        break




