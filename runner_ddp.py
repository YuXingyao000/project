import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from model import PoinTr
from utils import misc
import time
from model.chamfer_distance import ChamferDistanceL1, ChamferDistanceL2
from dataset.dataset import ABCDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp_orig

def setup(rank, world_size):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment"""
    dist.destroy_process_group()

def validate(model, val_dataloader, device, rank):
    """Validation function"""
    model.eval()
    total_sparse_loss = 0.0
    total_brep_loss = 0.0
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for gt_brep_grid, gt_point_cloud, partial_point_cloud, cropped_point_cloud in val_dataloader:
            gt = gt_point_cloud.to(device, non_blocking=True)
            gt_brep_grid = gt_brep_grid.to(device, non_blocking=True)
            partial = partial_point_cloud.to(device, non_blocking=True)
            cropped = cropped_point_cloud.to(device, non_blocking=True)

            coarse_point_cloud, rebuild_points, brep_grids = model(partial)
            
            sparse_loss, brep_loss = model.get_loss(coarse_point_cloud, rebuild_points, brep_grids, gt, gt_brep_grid)
            
            total_loss_val = sparse_loss + brep_loss
            
            total_sparse_loss += sparse_loss.item()
            total_brep_loss += brep_loss.item()
            total_loss += total_loss_val.item()
            num_batches += 1
    
    # Calculate average losses
    avg_sparse_loss = total_sparse_loss / num_batches
    avg_brep_loss = total_brep_loss / num_batches
    avg_total_loss = total_loss / num_batches
    
    return avg_sparse_loss, avg_brep_loss, avg_total_loss

def train(rank, world_size):
    """Training function for each process"""
    setup(rank, world_size)
    
    # Create tensorboard writer (only on rank 0)
    if rank == 0:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('runs', f'training_ddp_{current_time}')
        writer = SummaryWriter(log_dir)
        print(f"Found {world_size} GPUs")
    
    # Set device for this process
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Build model and wrap with DDP
    base_model = PoinTr().to(device)
    base_model = DDP(base_model, device_ids=[rank])
    
    # Optimize CPU usage for data loading
    num_workers = min(16, int(mp_orig.cpu_count() // world_size))  # Distribute workers across GPUs
    if rank == 0:
        print(f"Using {num_workers} CPU workers per GPU for data loading")
    
    # Batch size per GPU
    batch_size_per_gpu = 32
    total_batch_size = batch_size_per_gpu * world_size
    if rank == 0:
        print(f"Total batch size: {total_batch_size} ({batch_size_per_gpu} per GPU)")

    # Create datasets and samplers
    train_dataset = ABCDataset(root='/mnt/d/data/1000_16_brep_sample_rate_processed_data', mode='train')
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size_per_gpu, 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True,
        persistent_workers=True
    )
    
    val_dataset = ABCDataset(root='/mnt/d/data/1000_16_brep_sample_rate_processed_data', mode='val')
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size_per_gpu, 
        sampler=val_sampler,
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True,
        persistent_workers=True
    )
    
    if rank == 0:
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Training batches per epoch: {len(train_dataloader)}")
        print(f"Validation batches per epoch: {len(val_dataloader)}")
    
    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-4, weight_decay=5e-4)
    
    lambda_lr = lambda e: max(0.9 ** ((e - 0) / 21), 0.02) if e >= 0 else max(e / 0, 0.001)
    scheduler = [torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr, last_epoch=-1), misc.BNMomentumScheduler(base_model, lambda_lr)]

    # Training
    base_model.zero_grad()
    best_val_loss = float('inf')
    
    for epoch in range(600):
        train_sampler.set_epoch(epoch)  # Important for shuffling
        
        base_model.train()
        epoch_start_time = time.time()
        
        # Training phase
        total_train_sparse_loss = 0.0
        total_train_brep_loss = 0.0
        total_train_loss = 0.0
        num_train_batches = 0
        
        # Training loop with progress bar (only on rank 0)
        if rank == 0:
            train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/600", leave=False)
        else:
            train_pbar = train_dataloader
            
        for idx, (gt_brep_grid, gt_point_cloud, partial_point_cloud, cropped_point_cloud) in enumerate(train_pbar):
            gt = gt_point_cloud.to(device, non_blocking=True)
            gt_brep_grid = gt_brep_grid.to(device, non_blocking=True)
            partial = partial_point_cloud.to(device, non_blocking=True)
            cropped = cropped_point_cloud.to(device, non_blocking=True)

            coarse_point_cloud, rebuild_points, brep_grids = base_model(partial)
            
            sparse_loss, brep_loss = base_model.module.get_loss(coarse_point_cloud, rebuild_points, brep_grids, gt, gt_brep_grid)
         
            _loss = sparse_loss + brep_loss 
            _loss.backward()

            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 10, norm_type=2)
            optimizer.step()
            base_model.zero_grad()
            
            # Accumulate losses for logging
            total_train_sparse_loss += sparse_loss.item()
            total_train_brep_loss += brep_loss.item()
            total_train_loss += _loss.item()
            num_train_batches += 1
            
            # Update progress bar (only on rank 0)
            if rank == 0 and hasattr(train_pbar, 'set_postfix'):
                train_pbar.set_postfix({
                    'Loss': f'{_loss.item():.6f}',
                    'Sparse': f'{sparse_loss.item():.6f}',
                    'Brep': f'{brep_loss.item():.6f}'
                })

        # Calculate average training losses
        avg_train_sparse_loss = total_train_sparse_loss / num_train_batches
        avg_train_brep_loss = total_train_brep_loss / num_train_batches
        avg_train_loss = total_train_loss / num_train_batches
        
        # Validation phase
        val_sparse_loss, val_brep_loss, val_total_loss = validate(base_model, val_dataloader, device, rank)
        
        # Update schedulers
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to tensorboard (only on rank 0)
        if rank == 0:
            writer.add_scalar('Loss/Train_Total', avg_train_loss, epoch)
            writer.add_scalar('Loss/Train_Sparse', avg_train_sparse_loss, epoch)
            writer.add_scalar('Loss/Train_Brep', avg_train_brep_loss, epoch)
            writer.add_scalar('Loss/Val_Total', val_total_loss, epoch)
            writer.add_scalar('Loss/Val_Sparse', val_sparse_loss, epoch)
            writer.add_scalar('Loss/Val_Brep', val_brep_loss, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f'Epoch [{epoch+1}/600] ({epoch_time:.2f}s) '
                  f'Train Loss: {avg_train_loss:.6f} '
                  f'Val Loss: {val_total_loss:.6f} '
                  f'LR: {current_lr:.2e}')
            
            # Save best model
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': base_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_total_loss,
                    'train_loss': avg_train_loss,
                }, os.path.join(log_dir, 'best_model.pth'))
                print(f'New best model saved! Val Loss: {val_total_loss:.6f}')
            
            # Save checkpoint every 50 epochs
            if (epoch + 1) % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': base_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_total_loss,
                    'train_loss': avg_train_loss,
                }, os.path.join(log_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    if rank == 0:
        writer.close()
        print(f"Training completed! Logs saved to: {log_dir}")
    
    cleanup()

def run_net():
    """Main function to start distributed training"""
    world_size = torch.cuda.device_count()
    print(f"Starting distributed training with {world_size} GPUs")
    
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    run_net() 