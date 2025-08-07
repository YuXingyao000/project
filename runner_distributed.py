import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
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
import multiprocessing as mp

def validate(model, val_dataloader, device):
    """
    Validation function
    """
    model.eval()
    total_sparse_loss = 0.0
    # total_brep_loss = 0.0
    total_loss = 0.0
    num_batches = 0
    
    print(f"Starting validation with {len(val_dataloader)} batches")
    
    # Get the underlying model for loss calculation
    model_for_loss = model.module if hasattr(model, 'module') else model
    
    with torch.no_grad():
        for gt_brep_grid, gt_point_cloud, partial_point_cloud, cropped_point_cloud in tqdm(val_dataloader, desc="Validating", leave=False):
            gt = gt_point_cloud.to(device)
            gt_brep_grid = gt_brep_grid.to(device)
            partial = partial_point_cloud.to(device)
            cropped = cropped_point_cloud.to(device)

            coarse_point_cloud, rebuild_points = model(partial)
            
            # Get actual batch size and reshape BREP grids
            actual_batch_size = gt.shape[0]
            # brep_grids_flat = brep_grids.reshape(actual_batch_size, -1, 3)
            # gt_brep_grid_flat = gt_brep_grid.reshape(actual_batch_size, -1, 3)
            
            sparse_loss, fine_loss = model_for_loss.get_loss(coarse_point_cloud, rebuild_points, gt)
            
            total_loss_val = sparse_loss + fine_loss
            
            total_sparse_loss += sparse_loss.item()
            # total_brep_loss += brep_loss.item()
            total_loss += total_loss_val.item()
            num_batches += 1
    
    # Calculate average losses
    if num_batches == 0:
        print("Warning: No validation batches found. Returning zero losses.")
        return 0.0, 0.0
    
    avg_sparse_loss = total_sparse_loss / num_batches
    # avg_brep_loss = total_brep_loss / num_batches
    avg_total_loss = total_loss / num_batches
    
    print(f"Validation completed: {num_batches} batches processed")
    
    return avg_sparse_loss, avg_total_loss

def run_net():
    # Create tensorboard writer
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', f'training_distributed_{current_time}')
    writer = SummaryWriter(log_dir)
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    
    # Use all available GPUs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build model and wrap with DataParallel
    base_model = PoinTr()
    if num_gpus > 1:
        base_model = DataParallel(base_model)
    base_model.to(device)
    
    # Use reasonable number of CPU workers (leaving most cores free for other work)
    num_workers = min(16, int(mp.cpu_count() * 0.15))  # Use only 15% of CPU cores
    print(f"Using {num_workers} CPU workers for data loading (leaving cores free for other work)")
    
    # Reduce batch size to prevent OOM
    # Each GPU will handle batch_size/num_gpus
    base_batch_size = 2# Use 1 sample per GPU to avoid memory issues
    total_batch_size = base_batch_size * num_gpus
    print(f"Total batch size: {total_batch_size} ({base_batch_size} per GPU)")
    
    # Set memory optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_dataset = ABCDataset(root='/mnt/d/data/1000_16_brep_sample_rate_processed_data', mode='train')
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=total_batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True,
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    val_dataset = ABCDataset(root='/mnt/d/data/1000_16_brep_sample_rate_processed_data', mode='val')
    # Calculate validation batch size to ensure at least 3 batches with drop_last=True
    val_samples = len(val_dataset)
    val_batch_size = 2  # Reduced for memory efficiency
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=val_batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True,  # Keep drop_last=True for distributed training consistency
        persistent_workers=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training batch size: {total_batch_size}")
    print(f"Validation batch size: {val_batch_size}")
    print(f"Training batches per epoch: {len(train_dataloader)}")
    print(f"Validation batches per epoch: {len(val_dataloader)}")
    
    # Safety check for validation batches
    if len(val_dataloader) == 0:
        raise ValueError(f"Validation dataloader has 0 batches! Val samples: {len(val_dataset)}, Val batch size: {val_batch_size}")
    elif len(val_dataloader) < 2:
        print(f"Warning: Only {len(val_dataloader)} validation batches available. Consider reducing batch size further.")
    
    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-4, weight_decay=5e-4)  # Reduced learning rate
    
    lambda_lr = lambda e: max(0.9 ** ((e - 0) / 21), 0.02) if e >= 0 else max(e / 0, 0.001)
    scheduler = [torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr, last_epoch=-1), misc.BNMomentumScheduler(base_model, lambda_lr)]

    # Training
    base_model.zero_grad()
    best_val_loss = float('inf')
    
    # Gradient accumulation to maintain effective batch size
    accumulation_steps = 2  # Reduced for memory efficiency
    print(f"Using gradient accumulation: {accumulation_steps} steps")
    print(f"Effective batch size: {total_batch_size * accumulation_steps}")
    
    for epoch in tqdm(range(600), desc="Training Progress"):
        base_model.train()
        epoch_start_time = time.time()
        
        # Clear GPU cache at the start of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Training phase
        total_train_sparse_loss = 0.0
        # total_train_brep_loss = 0.0
        total_train_loss = 0.0
        num_train_batches = 0
        
        # Training loop with progress bar
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/600", leave=False)
        for idx, (gt_brep_grid, gt_point_cloud, partial_point_cloud, cropped_point_cloud) in enumerate(train_pbar):
            gt = gt_point_cloud.to(device, non_blocking=True)
            gt_brep_grid = gt_brep_grid.to(device, non_blocking=True)
            partial = partial_point_cloud.to(device, non_blocking=True)
            cropped = cropped_point_cloud.to(device, non_blocking=True)

            coarse_point_cloud, rebuild_points = base_model(partial)
            
            # Get actual batch size and reshape BREP grids
            # actual_batch_size = gt.shape[0]
            # brep_grids_flat = brep_grids.reshape(actual_batch_size, -1, 3)
            # gt_brep_grid_flat = gt_brep_grid.reshape(actual_batch_size, -1, 3)
            
            # Get the underlying model for loss calculation
            model_for_loss = base_model.module if isinstance(base_model, DataParallel) else base_model
            
            # Debug tensor shapes and devices for distributed training
    
            sparse_loss, fine_loss = model_for_loss.get_loss(coarse_point_cloud, rebuild_points, gt)
         
            _loss = (sparse_loss + fine_loss) / accumulation_steps  # Scale loss for accumulation
            _loss.backward()

            # Update weights every accumulation_steps
            if (idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 10, norm_type=2)
                optimizer.step()
                base_model.zero_grad()
            
            # Accumulate losses for logging
            total_train_sparse_loss += sparse_loss.item()
            # total_train_brep_loss += brep_loss.item()
            total_train_loss += _loss.item()
            num_train_batches += 1
            
            # Update progress bar with current loss
            train_pbar.set_postfix({
                'Loss': f'{_loss.item():.6f}',
                'Sparse': f'{sparse_loss.item():.6f}',
                # 'Brep': f'{brep_loss.item():.6f}',
                'GPU Mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
            })
            
            # Clear cache periodically to prevent memory fragmentation
            if idx % 10 == 0:
                torch.cuda.empty_cache()
            
            # Clear variables to free memory
            del gt, gt_brep_grid, partial, cropped, coarse_point_cloud, rebuild_points
            del sparse_loss, fine_loss, _loss

        # Calculate average training losses
        avg_train_sparse_loss = total_train_sparse_loss / num_train_batches
        # avg_train_brep_loss = total_train_brep_loss / num_train_batches
        avg_train_loss = total_train_loss / num_train_batches
        
        # Clear cache before validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        # Validation phase
        val_sparse_loss, val_total_loss = validate(base_model, val_dataloader, device)
        
        # Update schedulers
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to tensorboard
        writer.add_scalar('Loss/Train_Total', avg_train_loss, epoch)
        writer.add_scalar('Loss/Train_Sparse', avg_train_sparse_loss, epoch)
        # writer.add_scalar('Loss/Train_Brep', avg_train_brep_loss, epoch)
        writer.add_scalar('Loss/Val_Total', val_total_loss, epoch)
        writer.add_scalar('Loss/Val_Sparse', val_sparse_loss, epoch)
        # writer.add_scalar('Loss/Val_Brep', val_brep_loss, epoch)
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
            # Save the underlying model (not DataParallel wrapper)
            model_to_save = base_model.module if isinstance(base_model, DataParallel) else base_model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total_loss,
                'train_loss': avg_train_loss,
            }, os.path.join(log_dir, 'best_model.pth'))
            print(f'New best model saved! Val Loss: {val_total_loss:.6f}')
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            model_to_save = base_model.module if isinstance(base_model, DataParallel) else base_model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total_loss,
                'train_loss': avg_train_loss,
            }, os.path.join(log_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    print(f"Training completed! Logs saved to: {log_dir}")


if __name__ == "__main__":
    run_net() 