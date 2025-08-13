import torch
import os
# Set environment variable to help with memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from model import AdaPoinTr
from utils import misc
import time
from dataset.dataset import ABCDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from tqdm import tqdm
import gc

def validate(model, val_dataloader, device):
    """
    Validation function with memory management
    """
    model.eval()
    total_sparse_loss = 0.0
    total_brep_loss = 0.0
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for gt_brep_grid, gt_point_cloud, partial_point_cloud, cropped_point_cloud in tqdm(val_dataloader, desc="Validating", leave=False):
            # Clear cache before processing each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gt = gt_point_cloud.to(device)
            gt_brep_grid = gt_brep_grid.to(device)
            partial = partial_point_cloud.to(device)
            cropped = cropped_point_cloud.to(device)

            ret = model(partial)
            
            sparse_loss, brep_loss = model.get_loss(ret, gt)
            
            total_loss_val = sparse_loss + brep_loss
            
            total_sparse_loss += sparse_loss.item()
            total_loss += total_loss_val.item()
            num_batches += 1
            
            # Clear variables to free memory
            del gt, gt_brep_grid, partial, cropped, ret
            del sparse_loss, total_loss_val
    
    # Calculate average losses
    avg_sparse_loss = total_sparse_loss / num_batches
    avg_total_loss = total_loss / num_batches
    
    return avg_sparse_loss, avg_total_loss

def train():
    # Create tensorboard writer
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', f'training_{current_time}')
    writer = SummaryWriter(log_dir)
    
    # build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check available GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total GPU memory: {gpu_memory:.2f} GB")
        free_memory = torch.cuda.memory_allocated(0) / 1024**3
        print(f"Currently allocated: {free_memory:.2f} GB")
    
    base_model = AdaPoinTr()
    base_model.to(device)

    # Reduce batch sizes to prevent OOM
    train_batch_size = 16  # Reduced from 64
    val_batch_size = 8    # Reduced from 64
    
    train_dataset = ABCDataset(root='/mnt/d/data/1000_16_brep_sample_rate_processed_data', mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_dataset = ABCDataset(root='/mnt/d/data/1000_16_brep_sample_rate_processed_data', mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training batch size: {train_batch_size}")
    print(f"Validation batch size: {val_batch_size}")
    
    # optimizer & scheduler
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    lambda_lr = lambda e: max(0.9 ** ((e - 0) / 21), 0.02) if e >= 0 else max(e / 0, 0.001)
    scheduler = [torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr, last_epoch=-1), misc.BNMomentumScheduler(base_model, lambda_lr)]

    # training
    base_model.zero_grad()
    best_val_loss = float('inf')
    
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
            # Clear cache periodically during training
            if idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gt = gt_point_cloud.to(device)
            gt_brep_grid = gt_brep_grid.to(device)
            partial = partial_point_cloud.to(device)
            cropped = cropped_point_cloud.to(device)

            ret = base_model(partial)
            
            sparse_loss, fine_loss = base_model.get_loss(ret, gt)
         
            _loss = sparse_loss + fine_loss
            _loss.backward()

            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 10, norm_type=2)
            optimizer.step()
            base_model.zero_grad()
            
            # Accumulate losses for logging
            total_train_sparse_loss += sparse_loss.item()
            total_train_loss += _loss.item()
            num_train_batches += 1
            
            # Update progress bar with current loss
            train_pbar.set_postfix({
                'Loss': f'{_loss.item():.6f}',
                'Sparse': f'{sparse_loss.item():.6f}',
            })
            
            # Clear variables to free memory
            del gt, gt_brep_grid, partial, cropped, ret
            del sparse_loss, _loss

        # Calculate average training losses
        avg_train_sparse_loss = total_train_sparse_loss / num_train_batches
        # avg_train_brep_loss = total_train_brep_loss / num_train_batches
        avg_train_loss = total_train_loss / num_train_batches
        
        # Clear cache before validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        writer.add_scalar('Loss/Val_Total', val_total_loss, epoch)
        writer.add_scalar('Loss/Val_Sparse', val_sparse_loss, epoch)
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
                'model_state_dict': base_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total_loss,
                'train_loss': avg_train_loss,
            }, os.path.join(log_dir, 'best_model.pth'))
            print(f'New best model saved! Val Loss: {val_total_loss:.6f}')
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': base_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total_loss,
                'train_loss': avg_train_loss,
            }, os.path.join(log_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    print(f"Training completed! Logs saved to: {log_dir}")


if __name__ == "__main__":
    train()

# crop_ratio = {
#     'easy': 1/4,
#     'median' :1/2,
#     'hard':3/4
# }
# 
# def test_net(args, config):
#     logger = get_logger(args.log_name)
#     print_log('Tester start ... ', logger = logger)
#     _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
#     base_model = builder.model_builder(config.model)
#     # load checkpoints
#     builder.load_model(base_model, args.ckpts, logger = logger)
#     if args.use_gpu:
#         base_model.to(args.local_rank)

#     #  DDP    
#     if args.distributed:
#         raise NotImplementedError()

#     # Criterion
#     ChamferDisL1 = ChamferDistanceL1()
#     ChamferDisL2 = ChamferDistanceL2()

#     test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

# def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger = None):

#     base_model.eval()  # set model to eval mode

#     test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
#     test_metrics = AverageMeter(Metrics.names())
#     category_metrics = dict()
#     n_samples = len(test_dataloader) # bs is 1

#     with torch.no_grad():
#         for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
#             taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
#             model_id = model_ids[0]

#             npoints = config.dataset.test._base_.N_POINTS
#             dataset_name = config.dataset.test._base_.NAME
#             if dataset_name == 'PCN' or dataset_name == 'Projected_ShapeNet':
#                 partial = data[0].cuda()
#                 gt = data[1].cuda()

#                 ret = base_model(partial)
#                 coarse_points = ret[0]
#                 dense_points = ret[-1]

#                 sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
#                 sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
#                 dense_loss_l1 =  ChamferDisL1(dense_points, gt)
#                 dense_loss_l2 =  ChamferDisL2(dense_points, gt)

#                 test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

#                 _metrics = Metrics.get(dense_points, gt, require_emd=True)
#                 # test_metrics.update(_metrics)

#                 if taxonomy_id not in category_metrics:
#                     category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
#                 category_metrics[taxonomy_id].update(_metrics)

#             elif dataset_name == 'ShapeNet':
#                 gt = data.cuda()
#                 choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
#                             torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
#                 num_crop = int(npoints * crop_ratio[args.mode])
#                 for item in choice:           
#                     partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
#                     # NOTE: subsample the input
#                     partial = misc.fps(partial, 2048)
#                     ret = base_model(partial)
#                     coarse_points = ret[0]
#                     dense_points = ret[-1]

#                     sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
#                     sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
#                     dense_loss_l1 =  ChamferDisL1(dense_points, gt)
#                     dense_loss_l2 =  ChamferDisL2(dense_points, gt)

#                     test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

#                     _metrics = Metrics.get(dense_points ,gt)



#                     if taxonomy_id not in category_metrics:
#                         category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
#                     category_metrics[taxonomy_id].update(_metrics)
#             elif dataset_name == 'KITTI':
#                 partial = data.cuda()
#                 ret = base_model(partial)
#                 dense_points = ret[-1]
#                 target_path = os.path.join(args.experiment_path, 'vis_result')
#                 if not os.path.exists(target_path):
#                     os.mkdir(target_path)
#                 misc.visualize_KITTI(
#                     os.path.join(target_path, f'{model_id}_{idx:03d}'),
#                     [partial[0].cpu(), dense_points[0].cpu()]
#                 )
#                 continue
#             else:
#                 raise NotImplementedError(f'Train phase do not support {dataset_name}')

#             if (idx+1) % 200 == 0:
#                 print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
#                             (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
#                             ['%.4f' % m for m in _metrics]), logger=logger)
#         if dataset_name == 'KITTI':
#             return
#         for _,v in category_metrics.items():
#             test_metrics.update(v.avg())
#         print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

     

#     # Print testing results
#     shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
#     print_log('============================ TEST RESULTS ============================',logger=logger)
#     msg = ''
#     msg += 'Taxonomy\t'
#     msg += '#Sample\t'
#     for metric in test_metrics.items:
#         msg += metric + '\t'
#     msg += '#ModelName\t'
#     print_log(msg, logger=logger)


#     for taxonomy_id in category_metrics:
#         msg = ''
#         msg += (taxonomy_id + '\t')
#         msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
#         for value in category_metrics[taxonomy_id].avg():
#             msg += '%.3f \t' % value
#         msg += shapenet_dict[taxonomy_id] + '\t'
#         print_log(msg, logger=logger)

#     msg = ''
#     msg += 'Overall \t\t'
#     for value in test_metrics.avg():
#         msg += '%.3f \t' % value
#     print_log(msg, logger=logger)
#     return 
