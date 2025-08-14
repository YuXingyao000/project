import os
import time
import gc
import random
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import AdaPoinTr
from utils import misc
from dataset.dataset import DeepCADDataset  # adjust import if needed


class DDPTrainer:
    def __init__(self, config=None):  # config unused per your note
        self.config = config
        self.train_batch = 32
        self.val_batch = 64
        self.epochs = 600
        self.num_workers = 4
        self.seed = 114514

        # --- distributed + device first ---
        self.setup_distributed()
        self.device, self.local_rank = self.setup_device()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # --- model/optim/scheduler next ---
        self.model = self.setup_model()
        self.optimizer = self.setup_optimizer()
        self.schedulers = self.setup_schedulers()  # list

        # --- datasets & loaders ---
        self.train_dataset = DeepCADDataset(
            data_root='/mnt/d/data/processed_deepcad',
            index_path='/mnt/d/data/DeepCAD/data_index/deduplicated_deepcad_training_7_30.txt'
        )
        self.val_dataset = DeepCADDataset(
            data_root='/mnt/d/data/processed_deepcad',
            index_path='/mnt/d/data/DeepCAD/data_index/deduplicated_deepcad_validation_7_30.txt'
        )

        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)
        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False, drop_last=False)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            worker_init_fn=self.setup_seed_worker(self.seed),
            generator=torch.Generator().manual_seed(self.seed),
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.val_batch,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            worker_init_fn=self.setup_seed_worker(self.seed),
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,   
        )

        # --- logging/checkpointing only on rank 0 ---
        self.best_val_loss = float('inf')
        self.log_dir = Path('runs') / f'AdaPoinTr_DeepCAD_{datetime.now().strftime("%b%d_%H-%M-%S")}'
        self.writer = SummaryWriter(self.log_dir) if self.rank == 0 else None

    # ---------- setup ----------
    def setup_distributed(self):
        # Check if we're in a distributed environment
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # We're in a proper distributed environment
            dist.init_process_group(backend='nccl', init_method='env://')
        else:
            # We're running in single-GPU mode (e.g., for debugging)
            # Set up environment variables for single-GPU DDP
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            os.environ['LOCAL_RANK'] = '0'
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group(backend='nccl', init_method='env://')

    def setup_device(self):
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        if local_rank == 0:
            print(f"[DDP] World size: {dist.get_world_size()}")
        return device, local_rank

    def setup_seed_worker(self, base_seed):
        # Get rank (0 if not using DDP)
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Unique seed per rank
        seed = base_seed + rank

        # Python
        random.seed(seed)
        # NumPy
        np.random.seed(seed)
        # Torch (CPU + GPU)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Make CUDA deterministic (slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Ensure dataloader workers are seeded the same way
        def seed_worker(worker_id):
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        return seed_worker
    
    def setup_model(self):
        model = AdaPoinTr().to(self.device)
        model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        return model

    def setup_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=5e-4)

    def setup_schedulers(self):
        def lr_lambda(epoch):
            return max(0.9 ** ((epoch - 0) / 21), 0.02) if epoch >= 0 else max(epoch / 0, 0.001)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, last_epoch=-1)
        bn_scheduler = misc.BNMomentumScheduler(self.model.module, lr_lambda)
        return [lr_scheduler, bn_scheduler]

    def cleanup_distributed(self):
        dist.destroy_process_group()

    # ---------- validation with global reduction ----------
    @torch.no_grad()
    def validate(self, model, epoch):
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        for gt_brep_grid, gt_point_cloud, partial_point_cloud, cropped_point_cloud in tqdm(
            self.val_dataloader, desc="Validating", leave=False, disable=(self.rank != 0)
        ):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gt = gt_point_cloud.to(self.device, non_blocking=True)
            partial = partial_point_cloud.to(self.device, non_blocking=True)

            # forward
            # adapt to your model’s actual return signature
            out = model(partial)
            recon_loss = model.module.get_loss(out, gt)  # call underlying module for clarity

            val_loss_sum += float(recon_loss.item())
            val_batches += 1

            del gt, partial, out, recon_loss

        # --- All-reduce across ranks to get global mean loss ---
        loss_tensor = torch.tensor([val_loss_sum, val_batches], device=self.device, dtype=torch.float32)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        global_loss_sum, global_batches = loss_tensor.tolist()
        avg_val_loss = global_loss_sum / max(global_batches, 1)

        # --- write/checkpoint on rank 0 only ---
        if self.rank == 0 and self.writer is not None:
            self.writer.add_scalar('Loss/Val_Total', avg_val_loss, epoch)

        if self.rank == 0 and avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),      # .module
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, self.log_dir / 'best_model.pth')


    # ---------- training ----------
    def train(self):
        base_model = self.model

        for epoch in tqdm(range(self.epochs), desc="Training Progress", disable=(self.rank != 0)):
            epoch_start_time = time.time()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ensure different (but aligned) shuffles per epoch across ranks
            self.train_sampler.set_epoch(epoch)

            train_denoised_sum = 0.0
            train_recon_sum = 0.0
            train_total_sum = 0.0
            train_batches = 0

            pbar = tqdm(self.train_dataloader,
                        desc=f"Epoch {epoch+1}/{self.epochs}",
                        leave=False,
                        disable=(self.rank != 0))
            for gt_brep_grid, gt_point_cloud, partial_point_cloud, cropped_point_cloud in pbar:
                base_model.train()
                base_model.zero_grad()

                gt = gt_point_cloud.to(self.device, non_blocking=True)
                partial = partial_point_cloud.to(self.device, non_blocking=True)

                # forward (adapt to your actual outputs)
                pred_coarse, denoised_coarse, pred_fine, denoised_fine = base_model(partial)

                denoised_loss, recon_loss = base_model.module.get_loss(
                    (pred_coarse, denoised_coarse, pred_fine, denoised_fine), gt
                )
                loss = denoised_loss + recon_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 10, norm_type=2)
                self.optimizer.step()

                train_denoised_sum += float(denoised_loss.item())
                train_recon_sum += float(recon_loss.item())
                train_total_sum += float(loss.item())
                train_batches += 1

                if self.rank == 0:
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.6f}',
                        'Denoised': f'{denoised_loss.item():.6f}',
                        'Recon': f'{recon_loss.item():.6f}',
                    })

                del gt, partial, pred_coarse, denoised_coarse, pred_fine, denoised_fine, denoised_loss, recon_loss, loss

            # --- reduce train metrics to global averages for logging ---
            train_tensor = torch.tensor([train_denoised_sum, train_recon_sum, train_total_sum, train_batches],
                                        device=self.device, dtype=torch.float32)
            dist.all_reduce(train_tensor, op=dist.ReduceOp.SUM)
            g_denoised, g_recon, g_total, g_batches = train_tensor.tolist()
            train_avg_denoised = g_denoised / max(g_batches, 1)
            train_avg_recon = g_recon / max(g_batches, 1)
            train_avg_total = g_total / max(g_batches, 1)

            # step schedulers (same on every rank)
            for sch in self.schedulers:
                sch.step()

            if self.rank == 0 and self.writer is not None:
                self.writer.add_scalar('Loss/Train_Total', train_avg_total, epoch)
                self.writer.add_scalar('Loss/Train_Denoised', train_avg_denoised, epoch)
                self.writer.add_scalar('Loss/Train_Recon', train_avg_recon, epoch)

            if self.rank == 0:
                epoch_time = time.time() - epoch_start_time
                print(f'Epoch [{epoch+1}/{self.epochs}] ({epoch_time:.2f}s) '
                      f'Train={train_avg_total:.6f}')

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # --- validation ---
            self.validate(base_model, epoch)

            # checkpoints every 50 epochs (rank 0 only)
            if self.rank == 0 and (epoch + 1) % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': base_model.module.state_dict(),  # .module
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': self.best_val_loss,
                    'train_loss': train_avg_total,
                }, self.log_dir / f'checkpoint_epoch_{epoch+1}.pth')

        if self.rank == 0 and self.writer is not None:
            self.writer.close()

        self.cleanup_distributed()
        
if __name__ == "__main__":
    trainer = DDPTrainer()
    trainer.train()