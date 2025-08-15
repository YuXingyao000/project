import os
import time
import gc
import random
import importlib
import numpy as np
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class DDPTrainer(ABC):
    def __init__(self, config: OmegaConf):  # config unused per your note
        self.config = config
        self.epochs = config.trainer.epochs
        self.num_workers = config.trainer.num_workers
        self.seed = config.random_seed

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
        self.train_dataset = self.setup_datasets(self.config.datasets.train)
        self.val_dataset = self.setup_datasets(self.config.datasets.val)

        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)
        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False, drop_last=False)

        # --- dataloaders ---
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.trainer.train_batch,
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
            batch_size=self.config.trainer.val_batch,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            worker_init_fn=self.setup_seed_worker(self.seed),
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,   
        )

        # --- logging/checkpointing only on rank 0 ---
        self.best_val_loss = float('inf')
        # Generate timestamp for log directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = Path(f"{self.config.trainer.log_dir}_{timestamp}")
        
        self.writer = SummaryWriter(self.log_dir) if self.rank == 0 else None

    # ---------- setup ----------
    @abstractmethod
    def setup_schedulers(self):
        pass
    
    @abstractmethod
    def extract_input_data(self, train_data):
        pass
    
    @abstractmethod
    def extract_gt_data(self, train_data):
        pass

    def setup_distributed(self):
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            dist.init_process_group(backend='nccl', init_method='env://')
        else:
            # Debugging mode
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
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = base_seed + rank

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        def seed_worker(worker_id):
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        return seed_worker
    
    def setup_model(self):
        model_class = getattr(importlib.import_module(self.config.model.module_name), self.config.model.class_name)
        model = model_class(**self.config.model.params).to(self.device)
        model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        return model

    def setup_optimizer(self):
        module_name = '.'.join(['torch.optim', self.config.trainer.optimizer.class_name.lower()])
        optimizer_class = getattr(importlib.import_module(module_name), self.config.trainer.optimizer.class_name)
        return optimizer_class(self.model.parameters(), **self.config.trainer.optimizer.params)

    def setup_datasets(self, params):
        dataset_class = getattr(importlib.import_module(self.config.datasets.module_name), self.config.datasets.class_name)
        return dataset_class(**params)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    # ---------- validation with global reduction ----------
    @torch.no_grad()
    def validate(self, model, epoch):
        model.eval()
        loss_dict = {}
        val_batches = 0

        for train_data in tqdm(self.val_dataloader, desc="Validating", leave=False, disable=(self.rank != 0)):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            input_data = self.extract_input_data(train_data)
            gt_data = self.extract_gt_data(train_data)

            input_data = input_data.to(self.device, non_blocking=True)
            gt_data = gt_data.to(self.device, non_blocking=True)

            # forward
            pred = model(input_data)
            all_loss = model.module.get_loss(pred, gt_data)

            for loss_name, loss_value in all_loss.items():
                if loss_name not in loss_dict:
                    loss_dict[loss_name] = 0.0
                loss_dict[loss_name] += float(loss_value.item())

            val_batches += 1
            del input_data, gt_data, pred, all_loss

        # --- All-reduce across ranks to get global mean loss ---
        loss_tensor = torch.tensor(list(loss_dict.values()) + [val_batches], device=self.device, dtype=torch.float32)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        global_loss_sum, global_batches = loss_tensor.tolist()[:-1], loss_tensor.tolist()[-1]
        avg_val_loss = {k: v / max(global_batches, 1) for k, v in zip(loss_dict.keys(), global_loss_sum)}
        avg_val_loss['total'] = sum(avg_val_loss.values())

        # --- write/checkpoint on rank 0 only ---
        if self.rank == 0 and self.writer is not None:
            for k, v in avg_val_loss.items():
                self.writer.add_scalar(f'Loss/Val_{k}', v, epoch)

        if self.rank == 0 and avg_val_loss['total'] < self.best_val_loss:
            self.best_val_loss = avg_val_loss['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': avg_val_loss['total'],
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

            train_loss_dict = {}
            train_batches = 0
            pbar = tqdm(self.train_dataloader,
                        desc=f"Epoch {epoch+1}/{self.epochs}",
                        leave=False,
                        disable=(self.rank != 0))
            for train_data in pbar:
                base_model.train()
                base_model.zero_grad()

                input_data = self.extract_input_data(train_data)
                gt_data = self.extract_gt_data(train_data)

                input_data = input_data.to(self.device, non_blocking=True)
                gt_data = gt_data.to(self.device, non_blocking=True)

                # forward (adapt to your actual outputs)
                pred = base_model(input_data)

                all_loss = base_model.module.get_loss(pred, gt_data)
                loss = sum(all_loss.values())
                loss.backward()

                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 10, norm_type=2)
                self.optimizer.step()

                for loss_name, loss_value in all_loss.items():
                    if loss_name not in train_loss_dict:
                        train_loss_dict[loss_name] = 0.0
                    train_loss_dict[loss_name] += float(loss_value.item())

                train_batches += 1

                if self.rank == 0:
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.6f}',
                        **{f'{k}': f'{v:.6f}' for k, v in train_loss_dict.items()},
                    })

                del input_data, gt_data, pred, all_loss, loss

            # --- reduce train metrics to global averages for logging ---
            train_tensor = torch.tensor(list(train_loss_dict.values()) + [train_batches],
                                        device=self.device, dtype=torch.float32)
            dist.all_reduce(train_tensor, op=dist.ReduceOp.SUM)
            g_loss_sum, g_batches = train_tensor.tolist()[:-1], train_tensor.tolist()[-1]
            train_avg_loss = {k: v / max(g_batches, 1) for k, v in zip(train_loss_dict.keys(), g_loss_sum)}
            train_avg_loss['total'] = sum(train_avg_loss.values())

            # step schedulers (same on every rank)
            for sch in self.schedulers:
                sch.step()

            if self.rank == 0 and self.writer is not None:
                for k, v in train_avg_loss.items():
                    self.writer.add_scalar(f'Loss/Train_{k}', v, epoch)

            if self.rank == 0:
                epoch_time = time.time() - epoch_start_time
                print(f'Epoch [{epoch+1}/{self.epochs}] ({epoch_time:.2f}s) '
                      f'Train={train_avg_loss["total"]:.6f}')

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
                    'train_loss': train_avg_loss['total'],
                }, self.log_dir / f'checkpoint_epoch_{epoch+1}.pth')

        if self.rank == 0 and self.writer is not None:
            self.writer.close()

        self.cleanup_distributed()
