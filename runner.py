import torch
import torch.nn as nn
import os
import json
from model import PCTransformer
from utils import misc, dist_utils
import time
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from chamfer import ChamferDistanceL1, ChamferDistanceL2
from dataset.dataset import ABCDataset
from torch.utils.data import DataLoader

def run_net(args, config, train_writer=None, val_writer=None):
    # build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = PCTransformer()
    base_model.to(device)


    train_dataset = ABCDataset(root='/mnt/d/data/processed_data_step', mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_dataset = ABCDataset(root='/mnt/d/data/processed_data_step', mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_dataset = ABCDataset(root='/mnt/d/data/processed_data_step', mode='test', level='hard')
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    # optimizer & scheduler
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-4, weight_decay=5e-4)
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    lambda_lr = lambda e: max(0.9 ** ((e - 0) / 21), 0.02) if e >= 0 else max(e / 0, 0.001)
    scheduler = [torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr, last_epoch=-1), misc.BNMomentumScheduler(base_model, lambda_lr)]

    # training
    base_model.zero_grad()
    for epoch in range(600):
        base_model.train()
        batch_start_time = time.time()

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (gt_brep_grid, gt_point_cloud, partial_point_cloud, cropped_point_cloud) in enumerate(train_dataloader):
            gt = gt_point_cloud.cuda()
            gt_brep_grid = gt_brep_grid.cuda()
            partial = partial_point_cloud.cuda()
            cropped = cropped_point_cloud.cuda()

            coarse_point_cloud, rebuild_points, brep_grids = base_model(partial)
            
            sparse_loss, dense_loss, brep_loss = base_model.get_loss(coarse_point_cloud, rebuild_points, brep_grids, gt, gt_brep_grid)
         
            _loss = sparse_loss + dense_loss + brep_loss 
            _loss.backward()

            torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
            optimizer.step()
            base_model.zero_grad()


        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()


        if epoch % args.val_freq == 0:
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)



def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    interval =  n_samples // 10

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'PCN' or dataset_name == 'Completion3D' or dataset_name == 'Projected_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            ret = base_model(partial)
            coarse_points = ret[0]
            dense_points = ret[-1]

            sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
            dense_loss_l1 =  ChamferDisL1(dense_points, gt)
            dense_loss_l2 =  ChamferDisL2(dense_points, gt)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])


            # dense_points_all = dist_utils.gather_tensor(dense_points, args)
            # gt_all = dist_utils.gather_tensor(gt, args)

            # _metrics = Metrics.get(dense_points_all, gt_all)
            _metrics = Metrics.get(dense_points, gt)
            if args.distributed:
                _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
            else:
                _metrics = [_metric.item() for _metric in _metrics]

            for _taxonomy_id in taxonomy_ids:
                if _taxonomy_id not in category_metrics:
                    category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[_taxonomy_id].update(_metrics)


            # if val_writer is not None and idx % 200 == 0:
            #     input_pc = partial.squeeze().detach().cpu().numpy()
            #     input_pc = misc.get_ptcloud_img(input_pc)
            #     val_writer.add_image('Model%02d/Input'% idx , input_pc, epoch, dataformats='HWC')

            #     sparse = coarse_points.squeeze().cpu().numpy()
            #     sparse_img = misc.get_ptcloud_img(sparse)
            #     val_writer.add_image('Model%02d/Sparse' % idx, sparse_img, epoch, dataformats='HWC')

            #     dense = dense_points.squeeze().cpu().numpy()
            #     dense_img = misc.get_ptcloud_img(dense)
            #     val_writer.add_image('Model%02d/Dense' % idx, dense_img, epoch, dataformats='HWC')
                
            #     gt_ptcloud = gt.squeeze().cpu().numpy()
            #     gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
            #     val_writer.add_image('Model%02d/DenseGT' % idx, gt_ptcloud_img, epoch, dataformats='HWC')
        
            if (idx+1) % interval == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


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
