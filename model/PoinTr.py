import torch
from torch import nn

from pointnet2_ops import pointnet2_utils
from model.PCTransformer import PCTransformer
from model.geometry import extract_coordinates_and_features


def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous() # type: ignore
    return sub_pc


class Fold(nn.Module):
    def __init__(self, in_channel , step , hidden_dim = 512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2


class PoinTr(nn.Module):
    def __init__(self):
        super().__init__()
        self.trans_dim = 384
        self.knn_layer = 1
        self.num_pred = 14336 // 4
        self.num_query = 224
        self.global_feature_dim = 1024

        self.fold_step = int(pow(self.num_pred//self.num_query, 0.5) + 0.5)
        self.base_model = PCTransformer(in_chans = 3, embed_dim = self.trans_dim, depth = [[1, 5], [1, 7]], num_heads = 6, num_query = self.num_query)
        
        self.foldingnet = Fold(self.trans_dim, step = self.fold_step, hidden_dim = 256)  # rebuild a cluster point

        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, self.global_feature_dim, 1),
            nn.BatchNorm1d(self.global_feature_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.global_feature_dim, self.global_feature_dim, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + self.global_feature_dim + 3, self.trans_dim)

        # self.brep_grid_test = nn.Sequential(
        #     nn.Conv1d(self.num_query, 30, 1),
        #     nn.LayerNorm(self.trans_dim + self.global_feature_dim + 3),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Linear(self.trans_dim + self.global_feature_dim + 3, 16 * 16 * 3),
        #     nn.ReLU(inplace=True)
        # )
        
        self.build_loss_func()

    def build_loss_func(self):
        from chamfer_distance import ChamferDistance as chamfer_dist
        self.loss_func = chamfer_dist()

    def get_loss(self, coarse_point_cloud, rebuild_points, gt_point_cloud):
        loss_coarse_l, loss_coarse_r, loss_coarse_idx_l, loss_coarse_idx_r = self.loss_func(coarse_point_cloud, gt_point_cloud)
        loss_fine_l, loss_fine_r, loss_fine_idx_l, loss_fine_idx_r = self.loss_func(rebuild_points, gt_point_cloud)
        # loss_brep = self.loss_func(pred_brep_grid, gt_brep_grid)
        return (loss_coarse_l.mean(dim=1) + loss_coarse_r.mean(dim=1)).mean(), (loss_fine_l.mean(dim=1) + loss_fine_r.mean(dim=1)).mean()

    def forward(self, xyz):
        query_features, coarse_point_cloud = self.base_model(xyz)  # bs, [3 + 384], num_query

        batch_size, num_query, feature_dim = query_features.shape

        global_feature = self.increase_dim(query_features.transpose(1,2)).transpose(1,2) # bs num_query 1024
        global_feature = torch.max(global_feature, dim=1)[0] # bs 1024

        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, num_query, -1),
            query_features,
            coarse_point_cloud], dim=-1) # bs num_query 1027 + 384

        # brep_grid_test = self.brep_grid_test(rebuild_feature)
        # brep_grid_test = brep_grid_test.reshape(batch_size, 30, 16, 16, 3)
        rebuild_feature = self.reduce_map(rebuild_feature.reshape(batch_size * num_query, -1)) # bs num_query 1027 + 384
        
        # # NOTE: try to rebuild pc
        # coarse_point_cloud = self.refine_coarse(rebuild_feature).reshape(B, M, 3)

        # NOTE: foldingNet
        relative_xyz = self.foldingnet(rebuild_feature).reshape(batch_size, num_query, 3, -1)    # bs num_query 3 S
        rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2,3).reshape(batch_size, -1, 3)  # bs num_pred 3

        # cat the input
        inp_sparse = fps(xyz, self.num_query)
        coarse_point_cloud = torch.cat([coarse_point_cloud, inp_sparse], dim=1).contiguous()
        rebuild_points = torch.cat([rebuild_points, xyz],dim=1).contiguous()

        return coarse_point_cloud, rebuild_points

