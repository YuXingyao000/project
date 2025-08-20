import torch
import torch.nn as nn

from .AdaPoinTrPCTransformer import AdaPoinTrPCTransformer
from .Utils import knn_index

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SimpleRebuildFCLayer(nn.Module):
    def __init__(self, input_dims, step, hidden_dim=512):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        self.layer = Mlp(self.input_dims, hidden_dim, step * 3)

    def forward(self, rec_feature):
        '''
        Input BNC
        '''
        batch_size = rec_feature.size(0)
        g_feature = rec_feature.max(1)[0]
        token_feature = rec_feature
            
        patch_feature = torch.cat([
                g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1),
                token_feature
            ], dim = -1)
        rebuild_pc = self.layer(patch_feature).reshape(batch_size, -1, self.step , 3)
        assert rebuild_pc.size(1) == rec_feature.size(1)
        return rebuild_pc

class AdaPoinTr(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.trans_dim = kwargs.get('trans_dim', 384)
        self.num_query = kwargs.get('num_query', 512)
        self.num_points = kwargs.get('num_points', 8192)

        self.base_model = AdaPoinTrPCTransformer(
            in_chans=kwargs.get('in_chans', 3),
            embed_dim=kwargs.get('embed_dim', 384),
            encoder_depth=kwargs.get('encoder_depth', [1, 5]),
            decoder_depth=kwargs.get('decoder_depth', [1, 7]),
            num_heads=kwargs.get('num_heads', 6),
            grouper_k_nearest_neighbors=kwargs.get('grouper_k_nearest_neighbors', 16),
            grouper_downsample=kwargs.get('grouper_downsample', [4, 8]),
            attention_k_nearest_neighbors=kwargs.get('attention_k_nearest_neighbors', 8),
            num_noised_query=kwargs.get('num_noised_query', 64),
            norm_eps=kwargs.get('norm_eps', 0.000001),
        )
        
        self.factor = self.num_points // self.num_query
        assert self.num_points % self.num_query == 0
        self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.num_points // self.num_query)  # rebuild a cluster point

        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        self.loss_func = self.setup_loss_func()

    def setup_loss_func(self):
        from chamfer_distance import ChamferDistance
        return ChamferDistance()

    def get_loss(self, ret, gt):
        if self.training:
            pred_coarse, denoised_coarse, pred_fine, denoised_fine = ret
            assert pred_fine.size(1) == gt.size(1)
            # denoise loss
            idx = knn_index(self.factor, denoised_coarse.transpose(1, 2), gt.transpose(1, 2)) # B n k 
            denoised_target = index_points(gt, idx) # B n k 3 
            denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
            assert denoised_target.size(1) == denoised_fine.size(1)
            loss_denoised_l, loss_denoised_r, _, _ = self.loss_func(denoised_fine, denoised_target)
            loss_denoised = (loss_denoised_l.mean(dim=1) + loss_denoised_r.mean(dim=1)).mean()

            # recon loss
            loss_coarse_l, loss_coarse_r, _, _ = self.loss_func(pred_coarse, gt)
            loss_fine_l, loss_fine_r, _, _ = self.loss_func(pred_fine, gt)
            loss_recon = (loss_coarse_l.mean(dim=1) + loss_coarse_r.mean(dim=1)).mean() + (loss_fine_l.mean(dim=1) + loss_fine_r.mean(dim=1)).mean()
            return {'denoised': loss_denoised, 'recon': loss_recon}
        else:
            pred_coarse, pred_fine = ret
            assert pred_fine.size(1) == gt.size(1)
            # recon loss
            loss_coarse_l, loss_coarse_r, _, _ = self.loss_func(pred_coarse, gt)
            loss_fine_l, loss_fine_r, _, _ = self.loss_func(pred_fine, gt)
            loss_recon = (loss_coarse_l.mean(dim=1) + loss_coarse_r.mean(dim=1)).mean() + (loss_fine_l.mean(dim=1) + loss_fine_r.mean(dim=1)).mean()
            return {'recon': loss_recon}

    def forward(self, xyz):
        q, coarse_point_cloud = self.base_model(xyz) # B M C and B M 3
    
        B, M ,C = q.shape

        global_feature = self.increase_dim(q.transpose(1,2)).transpose(1,2) # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024

        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C

        rebuild_feature = self.reduce_map(rebuild_feature) # B M C
        relative_xyz = self.decode_head(rebuild_feature)   # B M S 3
        rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-2))  # B M S 3

        if self.training:
            denoise_length = self.base_model.num_noised_query
            # split the reconstruction and denoise task
            pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, 3).contiguous()
            pred_coarse = coarse_point_cloud[:, :-denoise_length].contiguous()

            denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, 3).contiguous()
            denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()

            assert pred_fine.size(1) == self.num_query * self.factor
            assert pred_coarse.size(1) == self.num_query

            return pred_coarse, denoised_coarse, pred_fine, denoised_fine

        else:
            denoise_length = 0
            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()  # B N 3

            assert rebuild_points.size(1) == self.num_query * self.factor
            assert coarse_point_cloud.size(1) == self.num_query

            return coarse_point_cloud, rebuild_points