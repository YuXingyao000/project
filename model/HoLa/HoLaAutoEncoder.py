"""
This module is from our last work: HoLa-BRep.
The code is cleaned and optimized for our new work.
HoLa-BRep: https://arxiv.org/abs/2504.14257
"""

import copy
import time
import torch
from torch import nn, Tensor
from torch.nn import ModuleList
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from einops import rearrange

from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter_mean


def add_timer(time_statics, v_attr, timer):
    if v_attr not in time_statics:
        time_statics[v_attr] = 0.
    time_statics[v_attr] += time.time() - timer
    return time.time()


def profile_time(time_dict, key, v_timer):
    torch.cuda.synchronize()
    cur = time.time()
    time_dict[key] += cur - v_timer
    return cur


def denormalize_coord(normalized_points_with_normals, bounding_box_info):
    """
    Convert normalized coordinates back to original scale and position.
    
    Args:
        normalized_points_with_normals: Tensor [..., 6] with normalized coords + normals
        bounding_box_info: Tensor [..., 4] with [center_x, center_y, center_z, scale]
    
    Returns:
        original_coords_with_normals: Points restored to original coordinate system
    """
    # Extract components
    normalized_coords = normalized_points_with_normals[..., :3]
    normalized_normals = normalized_points_with_normals[..., 3:]
    
    # Extract transformation parameters
    original_center = bounding_box_info[..., :3]  # [N, 3]
    original_scale = bounding_box_info[..., 3:4]  # [N, 1]
    
    # Handle dimension broadcasting for batch processing
    while len(normalized_coords.shape) > len(original_center.shape):
        original_center = original_center.unsqueeze(1)
        original_scale = original_scale.unsqueeze(1)
    
    # Calculate target points in normalized space
    normalized_targets = normalized_coords + normalized_normals
    
    # Scale back to original coordinate system
    original_coords = normalized_coords * original_scale + original_center
    original_targets = normalized_targets * original_scale + original_center
    
    # Recalculate normals in original space
    original_normals = original_targets - original_coords
    # Ensure unit length
    original_normals = original_normals / (1e-6 + torch.linalg.norm(original_normals, dim=-1, keepdim=True))
    
    # Combine coordinates and normals
    result = torch.cat((original_coords, original_normals), dim=-1)
    return result


class res_block_xd(nn.Module):
    def __init__(self, dim, dim_in, dim_out, kernel_size=3, stride=1, padding=1, v_norm=None, v_norm_shape=None):
        super(res_block_xd, self).__init__()
        self.downsample = None
        if v_norm is None or  v_norm == "none":
            norm = nn.Identity()
        elif v_norm == "layer":
            norm = nn.LayerNorm(v_norm_shape)
            
        if dim == 0:
            self.conv1 = nn.Linear(dim_in, dim_out)
            self.norm1 = copy.deepcopy(norm)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Linear(dim_out, dim_out)
            self.norm2 = copy.deepcopy(norm)
            if dim_in != dim_out:
                self.downsample = nn.Sequential(
                    nn.Linear(dim_in, dim_out),
                    copy.deepcopy(norm),
                )
        if dim == 1:
            self.conv1 = nn.Conv1d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm1 = copy.deepcopy(norm)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv1d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm2 = copy.deepcopy(norm)
            if dim_in != dim_out:
                self.downsample = nn.Sequential(
                    nn.Conv1d(dim_in, dim_out, kernel_size=1, stride=1, bias=False),
                    copy.deepcopy(norm),
                )
        elif dim == 2:
            self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm1 = copy.deepcopy(norm)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm2 = copy.deepcopy(norm)
            if dim_in != dim_out:
                self.downsample = nn.Sequential(
                    nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False),
                    copy.deepcopy(norm),
                )
        elif dim == 3:
            self.conv1 = nn.Conv3d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm1 = copy.deepcopy(norm)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv3d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm2 = copy.deepcopy(norm)
            if dim_in != dim_out:
                self.downsample = nn.Sequential(
                    nn.Conv3d(dim_in, dim_out, kernel_size=1, stride=1, bias=False),
                    copy.deepcopy(norm),
                )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Neural intersection module
class AttnIntersection(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_latent,
        num_layers,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.attn_proj_in = nn.Linear(dim_in, dim_latent)
        layer = nn.TransformerDecoderLayer(
            dim_latent, 8, dim_feedforward=2048, dropout=0.1, 
            batch_first=True, norm_first=True)
        self.layers = ModuleList([copy.deepcopy(layer) for i in range(num_layers)])
        self.attn_proj_out = nn.Linear(dim_latent, dim_in * 2)

        self.pos_encoding = nn.Parameter(torch.randn(1, 2, dim_latent))


    def forward(
            self,
            src) -> Tensor:
        output = self.attn_proj_in(src) + self.pos_encoding
        tgt = output[:,0:1]
        mem = output[:,1:2]
        
        for mod in self.layers:
            tgt = mod(tgt, mem)

        output = self.attn_proj_out(tgt)[:,0]
        return output


# Brep Grid Autoencoder
# ref: https://arxiv.org/abs/2504.14257
# Modified and light-weight version
class HoLaAutoEncoder(nn.Module):
    def __init__(self, in_channels=6, primitive_feature_dim=768, face_latent_dim=8, gaussian_weights=1e-6):
        super().__init__()
        self.dim_shape = primitive_feature_dim
        self.dim_latent = face_latent_dim
        norm = "layer" # LayerNorm
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = self.dim_latent * 2 * 2
        df = self.df

        self.in_channels = in_channels

        self.face_coords = nn.Sequential(
            nn.Conv2d(self.in_channels, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((ds // 8, 16,16)),
            nn.LeakyReLU(),
            res_block_xd(2, ds // 8, ds // 4, 3, 1, 1, v_norm=norm , v_norm_shape = (ds // 4, 16, 16)),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8
            res_block_xd(2, ds // 4, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 2, 8, 8)),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4
            res_block_xd(2, ds // 2, ds // 1, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 4, 4)),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2
            res_block_xd(2, ds // 1, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 2, 2)),
            nn.Conv2d(ds, dl, kernel_size=1, stride=1, padding=0),
            Rearrange("b n h w -> b (n h w)")
        )
        self.edge_coords = nn.Sequential(
            nn.Conv1d(self.in_channels, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((ds // 8, 16,)),
            nn.LeakyReLU(),
            res_block_xd(1, ds // 8, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 4, 16,)),
            nn.MaxPool1d(kernel_size=2, stride=2), # 8
            res_block_xd(1, ds // 4, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 2, 8,)),
            nn.MaxPool1d(kernel_size=2, stride=2), # 4
            res_block_xd(1, ds // 2, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 4,)),
            nn.MaxPool1d(kernel_size=2, stride=2), # 2
            res_block_xd(1, ds, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 2,)),
            nn.Conv1d(ds, df, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        ) # b c 1

        self.graph_face_edge = nn.ModuleList()
        for i in range(5):
            self.graph_face_edge.append(GATv2Conv(
                df, df, 
                heads=1, edge_dim=df * 2,
            ))
            self.graph_face_edge.append(nn.LeakyReLU())
        
        bd = 768 # bottlenek_dim
        self.face_attn_proj_in = nn.Linear(df, bd)
        self.face_attn_proj_out = nn.Linear(bd, df)
        layer = nn.TransformerEncoderLayer(
            bd, 16, dim_feedforward=2048, dropout=0.1, 
            batch_first=True, norm_first=True)
        self.face_attn = nn.TransformerEncoder(layer, 8, nn.LayerNorm(bd))

        self.global_feature1 = nn.Sequential(
            nn.Linear(df, df),
            nn.LeakyReLU(),
            nn.Linear(df, df),
        )
        self.global_feature2 = nn.Sequential(
            nn.Linear(df * 2, df),
            nn.LeakyReLU(),
            nn.Linear(df, df),
        )

        self.inter = AttnIntersection(df, 512, 8)
        self.classifier = nn.Linear(df*2, 1)

        self.face_attn_proj_in2 = nn.Linear(df, bd)
        self.face_attn_proj_out2 = nn.Linear(bd, df)
        layer2 = nn.TransformerEncoderLayer(
            bd, 16, dim_feedforward=2048, dropout=0.1, 
            batch_first=True, norm_first=True)
        self.face_attn2 = nn.TransformerEncoder(layer2, 8)

        # Decoder
        self.face_points_decoder = nn.Sequential(
            Rearrange("b (n h w) -> b n h w", h=2, w=2),
            res_block_xd(2, dl, ds, 3, 1, 1, v_norm=norm, v_norm_shape=(ds, 2, 2)),
            nn.ConvTranspose2d(ds // 1, ds // 2, kernel_size=2, stride=2),
            res_block_xd(2, ds // 2, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 2, 4, 4)),
            nn.ConvTranspose2d(ds // 2, ds // 4, kernel_size=2, stride=2),
            res_block_xd(2, ds // 4, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 4, 8, 8)),
            nn.ConvTranspose2d(ds // 4, ds // 8, kernel_size=2, stride=2),
            res_block_xd(2, ds // 8, ds // 8, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 8, 16, 16)),
            nn.Conv2d(ds // 8, self.in_channels, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w h -> ... w h c',c=self.in_channels),
        )
        self.face_center_scale_decoder = nn.Sequential(
            res_block_xd(0, dl * 2 * 2, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            nn.Linear(ds, 4),
        )
        
        self.edge_points_decoder = nn.Sequential(
            Rearrange("b (n w)-> b n w", n=df, w=2),
            res_block_xd(1, df, ds, 3, 1, 1, v_norm=norm, v_norm_shape=(ds, 2,)),
            nn.ConvTranspose1d(ds, ds // 2, kernel_size=2, stride=2),
            res_block_xd(1, ds // 2, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 2, 4,)),
            nn.ConvTranspose1d(ds // 2, ds // 4, kernel_size=2, stride=2),
            res_block_xd(1, ds // 4, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 4, 8,)),
            nn.ConvTranspose1d(ds // 4, ds // 8, kernel_size=2, stride=2),
            res_block_xd(1, ds // 8, ds // 8, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 8, 16,)),
            nn.Conv1d(ds // 8, self.in_channels, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w -> ... w c',c=self.in_channels),
        )
        self.edge_center_scale_decoder = nn.Sequential(
            res_block_xd(0, df * 2, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            nn.Linear(ds, 4),
        )
        
        self.gaussian_weights = gaussian_weights
        self.gaussian_proj = nn.Sequential(
            nn.Linear(self.df, self.df*2),
            nn.LeakyReLU(),
            nn.Linear(self.df*2, self.df*2),
        )

        self.times = {
            "Encoder": 0,
            "Fuser": 0,
            "Sample": 0,
            "global": 0,
            "Decoder": 0,
            "Intersection": 0,
            "Loss": 0,
        }

        self.loss_fn = nn.MSELoss()


    def sample(self, face_latent_features):
        if self.gaussian_weights <= 0:
            return self.gaussian_proj(face_latent_features), torch.zeros_like(face_latent_features[0,0])

        fused_face_features_gau = self.gaussian_proj(face_latent_features)
        fused_face_features_gau = fused_face_features_gau.reshape(-1, self.df, 2)
        mean = fused_face_features_gau[:, :, 0]
        logvar = fused_face_features_gau[:, :, 1]
        std = torch.exp(0.5 * logvar)

        if self.training:
            eps = torch.randn_like(std)
            fused_face_features = eps.mul(std).add_(mean)
        else:
            fused_face_features = mean
        
        return fused_face_features, mean, std

    
    def profile_time(self, timer, key):
        torch.cuda.synchronize()
        self.times[key] += time.time() - timer
        timer = time.time()
        return timer

    
    def encode(self, face_points, edge_points, face_attn_mask, edge_face_connectivity, face_index):
        face_points = rearrange(face_points[..., :self.in_channels], 'b h w n -> b n h w').contiguous()
        edge_points = rearrange(edge_points[..., :self.in_channels], 'b h n -> b n h').contiguous()      
        face_features = self.face_coords(face_points)        
        half_edge_features = self.edge_coords(edge_points)

        # ---- Face self-attention ----
        attn_x = self.face_attn_proj_in(face_features)
        attn_x = self.face_attn(attn_x, face_attn_mask)
        attn_x = self.face_attn_proj_out(attn_x)
        fused_face_features = face_features + attn_x

        # ---- Face-edge GAT ----
        x = fused_face_features
        edge_index = edge_face_connectivity[:, 1:].permute(1, 0)
        edge_attr = half_edge_features[edge_face_connectivity[:, 0]]
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)
        fused_face_features = x + fused_face_features

        # ---- Global feature ----
        batch_size = face_index.shape[0]
        index = torch.arange(batch_size, device=fused_face_features.device).repeat_interleave(face_index)
        global_feature = scatter_mean(fused_face_features, index, dim=0)
        global_feature = self.global_feature1(global_feature)
        global_feature = global_feature.repeat_interleave(face_index, dim=0)
        
        # ---- Output projection ----
        face_latent_features = torch.cat((fused_face_features, global_feature), dim=1)
        face_latent_features = self.global_feature2(face_latent_features) + fused_face_features

        # Half edge features will be used to compute the loss
        return face_latent_features, half_edge_features

    def decode(self, face_features, encoded_half_edge_feature=None, face_attn_mask=None, edge_face_connectivity=None, non_intersection_faces=None):
        face_z = face_features
        face_feature = self.face_attn_proj_in2(face_z)
        
        # ---- Face processing ----
        if face_attn_mask is None:
            # This branch will be reached during the inference.
            # During the training, faces can only attend to the faces in the same model, 
            # while during the inference, we input one single model at once so no mask is needed.
            
            # Training: [num_face*num_models, feature_dim]
            # Inference: [num_face, feature_dim]
            num_faces = face_z.shape[0]
            face_attn_mask = torch.zeros((num_faces, num_faces), dtype=bool, device=face_z.device)
            
        face_feature = self.face_attn2(face_feature, face_attn_mask)
        face_z = self.face_attn_proj_out2(face_feature)
        
        face_points_local = self.face_points_decoder(face_z)
        face_bounding_box = self.face_center_scale_decoder(face_z)
        
        # ---- Deduplication (inference only) ----
        if not self.training: 
            pred_face_points = denormalize_coord(face_points_local, face_bounding_box)
            num_faces = face_z.shape[0]

            deduplicate_face_id = []
            for i in range(num_faces):
                is_duplicate = False
                for j in deduplicate_face_id:
                    if torch.sqrt(((pred_face_points[i]-pred_face_points[j])**2).sum(dim=-1)).mean() < 1e-3:
                        is_duplicate=True
                        break
                if not is_duplicate:
                    deduplicate_face_id.append(i)
            face_z = face_z[deduplicate_face_id]
            
        # ---- Intersection prediction ----
        if not self.training:
            num_faces = face_z.shape[0]
            device = face_z.device

            indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)
            indexes = indexes.reshape(-1,2).to(device)

            feature_pair = face_z[indexes]
            feature_pair = self.inter(feature_pair)
            
            is_intersect = self.classifier(feature_pair)[...,0]
            is_intersect = torch.sigmoid(is_intersect) > 0.5

            half_edge_feature = feature_pair[is_intersect]
        else:
            true_intersection_embedding = face_z[edge_face_connectivity[:, 1:]]
            false_intersection_embedding = face_z[non_intersection_faces]
            id_false_start = true_intersection_embedding.shape[0]

            feature_pair = torch.cat((true_intersection_embedding, false_intersection_embedding), dim=0)
            feature_pair = self.inter(feature_pair)
            
            is_intersect = self.classifier(feature_pair)
            half_edge_feature = feature_pair[:id_false_start]

        # ---- Edge Decode ----
        intersected_half_edge_coords = self.edge_points_decoder(half_edge_feature)
        intersected_half_edge_bbox = self.edge_center_scale_decoder(half_edge_feature)
        if self.training and encoded_half_edge_feature:
            # We will decode the encoded edge features directly from the encoder as well during the training.
            # Edges generated from the face intersection are "conditional" predicted edges.
            # We also need stable predicted to make sure the decoder can learn the true edge information, making training easier.
            decoded_half_edge_coords = self.edge_points_decoder(encoded_half_edge_feature)
            decoded_half_edge_bbox = self.edge_center_scale_decoder(encoded_half_edge_feature)
            return face_z, is_intersect, half_edge_feature, (intersected_half_edge_coords, decoded_half_edge_coords), (intersected_half_edge_bbox, decoded_half_edge_bbox)
        
        return face_z, is_intersect, half_edge_feature, intersected_half_edge_coords, intersected_half_edge_bbox
        
        
    # TODO: Coming Soon (This loss function is really a mess, but at least I think you can understand what kinds of losses are to compute...)
    # def loss(self, pred, gt):
    #     # Loss
    #     loss={}
    #     loss["face_norm"] = self.loss_fn(
    #         pred["face_points_local"],
    #         gt["face_norm"]
    #     )
    #     loss["face_bbox"] = self.loss_fn(
    #         pred["face_center_scale"],
    #         gt["face_bbox"]
    #     )
    #
    # I can give you a hint, "edge_points_local1" is the edge points directly decoded from the encoded edge features,
    # while "edge_points_local" is the conditional predicted one
    #     loss["edge_norm1"] = self.loss_fn(
    #         pred["edge_points_local1"],
    #         gt["edge_norm"]
    #     )
    #     loss["edge_bbox1"] = self.loss_fn(
    #         pred["edge_center_scale1"],
    #         gt["edge_bbox"]
    #     )
    #     loss["edge_feature"] = pred["loss_edge_feature"]
    #     loss["edge_classification"] = pred["loss_edge"] * 0.1
    #     edge_face_connectivity = gt["edge_face_connectivity"]
    #     loss["edge_norm"] = self.loss_fn(
    #         pred["edge_points_local"],
    #         gt["edge_norm"][edge_face_connectivity[:, 0]]
    #     )
    #     loss["edge_bbox"] = self.loss_fn(
    #         pred["edge_center_scale"],
    #         gt["edge_bbox"][edge_face_connectivity[:, 0]]
    #     )
    #     if self.gaussian_weights > 0:
    #         loss["kl_loss"] = (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())) * self.gaussian_weights
    #     return loss
    
    # I tried my best, the loss function is really impossible to handle...
    # But I can assure you that all the candidate variables are reserved for you.
    # And by the way, I cannot even find the trainer for this VAE.
    # Only god can know what have they done ...
    # But the forwarding can work now
    def forward(self, 
                face_points, 
                edge_points,
                edge_face_connectivity, 
                zero_positions,
                attn_mask, 
                num_face_record):
        
        face_latent_feature, edge_feature = self.encode(face_points, edge_points, attn_mask, edge_face_connectivity, num_face_record)
        face_feature, mean, logvar = self.sample(face_latent_feature)
        face_z, is_intersect, half_edge_feature, half_edge_coords, half_edge_bbox = self.decode(face_feature, edge_feature, attn_mask, edge_face_connectivity, zero_positions)

        return face_z, is_intersect, half_edge_feature, half_edge_coords, half_edge_bbox, mean, logvar
    
