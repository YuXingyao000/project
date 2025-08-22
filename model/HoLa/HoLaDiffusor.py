import importlib
import math
import torch
import torch.nn as nn
import numpy as np

from diffusers import DDPMScheduler
from pointnet2_ops.pointnet2_modules import PointnetSAModuleMSG

from .HoLaAutoEncoder import HoLaAutoEncoder


def sincos_embedding(input, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param input: a N-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=input.dtype, device=input.device) / half
    )
    for _ in range(len(input.size())):
        freqs = freqs[None]
    args = input.unsqueeze(-1).float() * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Diffusion_condition(nn.Module):
    def __init__(self,
                 in_channels=6,
                 latent_dim=768, condition_dim=256, diffusion_type='epsilon',
                 ae_bottleneck_dim=768, ae_face_latent_dim=8, ae_gaussian_weights=1e-6,
                 ae_weights_path=None,
                 ):
        super().__init__()
        self.dim_input = 8 * 2 * 2
        self.dim_latent = latent_dim
        self.dim_condition = condition_dim
        self.dim_total = self.dim_latent + self.dim_condition

        # Diffusion input pre-process
        self.p_embed = nn.Sequential(
            nn.Linear(self.dim_input, self.dim_latent),
            nn.LayerNorm(self.dim_latent),
            nn.SiLU(),
            nn.Linear(self.dim_latent, self.dim_latent),
        )
        self.time_embed = nn.Sequential(
            nn.Linear(self.dim_total, self.dim_total),
            nn.LayerNorm(self.dim_total),
            nn.SiLU(),
            nn.Linear(self.dim_total, self.dim_total),
        )

        # Diffusion backbone
        layer1 = nn.TransformerEncoderLayer(
                d_model=self.dim_total,
                nhead=self.dim_total // 64, norm_first=True, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.net1 = nn.TransformerEncoder(layer1, 24, nn.LayerNorm(self.dim_total))
        self.fc_out = nn.Sequential(
                nn.Linear(self.dim_total, self.dim_total),
                nn.LayerNorm(self.dim_total),
                nn.SiLU(),
                nn.Linear(self.dim_total, self.dim_input),
        )

        # Point Cloud Encoder
        self.SA_modules = nn.ModuleList()
        c_in = 6
        with_bn = False
        self.SA_modules.append(
                PointnetSAModuleMSG(
                        npoint=1024,
                        radii=[0.05, 0.1],
                        nsamples=[16, 32],
                        mlps=[[c_in, 32], [c_in, 64]],
                        use_xyz=True,
                        bn=with_bn
                )
        )
        c_out_0 = 32 + 64
        c_in = c_out_0
        self.SA_modules.append(
                PointnetSAModuleMSG(
                        npoint=256,
                        radii=[0.1, 0.2],
                        nsamples=[16, 32],
                        mlps=[[c_in, 64], [c_in, 128]],
                        use_xyz=True,
                        bn=with_bn
                )
        )
        c_out_1 = 64 + 128
        c_in = c_out_1
        self.SA_modules.append(
                PointnetSAModuleMSG(
                        npoint=64,
                        radii=[0.2, 0.4],
                        nsamples=[16, 32],
                        mlps=[[c_in, 128], [c_in, 256]],
                        use_xyz=True,
                        bn=with_bn
                )
        )
        c_out_2 = 128 + 256
        c_in = c_out_2
        self.SA_modules.append(
                PointnetSAModuleMSG(
                        npoint=16,
                        radii=[0.4, 0.8],
                        nsamples=[16, 32],
                        mlps=[[c_in, 512], [c_in, 512]],
                        use_xyz=True,
                        bn=with_bn
                )
        )
        self.fc_lyaer = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Linear(1024, self.dim_condition),
        )
        # End of Point Cloud Encoder

  
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='squaredcos_cap_v2',
            prediction_type=diffusion_type,
            beta_start=1e-4,
            beta_end=0.02,
            variance_type='fixed_small',
            clip_sample=False,
        )


        self.num_max_faces = 30
        self.loss = nn.functional.mse_loss
        self.diffusion_type = diffusion_type

        self.use_mean = True
        self.ae_model = self._load_frozen_ae(
            ae_weights_path,
            in_channels=in_channels,
            bottleneck_dim=ae_bottleneck_dim,
            face_latent_dim=ae_face_latent_dim,
            gaussian_weights=ae_gaussian_weights,
        )
    
    def _load_frozen_ae(self, weights_path, in_channels, bottleneck_dim, face_latent_dim, gaussian_weights):
        # Init
        ae = HoLaAutoEncoder(
            in_channels=in_channels,
            primitive_feature_dim=bottleneck_dim,
            face_latent_dim=face_latent_dim,
            gaussian_weights=gaussian_weights,
        )
        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        ae.load_state_dict(state_dict, strict=True)

        # Freeze
        ae.requires_grad_(False)
        ae.eval()

        return ae
    
    def get_latent_feature(self, face_points, edge_points, face_attn_mask, edge_face_connectivity, face_index):
        data = {}
        with torch.no_grad() and torch.autocast(device_type='cuda', dtype=torch.float32):
            face_latents, _ = self.ae_model.encode(face_points, edge_points, face_attn_mask, edge_face_connectivity, face_index)
            face_features, _, _ = self.ae_model.sample(face_latents, self.use_mean)
        dim_latent = face_features.shape[-1]
        num_faces = face_index
        bs = num_faces.shape[0]
        
        positions = torch.arange(self.num_max_faces, device=face_features.device).unsqueeze(0).repeat(bs, 1)
        mandatory_mask = positions < num_faces[:,None]
        random_indices = (torch.rand((bs, self.num_max_faces), device=face_features.device) * num_faces[:,None]).long()
        indices = torch.where(mandatory_mask, positions, random_indices)
        num_faces_cum = num_faces.cumsum(dim=0).roll(1)
        num_faces_cum[0] = 0
        indices += num_faces_cum[:,None]
        
        # Permute the indices
        r_indices = torch.argsort(torch.rand((bs, self.num_max_faces), device=face_features.device), dim=1)
        indices = indices.gather(1, r_indices)
        return face_features[indices]

    
    # TODO: This is going to be replaced by DiT
    def diffuse(self, v_feature, v_timesteps, v_condition=None):
        bs = v_feature.size(0)
        de = v_feature.device
        dt = v_feature.dtype
        time_embeds = self.time_embed(sincos_embedding(v_timesteps, self.dim_total)).unsqueeze(1)
        noise_features = self.p_embed(v_feature)
        v_condition = torch.zeros((bs, 1, self.dim_condition), device=de, dtype=dt) if v_condition is None else v_condition
        v_condition = v_condition.repeat(1, v_feature.shape[1], 1)
        noise_features = torch.cat([noise_features, v_condition], dim=-1)
        noise_features = noise_features + time_embeds

        pred_x0 = self.net1(noise_features)
        pred_x0 = self.fc_out(pred_x0)
        return pred_x0

    def encode_condition(self, point_cloud):
        condition = None

        pc = point_cloud
        
        if pc.shape[2] > 2048:
            num_points = pc.shape[2]
            index = np.arange(num_points)
            np.random.shuffle(index)
            pc = pc[:, :, index[:2048]]
        assert pc.shape[2] == 2048
        points = pc[:, 0, :, :3]
        normals = pc[:, 0, :, 3:6]
        pc = torch.cat([points, normals], dim=-1)
        
        l_xyz, l_features = [pc[:, :, :3].contiguous().float()], [pc.permute(0, 2, 1).contiguous().float()]
        with torch.autocast(device_type=pc.device.type, dtype=torch.float32):
            for i in range(len(self.SA_modules)):
                li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
                l_xyz.append(li_xyz)
                l_features.append(li_features)
            features = self.fc_lyaer(l_features[-1].mean(dim=-1))
            condition = features[:, None]
        
        return condition

    def forward(self, fused_latent, condition=None):
        device = fused_latent.device
        bs = fused_latent.size(0)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

        if condition is not None:
            condition = self.encode_condition(condition)
        noise = torch.randn(fused_latent.shape, device=device)
        noise_input = self.noise_scheduler.add_noise(fused_latent, noise, timesteps)

        # ---- Diffuse ----
        pred = self.diffuse(noise_input, timesteps, condition)

        return pred