import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
import copy
from models.encoders.side_encoder import SideEncoder_Var
import os


class NormCausalAttention(nn.Module):
    """
    Attention module of LightningDiT.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.1,
            proj_drop: float = 0.1,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5


        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, is_causal: bool) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.to(v.dtype)
        k = k.to(v.dtype)  # rope may change the q,k's dtype
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
            is_causal=is_causal)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
    ):
        super().__init__()
        mlp_ratio = 4.0
        # Initialize normalization layers
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Initialize attention layer
        self.attn = NormCausalAttention(hidden_size, num_heads=num_heads)

        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim), hidden_size)

    def forward(self, x, is_causal):
        x = x + self.attn(self.norm1(x), is_causal)
        x = x + self.mlp(self.norm2(x))
        return x



def get_torch_trans(heads=8, layers=1, channels=64):
    # encoder_layer = nn.TransformerEncoderLayer(
    #     d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
    # )
    # encoder_layer = EncoderLayer(
    #     hidden_size=channels, num_heads=heads,
    # )
    return EncoderLayer(hidden_size=channels, num_heads=heads)


# def get_torch_trans(heads=8, layers=1, channels=64):
#     encoder_layer = nn.TransformerEncoderLayer(
#         d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
#     )
#     return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def get_torch_cross_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerDecoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
    )
    return nn.TransformerDecoder(encoder_layer, num_layers=layers)

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class TsPatchEmbedding(nn.Module):
    def __init__(self, L_patch_len, channels, d_model, dropout):
        super(TsPatchEmbedding, self).__init__()
        self.L_patch_len = L_patch_len
        self.padding_patch_layer = nn.ReplicationPad2d((0, L_patch_len, 0, 0))
        self.value_embedding = nn.Sequential(
            nn.Linear(L_patch_len*channels, d_model),
            nn.ReLU(),
        )

    def forward(self, x_in):
        # x_in (B, c, n_vars, t)
        if x_in.shape[-1] % self.L_patch_len:
            x = self.padding_patch_layer(x_in)
        else:
            x = x_in
        x = x.unfold(dimension=3, size=self.L_patch_len, step=self.L_patch_len) # (B, c, n_var, t/L_patch_len, L_patch_len)
        B, C, n_var, Nl, Pl = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().reshape(B, n_var, Nl, Pl*C)
        # above they did patchify
        x = self.value_embedding(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SidePatchEmbedding(nn.Module):
    def __init__(self, L_patch_len, channels, d_model, dropout):
        super(SidePatchEmbedding, self).__init__()
        self.L_patch_len = L_patch_len
        self.padding_patch_layer = nn.ReplicationPad2d((0, L_patch_len, 0, 0))
        self.value_embedding = nn.Sequential(
            nn.Linear(L_patch_len*channels, d_model),
        )

    def forward(self, x_in):
        if x_in.shape[-1] % self.L_patch_len:
            x = self.padding_patch_layer(x_in)
        else:
            x = x_in
        x = x.unfold(dimension=3, size=self.L_patch_len, step=self.L_patch_len)
        B, C, n_var, Nl, Pl = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().reshape(B, n_var, Nl, Pl*C)
        x = self.value_embedding(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchDecoder(nn.Module):
    def __init__(self, L_patch_len, d_model, channels):
        super().__init__()
        self.L_patch_len = L_patch_len
        self.channels = channels
        self.linear = nn.Linear(d_model, L_patch_len*channels)

    def forward(self, x):
        B, D, n_var, Nl = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.linear(x)
        x = x.reshape(B, n_var, Nl, self.L_patch_len, self.channels).permute(0, 4, 1, 2, 3).contiguous()
        x = x.reshape(B, self.channels, n_var, Nl*self.L_patch_len)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, condition_type="add"):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.side_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        if condition_type == "add":
            pass
        elif condition_type == "cross_attention":
            self.condition_cross_attention = get_torch_cross_trans(heads=nheads, layers=1, channels=channels)
        elif condition_type == "adaLN":
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 3 * channels, bias=True)
            )

    def forward_time(self, y, base_shape):
        # do attention in time direction
        B, channel, K, L = base_shape # torch.Size([512, 64, 1, 47])
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L) # aggregate all time_vars
        y = y.permute(0, 2, 1)
        y = self.time_layer(y, is_causal=os.environ["USE_CAUSAL"]=="true").permute(0, 2, 1)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(0, 2, 1), is_causal=False).permute(0, 2, 1)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y
    
    def foward_cross_attention(self, y, cond, attetion_mask=None):
        B, channel, K, L = y.shape
        y = y.reshape(B, channel, K, L).permute(0, 2, 3, 1).reshape(B * K, L, channel)
        cond = cond.reshape(B, channel, K, L).permute(0, 2, 3, 1).reshape(B * K, L, channel)
        y = self.condition_cross_attention(tgt=y, memory=cond, memory_mask=attetion_mask).permute(0, 2, 1)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3)
        return y

    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, x, side_emb, attr_emb, diffusion_emb, attention_mask=None, condition_type="add"):

        gama, beta, alpha = self.adaLN_modulation(attr_emb.permute(0,2,3,1)).chunk(3, dim=-1)
        gama, beta, alpha = gama.permute(0,3,1,2), beta.permute(0,3,1,2), alpha.permute(0,3,1,2)

        B, channel, K, L = x.shape
        base_shape = x.shape

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1).unsqueeze(-1)
        y = x + diffusion_emb
        if condition_type == "adaLN":
            y = self.modulate(y, gama, beta)

        # y.shape == torch.Size([512, 64, 1, 47])
        # base_shape == torch.Size([512, 64, 1, 47])
        y = self.forward_time(y, base_shape) # set to attention_mask==None for now.
        y = self.forward_feature(y, base_shape)

        if condition_type == "adaLN":
            y = y.reshape(B,channel,K,L)
            y = alpha * y
            y = y.reshape(B,channel,K*L)

        y = y.reshape(B,channel,K*L)
        y = self.mid_projection(y)

        _, side_dim, _, _ = side_emb.shape
        side_emb = side_emb.reshape(B, side_dim, K * L)
        side_emb = self.side_projection(side_emb)
        y = y + side_emb

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class VerbalTSQwen(nn.Module):
    def __init__(self, config, inputdim=1):
        super().__init__()
        self.config = config
        self.n_var = config["n_var"]
        self.var_dep_type = config["var_dep_type"]
        self.channels = config["channels"]
        self.multipatch_num = config["multipatch_num"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        config["side"]["device"] = config["device"]
        self.side_encoder = SideEncoder_Var(configs=config["side"])
        side_dim = self.side_encoder.total_emb_dim

        self.attention_mask_type = config["attention_mask_type"]
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.ts_downsample = nn.ModuleList([])
        self.side_downsample = nn.ModuleList([])
        self.patch_decoder = nn.ModuleList([])

        for i in range(self.multipatch_num):
            self.ts_downsample.append(TsPatchEmbedding(L_patch_len=config["base_patch"]*config["L_patch_len"]**i, channels=inputdim, d_model=self.channels, dropout=0))
            self.patch_decoder.append(PatchDecoder(L_patch_len=config["base_patch"]*config["L_patch_len"]**i, d_model=self.channels, channels=1))
            self.side_downsample.append(SidePatchEmbedding(L_patch_len=config["base_patch"]*config["L_patch_len"]**i, channels=side_dim, d_model=side_dim, dropout=0))

        self.attn_mask_initialized = False

        self.multipatch_mixer = nn.Linear(self.multipatch_num, 1)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=side_dim,
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    condition_type=config["condition_type"]
                )
                for _ in range(config["layers"])
            ]
        )
        
    def forward(self, x_raw, tp, attr_emb_raw, diffusion_step):
        B_raw, inputdim, n_var, L = x_raw.shape

        side_emb_raw = self.side_encoder(tp)
        diffusion_emb = self.diffusion_embedding(diffusion_step)


        x_list = []
        side_list = []
        scale_length = []
        # attr_emb_list = []
        for i in reversed(range(self.multipatch_num)):
            x = self.ts_downsample[i](x_raw)
            side_emb = self.side_downsample[i](side_emb_raw)
            x_list.append(x)
            side_list.append(side_emb)
            scale_length.append(x.shape[-1])
        x_in = torch.cat(x_list, dim=-1)
        side_in = torch.cat(side_list, dim=-1)


        if attr_emb_raw is not None:
            attr_emb_raw = attr_emb_raw.permute(0, 3, 2, 1)
            attr_emb_list = []
            for i in reversed(range(self.multipatch_num)):
                patch_length = self.config["base_patch"] * self.config["L_patch_len"] ** i
                attr_emb_list.append(attr_emb_raw.repeat_interleave(32 // patch_length, dim=-1))
            attr_emb = torch.cat(attr_emb_list, dim=-1)
        else:
            attr_emb = torch.zeros_like(x_in)


        B, _, Nk, Nl = x_in.shape

        _x_in = x_in
        skip = []
        for layer in self.residual_layers:
            x_in, skip_connection = layer(x_in+_x_in, side_in, attr_emb, diffusion_emb, condition_type=self.config["condition_type"])
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))

        x = x.reshape(B, self.channels, Nk * Nl)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, Nk, Nl)



        start_id = 0
        all_out = []
        for i in range(len(x_list)):
            x_out = x[:,:,:,start_id:start_id+x_list[i].shape[-1]]
            # print(f"x_out.shape = {x_out.shape}")
            x_out = self.patch_decoder[len(x_list)-1-i](x_out)
            x_out = x_out[:, :, :, :L]
            all_out.append(x_out)
            start_id += x_list[i].shape[-1]

        all_out = torch.cat(all_out, dim=1)
        all_out = all_out.permute(0, 2, 3, 1).contiguous()
        # print(f"all_out.shape = {all_out.shape}")
        all_out = self.multipatch_mixer(all_out)
        # print(f"all_out.shape = {all_out.shape}")
        # breakpoint()
        all_out = all_out.reshape((B_raw, n_var, L))
        return all_out, {}


