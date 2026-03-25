import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
import copy
from models.model_1d.side_encoder import SideEncoder_Var
import yaml
import numpy as np
import torch
from utils import persistence
from torch.nn.functional import silu
from models.ema import LitEma
from contextlib import contextmanager




@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x




def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

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

    def forward_time(self, y, base_shape, attention_mask=None):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(0, 2, 1), mask=attention_mask).permute(0, 2, 1)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape, attention_mask=None):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(0, 2, 1), mask=attention_mask).permute(0, 2, 1)
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
    
        if condition_type == "add":
            x = x + attr_emb
        elif condition_type == "cross_attention":
            x = self.foward_cross_attention(x, attr_emb, attetion_mask=attention_mask)
        elif condition_type == "adaLN":
            gama, beta, alpha = self.adaLN_modulation(attr_emb.permute(0,2,3,1)).chunk(3, dim=-1)
            gama, beta, alpha = gama.permute(0,3,1,2), beta.permute(0,3,1,2), alpha.permute(0,3,1,2)

        B, channel, K, L = x.shape
        base_shape = x.shape

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1).unsqueeze(-1)
        y = x + diffusion_emb
        if condition_type == "adaLN":
            y = self.modulate(y, gama, beta)
        
        y = self.forward_time(y, base_shape, attention_mask)
        y = self.forward_feature(y, base_shape, None)

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

class VerbalTS(nn.Module):
    def __init__(self, inputdim=1):
        super().__init__()
        self.config = yaml.safe_load(open("models/model_1d/model.yaml"))["diffusion"]

        config = self.config
        self.n_var = config["n_var"]
        self.var_dep_type = config["var_dep_type"]
        self.channels = config["channels"]
        self.multipatch_num = config["multipatch_num"]
        # self.diffusion_embedding = DiffusionEmbedding(
        #     num_steps=config["num_steps"],
        #     embedding_dim=config["diffusion_embedding_dim"],
        # )
        self.map_noise = PositionalEmbedding(num_channels=config["diffusion_embedding_dim"])

        config["side"]["device"] = config["device"] if torch.cuda.is_available() else 'cpu'
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
        
    def forward(self, x_raw, noise_label, attr_emb_raw=None):
        x_raw = x_raw.unsqueeze(1)

        B_raw, inputdim, n_var, L = x_raw.shape
        tp = torch.arange(L).repeat(B_raw, 1).to(x_raw.device).float()
        side_emb_raw = self.side_encoder(tp)
        diffusion_emb = self.map_noise(noise_label)
        
        x_list = []
        side_list = []
        scale_length = []
        for i in range(self.multipatch_num):
            x = self.ts_downsample[i](x_raw)
            side_emb = self.side_downsample[i](side_emb_raw)
            x_list.append(x)
            side_list.append(side_emb)
            scale_length.append(x.shape[-1])

        if self.attention_mask_type == "full" or attr_emb_raw is None:
            attention_mask = None
        elif self.attention_mask_type == "parallel":
            attention_mask = self.get_mask(0, [x_list[i].shape[-1] for i in range(len(x_list))], device=x_raw.device)
        
        x_in = torch.cat(x_list, dim=-1)
        side_in = torch.cat(side_list, dim=-1)

        if attr_emb_raw is None:
            attr_emb = torch.zeros_like(x_in)
        else:
            if "scale" in self.config["text_projector"]:
                assert len(scale_length) == attr_emb_raw.shape[2]
                mscale_attr_list = []
                for i in range(len(scale_length)):
                    tmp_scale_attr = attr_emb_raw[:,:,i:i+1,:].expand([-1, -1, scale_length[i], -1])
                    mscale_attr_list.append(tmp_scale_attr)
                attr_emb = torch.cat(mscale_attr_list, dim=2)
            else:
                attr_emb = attr_emb_raw.expand([-1, -1, x_in.shape[-1], -1])
            attr_emb = attr_emb.permute(0,3,1,2)

        B, _, Nk, Nl = x_in.shape
        _x_in = x_in
        skip = []
        for layer in self.residual_layers:
            x_in, skip_connection = layer(x_in+_x_in, side_in, attr_emb, diffusion_emb, attention_mask=attention_mask, condition_type=self.config["condition_type"])
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
            x_out = self.patch_decoder[i](x_out)
            x_out = x_out[:, :, :, :L]
            all_out.append(x_out)
            start_id += x_list[i].shape[-1]

        all_out = torch.cat(all_out, dim=1)
        all_out = self.multipatch_mixer(all_out.permute(0, 2, 3, 1).contiguous())
        all_out = all_out.reshape((B_raw, n_var, L))
        return all_out
    
    def get_mask(self, attr_len, len_list, device="cuda:0"):
        total_len = sum(len_list) + attr_len
        mask = torch.zeros(total_len, total_len, device=device) - float("inf")
        mask[:attr_len, :] = 0
        mask[:, :attr_len] = 0
        start_id = attr_len
        for i in range(len(len_list)):
            mask[start_id:start_id+len_list[i], start_id:start_id+len_list[i]] = 0
            start_id += len_list[i]
        return mask



@persistence.persistent_class
class EDMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = VerbalTS(inputdim=img_channels)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten())
        # print(F_x.shape)
        # breakpoint()
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class Model1D(nn.Module):
    def __init__(self, args, device):
        '''
        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        '''

        super().__init__()
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.T = args.diffusion_steps

        self.device = device
        self.net = EDMPrecond(args.img_resolution, args.input_channels, channel_mult=args.ch_mult,
                              model_channels=args.unet_channels, attn_resolutions=args.attn_resolution)

        # delay embedding is used
        if not args.use_stft:
            self.delay = args.delay
            self.embedding = args.embedding
            self.seq_len = args.seq_len

        if args.ema:
            self.use_ema = True
            self.model_ema = LitEma(self.net, decay=0.9999, use_num_upates=True, warmup=args.ema_warmup)
        else:
            self.use_ema = False


    def loss_fn(self, x):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        '''

        to_log = {}

        output, weight = self.forward(x)

        # denoising matching term
        # loss = weight * ((output - x) ** 2)

        loss = (weight * (output - x).square()).mean()
        to_log['karras loss'] = loss.detach().item()

        return loss, to_log

    def loss_fn_impute(self, x, mask):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        '''

        to_log = {}
        output, weight = self.forward_impute(x, mask)
        x = self.unpad(x * (1 - mask), x.shape)
        output = self.unpad(output * (1 - mask), x.shape)
        loss = (weight * (output - x).square()).mean()
        to_log['karras loss'] = loss.detach().item()

        return loss, to_log

    def forward(self, x, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([x.shape[0], 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(x) if augment_pipe is not None else (x, None)
        n = torch.randn_like(y) * sigma


        D_yn = self.net(y + n, sigma)
        return D_yn, weight

    def forward_impute(self, x, mask, labels=None, augment_pipe=None):

        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # noisy impute part
        n = torch.randn_like(x) * sigma
        noise_impute = n * (1 - mask)
        x_to_impute = x * (1 - mask) + noise_impute

        # clear image
        x = x * mask
        y, augment_labels = augment_pipe(x) if augment_pipe is not None else (x, None)

        D_yn = self.net(y + x_to_impute, sigma, labels, augment_labels=augment_labels)
        return D_yn, weight

    def forward_forecast(self, past, future, labels=None, augment_pipe=None):
        s, e = past.shape[-1], future.shape[-1]
        rnd_normal = torch.randn([past.shape[0], 1, 1, 1], device=past.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(past) if augment_pipe is not None else (past, None)
        n = torch.randn_like(future) * sigma
        full_seq = self.pad_f(torch.cat([past, future + n], dim=-1))
        D_yn = self.net(full_seq, sigma, labels, augment_labels=augment_labels)[..., s:(s + e)]
        return D_yn, weight

    def pad_f(self, x):
        """
        Pads the input tensor x to make it square along the last two dimensions.
        """
        _, _, cols, rows = x.shape
        max_side = max(32, rows)
        padding = (
            0, max_side - rows, 0, 0)  # Padding format: (pad_left, pad_right, pad_top, pad_bottom)

        # Padding the last two dimensions to make them square
        x_padded = torch.nn.functional.pad(x, padding, mode='constant', value=0)
        return x_padded

    def unpad(self, x, original_shape):
        """
        Removes the padding from the tensor x to get back to its original shape.
        """
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    @contextmanager
    def ema_scope(self, context=None):
        """
        Context manager to temporarily switch to EMA weights during inference.
        Args:
            context: some string to print when switching to EMA weights

        Returns:

        """
        if self.use_ema:
            self.model_ema.store(self.net.parameters())
            self.model_ema.copy_to(self.net)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.net.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args):
        """
        this function updates the EMA model, if it is used
        Args:
            *args:

        Returns:

        """
        if self.use_ema:
            self.model_ema(self.net)

