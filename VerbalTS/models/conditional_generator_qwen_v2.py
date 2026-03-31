import torch
import torch.nn as nn

# from models.encoders.attr_encoder import AttributeEncoder
from models.encoders.text_encoder import QwenTextEncoder
from models.encoders.cond_projector import TextProjectorMVarMScaleMStep, AttrProjectorAvg,QwenProjector
from models.unconditional_generator_qwen_v2 import UnConditionalGeneratorQwenV2
from models.cttp.cttp_model import CTTP
import time
import random
import yaml
import os
import re


def organize_caps(batch_caps, n_channels, n_segments):
    """
    Args:
        batch_caps: list of size B
            each element: list of dicts [{key: caption}, ...]
        n_channels: int
        n_segments: int

    Returns:
        caps_out: list of size (B, C, S)
            caps_out[b][c][s] = string
    """

    B = len(batch_caps)

    # 初始化
    caps_out = [
        [
            ["" for _ in range(n_segments)]  # segments
            for _ in range(n_channels)  # channels
        ]
        for _ in range(B)
    ]

    for b in range(B):
        caps_list = batch_caps[b]

        for d in caps_list:
            for k, v in d.items():
                # 解析 key: segX_channelY
                # e.g. seg1_channel0
                seg_match = re.search(r"seg(\d+)", k)
                ch_match = re.search(r"channel(\d+)", k)

                if seg_match is None or ch_match is None:
                    raise ValueError(f"Bad key: {k}")

                s = int(seg_match.group(1)) - 1  # 0-based
                c = int(ch_match.group(1))

                caps_out[b][c][s] = v

    return caps_out


def flatten_caps(caps):
    """
    caps: (B, C, S)
    return:
        flat_text: List[str] of size B*C*S
        shape info
    """
    B = len(caps)
    C = len(caps[0])
    S = len(caps[0][0])

    flat_text = [
        caps[b][c][s]
        for b in range(B)
        for c in range(C)
        for s in range(S)
    ]

    return flat_text, (B, C, S)


class ConditionalGeneratorQwenV2(nn.Module):
    def __init__(self, diff_configs, cond_configs):
        super().__init__()
        self.device = diff_configs["device"] if torch.cuda.is_available() else "cpu"
        self.diff_configs = diff_configs
        self.cond_configs = cond_configs
        self._init_condition_encoders(diff_configs, cond_configs)
        self._init_diff(diff_configs)

    def _init_condition_encoders(self, diff_configs, cond_configs):
        if cond_configs["cond_modal"] == "text":
            cond_configs["text"]["device"] = self.device
            self.cond_projector = nn.Sequential(
                nn.Linear(cond_configs["text"]["pretrain_model_dim"],
                          cond_configs["text"]["vl_emb_hidden_dim"]),
                nn.LayerNorm(cond_configs["text"]["vl_emb_hidden_dim"]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(cond_configs["text"]["vl_emb_hidden_dim"], cond_configs["text"]["vl_emb"])
            )
            self.cond_projector = QwenProjector(
                n_var=diff_configs["diffusion"]["n_var"],
                n_scale=diff_configs["diffusion"]["multipatch_num"],
                n_steps=diff_configs["diffusion"]["num_steps"],
                n_stages=cond_configs["text"]["num_stages"],
                dim_in=cond_configs["text"]["vl_emb"],
                dim_out=diff_configs["diffusion"]["channels"]
            )
            self.cond_projector = self.cond_projector.to(self.device)
            self.attr_en = QwenTextEncoder(cond_configs["text"]).to(self.device)


        elif cond_configs["cond_modal"] == "vae_embed":
            self.cond_projector = TextProjectorMVarMScaleMStep(n_var=diff_configs["diffusion"]["n_var"],
                                                               n_scale=diff_configs["diffusion"]["multipatch_num"],
                                                               n_steps=diff_configs["diffusion"]["num_steps"],
                                                               n_stages=cond_configs["vae_embed"]["num_stages"],
                                                               dim_in=cond_configs["vae_embed"]["embed_dim"],
                                                               dim_out=diff_configs["diffusion"]["channels"])
            self.cond_projector = self.cond_projector.to(self.device)

        else:
            raise NotImplementedError

    def _init_diff(self, configs):
        configs["device"] = self.device

        configs["diffusion"]["text_projector"] = self.cond_configs["text"]["text_projector"]

        self.generator = UnConditionalGeneratorQwenV2(configs=configs)
        if configs["generator_pretrain_path"] != "":
            self.generator.load_state_dict(torch.load(configs["generator_pretrain_path"]))
            print("Load the pretrain unconditional generator")
        else:
            print("Learn from scratch")

    def forward(self, batch, is_train):
        # x, tp, text_embedding_all_segments, moment_embeds = self._unpack_data_cond_gen(batch)
        x, tp, attr_embed_raw = self._unpack_data_cond_gen(batch)
        # print(attr_embed_raw.shape)
        attr_embed_raw = self.attr_en(attr_embed_raw)
        # attr_len, attr_dim = attr_embed_raw.shape[-2:]
        # B, C, S = text_caps_base_shape
        # attr_embed_raw = attr_embed_raw.view(B, C, S, attr_len, attr_dim)

        B = x.shape[0]
        if is_train:
            t = torch.randint(0, self.generator.num_steps, [B], device=self.device)
            attr_embed = self.cond_projector(attr_embed_raw, t)  # for now we are not using projector.
            loss = self.generator._noise_estimation_loss(x, tp, attr_embed, t)
            return loss

        loss_dict = {}
        for t in range(self.generator.num_steps):
            t = (torch.ones(B, device=self.device) * t).long()
            attr_embed = self.cond_projector(attr_embed_raw, t)  # for now we are not using projector.
            tmp_loss_dict = self.generator._noise_estimation_loss(x, tp, attr_embed, t)
            for k in tmp_loss_dict:
                if k in loss_dict.keys():
                    loss_dict[k] += tmp_loss_dict[k]
                else:
                    loss_dict[k] = tmp_loss_dict[k]
        for k in loss_dict:
            loss_dict[k] = loss_dict[k] / self.generator.num_steps
        return loss_dict


    def _unpack_data_cond_gen(self, batch):
        ts = batch["ts"].to(self.device).float()  # batch_size, num_channels, seq_len
        B, C, T = ts.shape
        # print("ts.shape in _unpack_data_cond_gen", ts.shape)
        tp = torch.arange(T).repeat(B, 1).to(self.device).float()
        # text_embedding_all_segments = batch["text_embedding_all_segments"].to(self.device).float()
        # vae_embeds = batch["vae_embeds"].to(self.device).float() if batch["vae_embeds"] is not None else None
        # moment_embeds = batch["moment_embed"].to(self.device).float() if batch["moment_embed"] is not None else None

        attrs = organize_caps(
            batch['caps'],
            n_channels=C,  # 或者你的C
            n_segments=4
        )

        # attrs, attrs_base_shape = flatten_caps(attrs)
        attr_embed_raw = batch["my_caps_qwen_embeds"].to(self.device).float()
        return ts, tp, attr_embed_raw

    def generate(self, batch, n_samples, sampler="ddim"):
        if self.cond_configs["cond_modal"] == "constraint":
            raise ValueError("Not Changed for precomputed attr_embed yet")
            # return self.generate_constraint(batch, n_samples, sampler)
        else:
            return self.generate_text(batch, n_samples, sampler)

    @torch.no_grad()
    def generate_text(self, batch, n_samples, sampler="ddim"):
        ts, tp, attr_embed_raw = self._unpack_data_cond_gen(batch)
        B, C, T = ts.shape

        attr_embed_raw = self.attr_en(attr_embed_raw)
        # attr_len, attr_dim = attr_embed_raw.shape[-2:]
        # B, C, S = text_caps_base_shape
        # attr_embed_raw = attr_embed_raw.view(B, C, S, attr_len, attr_dim)

        samples = []
        for i in range(n_samples):
            x = torch.randn_like(ts)

            for t in range(self.generator.num_steps - 1, -1, -1):
                noise = torch.randn_like(x)
                t = (torch.ones(B, device=self.device) * t).long()
                attr_embed = self.cond_projector(attr_embed_raw, t)

                pred_noise, _ = self.generator.predict_noise(x, tp, attr_embed, t)
                if sampler == "ddpm":
                    x = self.generator.ddpm.reverse(x, pred_noise, t, noise)
                else:
                    x = self.generator.ddim.reverse(x, pred_noise, t, noise, is_determin=True)

            samples.append(x)
        return torch.stack(samples)


