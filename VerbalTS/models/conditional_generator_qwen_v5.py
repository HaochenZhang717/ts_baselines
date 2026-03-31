import torch
import torch.nn as nn

# from models.encoders.attr_encoder import AttributeEncoder
# from models.encoders.text_encoder import TextEncoder, CLIPTextEncoder
from models.encoders.cond_projector import TextProjectorMVarMScaleMStep, AttrProjectorAvg
from models.unconditional_generator_qwen_v5 import UnConditionalGeneratorQwenV5
from models.cttp.cttp_model import CTTP
import time
import random
import yaml
import os


class ConditionalGeneratorQwenV5(nn.Module):
    def __init__(self, diff_configs, cond_configs):
        super().__init__()
        self.device = diff_configs["device"] if torch.cuda.is_available() else "cpu"
        self.diff_configs = diff_configs
        self.cond_configs = cond_configs
        self._init_condition_encoders(diff_configs, cond_configs)
        self._init_diff(diff_configs)
        self.mu = -0.8
        self.sigma = 0.8

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
            self.cond_projector = self.cond_projector.to(self.device)
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
        if "vae_embed" in self.cond_configs["cond_modal"]:
            configs["diffusion"]["text_projector"] = self.cond_configs["vae_embed"]["text_projector"]

        self.generator = UnConditionalGeneratorQwenV5(configs=configs)
        if configs["generator_pretrain_path"] != "":
            self.generator.load_state_dict(torch.load(configs["generator_pretrain_path"]))
            print("Load the pretrain unconditional generator")
        else:
            print("Learn from scratch")

    def build_t_schedule(self, num_steps):
        # 在 logit space 均匀采样
        s = torch.linspace(-3, 3, num_steps, device=self.device)  # 覆盖大部分概率质量
        t = torch.sigmoid(self.mu + self.sigma * s)
        return t

    def forward(self, batch, is_train):
        x, tp, text_embedding_all_segments, vae_embeds, moment_embeds  = self._unpack_data_cond_gen(batch)
        B, C, T = x.shape
        attr_embed_raw = text_embedding_all_segments
        t_float = torch.sigmoid(self.mu + self.sigma * torch.randn(B).to(self.device)) # lognormal distribution
        t_long = (t_float * (self.generator.num_steps - 1)).long()
        attr_embed = self.cond_projector(attr_embed_raw)
        loss = self.generator._xpred_vloss(x, tp, attr_embed, t_float, t_long)
        return loss

    def _unpack_data_cond_gen(self, batch):
        ts = batch["ts"].to(self.device).float()  # batch_size, num_channels, seq_len
        B, C, T = ts.shape
        # print("ts.shape in _unpack_data_cond_gen", ts.shape)
        tp = torch.arange(T).repeat(B, 1).to(self.device).float()
        text_embedding_all_segments = batch["text_embedding_all_segments"].to(self.device).float()
        vae_embeds = batch["vae_embeds"].to(self.device).float() if batch["vae_embeds"] is not None else None
        moment_embeds = batch["moment_embed"].to(self.device).float() if batch["moment_embed"] is not None else None
        return ts, tp, text_embedding_all_segments, vae_embeds, moment_embeds

    def generate(self, batch, n_samples, sampler="ddim"):
        if self.cond_configs["cond_modal"] == "constraint":
            raise ValueError("Not Changed for precomputed attr_embed yet")
            # return self.generate_constraint(batch, n_samples, sampler)
        else:
            return self.generate_text(batch, n_samples, sampler)

    @torch.no_grad()
    def generate_text(self, batch, n_samples, sampler="ddim"):

        ts, tp, text_embedding_all_segments, vae_embeds, moment_embeds = self._unpack_data_cond_gen(batch)
        B, C, T = ts.shape
        attr_embed_raw = text_embedding_all_segments
        t_schedule = self.build_t_schedule(self.generator.num_steps)
        samples = []
        for i in range(n_samples):
            x = torch.randn_like(ts)
            for i_t in range(self.generator.num_steps-1):
                t_float = t_schedule[i_t]
                t_float_next = t_schedule[i_t+1]
                dt = t_float_next - t_float
                t_long = (torch.ones(B, device=self.device) * t_float * (self.generator.num_steps - 1)).long()
                t_long_next = (torch.ones(B, device=self.device) * t_float_next * (self.generator.num_steps - 1)).long()

                attr_embed = self.cond_projector(attr_embed_raw)  # for now we are not using projector.
                pred_x, _ = self.generator.predict_x(x, tp, attr_embed, t_long)
                eps = 1e-5
                denom = (1 - t_float).clamp(min=eps)
                pred_v = (pred_x - x) / denom
                x_updated = x + dt * pred_v

                pred_x_next, _ = self.generator.predict_x(x_updated, tp, attr_embed, t_long_next)
                denom_next = (1 - t_float_next).clamp(min=1e-5)
                pred_v_next = (pred_x_next - x_updated) / denom_next

                x = x + 0.5 * dt * (pred_v + pred_v_next)

            samples.append(x)
        return torch.stack(samples)