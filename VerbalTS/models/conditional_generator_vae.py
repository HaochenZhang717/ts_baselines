import torch
import torch.nn as nn

from models.encoders.cond_projector import TextProjectorMVarMScaleMStep
from models.unconditional_generator_vae import UnConditionalGeneratorVAE
from models.cttp.cttp_model import CTTP
import time
import random
import yaml
import os


class ConditionalGeneratorVAE(nn.Module):
    def __init__(self, diff_configs, cond_configs):
        super().__init__()
        self.device = diff_configs["device"] if torch.cuda.is_available() else "cpu"
        self.diff_configs = diff_configs
        self.cond_configs = cond_configs
        self._init_condition_encoders(diff_configs, cond_configs)
        self._init_diff(diff_configs)

    def _init_condition_encoders(self, diff_configs, cond_configs):
        if cond_configs["cond_modal"] == "vae_embed":
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

        self.generator = UnConditionalGeneratorVAE(configs=configs)
        if configs["generator_pretrain_path"] != "":
            self.generator.load_state_dict(torch.load(configs["generator_pretrain_path"]))
            print("Load the pretrain unconditional generator")
        else:
            print("Learn from scratch")

    def forward(self, batch, is_train):
        x, tp, vae_embeds  = self._unpack_data_cond_gen(batch)
        B, C, T = x.shape

        attr_embed_raw = vae_embeds

        B = x.shape[0]
        if is_train:
            t = torch.randint(0, self.generator.num_steps, [B], device=self.device)
            if self.cond_configs["cond_modal"] == "vae_embed":
                attr_embed = self.cond_projector(attr_embed_raw, t)
            else:
                raise NotImplementedError
            loss = self.generator._noise_estimation_loss(x, tp, attr_embed, t)
            return loss

        loss_dict = {}
        for t in range(self.generator.num_steps):
            t = (torch.ones(B, device=self.device) * t).long()
            if self.cond_configs["cond_modal"] == "vae_embed":
                attr_embed = self.cond_projector(attr_embed_raw, t)
            else:
                raise NotImplementedError
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
        tp = torch.arange(T).repeat(B, 1).to(self.device).float()
        vae_embeds = batch["vae_embeds"].to(self.device).float() if batch["vae_embeds"] is not None else None
        return ts, tp, vae_embeds

    def generate(self, batch, n_samples, sampler="ddim"):
        if self.cond_configs["cond_modal"] == "constraint":
            raise ValueError("Not Changed for precomputed attr_embed yet")
            # return self.generate_constraint(batch, n_samples, sampler)
        else:
            return self.generate_text(batch, n_samples, sampler)

    @torch.no_grad()
    def generate_text(self, batch, n_samples, sampler="ddim"):
        ts, tp, vae_embeds = self._unpack_data_cond_gen(batch)
        B, C, T = ts.shape

        if self.cond_configs["cond_modal"] == "vae_embed":
            attr_embed_raw = vae_embeds
        else:
            raise NotImplementedError

        samples = []
        for i in range(n_samples):
            x = torch.randn_like(ts)

            for t in range(self.generator.num_steps - 1, -1, -1):
                noise = torch.randn_like(x)
                t = (torch.ones(B, device=self.device) * t).long()
                if self.cond_configs["cond_modal"] == "vae_embed":
                    attr_embed = self.cond_projector(attr_embed_raw, t)
                else:
                    raise NotImplementedError

                pred_noise, _ = self.generator.predict_noise(x, tp, attr_embed, t)
                if sampler == "ddpm":
                    x = self.generator.ddpm.reverse(x, pred_noise, t, noise)
                else:
                    x = self.generator.ddim.reverse(x, pred_noise, t, noise, is_determin=True)

            samples.append(x)
        return torch.stack(samples)