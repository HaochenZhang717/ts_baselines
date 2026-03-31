import torch
import torch.nn as nn

from models.diffusion.verbalts_qwen import VerbalTSQwen
from models.diffusion.verbalts import VerbalTS
from samplers import DDPMSampler, DDIMSampler
import numpy as np
import time
import random

class UnConditionalGeneratorQwenV5(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.device = configs["device"]
        self.configs = configs
        self._init_diff(configs["diffusion"])

    def _init_diff(self, configs):
        configs["device"] = self.device
        self.diff_model = VerbalTSQwen(configs, inputdim=1).to(self.device)
        self.num_steps = configs["num_steps"]

    def _xpred_vloss(self, x, tp, text_embed, t_float, t_long):
        # ====== step 2: sample noise ======
        noise = torch.randn_like(x)
        # ====== step 3: construct z_t (JiT style) ======
        z_t = t_float[:, None, None] * x + (1 - t_float[:, None, None]) * noise
        # ====== step 4: predict x ======
        x_pred, loss_dict = self.predict_x(z_t, tp, text_embed, t_long)
        # ====== step 5: compute v ======
        eps = 1e-5
        denom = (1 - t_float[:, None, None]).clamp(min=eps)
        v_pred = (x_pred - z_t) / denom
        v_target = (x - z_t) / denom
        loss_dict["v_loss"] = ((v_pred - v_target) ** 2).mean()
        # ====== aggregate ======
        all_loss = torch.zeros_like(loss_dict["v_loss"])
        for k in loss_dict.keys():
            all_loss += loss_dict[k]
        loss_dict["all"] = all_loss

        return loss_dict

    """
    Pretrain.
    """
    def forward(self, batch, is_train):
        pass

    def _unpack_data_uncond_gen(self, batch):
        ts = batch["ts"].to(self.device).float()
        tp = batch["tp"].to(self.device).float()
        ts = ts.permute(0, 2, 1)
        return ts, tp

    """
    Generation.
    """
    @torch.no_grad()
    def generate(self, batch, n_samples):
        pass

    def predict_x(self, xt, tp, text_embed, t):
        noisy_x = torch.unsqueeze(xt, 1)
        pred_x, loss_dict = self.diff_model(noisy_x, tp, text_embed, t)
        return pred_x, loss_dict