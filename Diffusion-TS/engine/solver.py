import os
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info
import copy


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


class Trainer(object):
    def __init__(self, config, args, model, dataloader):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_num_epochs = config['solver']['max_epochs']
        self.train_dataloader = dataloader['train_dataloader']
        self.valid_dataloader = dataloader['valid_dataloader']
        self.step = 0
        self.milestone = 0
        self.args, self.config = args, config

        self.results_folder = Path(config['solver']['results_folder'])
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr)

        self.ema_decay = 0.999
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.ema_model.eval()

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def update_ema(self):
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema_model.load_state_dict(data['ema'])
        self.milestone = milestone

    def train(self):
        step = 0
        best_val_loss = float('inf')
        for epoch in range(step, self.train_num_epochs):
            train_loss_avg = 0.0
            tic = time.time()
            self.model.train()
            for batch in tqdm(self.train_dataloader):
                batch = batch.to(self.device)
                loss = self.model(batch, target=batch)
                train_loss_avg += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.update_ema()
            train_loss_avg = train_loss_avg / len(self.train_dataloader)

            if epoch % 100 == 0:
                val_loss_avg = 0.0
                self.ema_model.eval()
                for batch in self.valid_dataloader:
                    batch = batch.to(self.device)
                    with torch.no_grad():
                        loss = self.ema_model(batch, target=batch)
                    val_loss_avg += loss.item()
                val_loss_avg = val_loss_avg / len(self.valid_dataloader)
                toc = time.time()
                print(f"Epoch {epoch}: Train Loss: {train_loss_avg:.6f}, Val Loss: {val_loss_avg:.6f}, Time: {toc - tic:.2f}s")
                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    self.save("best")

    def sample(self, num, size_every, shape=None, model_kwargs=None, cond_fn=None):
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1
        tic = time.time()
        for _ in range(num_cycle):
            sample = self.ema_model.generate_mts(batch_size=size_every, model_kwargs=model_kwargs, cond_fn=cond_fn)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()
        print('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples

    # def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        # tic = time.time()
        # print('Begin to restore...')
        # model_kwargs = {}
        # model_kwargs['coef'] = coef
        # model_kwargs['learning_rate'] = stepsize
        # samples = np.empty([0, shape[0], shape[1]])
        # reals = np.empty([0, shape[0], shape[1]])
        # masks = np.empty([0, shape[0], shape[1]])
        #
        # for idx, (x, t_m) in enumerate(raw_dataloader):
        #     x, t_m = x.to(self.device), t_m.to(self.device)
        #     if sampling_steps == self.model.num_timesteps:
        #         sample = self.ema_model.sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m,
        #                                                   model_kwargs=model_kwargs)
        #     else:
        #         sample = self.ema_model.fast_sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m, model_kwargs=model_kwargs,
        #                                                        sampling_timesteps=sampling_steps)
        #
        #     samples = np.row_stack([samples, sample.detach().cpu().numpy()])
        #     reals = np.row_stack([reals, x.detach().cpu().numpy()])
        #     masks = np.row_stack([masks, t_m.detach().cpu().numpy()])

        # if self.logger is not None:
        #     self.logger.log_info('Imputation done, time: {:.2f}'.format(time.time() - tic))
        # return samples, reals, masks
        # return samples

    # def forward_sample(self, x_start):
    #    b, c, h = x_start.shape
    #    noise = torch.randn_like(x_start, device=self.device)
    #    t = torch.randint(0, self.model.num_timesteps, (b,), device=self.device).long()
    #    x_t = self.model.q_sample(x_start=x_start, t=t, noise=noise).detach()
    #    return x_t, t