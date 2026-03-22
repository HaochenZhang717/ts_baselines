import os
import torch
from tqdm.auto import tqdm
import copy
import time
import numpy as np
import wandb


class Trainer:
    def __init__(self, config, args, model, dataloader):
        self.config = config
        self.args = args
        self.model = model
        # self.dataloader = dataloader["dataloader"]
        self.train_dataloader = dataloader['train_dataloader']
        self.valid_dataloader = dataloader['valid_dataloader']
        self.step = 0
        self.milestone = 0
        self.train_num_epochs = config["solver"]["max_epochs"]

        self.device = next(self.model.parameters()).device

        self.base_lr = config["solver"]["base_lr"]
        self.max_epochs = config["solver"]["max_epochs"]
        # self.save_cycle = config["solver"]["save_cycle"]

        self.results_folder = os.environ.get(
            "results_folder",
            config["solver"].get("results_folder", "./Checkpoints_default")
        )
        os.makedirs(self.results_folder, exist_ok=True)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.base_lr)

        self.ema_decay = 0.999
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.ema_model.eval()

        wandb.init(
            project="FlowTS-Neurips-Baseline",  # 你可以改名字
            name=os.getenv("WANDB_NAME", "no_name"),  # 自动用实验名
            config=self.config
        )


    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, f'{self.results_folder}/checkpoint-{milestone}.pt')

    def update_ema(self):
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def load(self, milestone):
        device = self.device
        data = torch.load(f'{self.results_folder}/checkpoint-{milestone}.pt', map_location=device)
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
                loss = self.model(batch)
                train_loss_avg += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.update_ema()
            train_loss_avg = train_loss_avg / len(self.train_dataloader)
            toc = time.time()
            print(f"Epoch {epoch}: Train Loss: {train_loss_avg:.6f}, Time: {toc - tic:.2f}s")
            wandb.log({
                "train/epoch_loss": train_loss_avg,
                "train/learning_rate": self.opt.param_groups[0]['lr'],
                "epoch": epoch
            })

            if epoch % 100 == 0:
                val_loss_avg = 0.0
                self.ema_model.eval()
                for batch in self.valid_dataloader:
                    batch = batch.to(self.device)
                    with torch.no_grad():
                        loss = self.ema_model(batch)
                    val_loss_avg += loss.item()
                val_loss_avg = val_loss_avg / len(self.valid_dataloader)

                print(
                    f"Epoch {epoch}: Train Loss: {train_loss_avg:.6f}, Val Loss: {val_loss_avg:.6f}, Time: {toc - tic:.2f}s")

                wandb.log({
                    "valid/loss": val_loss_avg,
                    "epoch": epoch
                })

                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    self.save("best")

    def sample(self, num, size_every, shape=None):
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1
        tic = time.time()
        for _ in range(num_cycle):
            sample = self.ema_model.generate_mts(batch_size=size_every)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()
        print('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples