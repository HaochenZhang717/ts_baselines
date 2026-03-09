import os
import torch
from tqdm.auto import tqdm


class Trainer:
    def __init__(self, config, args, model, dataloader):
        self.config = config
        self.args = args
        self.model = model
        self.dataloader = dataloader["dataloader"]

        self.device = next(self.model.parameters()).device

        self.base_lr = config["solver"]["base_lr"]
        self.max_epochs = config["solver"]["max_epochs"]
        self.save_cycle = config["solver"]["save_cycle"]

        self.results_folder = os.environ.get(
            "results_folder",
            config["solver"].get("results_folder", "./Checkpoints_default")
        )
        os.makedirs(self.results_folder, exist_ok=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_lr)

    def save(self, milestone):
        data = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "milestone": milestone,
        }
        torch.save(data, os.path.join(self.results_folder, f"model-{milestone}.pt"))

    def load(self, milestone):
        path = os.path.join(self.results_folder, f"model-{milestone}.pt")
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data["model"])
        if "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
        print(f"Loaded checkpoint from {path}")

    def train(self):
        self.model.train()

        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            total_num = 0

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{self.max_epochs}", leave=False)
            for batch in pbar:
                x = batch.to(self.device)  # train dataloader only returns x
                loss = self.model(x)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                bs = x.shape[0]
                total_loss += loss.item() * bs
                total_num += bs
                pbar.set_postfix(loss=f"{loss.item():.6f}")

            avg_loss = total_loss / max(total_num, 1)
            print(f"[Epoch {epoch}] train loss = {avg_loss:.6f}")

            if epoch % self.save_cycle == 0 or epoch == self.max_epochs:
                self.save(epoch)

    @torch.no_grad()
    def restore(self, test_dataloader, shape=None):
        """
        test_dataloader yields (x, mask)
        x: [B, L, D] in [-1,1]
        mask: boolean, True means observed, False means target to predict
        """
        self.model.eval()

        all_samples = []

        for batch in test_dataloader:
            x, mask = batch
            x = x.to(self.device)
            mask = mask.to(self.device).bool()

            sample = self.model.fast_sample_infill(
                shape=x.shape,
                target=x,
                partial_mask=mask
            )
            all_samples.append(sample.detach().cpu())

        all_samples = torch.cat(all_samples, dim=0).numpy()
        return all_samples, None