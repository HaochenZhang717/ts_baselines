from torch import nn
import torch
import torch.optim as optim

import numpy as np
from tqdm.auto import tqdm
from models.testing_models.s4d import S4D, dropout_fn
from scipy.linalg import sqrtm

import torch
import torch.nn.functional as F


class NormCausalAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.
        )
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
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        mlp_ratio = 4.0

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = NormCausalAttention(hidden_size, num_heads=num_heads)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(
            hidden_size,
            int(2 / 3 * mlp_hidden_dim),
            hidden_size
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FIDEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        latent_dim=64,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList([
            EncoderLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)
        self.to_mu = nn.Linear(hidden_size, latent_dim)
        self.to_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        """
        x: (B, C, T)
        return:
            mu:     (B, latent_dim)
            logvar: (B, latent_dim)
        """
        x = self.conv(x)              # (B, hidden, T')
        x = x.permute(0, 2, 1)        # (B, T', hidden)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        # global pooling over time
        x = x.mean(dim=1)             # (B, hidden)

        mu = self.to_mu(x)            # (B, latent_dim)
        logvar = self.to_logvar(x)    # (B, latent_dim)

        # 可选：数值稳定
        logvar = torch.clamp(logvar, min=-6.0, max=6.0)

        return mu, logvar


class FIDDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        output_dim,
        hidden_size=128,
        seq_len=128,
    ):
        super().__init__()

        assert seq_len % 4 == 0, "seq_len must be divisible by 4"
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.base_len = seq_len // 4

        self.fc = nn.Linear(latent_dim, hidden_size * self.base_len)

        self.net = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.Conv1d(hidden_size // 2, output_dim, kernel_size=5, padding=2),
        )

    def forward(self, z):
        """
        z: (B, latent_dim)
        return: (B, C, T)
        """
        B = z.shape[0]

        x = self.fc(z)                                # (B, hidden * T/4)
        x = x.view(B, self.hidden_size, self.base_len)
        x = self.net(x)                               # (B, C, T)

        return x


class FIDVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        seq_len,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        latent_dim=64,
        beta=0.001,
    ):
        super().__init__()

        self.beta = beta

        self.encoder = FIDEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            latent_dim=latent_dim,
        )

        self.decoder = FIDDecoder(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
            seq_len=seq_len,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """
        x: (B, C, T)
        returns:
            mu, logvar, z with shape (B, latent_dim)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

    def decode(self, z):
        return self.decoder(z)

    def get_embedding(self, x, use_mu=True):
        """
        x: (B, C, T)
        return: (B, latent_dim)
        """
        mu, logvar = self.encoder(x)
        if use_mu:
            return mu
        return self.reparameterize(mu, logvar)

    def forward(self, x):
        """
        x: (B, C, T)
        """
        mu, logvar, z = self.encode(x)
        recon = self.decode(z)

        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def loss_function(self, x, recon, mu, logvar):
        recon_loss = F.mse_loss(recon, x)

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl.mean()

        loss = recon_loss + self.beta * kl_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }


class S4Model(nn.Module):

    def __init__(
            self,
            d_input,
            d_state,
            d_output=10,
            d_model=256,
            n_layers=4,
            dropout=0.2,
            prenorm=False,
            bidirectional=False,
            seq2seq=False,
            lr=0.001,
            activation=nn.Identity()
    ):
        super().__init__()

        self.prenorm = prenorm
        self.seq2seq = seq2seq

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(
                    d_model=d_model,
                    d_state=d_state,
                    bidirectional=bidirectional,
                    # postact='glu' if glu else None,
                    dropout=dropout,
                    transposed=True,
                    lr=min(0.001, lr)
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout) if dropout > 0 else nn.Identity())

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)
        self.act = activation

    def forward(self, x, aux=None, t=None, **kwargs):
        """
        Input x is shape (B, L, d_input)
        """

        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        if not self.seq2seq:
            x = x.mean(1)
        # Decode the outputs
        x = self.decoder(x)  # (B, L, d_model) -> (B, L, d_output)
        x = self.act(x)
        return x, None

    def default_state(self, *args, **kwargs):
        return [layer.default_state(*args, **kwargs) for layer in self.s4_layers]


class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)


def histogram_torch(x, n_bins, density=True):
    a, b = x.min().item(), x.max().item()
    delta = (b - a) / n_bins
    bins = torch.arange(a, b + 1e-8, step=delta)
    count = torch.histc(x, n_bins).float()
    if density:
        count = count / delta / float(x.shape[0] * x.shape[1])
    return count, bins


class HistoLoss(Loss):
    def __init__(self, x_real, n_bins, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.densities = list()
        self.locs = list()
        self.deltas = list()
        for i in range(x_real.shape[2]):
            x_i = x_real[..., i].reshape(-1, 1)
            d, b = histogram_torch(x_i, n_bins, density=True)
            self.densities.append(nn.Parameter(d).to(x_real.device))
            delta = b[1:2] - b[:1]
            loc = 0.5 * (b[1:] + b[:-1])
            self.locs.append(loc)
            self.deltas.append(delta)

    def compute(self, x_fake):
        loss = list()

        def relu(x):
            return x * (x >= 0.).float()

        for i in range(x_fake.shape[2]):
            loc = self.locs[i].view(1, -1).to(x_fake.device)
            x_i = x_fake[:, :, i].contiguous().view(-1, 1).repeat(1, loc.shape[1])
            dist = torch.abs(x_i - loc)
            counter = (relu(self.deltas[i].to(x_fake.device) / 2. - dist) > 0.).float()
            density = counter.mean(0) / self.deltas[i].to(x_fake.device)
            abs_metric = torch.abs(density - self.densities[i].to(x_fake.device))
            loss.append(torch.mean(abs_metric, 0))
        loss_componentwise = torch.stack(loss)
        return loss_componentwise


def compute_classification_score(x_fake, x_real, get_optim_func, device):
    x_fake = x_fake.detach().cpu()
    x_real = x_real.detach().cpu()
    X = torch.cat([x_fake, x_real], dim=0)
    Y = torch.cat([torch.ones_like(x_fake[:, 0, 0]), torch.zeros_like(x_real[:, 0, 0])], dim=0)

    randperm = torch.randperm(X.shape[0])

    X_train, Y_train = X[randperm[:int(X.shape[0] * 0.8)]], Y[randperm[:int(X.shape[0] * 0.8)]]

    X_test, Y_test = X[randperm[int(X.shape[0] * 0.8):]], Y[randperm[int(X.shape[0] * 0.8):]]
    model = S4Model(d_input=X.shape[-1], d_state=16, d_output=1, d_model=16, n_layers=1,
                    dropout=0.0, seq2seq=False).to(device)
    trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), shuffle=True,
                                              batch_size=128)
    testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, Y_test), batch_size=128)
    optimizer, _ = get_optim_func(model, lr=0.01, weight_decay=0.0, epochs=100)

    pbar = tqdm(range(100))
    for i in range(100):
        for data, label in trainloader:
            optimizer.zero_grad()
            pred, _ = model(data.to(device))

            loss = torch.nn.BCEWithLogitsLoss()(pred.squeeze(-1), label.to(device))
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            test_loss = 0
            for ind, (data, label) in enumerate(testloader):
                pred, _ = model(data.to(device))
                loss = torch.nn.BCEWithLogitsLoss()(pred.squeeze(-1), label.to(device)).detach().cpu()
                test_loss += loss

            pbar.set_description(f'Epoch {i} Test loss: {test_loss / (ind + 1)}')

    return test_loss


def compute_predictive_score(x_real, x_fake, pred_step, get_optim_func, device, pred_activation):
    x_fake = x_fake.detach().cpu()
    x_real = x_real.detach().cpu()
    X = x_fake[:, :-1]
    Y = x_fake[:, 1:]
    masks = torch.ones_like(X, dtype=torch.bool)
    masks[:, :-pred_step] = 0
    X_test = x_real[:, :-1]
    Y_test = x_real[:, 1:]

    model = S4Model(d_input=X.shape[-1], d_state=16, d_output=1, d_model=16, n_layers=1,
                    dropout=0.0, seq2seq=True, activation=pred_activation).to(device)
    trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y, masks), shuffle=True, batch_size=128)
    testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, Y_test, masks), batch_size=128)
    optimizer, _ = get_optim_func(model, lr=0.01, weight_decay=0.0, epochs=100)

    pbar = tqdm(range(100))
    for i in range(100):
        for data, target, mask in trainloader:
            mask = mask.to(device)
            optimizer.zero_grad()
            pred, _ = model(data.to(device))
            loss = torch.nn.MSELoss()(pred[mask], target.to(device)[mask])
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            test_loss = 0
            for ind, (data, target, mask) in enumerate(testloader):
                pred, _ = model(data.to(device))
                loss = torch.nn.MSELoss()(pred[mask], target.to(device)[mask]).detach().cpu()
                test_loss += loss

            pbar.set_description(f'Epoch {i} Test loss: {test_loss / (ind + 1)}')
    return test_loss


def compute_test_metrics(x_fake, x_real):
    res = dict()
    res['marginal_loss'] = HistoLoss(x_real=x_real, n_bins=50, name='marginal_loss')(x_fake).item()

    return res


def compute_all_metrics(x_real, gens, get_optim_func, pred_activation, device):
    pred_step = 10
    cls = []
    pred = []
    marg = []
    for i in range(10):
        clfscore = compute_classification_score(gens, x_real, get_optim_func, device)
        predscore = compute_predictive_score(x_real, gens, pred_step, get_optim_func, device, pred_activation)
        marginalscore = compute_test_metrics(gens, x_real)['marginal_loss']
        cls.append(clfscore)
        pred.append(predscore)
        marg.append(marginalscore)
    clfscore, clf_std = np.mean(cls), np.std(cls)
    predscore, pred_std = np.mean(pred), np.std(pred)
    marginalscore = np.mean(marg)

    res = {'clf_score_mean': clfscore,
           'clf_score_std': clf_std,
           'marginal_score_mean': marginalscore,
           'predictive_score_mean': predscore,
           'predictive_score_std': pred_std
           }

    return res




def compute_fid_given_embeds(real, fake):
    """
    real, fake: (N, D)
    """

    mu_r = np.mean(real, axis=0)
    mu_f = np.mean(fake, axis=0)

    sigma_r = np.cov(real, rowvar=False)
    sigma_f = np.cov(fake, rowvar=False)

    covmean = sqrtm(sigma_r @ sigma_f)

    # 数值稳定（必须）
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum((mu_r - mu_f) ** 2) + np.trace(
        sigma_r + sigma_f - 2 * covmean
    )

    return float(fid)


@torch.no_grad()
def extract_embeddings(model, data, device):

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data.to(device)),
        batch_size=128, shuffle=False,
        drop_last=False
    )
    model.eval()

    all_embeddings = []

    for batch in dataloader:
        x = batch[0].to(device)
        with torch.no_grad():
            out = model(x)
        mu = out["mu"]
        emb = mu
        all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0).cpu().numpy()


def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.
    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.
    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
                             f"Optimizer group {i}",
                             f"{len(g['params'])} tensors",
                         ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler















