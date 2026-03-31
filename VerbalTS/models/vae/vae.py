import torch
import torch.nn as nn
import torch.nn.functional as F



class NormAttention(nn.Module):
    """
    Attention module of LightningDiT.
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.to(v.dtype)
        k = k.to(v.dtype)  # rope may change the q,k's dtype
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
        self.attn = NormAttention(hidden_size, num_heads=num_heads)

        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim), hidden_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x



class TimeSeriesEncoder(nn.Module):
    def __init__(
        self,
        input_dim,        # channel数
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        latent_dim=64,
    ):
        super().__init__()

        # ===== CNN 前端（抓局部结构）=====
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size // 2, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=2, stride=2, padding=0),
        )

        # ===== Transformer blocks（用你的 EncoderLayer）=====
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

        # ===== VAE head =====
        self.to_mu = nn.Linear(hidden_size, latent_dim)
        self.to_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        """
        x: (B, C, T)
        """

        # CNN
        x = self.conv(x)  # (B, hidden, T)

        # → Transformer 输入格式
        x = x.permute(0, 2, 1)  # (B, T, hidden)

        # Transformer
        for layer in self.layers:
            x = layer(x)


        # VAE
        mu = self.to_mu(x)
        logvar = self.to_logvar(x)

        return mu, logvar


class TimeSeriesDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        output_dim,
        hidden_size=128,
    ):
        super().__init__()
        # ===== latent → hidden =====
        self.input_proj = nn.Conv1d(latent_dim, hidden_size, kernel_size=1)
        # ===== CNN decoder =====
        self.net = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(hidden_size // 2, output_dim, kernel_size=5, padding=2),
        )

    def forward(self, z):
        """
        z: (B, latent_dim, T/4)
        """
        x = self.input_proj(z)   # (B, hidden, T/4)
        x = self.net(x)          # (B, C, T)
        return x


class TimeSeriesVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        latent_dim=64,
        beta=0.001,   # KL权重
    ):
        super().__init__()

        self.beta = beta

        # ===== Encoder =====
        self.encoder = TimeSeriesEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            latent_dim=latent_dim,
        )

        # ===== Decoder =====
        self.decoder = TimeSeriesDecoder(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
        )

    # =========================
    # reparameterization
    # =========================
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # =========================
    # forward
    # =========================
    def forward(self, x):
        """
        x: (B, C, T)
        """

        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        recon = self.decoder(z.permute(0, 2, 1))

        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    # =========================
    # loss
    # =========================
    def loss_function(self, x, recon, mu, logvar):
        """
        x:      (B, C, T)
        recon:  (B, C, T)
        """

        # ===== reconstruction loss =====
        recon_loss = F.mse_loss(recon, x)

        # ===== KL divergence =====
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        # ===== total =====
        loss = recon_loss + self.beta * kl_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }


if __name__ == "__main__":

    model = TimeSeriesVAE(
        input_dim=1,
        output_dim=1,
        hidden_size=128,
        num_layers=2,
        num_heads=8,
        latent_dim=128,
        beta=0.001,  # KL权重
    )

    x = torch.randn(8, 1, 128)

    out = model(x)

    loss_dict = model.loss_function(
        x,
        out["recon"],
        out["mu"],
        out["logvar"]
    )

    loss = loss_dict["loss"]
    loss.backward()