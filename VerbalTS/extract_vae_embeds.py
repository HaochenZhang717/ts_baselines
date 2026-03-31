import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.vae.vae import TimeSeriesVAE


# =========================
# Args
# =========================
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=128)

    # model config（必须一致）
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=64)

    # embedding 类型
    # parser.add_argument(
    #     "--mode",
    #     type=str,
    #     default="mean",
    #     choices=["mean", "full", "flatten"]
    # )

    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


# =========================
# Dataset
# =========================
def load_dataset(npy_path):
    data = np.load(npy_path)

    if data.ndim == 2:
        data = data[:, None, :]

    data = torch.tensor(data, dtype=torch.float32).permute(0, 2, 1)

    print(f"Loaded {npy_path}: {data.shape}")

    return TensorDataset(data)


# =========================
# Extract Embedding
# =========================
@torch.no_grad()
def extract_embeddings(model, dataloader, device):

    model.eval()

    all_embeddings = []

    for batch in tqdm(dataloader, desc="Extracting"):
        x = batch[0].to(device)

        out = model(x)
        mu = out["mu"]   # (B, C, T', latent)

        emb = mu

        all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0)


# =========================
# Main
# =========================
def main(args):

    device = args.device if torch.cuda.is_available() else "cpu"

    dataset = load_dataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    sample = dataset[0][0]
    C, T = sample.shape

    # ===== load model =====
    model = TimeSeriesVAE(
        input_dim=C,
        output_dim=C,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        latent_dim=args.latent_dim,
    ).to(device)

    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    print(f"Loaded checkpoint from {args.ckpt_path}")

    # ===== extract =====
    embeddings = extract_embeddings(
        model,
        dataloader,
        device,
    )

    print(f"Embedding shape: {embeddings.shape}")

    # ===== save =====
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    np.save(args.save_path, embeddings.numpy())

    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    args = get_args()
    main(args)