import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.vae.fid_vae import FIDVAE

from scipy.linalg import sqrtm

def compute_fid(real, fake):
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


# =========================
# Args
# =========================
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--real_path", type=str, required=True)
    parser.add_argument("--fake_path", type=str, required=True)
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
def load_dataset(dict_path, dict_key, idx=-1):
    data = torch.load(dict_path, weights_only=False)[dict_key]
    if dict_key == "sampled_ts":
        if idx > -1:
            data = data[idx]
    if data.shape[1] > data.shape[2]:
        data = data.permute(0,2,1)
    # print(f"Loaded {dict_path}, key: {dict_key}: {data.shape}")
    return TensorDataset(data.float())



# =========================
# Extract Embedding
# =========================
@torch.no_grad()
def extract_embeddings(model, dataloader, device):

    model.eval()

    all_embeddings = []

    for batch in dataloader:
        x = batch[0].to(device)
        with torch.no_grad():
            out = model(x)
        mu = out["mu"]   # (B, C, T', latent)

        emb = mu

        all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0).cpu().numpy()



def main(args):

    device = args.device if torch.cuda.is_available() else "cpu"

    # ===== load real =====
    real_dataset = load_dataset(args.real_path, "real_ts", idx=-1)
    real_dataloader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False)

    sample = real_dataset[0][0]
    C, T = sample.shape

    # ===== load model =====
    model = FIDVAE(
        input_dim=C,
        output_dim=C,
        seq_len=T,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        latent_dim=args.latent_dim,
    ).to(device).eval()

    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    print(f"Loaded checkpoint from {args.ckpt_path}")

    # ===== extract all real embeddings =====
    real_embeddings = extract_embeddings(
        model,
        real_dataloader,
        device,
    )  # (N, D)

    N = real_embeddings.shape[0]

    # ===== collect all fake embeddings =====
    fake_embeddings_list = []

    for i in range(10):  # num_repeat
        fake_dataset = load_dataset(args.fake_path, "sampled_ts", idx=i)
        fake_dataloader = DataLoader(fake_dataset, batch_size=args.batch_size, shuffle=False)

        fake_embeddings = extract_embeddings(
            model,
            fake_dataloader,
            device,
        )  # (N, D)

        fake_embeddings_list.append(fake_embeddings)

    # shape: (num_repeat, N, D)
    fake_embeddings_all = np.stack(fake_embeddings_list, axis=0)

    # =========================
    # compute context-FID
    # =========================
    context_fid_list = []

    for i in range(N):
        real_i = real_embeddings[i:i+1]   # (1, D)

        fake_i = fake_embeddings_all[:, i, :]  # (num_repeat, D)

        # repeat real to match fake sample size
        real_i_repeat = np.repeat(real_i, fake_i.shape[0], axis=0)

        fid_i = compute_fid(real_i_repeat, fake_i)
        context_fid_list.append(fid_i)

    context_fid_array = np.array(context_fid_list)

    mean_cfid = np.mean(context_fid_array)
    std_cfid = np.std(context_fid_array)

    print("\n==========================")
    print("real_path: {}".format(args.real_path))
    print("fake_path: {}".format(args.fake_path))
    print(f"Context-FID: ${mean_cfid:.4f} \pm {std_cfid:.4f}$")
    print("==========================\n")


if __name__ == "__main__":
    args = get_args()
    main(args)