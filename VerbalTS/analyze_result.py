import torch
from metrics.discriminative_torch import discriminative_score_metrics
import numpy as np

import json
import os
import numpy as np
import torch

import torch
import matplotlib.pyplot as plt
import numpy as np
from metrics.discriminative_torch import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
import torch
import numpy as np
from scipy.linalg import sqrtm
from momentfm import MOMENTPipeline



def calculate_disc_two_paths(real_path, fake_path, save_path="disc_results.jsonl"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_dict = torch.load(real_path, map_location="cpu", weights_only=False)
    real = real_dict["real_ts"]

    samples_dict = torch.load(fake_path, map_location="cpu", weights_only=False)

    # num_samples = min(len(real), len(samples_dict["sampled_ts"][0]))
    num_samples = 2850
    real = real[:num_samples]
    if real.shape[1] > real.shape[2]:
        real = real.permute(0,2,1)
    disc_score_list = []
    print(real.shape)
    print(samples_dict["sampled_ts"].shape)
    for i in range(samples_dict["sampled_ts"].shape[0]):
        fake = samples_dict["sampled_ts"][i, :num_samples]
        if fake.shape[1] > fake.shape[2]:
            fake = fake.permute(0,2,1)
        for _ in range(1):
            discriminative_score = discriminative_score_metrics(
                real.permute(0,2,1), fake.permute(0,2,1),
                real.shape[1],
                device,
            )
            disc_score_list.append(discriminative_score)
            # print(discriminative_score)

    disc_score_arr = np.array(disc_score_list)
    disc_mean = float(disc_score_arr.mean())
    disc_std = float(disc_score_arr.std(ddof=1))

    # ===== 构造结果 =====
    result = {
        "real_path": real_path,
        "fake_path": fake_path,
        "num_samples": int(num_samples),
        "disc_scores": disc_score_list,
        "disc_mean": disc_mean,
        "disc_std": disc_std,
    }

    # ===== 写入 jsonl（append）=====
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None

    with open(save_path, "a") as f:
        f.write(json.dumps(result) + "\n")

    # ===== 仍然print（方便你看）=====
    print(fake_path)
    print(f"Disc Score: mean = {disc_mean:.4f}, std = {disc_std:.4f}")
    print("---" * 50)


def calculate_pred_two_paths(real_path, fake_path, look_real=False, save_path="pred_results.jsonl"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_dict = torch.load(real_path, map_location="cpu", weights_only=False)
    real = real_dict["real_ts"]
    if real.shape[1] > real.shape[2]:
        real = real.permute(0,2,1)

    samples_dict = torch.load(fake_path, map_location="cpu", weights_only=False)

    # num_samples = min(len(real), len(samples_dict["sampled_ts"][0]))
    num_samples = 2850
    real = real[:num_samples]

    pred_score_list = []
    print(real.shape)
    print(samples_dict["sampled_ts"].shape)
    for i in range(samples_dict["sampled_ts"].shape[0]):
        fake = samples_dict["sampled_ts"][i, :num_samples]
        if fake.shape[1] > fake.shape[2]:
            fake = fake.permute(0,2,1)
        for _ in range(1):
            if not look_real:
                pred_score = predictive_score_metrics(
                    real.permute(0,2,1),
                    fake.permute(0,2,1),
                    device=device,
                )
            else:
                pred_score = predictive_score_metrics(
                    real.permute(0, 2, 1),
                    real.permute(0, 2, 1),
                    device,
                )
                print(pred_score)
            pred_score_list.append(pred_score)
            # print(discriminative_score)

    pred_score_arr = np.array(pred_score_list)
    pred_score_mean = float(pred_score_arr.mean())
    pred_score_std = float(pred_score_arr.std(ddof=1))

    # ===== 构造结果 =====
    result = {
        "real_path": real_path,
        "fake_path": fake_path,
        "num_samples": int(num_samples),
        # "pred_scores": pred_score_list,
        "pred_mean": pred_score_mean,
        "pred_std": pred_score_std,
    }

    # ===== 写入 jsonl（append）=====
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None

    with open(save_path, "a") as f:
        f.write(json.dumps(result) + "\n")

    # ===== 仍然print（方便你看）=====
    print(fake_path)
    print(f"Pred Score: mean = {pred_score_mean:.4f}, std = {pred_score_std:.4f}")
    print("---" * 50)



def _moment_embed(moment_model, x, device, batch_size=64):
    """
    x: torch.Tensor, shape (N, n_var, seq_len)
    return: np.ndarray, shape (N, dim)
    """
    moment_model.eval()
    emb_list = []

    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = x[start:start + batch_size].to(device).float()

            out = moment_model(x_enc=batch, reduction="none").embeddings
            # out shape: (B, n_var, seq_len, dim)
            out = out.mean(dim=(1, 2))   # -> (B, dim)

            emb_list.append(out.cpu().numpy())

    return np.concatenate(emb_list, axis=0)


def _calculate_fid_from_embeddings(real_emb, fake_emb, eps=1e-6):
    """
    real_emb: np.ndarray, shape (N, D)
    fake_emb: np.ndarray, shape (M, D)
    """
    mu_r = np.mean(real_emb, axis=0)
    mu_f = np.mean(fake_emb, axis=0)

    sigma_r = np.cov(real_emb, rowvar=False)
    sigma_f = np.cov(fake_emb, rowvar=False)

    if sigma_r.ndim == 0:
        sigma_r = np.array([[sigma_r]])
    if sigma_f.ndim == 0:
        sigma_f = np.array([[sigma_f]])

    sigma_r = sigma_r + eps * np.eye(sigma_r.shape[0])
    sigma_f = sigma_f + eps * np.eye(sigma_f.shape[0])

    diff = mu_r - mu_f
    covmean = sqrtm(sigma_r @ sigma_f)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2.0 * covmean)
    return float(fid)


def calculate_fid_two_paths(real_path, fake_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_dict = torch.load(real_path, map_location="cpu", weights_only=False)
    real = real_dict["real_ts"]
    print(f"real shape = {real.shape}")
    samples_dict = torch.load(fake_path, map_location="cpu", weights_only=False)
    print(f"fake shape = {samples_dict['sampled_ts'].shape}")
    num_samples = min(len(real), len(samples_dict["sampled_ts"][0]))
    real = real[:num_samples]
    fake_raw = samples_dict["sampled_ts"]
    print(f"real shape: {real.shape}")
    # ----------------------------
    # load MOMENT once
    # ----------------------------
    moment_model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "embedding"},
    )
    moment_model.init()
    moment_model = moment_model.to(device)
    moment_model.eval()

    fid_score_list = []
    # real embedding can be computed once
    real_emb = _moment_embed(moment_model, real.permute(0,2,1), device)

    for i in range(10):
        fake = fake_raw[i, :num_samples].permute(0,2,1)
        fake_emb = _moment_embed(moment_model, fake, device)
        fid_score = _calculate_fid_from_embeddings(real_emb, fake_emb)
        fid_score_list.append(fid_score)

    fid_score_arr = np.array(fid_score_list)
    fid_mean = fid_score_arr.mean()
    fid_std = fid_score_arr.std(ddof=1)

    print(f"MOMENT-FID: mean = {fid_mean:.4f}, std = {fid_std:.4f}")
    print("---" * 50)




if __name__ == "__main__":


    # calculate_disc_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_vae_embed/text2ts_msmdiffmv/0/real_text_samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_vae_embed/text2ts_msmdiffmv/0/real_text_samples.pt"
    # )

    # calculate_pred_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen/text2ts_msmdiffmv/0/real_text_samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen/text2ts_msmdiffmv/0/real_text_samples.pt"
    # )
    #
    # calculate_pred_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen/text2ts_msmdiffmv/0/real_text_samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen/text2ts_msmdiffmv/0/fake_text_samples.pt"
    # )
    #
    # calculate_pred_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_u/text2ts_msmdiffmv/0/samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_u/text2ts_msmdiffmv/0/samples.pt"
    # )
    #
    # calculate_pred_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_u/text2ts_msmdiffmv/0/samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_u/text2ts_msmdiffmv/0/samples.pt",
    #     look_real=True
    # )

    calculate_pred_two_paths(
    "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u/text2ts_msmdiffmv/0/verbalts_caps_samples.pt",
    "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u/text2ts_msmdiffmv/0/verbalts_caps_samples.pt",
    look_real=False)

    calculate_disc_two_paths(
    "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u/text2ts_msmdiffmv/0/verbalts_caps_samples.pt",
    "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u/text2ts_msmdiffmv/0/verbalts_caps_samples.pt",
    )



    # calculate_pred_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_qwen/text2ts_msmdiffmv/0/real_text_samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_qwen/text2ts_msmdiffmv/0/real_text_samples.pt"
    # )
    #
    # calculate_pred_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_qwen/text2ts_msmdiffmv/0/real_text_samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_qwen/text2ts_msmdiffmv/0/fake_text_samples.pt"
    # )
    #
    # calculate_pred_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_m/text2ts_msmdiffmv/0/samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_m/text2ts_msmdiffmv/0/samples.pt"
    # )

    # calculate_pred_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_m/text2ts_msmdiffmv/0/samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_m/text2ts_msmdiffmv/0/samples.pt",
    #     look_real=True
    # )


    #
    # calculate_disc_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen/text2ts_msmdiffmv/0/real_text_samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen/text2ts_msmdiffmv/0/real_text_samples.pt"
    # )
    #
    # calculate_disc_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen/text2ts_msmdiffmv/0/real_text_samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen/text2ts_msmdiffmv/0/fake_text_samples.pt"
    # )
    #
    # calculate_disc_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_u/text2ts_msmdiffmv/0/samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_u/text2ts_msmdiffmv/0/samples.pt"
    # )


    # calculate_disc_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_qwen/text2ts_msmdiffmv/0/real_text_samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_qwen/text2ts_msmdiffmv/0/real_text_samples.pt"
    # )

    # calculate_disc_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_qwen/text2ts_msmdiffmv/0/real_text_samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_qwen/text2ts_msmdiffmv/0/fake_text_samples.pt"
    # )
    #
    # calculate_disc_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_m/text2ts_msmdiffmv/0/samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_m/text2ts_msmdiffmv/0/samples.pt"
    # )


    # calculate_disc_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_qwen/text2ts_msmdiffmv/0/real_text_samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_qwen/text2ts_msmdiffmv/0/real_text_samples.pt"
    # )
    #
    # calculate_disc_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_qwen/text2ts_msmdiffmv/0/real_text_samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_qwen/text2ts_msmdiffmv/0/fake_text_samples.pt"
    # )
    #
    # calculate_disc_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_m/text2ts_msmdiffmv/0/samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_m/text2ts_msmdiffmv/0/samples.pt"
    # )



    # calculate_disc_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u/text2ts_msmdiffmv/0/verbalts_caps_samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u/text2ts_msmdiffmv/0/verbalts_caps_samples.pt"
    # )
    #


    # calculate_fid_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_vae_embed/text2ts_msmdiffmv/0/real_text_samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_vae_embed/text2ts_msmdiffmv/0/real_text_samples.pt"
    # )
    #
    # calculate_fid_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen/text2ts_msmdiffmv/0/real_text_samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen/text2ts_msmdiffmv/0/real_text_samples.pt"
    # )
    #
    # calculate_fid_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u/text2ts_msmdiffmv/0/verbalts_caps_samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u/text2ts_msmdiffmv/0/verbalts_caps_samples.pt"
    # )
    #
    # calculate_fid_two_paths(
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_u/text2ts_msmdiffmv/0/samples.pt",
    #     "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_u/text2ts_msmdiffmv/0/samples.pt"
    # )
