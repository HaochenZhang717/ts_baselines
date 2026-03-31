#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

# =========================
# 路径
# =========================

# 👉 必须是真实数据（GT）
REAL_PATH="/playpen-shared/haochenz/ts_baselines/ImagenTime/logs/synth_u/conditional-bs=128-lr=0.0001-ch_mult=1-2-attn_res=16-8-4-unet_ch=64-delay=4-32/samples_epoch_1500.pt"
SAMPLE_PATH="/playpen/haochenz/VerbalTS_reimplement/baseline_results/diffusionts_results/synth_u/ddpm_fake_synth_u.npy"
SAVE_FILE="./fid_results/synth_u_diffusion_ts.txt"

python calculate_fid_diffusionTS.py \
    --real_path ${REAL_PATH} \
    --fake_path ${SAMPLE_PATH} \
    --ckpt_path "./fid_vae_ckpts/vae_synth_u/best.pt" \
    --batch_size 128 \
    --hidden_size 128 \
    --num_layers 2 \
    --num_heads 8 \
    --latent_dim 64 \
    --num_samples 2850 \
    --save_path "tmp.txt"
