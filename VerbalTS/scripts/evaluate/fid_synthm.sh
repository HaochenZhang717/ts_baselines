#!/bin/bash

# =========================
# GPU
# =========================
export CUDA_VISIBLE_DEVICES=1

REAL_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_qwen/text2ts_msmdiffmv/0/real_text_samples.pt"
FAKE_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_qwen/text2ts_msmdiffmv/0/real_text_samples.pt"
python calculate_fid.py \
    --real_path ${REAL_PATH} \
    --fake_path ${FAKE_PATH} \
    --ckpt_path "./fid_vae_ckpts/vae_synth_m/best.pt" \
    --batch_size 128 \
    --hidden_size 128 \
    --num_layers 2 \
    --num_heads 8 \
    --latent_dim 64 \
    --save_path "./fid_results/dummy_output.txt"

REAL_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_qwen/text2ts_msmdiffmv/0/real_text_samples.pt"
FAKE_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_qwen/text2ts_msmdiffmv/0/fake_text_samples.pt"
python calculate_fid.py \
    --real_path ${REAL_PATH} \
    --fake_path ${FAKE_PATH} \
    --ckpt_path "./fid_vae_ckpts/vae_synth_m/best.pt" \
    --batch_size 128 \
    --hidden_size 128 \
    --num_layers 2 \
    --num_heads 8 \
    --latent_dim 64 \
    --save_path "./fid_results/dummy_output.txt"


REAL_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_m/text2ts_msmdiffmv/0/samples.pt"
FAKE_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_m/text2ts_msmdiffmv/0/samples.pt"
python calculate_fid.py \
    --real_path ${REAL_PATH} \
    --fake_path ${FAKE_PATH} \
    --ckpt_path "./fid_vae_ckpts/vae_synth_m/best.pt" \
    --batch_size 128 \
    --hidden_size 128 \
    --num_layers 2 \
    --num_heads 8 \
    --latent_dim 64 \
    --save_path "./fid_results/dummy_output.txt"


