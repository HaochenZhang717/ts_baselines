#!/bin/bash

# =========================
# GPU
# =========================
export CUDA_VISIBLE_DEVICES=1


#REAL_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_vae_embed/text2ts_msmdiffmv/0/real_text_samples.pt"
#FAKE_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_vae_embed/text2ts_msmdiffmv/0/real_text_samples.pt"
#python calculate_fid.py \
#    --real_path ${REAL_PATH} \
#    --fake_path ${FAKE_PATH} \
#    --ckpt_path "./fid_vae_ckpts/vae_synth_u/best.pt" \
#    --batch_size 128 \
#    --hidden_size 128 \
#    --num_layers 2 \
#    --num_heads 8 \
#    --latent_dim 64 \
#    --save_path "./fid_results/vae_embed_generation.txt"


#REAL_PATH="../sweep_text2ts/synth_u/orig_verbalts_orig_cap/2/samples.pt"
#FAKE_PATH="../sweep_text2ts/synth_u/orig_verbalts_orig_cap/2/samples.pt"
##REAL_PATH="/playpen-shared/haochenz/sweep_text2ts/synth_u/orig_verbalts_orig_cap_precompute_embed-lr1e-3_bs512_multistep/0/samples.pt"
##FAKE_PATH="/playpen-shared/haochenz/sweep_text2ts/synth_u/orig_verbalts_orig_cap_precompute_embed-lr1e-3_bs512_multistep/0/samples.pt"
#
#python calculate_fid.py \
#    --real_path ${REAL_PATH} \
#    --fake_path ${FAKE_PATH} \
#    --ckpt_path "../fid_vae_ckpts/vae_synth_u/best.pt" \
#    --batch_size 128 \
#    --hidden_size 128 \
#    --num_layers 2 \
#    --num_heads 8 \
#    --latent_dim 64 \
#    --save_path "./fid_results/verbal_ts_generation_0324.txt"
#
#
#
#
#
#REAL_PATH="/playpen-shared/haochenz/sweep_text2ts/synth_u/orig_verbalts_orig_cap_precompute_embed-lr1e-3_bs512_multistep/0/samples.pt"
#FAKE_PATH="/playpen-shared/haochenz/sweep_text2ts/synth_u/orig_verbalts_orig_cap_precompute_embed-lr1e-3_bs512_multistep/0/samples.pt"
#
#python calculate_fid.py \
#    --real_path ${REAL_PATH} \
#    --fake_path ${FAKE_PATH} \
#    --ckpt_path "../fid_vae_ckpts/vae_synth_u/best.pt" \
#    --batch_size 128 \
#    --hidden_size 128 \
#    --num_layers 2 \
#    --num_heads 8 \
#    --latent_dim 64 \
#    --save_path "./fid_results/verbal_ts_generation_0324.txt"




#REAL_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen_v3/text2ts_msmdiffmv/1/fake_text_samples.pt"
#FAKE_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen_v3/text2ts_msmdiffmv/1/fake_text_samples.pt"
#python calculate_fid.py \
#    --real_path ${REAL_PATH} \
#    --fake_path ${FAKE_PATH} \
#    --ckpt_path "./fid_vae_ckpts/vae_synth_u/best.pt" \
#    --batch_size 128 \
#    --hidden_size 128 \
#    --num_layers 2 \
#    --num_heads 8 \
#    --latent_dim 64 \
#    --save_path "./fid_results/uncond_generation.txt"



#REAL_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen_v3/text2ts_msmdiffmv/0/fake_text_samples.pt"
#FAKE_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen_v3/text2ts_msmdiffmv/0/fake_text_samples.pt"
#python calculate_fid.py \
#    --real_path ${REAL_PATH} \
#    --fake_path ${FAKE_PATH} \
#    --ckpt_path "./fid_vae_ckpts/vae_synth_u/best.pt" \
#    --batch_size 128 \
#    --hidden_size 128 \
#    --num_layers 2 \
#    --num_heads 8 \
#    --latent_dim 64 \
#    --save_path "./fid_results/uncond_generation.txt"



#REAL_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen_v2/text2ts_msmdiffmv/0/real_text_samples.pt"
#FAKE_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen_v2/text2ts_msmdiffmv/0/real_text_samples.pt"
#python calculate_fid.py \
#    --real_path ${REAL_PATH} \
#    --fake_path ${FAKE_PATH} \
#    --ckpt_path "./fid_vae_ckpts/vae_synth_u/best.pt" \
#    --batch_size 128 \
#    --hidden_size 128 \
#    --num_layers 2 \
#    --num_heads 8 \
#    --latent_dim 64 \
#    --save_path "./fid_results/uncond_generation.txt"

#REAL_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen/text2ts_msmdiffmv/0/real_text_samples.pt"
#FAKE_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen/text2ts_msmdiffmv/0/real_text_samples.pt"
#python calculate_fid.py \
#    --real_path ${REAL_PATH} \
#    --fake_path ${FAKE_PATH} \
#    --ckpt_path "./fid_vae_ckpts/vae_synth_u/best.pt" \
#    --batch_size 128 \
#    --hidden_size 128 \
#    --num_layers 2 \
#    --num_heads 8 \
#    --latent_dim 64 \
#    --save_path "./fid_results/uncond_generation.txt"

#REAL_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen/text2ts_msmdiffmv/0/real_text_samples.pt"
#FAKE_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen/text2ts_msmdiffmv/0/fake_text_samples.pt"
#python calculate_fid.py \
#    --real_path ${REAL_PATH} \
#    --fake_path ${FAKE_PATH} \
#    --ckpt_path "./fid_vae_ckpts/vae_synth_u/best.pt" \
#    --batch_size 128 \
#    --hidden_size 128 \
#    --num_layers 2 \
#    --num_heads 8 \
#    --latent_dim 64 \
#    --save_path "./fid_results/uncond_generation.txt"


REAL_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u/text2ts_msmdiffmv/1/verbalts_caps_samples.pt"
FAKE_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u/text2ts_msmdiffmv/1/verbalts_caps_samples.pt"
python calculate_fid.py \
    --real_path ${REAL_PATH} \
    --fake_path ${FAKE_PATH} \
    --ckpt_path "../fid_vae_ckpts/vae_synth_u/best.pt" \
    --batch_size 128 \
    --hidden_size 128 \
    --num_layers 2 \
    --num_heads 8 \
    --latent_dim 64 \
    --save_path "./fid_results/verbalts_generation.txt"


REAL_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_u/text2ts_msmdiffmv/0/samples.pt"
FAKE_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_u/text2ts_msmdiffmv/0/samples.pt"
python calculate_fid.py \
    --real_path ${REAL_PATH} \
    --fake_path ${FAKE_PATH} \
    --ckpt_path "../fid_vae_ckpts/vae_synth_u/best.pt" \
    --batch_size 128 \
    --hidden_size 128 \
    --num_layers 2 \
    --num_heads 8 \
    --latent_dim 64 \
    --save_path "./fid_results/uncond_generation.txt"


#REAL_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_my_caps/text2ts_msmdiffmv/0/real_text_samples.pt"
#FAKE_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_my_caps/text2ts_msmdiffmv/0/fake_text_samples.pt"
#python calculate_fid.py \
#    --real_path ${REAL_PATH} \
#    --fake_path ${FAKE_PATH} \
#    --ckpt_path "./fid_vae_ckpts/vae_synth_u/best.pt" \
#    --batch_size 128 \
#    --hidden_size 128 \
#    --num_layers 2 \
#    --num_heads 8 \
#    --latent_dim 64 \
#    --save_path "./fid_results/synth_u_my_caps_generation.txt"


#REAL_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_my_caps/text2ts_msmdiffmv/0/real_text_samples.pt"
#FAKE_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_my_caps/text2ts_msmdiffmv/0/real_text_samples.pt"
#python calculate_fid.py \
#    --real_path ${REAL_PATH} \
#    --fake_path ${FAKE_PATH} \
#    --ckpt_path "./fid_vae_ckpts/vae_synth_u/best.pt" \
#    --batch_size 128 \
#    --hidden_size 128 \
#    --num_layers 2 \
#    --num_heads 8 \
#    --latent_dim 64 \
#    --save_path "./fid_results/synth_u_my_caps_generation.txt"



#REAL_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_vae_embed/text2ts_msmdiffmv/0/samples.pt"
#FAKE_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_vae_embed/text2ts_msmdiffmv/0/samples.pt"
#python calculate_fid.py \
#    --real_path ${REAL_PATH} \
#    --fake_path ${FAKE_PATH} \
#    --ckpt_path "./fid_vae_ckpts/vae_synth_m/best.pt" \
#    --batch_size 128 \
#    --hidden_size 128 \
#    --num_layers 2 \
#    --num_heads 8 \
#    --latent_dim 64 \
#    --save_path "./fid_results/vae_embed_synthm.txt"
#
#
#REAL_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_vae_embed/text2ts_msmdiffmv/0/samples.pt"
#FAKE_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_m_vae_embed/text2ts_msmdiffmv/0/samples.pt"
#python calculate_fid.py \
#    --real_path ${REAL_PATH} \
#    --fake_path ${FAKE_PATH} \
#    --ckpt_path "../fid_vae_ckpts/vae_synth_m/best.pt" \
#    --batch_size 128 \
#    --hidden_size 128 \
#    --num_layers 2 \
#    --num_heads 8 \
#    --latent_dim 64 \
#    --save_path "./fid_results/vae_embed_synthm.txt"