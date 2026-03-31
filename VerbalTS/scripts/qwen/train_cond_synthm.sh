export USE_CAUSAL=false

CUDA_VISIBLE_DEVICES=4 python run_qwen.py \
    --cond_modal text \
    --training_stage finetune \
    --save_folder ./verbalts_orig_save/synth_m_qwen/text2ts_msmdiffmv \
    --model_diff_config_path configs/synth_m_qwen/diff/model_text2ts_dep.yaml \
    --model_cond_config_path configs/synth_m_qwen/cond/text_msmdiffmv.yaml \
    --train_config_path configs/synth_m_qwen/train.yaml \
    --evaluate_config_path configs/synth_m_qwen/evaluate_fake_text.yaml \
    --data_folder /playpen/haochenz/LitsDatasets/128_len_ts/synthetic_m \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 2 \
    --base_patch 4 \
    --epochs 2500 \
    --batch_size 512 \
    --clip_cache_path "" \
    --samples_name "fake_text_samples.pt" \
    --model_ckpt_name "model_best_loss.pth" \
    --n_runs 1 \
    --only_evaluate True


CUDA_VISIBLE_DEVICES=4 python run_qwen.py \
    --cond_modal text \
    --training_stage finetune \
    --save_folder ./verbalts_orig_save/synth_m_qwen/text2ts_msmdiffmv \
    --model_diff_config_path configs/synth_m_qwen/diff/model_text2ts_dep.yaml \
    --model_cond_config_path configs/synth_m_qwen/cond/text_msmdiffmv.yaml \
    --train_config_path configs/synth_m_qwen/train.yaml \
    --evaluate_config_path configs/synth_m_qwen/evaluate_real_text.yaml \
    --data_folder /playpen/haochenz/LitsDatasets/128_len_ts/synthetic_m \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 2 \
    --base_patch 4 \
    --epochs 2500 \
    --batch_size 512 \
    --clip_cache_path "" \
    --samples_name "real_text_samples.pt" \
    --model_ckpt_name "model_best_loss.pth" \
    --n_runs 1 \
    --only_evaluate True

#CUDA_VISIBLE_DEVICES=4 python run_qwen.py \
#    --cond_modal vae_embed \
#    --training_stage finetune \
#    --save_folder ./verbalts_orig_save/synth_m_vae_embed/text2ts_msmdiffmv \
#    --model_diff_config_path configs/synth_m_qwen/diff/model_text2ts_dep.yaml \
#    --model_cond_config_path configs/synth_m_qwen/cond/text_msmdiffmv.yaml \
#    --train_config_path configs/synth_m_qwen/train.yaml \
#    --evaluate_config_path configs/synth_m_qwen/evaluate.yaml \
#    --data_folder /playpen/haochenz/LitsDatasets/128_len_ts/synthetic_m \
#    --clip_folder "" \
#    --multipatch_num 3 \
#    --L_patch_len 2 \
#    --base_patch 4 \
#    --epochs 2500 \
#    --batch_size 512 \
#    --clip_cache_path "" \
#    --samples_name "real_text_samples.pt"