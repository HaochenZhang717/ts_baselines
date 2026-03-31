export USE_CAUSAL=false

CUDA_VISIBLE_DEVICES=7 python run_qwen_v2.py \
    --cond_modal text \
    --training_stage finetune \
    --save_folder ./verbalts_orig_save/synth_u_qwen_v2/text2ts_msmdiffmv \
    --model_diff_config_path configs/synth_u_qwen_v2/diff/model_text2ts_dep.yaml \
    --model_cond_config_path configs/synth_u_qwen_v2/cond/text_msmdiffmv.yaml \
    --train_config_path configs/synth_u_qwen_v2/train.yaml \
    --evaluate_config_path configs/synth_u_qwen_v2/evaluate.yaml \
    --data_folder /playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 2 \
    --base_patch 4 \
    --epochs 2500 \
    --batch_size 512 \
    --clip_cache_path "" \
    --samples_name "real_text_samples.pt" \
    --model_ckpt_name "model_best_loss.pth" \
    --only_evaluate True \
    --n_runs 1