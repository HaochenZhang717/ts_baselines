export USE_CAUSAL=false
CUDA_VISIBLE_DEVICES=7 python run_qwen.py \
    --cond_modal text \
    --training_stage finetune \
    --save_folder ./verbalts_orig_save/istanbul_qwen/text2ts_msmdiffmv \
    --model_diff_config_path configs/istanbul_qwen/diff/model_text2ts_dep.yaml \
    --model_cond_config_path configs/istanbul_qwen/cond/text_msmdiffmv.yaml \
    --train_config_path configs/istanbul_qwen/train.yaml \
    --evaluate_config_path configs/istanbul_qwen/evaluate.yaml \
    --data_folder /playpen/haochenz/LitsDatasets/128_len_ts/istanbul_traffic \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 2 \
    --base_patch 4 \
    --epochs 2500 \
    --batch_size 512 \
    --clip_cache_path "" \
    --samples_name "real_text_samples.pt" \
    --only_evaluate True