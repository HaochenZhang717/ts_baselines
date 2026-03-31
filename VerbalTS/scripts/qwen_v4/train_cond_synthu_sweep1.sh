export USE_CAUSAL=false


LR=1e-3
BS=64
export WANDB_NAME="qwen_v4_synth_u_lr${LR}_bs${BS}"

CUDA_VISIBLE_DEVICES=6 python run_qwen_v4.py \
    --cond_modal text \
    --training_stage finetune \
    --save_folder ./verbalts_orig_save/synth_u_qwen_v4/text2ts_msmdiffmv \
    --model_diff_config_path configs/synth_u_qwen_v4/diff/model_text2ts_dep.yaml \
    --model_cond_config_path configs/synth_u_qwen_v4/cond/text_msmdiffmv.yaml \
    --train_config_path configs/synth_u_qwen_v4/train.yaml \
    --evaluate_config_path configs/synth_u_qwen_v4/evaluate.yaml \
    --data_folder /playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 2 \
    --base_patch 4 \
    --epochs 2500 \
    --lr ${LR} \
    --batch_size ${BS} \
    --clip_cache_path "" \
    --samples_name "real_text_samples.pt" \
    --model_ckpt_name "model_best_loss.pth"