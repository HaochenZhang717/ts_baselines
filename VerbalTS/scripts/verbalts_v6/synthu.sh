LR_LIST=(5e-4)
BS_LIST=(512)

for LR in "${LR_LIST[@]}"
do
  for BS in "${BS_LIST[@]}"
  do
    echo "Running lr=$LR bs=$BS"

    export WANDB_NAME="synth_u_v6_-lr${LR}_bs${BS}"

    CUDA_VISIBLE_DEVICES=1 python run_v6.py \
    --cond_modal text \
    --training_stage finetune \
    --save_folder ../sweep_text2ts_v6/synth_u/verbalts_v6 \
    --model_diff_config_path configs/synth_u/diff/model_text2ts_dep.yaml \
    --model_cond_config_path configs/synth_u/cond/text_msmdiffmv.yaml \
    --train_config_path configs/synth_u/train.yaml \
    --evaluate_config_path configs/synth_u/evaluate.yaml \
    --data_folder /playpen-shared/haochenz/LitsDatasets/128_len_ts/synthetic_u \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 2 \
    --base_patch 4 \
    --epochs 2500 \
    --batch_size ${BS} \
    --lr ${LR} \
    --clip_cache_path "" \
    --samples_name "samples.pt"
done
done