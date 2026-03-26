export CUDA_VISIBLE_DEVICES=4

export WANDB_PROJECT="text_conditional_baselines"
export WANDB_NAME="ImagenTime"

python run_unconditional_test_time.py \
  --config ./configs/neurips_baseline/synth_u_text_conditional.yaml \
  --log_dir ../edm_results_text_conditional/imagen_time