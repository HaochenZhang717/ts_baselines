export CUDA_VISIBLE_DEVICES=0

export WANDB_PROJECT="text_conditional_baselines"
export WANDB_NAME="LDM_debug"

python run_text_conditional_ldm.py \
  --config ./configs/neurips_baseline/synth_u_text_conditional_ldm.yaml \
  --log_dir ../edm_results_text_conditional/LDM_debug \
  --model_type LDMUNetModel
