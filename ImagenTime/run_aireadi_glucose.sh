export CUDA_VISIBLE_DEVICES=5

export WANDB_PROJECT="aireadi_prediction"
export WANDB_NAME="glucose"

python run_prediction.py \
  --config ./configs/extrapolation/aireadi_glucose.yaml





