export CUDA_VISIBLE_DEVICES=4

export WANDB_PROJECT="aireadi_prediction"
export WANDB_NAME="debug"

python run_prediction.py \
  --config ./configs/extropolation/aireadi.yaml





