export CUDA_VISIBLE_DEVICES=0

export WANDB_PROJECT="aireadi_prediction"
export WANDB_NAME="glucose_calorie"

#python run_prediction.py \
#  --config ./configs/extrapolation/aireadi_glucose_calorie.yaml \
#  --log_dir ./results_aireadi/glucose_calorie

python run_prediction.py \
  --config ./configs/extrapolation/aireadi_glucose_calorie.yaml \
  --log_dir ./results_aireadi/glucose_calorie \
  --resume True





