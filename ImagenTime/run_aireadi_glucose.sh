export CUDA_VISIBLE_DEVICES=8

export WANDB_PROJECT="aireadi_prediction"
export WANDB_NAME="glucose"

#python run_prediction.py \
#  --config ./configs/extrapolation/aireadi_glucose.yaml \
#  --log_dir ./results_aireadi/glucose


python show_prediction.py \
  --config ./configs/extrapolation/aireadi_glucose.yaml \
  --log_dir ./results_aireadi/glucose \
  --resume True




# scp -r haochenz@unites2.cs.unc.edu:/playpen-shared/haochenz/ts_baselines/ImagenTime/results_aireadi ../

