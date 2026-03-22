


LR=1e-4
BS=256
export CUDA_VISIBLE_DEVICES=5
export WANDB_NAME="synth_u_lr${LR}_bs${BS}"


python main.py \
  --name synth_u \
  --config_file Config/neurips_baselines/synth_u.yaml \
  --gpu 0 \
  --train \
  --lr ${LR} \
  --batch_size ${BS}

