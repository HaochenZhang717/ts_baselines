export HF_HOME=/playpen/haochenz/hf_cache
export CUDA_VISIBLE_DEVICES=1


IMAGE_PATH="/playpen-shared/haochenz/LitsDatasets/128_len_ts_trend_imgs/ETTh1"
SAVE_PATH="/playpen-shared/haochenz/LitsDatasets/123_len_ts_trend_imgs_caps/ETTh1"

python run_caption.py \
  --part_id 0 \
  --num_parts 8 \
  --image_folder "${IMAGE_PATH}/train" \
  --split "train" \
  --save_dir "${SAVE_PATH}/ETTh1" \
  --dataset_name "ETTh1"

python run_caption.py \
  --part_id 0 \
  --num_parts 8 \
  --image_folder "${IMAGE_PATH}/valid" \
  --split "valid" \
  --save_dir "${SAVE_PATH}/ETTh1" \
  --dataset_name "ETTh1"

python run_caption.py \
--part_id 0 \
--num_parts 8 \
--image_folder "${IMAGE_PATH}/test" \
--split "test" \
--save_dir "${SAVE_PATH}/ETTh1" \
--dataset_name "ETTh1"
