
#INPUT_PATH="../ETTh1"


INPUT_PATH="/playpen-shared/haochenz/LitsDatasets/128_len_ts_trend_imgs_caps/ETTh1/ETTh1"

python process_raw_caps.py \
 --folder ${INPUT_PATH} \
 --name_format "train_caps_*.jsonl" \
 --output_file "${INPUT_PATH}/train_caps.jsonl"

python process_raw_caps.py \
 --folder ${INPUT_PATH} \
 --name_format "valid_caps_*.jsonl" \
 --output_file "${INPUT_PATH}/valid_caps.jsonl"


python process_raw_caps.py \
 --folder ${INPUT_PATH} \
 --name_format "test_caps_*.jsonl" \
 --output_file "${INPUT_PATH}/test_caps.jsonl"