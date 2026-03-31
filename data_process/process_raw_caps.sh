
INPUT_PATH="../ETTh1"
#OUT_PATH="../ETTh1/train_caps_ready.jsonl"

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



python merge_caps_all_channels.py \
  --input "${INPUT_PATH}/train_caps.jsonl" \
  --output "${INPUT_PATH}/train_caps_ready.jsonl"

python merge_caps_all_channels.py \
  --input "${INPUT_PATH}/valid_caps.jsonl" \
  --output "${INPUT_PATH}/valid_caps_ready.jsonl"


python merge_caps_all_channels.py \
  --input "${INPUT_PATH}/test_caps.jsonl" \
  --output "${INPUT_PATH}/test_caps_ready.jsonl"