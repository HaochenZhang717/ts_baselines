export HF_HOME=/playpen/haochenz/hf_cache
export CUDA_VISIBLE_DEVICES=1
python run_caption.py --part_id 0 --num_parts 8 --image_folder "/playpen-shared/haochenz/LitsDatasets/128_len_img_one_per_image_0324/synth_u/train" --split "train" --save_dir "/playpen-shared/haochenz/LitsDatasets/128_len_caps_one_per_image_0324/synth_u" --dataset_name "synth_u"
#python run_caption.py --part_id 0 --num_parts 8 --image_folder "/playpen-shared/haochenz/LitsDatasets/128_len_img_one_per_image_0324/synth_u/valid" --split "valid" --save_dir "/playpen-shared/haochenz/LitsDatasets/128_len_caps_one_per_image_0324/synth_u" --dataset_name "synth_u"
#python run_caption.py --part_id 0 --num_parts 8 --image_folder "/playpen-shared/haochenz/LitsDatasets/128_len_img_one_per_image_0324/synth_u/test" --split "test" --save_dir "/playpen-shared/haochenz/LitsDatasets/128_len_caps_one_per_image_0324/synth_u" --dataset_name "synth_u"
