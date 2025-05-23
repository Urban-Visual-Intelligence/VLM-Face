#!/bin/bash
result_folder_path="/mnt/afs/machengqian/mllm/qwen2vl/results"
result_file_path="qwen2_5vl_7b_sft_1_grpo_bs16_lr_9e7_0327_direct_t_0_1000"
api="http://localhost:8000/v1"
cot=1
temp=0

# inference
# Remember to change the model path, image root, and annotation path in the script
python utils/inference_vllm_lfw.py --result "$result_folder_path/lfw_$result_file_path.json" --api $api --cot $cot --temp $temp &
python utils/inference_vllm_cp_lfw.py --result "$result_folder_path/cplfw_$result_file_path.json" --api $api --cot $cot --temp $temp &
python utils/inference_vllm_age30.py --result "$result_folder_path/age30_$result_file_path.json" --api $api --cot $cot --temp $temp 

# calculate metrics
python utils/calculate_acc_cot_args.py --result "$result_folder_path/lfw_$result_file_path.json"
python utils/calculate_acc_cot_args.py --result "$result_folder_path/cplfw_$result_file_path.json"
python utils/calculate_acc_cot_args.py --result "$result_folder_path/age30_$result_file_path.json"
