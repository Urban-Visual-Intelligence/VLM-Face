export DEBUG_MODE="true"

RUN_NAME="Qwen2.5-VL-7B-GRPO-Face_0305"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_face_cot.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct\
    --dataset_name face_0311 \
    --dataset_config_path data_config/face_0311.json \
    --max_prompt_length 4096 \
    --max_completion_length 512 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --beta 0.01 \
    --bf16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 4 \
    --run_name $RUN_NAME \
    --save_steps 200 \
    --max_pixels 802816 \
    --save_only_model True
