# Model settings
MODEL_NAME="Qwen/Qwen-VL-Chat-Int4"  # Using the Int4 quantized model as recommended
OUTPUT_DIR="output_qlora_longer"

# Data settings
DATA_PATH="/content/Zeus/training.json"

# Training settings
deepspeed --num_gpus 1 finetune.py \
    --model_name_or_path $MODEL_NAME \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --model_max_length 1024 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --use_lora \
    --q_lora \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fp16 \
    --gradient_checkpointing \
    --deepspeed ds_config_longer.json \
    --report_to none
