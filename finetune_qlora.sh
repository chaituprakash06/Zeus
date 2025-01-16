# Model settings
MODEL_NAME="Qwen/Qwen-VL-Chat-Int4"  # Using the Int4 quantized model as recommended
OUTPUT_DIR="output_qlora"

# Data settings
DATA_PATH="/content/Zeus/training.json"

# Training settings
deepspeed --num_gpus 1 finetune.py \
    --model_name_or_path $MODEL_NAME \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --use_lora \
    --q_lora \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed ds_config.json