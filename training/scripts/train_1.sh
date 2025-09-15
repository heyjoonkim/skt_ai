# export CUDA_VISIBLE_DEVICES=6,7
# num_processes=2

export CUDA_VISIBLE_DEVICES=6
num_processes=1

model='Qwen/Qwen2.5-7B-Instruct'

output_dir='/home/heyjoonkim/data/reasoning_abstention'

accumulation=4
batch_size=4

# num_epochs='1 2 3 5 10'
num_epochs='1 2'


lrs='1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6 5e-7 1e-7'
# lrs='5e-3 1e-3 5e-4'
# lrs='1e-4 5e-5 1e-5'
# lrs='5e-6 1e-6 5e-7'

lora_r='64'

for num_epoch in $num_epochs; do
    for lr in $lrs; do 

        lora_alpha=$(($lora_r*2))

        accelerate launch \
            --main_process_port 29502 \
            --num_processes $num_processes \
            --num_machines 1 \
            --dynamo_backend no \
            --mixed_precision bf16 \
            --use_deepspeed \
            --deepspeed_config_file /home/heyjoonkim/reasoning_abstention/configs/deepspeed/zero.json \
            -m train.train \
            --seed 1234 \
            --tf32 True \
            --bf16 True \
            --model_name_or_path $model \
            --model_cache HF_MODEL_CACHE \
            --output_dir $output_dir \
            --dataset_name 'heyjoonkim/hendrycks_math' \
            --inference_template_id 0 \
            --num_demonstrations 2 \
            --gradient_checkpointing True \
            --num_train_epochs $num_epoch \
            --per_device_train_batch_size $batch_size \
            --gradient_accumulation_steps $accumulation \
            --learning_rate $lr \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --optim "adamw_torch" \
            --lr_scheduler_type "cosine" \
            --save_strategy "no" \
            --save_steps 30000 \
            --save_total_limit 1 \
            --logging_steps 1 \
            --report_to wandb \
            --ddp_find_unused_parameters False \
            --lora_r $lora_r \
            --lora_alpha $lora_alpha \
            --lora_dropout 0.1 \
            --lora_bias "none" \
            --lora_task_type "CAUSAL_LM" \
            --lora_target_modules "q_proj" "v_proj" \
            --use_qlora False

            curl -d "Done SFT training: epoch=$num_epoch, lr=$lr, batch_size=$batch_size (LoRA alpha=$lora_alpha, r=$lora_r)" ntfy.sh/hjkim-experiments

    done        
done       

