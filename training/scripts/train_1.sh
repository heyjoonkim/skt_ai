# export CUDA_VISIBLE_DEVICES=6,7
# num_processes=2

export CUDA_VISIBLE_DEVICES=0
num_processes=1

model='meta-llama/Llama-3.2-1B-Instruct'

output_dir='/home/heyjoonkim/data/skt_ai'

accumulation=8
batch_size=1

# num_epochs='1 2 3 5 10'
num_epochs='1 2 3 5'
num_epochs='1 2'
lrs='5e-4 1e-4 5e-5 1e-5'



# transformers 4.51.3

for num_epoch in $num_epochs; do
    for lr in $lrs; do 

        accelerate launch \
            --main_process_port 29502 \
            --num_processes $num_processes \
            --num_machines 1 \
            --dynamo_backend no \
            --mixed_precision bf16 \
            -m train.train \
            --seed 1234 \
            --tf32 True \
            --bf16 True \
            --model_name_or_path $model \
            --model_cache HF_MODEL_CACHE \
            --output_dir $output_dir \
            --gradient_checkpointing True \
            --num_train_epochs $num_epoch \
            --per_device_train_batch_size $batch_size \
            --gradient_accumulation_steps $accumulation \
            --learning_rate $lr \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --save_strategy "no" \
            --save_steps 30000 \
            --save_total_limit 1 \
            --logging_steps 1 \
            --report_to wandb \
            --lora_r 8 \
            --lora_alpha 16 \
            --lora_dropout 0.1 \
            --lora_bias "none" \
            --lora_task_type "CAUSAL_LM" \
            --lora_target_modules "q_proj" "k_proj" "v_proj" "o_proj" "gate_proj" "up_proj" "down_proj" \
            --use_qlora True \
            --optim "adamw_bnb_8bit" \
            --gradient_checkpointing_kwargs '{"use_reentrant": false}'

            # --optim "adamw_torch" \
            

            curl -d "Done SFT training: epoch=$num_epoch, lr=$lr, batch_size=$batch_size (LoRA alpha=$lora_alpha, r=$lora_r)" ntfy.sh/hjkim-experiments

    done        
done       

