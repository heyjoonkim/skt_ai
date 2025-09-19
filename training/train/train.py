import os
import logging
from time import time
from typing import Optional, List
from dataclasses import dataclass, field

import torch
from transformers import HfArgumentParser, Trainer, TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
# from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
# from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model


from utils import seed_everything, logger_init, save_pkl, load_pkl
from models import _load_transformers_model, load_transformers_model_and_tokenizer

logger = logging.getLogger(__name__)


####################################
#                                  #
# PARAMETER ARGUMENTS FOR TRAINING #
#                                  #
####################################

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-7B-Instruct")
    offload_dir: Optional[str] = field(default=None)
    model_cache: Optional[str] = field(default='HF_MODEL_CACHE')

@dataclass
class DataArguments:
    dataset_name: str = field(default='heyjoonkim/hendrycks_math')

# @dataclass
# # class CustomTrainingArguments(SFTConfig):
# class CustomTrainingArguments(TrainingArguments):
#     output_dir: Optional[str] = field(default='/data1/heyjoonkim/reasoning_abstention')
#     optim: str = field(default="adamw_torch")
#     logging_level: Optional[str] = field(default='INFO')
#     # 원래 코드에 default 값이 있는데, 길이가 길어질 수 있으므로 truncate하지 않도록 None으로 설정.
#     max_prompt_length: int = field(default=None)
#     max_completion_length: int = field(default=None)
#     max_length: int = field(default=None)
#     # completion_only_loss: bool = field(default=True)
    
@dataclass
class CustomTrainingArguments(TrainingArguments):
    # override
    #   output_dir
    #   per_device_train_batch_size, per_device_eval_batch_size, gradient_accumulation_steps
    #   learning_rate, weight_decay, num_train_epochs
    #   lr_scheduler_type, warmup_ratio
    #   logging_steps
    #   save_strategy, save_steps, save_total_limit
    #   seed
    #   fp16
    cache_dir: Optional[str] = field(default=None)
    # select from : adamw_hf, adamw_torch, adamw_torch_fused, adamw_apex_fused, adamw_anyprecision or adafactor.
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=100000,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    remove_unused_columns: bool = field(default=False)
    logging_level: Optional[str] = field(default='INFO')
    
    
@dataclass
class LoraArguments:
    lora_task_type: str = field(default='CAUSAL_LM')
    inference_mode: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    lora_target_modules: List[str] = field(default=None)
    use_qlora: bool = field(default=False)
     

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    
    random_seed = training_args.seed   

    seed_everything(random_seed)
    
    cache_key = model_args.model_cache
    cache_dir = os.environ.get(cache_key)
    
    model_name = model_args.model_name_or_path
    model_name_for_dir = model_name.replace('/', '-')
    
    config_str = f'{model_name_for_dir}-{str(random_seed)}'    
    
    ## BASE PATH for outputs
    base_path = os.path.join(training_args.output_dir, config_str)
    assert os.path.isdir(base_path), f'Base path does not exist : {base_path}'

    ############################
    #                          #
    #       LOAD DATASET       #
    #                          #
    ############################
    
    PROMPT_INDEX = 4
    
    # selected_data_file = os.path.join(base_path, f'final_train_data_subset_{PROMPT_INDEX}.pkl')
    selected_data_file = os.path.join(base_path, 'final_train_data.pkl')
        
    data_module = load_pkl(path=selected_data_file)
    
    # TODO : for debugging. remove later.
    # train_data = train_data[:1]
        
    logger.info(f'Loaded data module : {selected_data_file}')


    ###########################################
    #                                         #
    # TRAINING STAGE                          #
    # - train model with new annotated labels #
    #                                         #
    ###########################################

    
    ## SET OUTPUT DIRECTORY ##
    TRAINING_CONFIG_STR = f'PROMPT-{PROMPT_INDEX}_epoch-{int(training_args.num_train_epochs)}_batch-{training_args.per_device_train_batch_size}_accumulation-{training_args.gradient_accumulation_steps}_lr-{training_args.learning_rate}'
    
    # set wandb run name
    training_args.run_name = f'SKT_AI_{TRAINING_CONFIG_STR}'
    
    model_checkpoint_path = os.path.join(base_path, TRAINING_CONFIG_STR)
    
    if not os.path.isdir(model_checkpoint_path):
        logger.info(f'Checkpoint path does not exist. Train model. Generate output directory: {model_checkpoint_path}')
        os.makedirs(model_checkpoint_path, exist_ok=True)
    
    ## INIT LOGGER
    logger_level = training_args.logging_level
    logger_init(logger, model_checkpoint_path, logger_level=logger_level, save_as_file=True)
    
    # check if checkpoint already exists
    logger.info(f'Checkpoint path already exists. Check if checkpoint exists. Path: {model_checkpoint_path}')
    dir_list = os.listdir(model_checkpoint_path)
    for file_name in dir_list:
        if file_name.endswith('.bin') or file_name.endswith('.safetensors'):
            logger.info(f'Checkpoint already exists. Skip training. File: {file_name}')
            return


    ## save config ## 
    training_config_file = os.path.join(model_checkpoint_path, 'training_config.pkl')
    save_pkl(data=[model_args, data_args, training_args, lora_args], path=training_config_file)
    
    
    # ## Train model to learn ambiguity ##
    # ## Initialize PEFT Configs ##
    # lora_config = None
    lora_config = LoraConfig(
                        r=lora_args.lora_r,
                        lora_alpha=lora_args.lora_alpha,
                        lora_dropout=lora_args.lora_dropout,
                        bias=lora_args.lora_bias,
                        inference_mode=False,
                        task_type=TaskType.CAUSAL_LM,
                        target_modules=lora_args.lora_target_modules,
                    )

    # QLoRA
    if lora_args.use_qlora:
        logger.info('** Use QLoRa quantization.')
        quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.bfloat16
                            )
    else:
        logger.info('** No QLoRa quantization.')
        quantization_config = None
    
    logger.info(f'Load model : {model_args.model_name_or_path}')
    ## LOAD MODEL and TOKENIZER ##

    model = _load_transformers_model(
                                    model_name=model_args.model_name_or_path,
                                    cache_path=cache_dir,
                                    quantization_config=quantization_config,
                                    train=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
    except:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", trust_remote_code=True, padding_side="right")

    # model, tokenizer = load_transformers_model_and_tokenizer(
    #     model_name=model_args.model_name_or_path,
    #     cache_path=cache_dir,
    #     load_model=True,
    #     quantization_config=quantization_config,
    #     train=True)
    

    model.config.use_cache = False  # Disable cache for DPO training
    
    
    if model is not None:
        logger.info('** Add PEFT parameters.')
        model.config.use_cache=False
        model.config.attn_implementation = "flash_attention_2"

        
        # model.gradient_checkpointing_enable()  # enable gradient checkpointing
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # from : https://github.com/huggingface/peft/issues/137
        model.enable_input_require_grads()

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        
    # set logs
    training_args.logging_dir = os.path.join(model_checkpoint_path, 'logs')
    # training_args.dataset_kwargs={'skip_prepare_dataset':True}
    
    training_args.output_dir = model_checkpoint_path
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    
    # trainer.model.print_trainable_parameters()
    
    
    logger.info('Start training.')
    start_time = time()
    trainer.train()
    end_time = time()
    logger.info(f'Training time : {end_time - start_time} seconds.')
    
    final_model = trainer.model
    final_model = final_model.merge_and_unload()
    
    logger.info('Save final model...')
    start_time = time()
    final_model.save_pretrained(model_checkpoint_path)
    end_time = time()
    logger.info(f'Save time : {end_time - start_time} seconds.')
    
    
    logger.info(f'Done.')


if __name__ == "__main__":
    train()