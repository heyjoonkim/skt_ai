
import os
from typing import List, Dict

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from vllm import LLM, SamplingParams
from huggingface_hub import login

## Transformer 모델 로드 ##

def _load_transformers_model(model_name:str=None, cache_path:str=None, quantization_config:BitsAndBytesConfig=None, train:bool=False):

    token = os.environ['HF_TOKEN']
    login(token=token)
    
    # device_map = 'auto' if quantization_config is None else None
    # Deepspeed를 쓰면 device_map='auto'하면 오류남. 아래 링크에서 해결책.
    # REF : https://github.com/huggingface/trl/issues/3311
    device_map = 'auto' if not train else None

    model = AutoModelForCausalLM.from_pretrained(
                                            model_name,
                                            cache_dir=cache_path,
                                            device_map=device_map,
                                            quantization_config=quantization_config,
                                        ).eval()
    
    return model
    
    
def _load_transformers_tokenizer(model_name:str=None, cache_path:str=None):

    token = os.environ['HF_TOKEN']
    login(token=token)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path)
    special_tokens_map = tokenizer.special_tokens_map
    assert special_tokens_map.get('eos_token', None) is not None
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer

def load_transformers_model_and_tokenizer(model_name:str=None, 
                                          cache_path:str=None, 
                                          load_model:bool=True, 
                                          quantization_config:BitsAndBytesConfig=None,
                                          train:bool=False) -> List:
    
    model = _load_transformers_model(model_name=model_name, cache_path=cache_path, quantization_config=quantization_config, train=train) if load_model else None
    tokenizer = _load_transformers_tokenizer(model_name=model_name, cache_path=cache_path)
    
    if not load_model:
        assert model is None, 'Model should be None if load_model is False'
    
    return model, tokenizer

    



## vLLM 모델 로드 ##
## LOAD MODEL AND TOKENIZER via VLLM ##
def load_vllm_model_and_tokenizer(model_name:str=None, 
                                  trained_path:str=None,
                                  cache_path:str=None,
                                  tensor_parallel_size:int=1, 
                                  use_lora:bool=False,) -> List:
    
    if trained_path is None:
        if use_lora:
            model = LLM(model=model_name, download_dir=cache_path, tensor_parallel_size=tensor_parallel_size, enforce_eager=True, enable_lora=True)
        else:
            model = LLM(model=model_name, download_dir=cache_path, tensor_parallel_size=tensor_parallel_size, enforce_eager=True)
    else:
        if use_lora:
            # load trained model
            model = LLM(model=trained_path, tokenizer=model_name, tensor_parallel_size=tensor_parallel_size, enforce_eager=True, enable_lora=True)
        else:
            # load trained model
            model = LLM(model=trained_path, tokenizer=model_name, tensor_parallel_size=tensor_parallel_size, enforce_eager=True)

    tokenizer = _load_transformers_tokenizer(model_name=model_name, cache_path=cache_path)
    

    return model, tokenizer


def get_vllm_param(args, 
                   do_sampling:bool=True,
                   num_generations:int=1,
                   temperature:float=0.0,
                   top_p:float=1.0,
                   max_tokens:int=3000) -> Dict:
    
    greedy_params = SamplingParams(n=1, temperature=0, max_tokens=args.generation.max_new_tokens)
        
    if do_sampling:
        sampling_params = SamplingParams(
                            n=num_generations, 
                            temperature=temperature, 
                            max_tokens=max_tokens,
                            top_p=top_p,)
    else:
        sampling_params = None
    
    return dict(
        greedy=greedy_params,
        sampling=sampling_params
    )