
from pdb import set_trace

from typing import Dict, List

from datasets import Dataset
from vllm import SamplingParams, LLM
from transformers import GenerationConfig


# BAD_WORDS = ['\n', '\t', '\r']
BAD_WORDS = []


def get_vllm_param(num_generations:int, max_new_tokens:int, temperature:float, top_p:float, top_k:int, do_sampling:bool=True) -> Dict:
    # n, top_p, tok_k, temperature, stop_token_ids, max_tokens, 
    # greedy_params = SamplingParams(n=1, temperature=1.0, stop=BAD_WORDS, max_tokens=max_new_tokens)
    
    # temperature: Float that controls the randomness of the sampling. Lower
    #               values make the model more deterministic, while higher values make
    #               the model more random. Zero means greedy sampling.
    greedy_params = SamplingParams(n=1, temperature=0.0, stop=BAD_WORDS, max_tokens=max_new_tokens)
      
    if do_sampling:
        sampling_params = SamplingParams(
                            n=num_generations, 
                            temperature=temperature, 
                            # top_p=top_p,
                            # top_k=top_k,
                            stop=BAD_WORDS, 
                            max_tokens=max_new_tokens)
    else:
        sampling_params = None
    
    return dict(
        greedy=greedy_params,
        sampling=sampling_params
    )
    

def get_hf_param(num_generations:int=1, 
                 max_new_tokens:int=30, 
                 temperature:float=0.7, 
                 bos_token_id:int=None,
                 eos_token_id:int=None,
                 pad_token_id:int=None,
                 top_p:float=1.0, 
                 top_k:int=50,  
                 do_sampling:bool=True,) -> Dict:
    # greedy
    greedy_params = GenerationConfig(
        max_new_tokens=max_new_tokens,
        stop_strings=BAD_WORDS,
        do_sample=False,
        num_beams=1,
        temperature=1.0,
        output_logits=True,
        return_dict_in_generate=True,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
    )
        
    if do_sampling:
        sampling_params = GenerationConfig(
            num_return_sequences=num_generations,
            max_new_tokens=max_new_tokens,
            stop_strings=BAD_WORDS,
            do_sample=True,
            num_beams=1,
            temperature=temperature,
            output_logits=True,
            return_dict_in_generate=True,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
        )
    else:
        sampling_params = None
    
    return dict(
        greedy=greedy_params,
        sampling=sampling_params,
    )
    

 