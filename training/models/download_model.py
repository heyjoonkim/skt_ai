
## 큰 모델 그냥 다운받게 돌려두기 위한 코드   ##
## tmux 같은데 그냥 코드 돌려두고 다른거 하기 ##

import os
from time import time

from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_cache_path = os.environ['HF_MODEL_CACHE']

    # model_name = 'deepseek-ai/deepseek-math-7b-instruct'
    model_name = 'meta-llama/Llama-3.1-8B-Instruct'


    print(f'Downloading model {model_name} to {model_cache_path}...')


    start_time = time()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=model_cache_path,
        device_map='auto',
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    end_time = time()
    total_time = end_time - start_time
    print(f'Model downloaded in {total_time:.2f} seconds.')
    print('Done.')



if __name__ == '__main__':
    
    main()
        