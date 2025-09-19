
import logging
import os
from time import time
from typing import Dict, List
from random import randint

import random
import hydra
from tqdm import tqdm
from omegaconf import DictConfig
from datasets import load_dataset

from utils import seed_everything, logger_init, save_pkl, load_pkl
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from models import load_transformers_model_and_tokenizer
from train.preprocess import make_supervised_data_module

from inference import (
    PROMPT, 
    SYSTEM_MESSAGE
)

logger = logging.getLogger(__name__)

# TODO : config 파일 경로 설정해주기
@hydra.main(config_path='../configs/', config_name='train.yaml', version_base=None)
def main(args: DictConfig) -> None:
    
    random_seed = args.seed
    # set random seed
    seed_everything(random_seed)

    ###########################################################################################################
    #                                                                                                         #
    # SET BASE PATHS : 이거 그냥 내가 학습된 결과물 저장할 위치 임의로 정의하는 코드임. 환경에 맞게 알아서 바꾸도됨. #
    #                                                                                                         #
    ###########################################################################################################

    model_name = args.model.name
    model_name_for_dir = model_name.replace('/', '-')

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
    except:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", trust_remote_code=True, padding_side="right")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # _, tokenizer = load_transformers_model_and_tokenizer(
    #                                         model_name=model_name,
    #                                         load_model=False,)
    
    config_str = f'{model_name_for_dir}-{str(random_seed)}'    
    output_dir = os.path.join(args.path.output, config_str)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    ## INIT LOGGER
    logger_level = args.logging_level
    logger_init(logger, output_dir=output_dir, logger_level=logger_level, save_as_file=True)
    
    
    ######################################
    ##                                  ##
    ##              STEP 1              ##
    ##                                  ##
    ######################################
    
    # 최종 결과가 있으면 실험 X
    output_filename = os.path.join(output_dir, 'train_data.pkl')
    
    logger.info(f'Initial prediction file : {output_filename}')
    if not os.path.isfile(output_filename):
        logger.info(f'Initial prediction file does not exists : {output_filename}')
           
        # 우리가 preprocess에서 저장한 파일 불러오기
        train_data_path = args.dataset.train
        dataset = load_dataset('json', data_files=train_data_path)
        
        # use train set
        dataset = None if "train" not in dataset else dataset['train']
        assert dataset is not None, f'No train set.'
        
        num_test_samples = len(dataset)
        logger.info(f'Loaded test dataset > train ({num_test_samples})')
        
        
        sft_sample_list = list()
        
        for sample in tqdm(dataset, total=len(dataset), desc='Formatting inputs'):
            # for sample in dataset:
            query = sample.get('query')
            function_call = sample.get('function_call')

            if query is None:
                continue

            if function_call is None:
                continue
                                    
            inference_str = PROMPT.format(query=query)
            
            prompt = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": inference_str},
            ]
            completion = [
                {"role": "assistant", "content": "Function : " + function_call},
            ]
                        
            # TODO : step이 이상하게 잘리는 문제 해결하기
            # Step 3: Substitute the expression for \( n \) from Step 1 into the equation from Step 2:
            sft_sample = dict(prompt=prompt, completion=completion, label=function_call) 
            sft_sample_list.append(sft_sample)
            
            
        # TODO : dataset으로 저장하려면 이렇게.
        # from datasets import Dataset
        # dataset = Dataset.from_list(dpo_sample_list)
        # save_pkl(data=dataset, path=output_filename)
        
        save_pkl(data=sft_sample_list, path=output_filename)   
    
    logger.info(f'Load : {output_filename}')
    sft_sample_list = load_pkl(path=output_filename)
    output_filename = os.path.join(output_dir, 'final_train_data.pkl')
    data_module = make_supervised_data_module(tokenizer=tokenizer, dataset=sft_sample_list)
    save_pkl(data=data_module, path=output_filename)

                
                
    logger.info(f'Trian data : {len(sft_sample_list)}')
    logger.info('Done.')
    
    
    
if __name__ == '__main__':
    try:
        start = time()
        main()
        end = time()
        print(f'Total time : {end - start} seconds')
        
    except Exception as error:
        os.system(f'curl -d "Error running preliminary_training/main.py. Error : {error}"')
        