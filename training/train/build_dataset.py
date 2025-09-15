
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


from models import load_vllm_model_and_tokenizer, get_vllm_param
from evaluation import MathEvaluator, StepsEvaluator
from utils import seed_everything, logger_init, save_pkl, load_pkl

from inference import (
    make_inference_template,
    get_demonstrations_str,
    DATASET_TO_SYS_MESSAGE,
    
    PROMPT, 
    SYSTEM_MESSAGE
)

logger = logging.getLogger(__name__)

  
        
validation_system_message = "The following is a full solution process for a given question. Based on the entire solution and the original question, evaluate whether Step {index} in the solution is logically valid and consistent. Your evaluation should focus only on the correctness of the specified step in the context of the whole solution."
validation_suffix = 'Is Step {index} logically valid? Answer only with "Correct" or "Incorrect".'
query_template = "Question: {question}\n"
steps_template = "Step {index}: {step}\n"




# 입력 형식 맞춰서 만들어주기.
def format_input(question:str, steps:List[str], index:int) -> List[Dict]:
    validation_system_message_str = validation_system_message.format(index=index+1)
    query_template_str = query_template.format(question=question)
    steps_str = "Solution:\n"
    for step_index, step in enumerate(steps):
        steps_str += steps_template.format(index=step_index+1, step=step)
        
    target_step = f'Target Step:\nStep {index+1}: {steps[index]}\n'
    validation_suffix_str = validation_suffix.format(index=index+1)
    
    final_user_msg = f'{query_template_str}\n{steps_str}\n{target_step}\n{validation_suffix_str}'
        
    messages = [
        {"role": "system", "content": validation_system_message_str},
        {"role": "user", "content": final_user_msg},
    ]
                
    return messages
  
# TODO : config 파일 경로 설정해주기
@hydra.main(config_path='../../configs/train/', config_name='qwen_2.5_7b_instruct.yaml', version_base=None)
def main(args: DictConfig) -> None:
    
    model = None
    tokenizer = None
    generation_param = None
    demonstrations = None
    system_message = None

    random_seed = args.seed
    # set random seed
    seed_everything(random_seed)

    ##################
    #                #
    # SET BASE PATHS #
    #                #
    ##################

    model_name = args.model.name
    model_name_for_dir = model_name.replace('/', '-')
    
    dataset_name = args.dataset.train
    dataset_name_for_dir = dataset_name.replace('/', '-')
    
    inference_template_id = args.generation.template_id
    num_demonstrations = args.generation.num_demonstrations
    
    config_str = f'{model_name_for_dir}-{dataset_name_for_dir}-template_{inference_template_id}-demo_{num_demonstrations}-{str(random_seed)}'    
    output_dir = os.path.join(args.path.output, 'train', config_str, 'sft')
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
    output_filename = os.path.join(output_dir, '1_initial_results.pkl')
    
    logger.info(f'Initial prediction file : {output_filename}')
    if not os.path.isfile(output_filename):
        logger.info(f'Initial prediction file does not exists : {output_filename}')
           
        ## load dataset
        dataset_cache = args.environ.dataset_cache
        # 우리가 preprocess에서 저장한 파일 불러오기
        dataset = load_dataset(dataset_name, cache_dir=dataset_cache)
        
        # use train set
        dataset = None if "test" not in dataset else dataset['test']
        assert dataset is not None, f'Dataset {dataset_name} does not have train set.'
        
        num_test_samples = len(dataset)
        logger.info(f'Loaded test dataset ({dataset_name}) > train ({num_test_samples})')
        
        if model is None:
            model, tokenizer = load_vllm_model_and_tokenizer(model_name=model_name, tensor_parallel_size=args.tensor_parallel_size)
        
        
        sft_sample_list = list()
        
        for sample in tqdm(dataset, total=len(dataset), desc='Formatting inputs'):
            # for sample in dataset:
            question = sample.get('question')
            function_call = sample.get('answer')
                                    
            inference_str = PROMPT.format(query=question)
            
            prompt = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": inference_str},
            ]
            completion = [
                {"role": "assistant", "content": function_call},
            ]
            
            
            # TODO : step이 이상하게 잘리는 문제 해결하기
            # Step 3: Substitute the expression for \( n \) from Step 1 into the equation from Step 2:
            sft_sample = dict(prompt=prompt, completion=completion) 
            sft_sample_list.append(sft_sample)
            
            
        # TODO : dataset으로 저장하려면 이렇게.
        # from datasets import Dataset
        # dataset = Dataset.from_list(dpo_sample_list)
        # save_pkl(data=dataset, path=output_filename)
        
        save_pkl(data=sft_sample_list, path=output_filename)   
                
    logger.info('Done.')
    
    
    
if __name__ == '__main__':
    try:
        start = time()
        main()
        end = time()
        print(f'Total time : {end - start} seconds')
        
    except Exception as error:
        os.system(f'curl -d "Error running preliminary_training/main.py. Error : {error}" ntfy.sh/hjkim-experiments')
        