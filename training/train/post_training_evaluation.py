
import logging
import os
from time import time

import hydra
from tqdm import tqdm
from omegaconf import DictConfig
from datasets import load_dataset


from models import load_vllm_model_and_tokenizer, get_vllm_param
from evaluation import MathEvaluator, StepsEvaluator, FeedbackRequestDetector
from utils import seed_everything, logger_init, save_pkl, load_pkl

from inference import (
    make_inference_template,
    get_demonstrations_str,
    DATASET_TO_SYS_MESSAGE,
)

logger = logging.getLogger(__name__)

  
@hydra.main(config_path='../../configs/train/', config_name='qwen_2.5_7b_instruct.yaml', version_base=None)
def main(args: DictConfig) -> None:
    
    model = None
    tokenizer = None
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
    
    train_dataset_name = args.dataset.train
    dataset_name_for_dir = train_dataset_name.replace('/', '-')
    
    inference_template_id = args.generation.template_id
    num_demonstrations = args.generation.num_demonstrations
    
    config_str = f'{model_name_for_dir}-{dataset_name_for_dir}-template_{inference_template_id}-demo_{num_demonstrations}-{str(random_seed)}'    
    base_dir = os.path.join(args.path.output, 'train', config_str, 'sft')
    assert os.path.isdir(base_dir), f'Base path does not exist : {base_dir}'
    
    
    r=args.training.lora.r
    lora_alpha=args.training.lora.alpha
    
    TRAINING_CONFIG_STR = f'epoch-{int(args.training.num_training_epochs)}_batch-{args.training.per_device_train_batch_size}_accumulation-{args.training.gradient_accumulation_steps}_lr-{args.training.learning_rate}_lora_r-{r}_lora_alpha-{lora_alpha}'

    is_balanced = args.test.is_balanced
    if is_balanced:
        TRAINING_CONFIG_STR = f'balanced_{TRAINING_CONFIG_STR}'
        
    if args.training.use_qlora:
        TRAINING_CONFIG_STR = f'_qlora{TRAINING_CONFIG_STR}'
    
    
    output_dir = os.path.join(base_dir, TRAINING_CONFIG_STR)
    assert os.path.isdir(output_dir), f'Output path does not exist : {output_dir}'
    
    ## INIT LOGGER
    logger_level = args.logging_level
    logger_init(logger, output_dir=output_dir, logger_level=logger_level, save_as_file=True)
    
    
    ## LOAD DATASET ##
    is_test = args.test.is_test
    dataset_cache = args.environ.dataset_cache
    evaluation_dataset_name_for_dir = None
    
    
    if is_test:
        logger.info('Final evaluation on test set...')
        test_dataset_name = args.test.test_dataset
        evaluation_dataset_name_for_dir = test_dataset_name.replace('/', '-')
        
        dataset = load_dataset(test_dataset_name, cache_dir=dataset_cache)
        assert 'test' in dataset.keys(), f'Test set not found in {test_dataset_name}. Available keys: {dataset.keys()}'
        
        dataset = dataset['test']
        logger.info(f'Loaded test dataset ({test_dataset_name}) : {len(dataset)} samples')
        
        generation_filename = os.path.join(output_dir, f'test_{evaluation_dataset_name_for_dir}.pkl')
        results_filename = os.path.join(output_dir, f'test_{evaluation_dataset_name_for_dir}_results.pkl')
    else:
        logger.info(f'Evaluation on validation set for hyperparameter tuning...')
        validation_dataset_name = args.test.validation_dataset
        evaluation_dataset_name_for_dir = validation_dataset_name.replace('/', '-')
        
        dataset = load_dataset(validation_dataset_name, cache_dir=dataset_cache)
        assert 'validation' in dataset.keys(), f'Validation set not found in {validation_dataset_name}. Available keys: {dataset.keys()}'
        
        dataset = dataset['validation']
        logger.info(f'Loaded validation dataset ({validation_dataset_name}) : {len(dataset)} samples')
        
        generation_filename = os.path.join(output_dir, f'validation_{evaluation_dataset_name_for_dir}.pkl')
        results_filename = os.path.join(output_dir, f'validation_{evaluation_dataset_name_for_dir}_results.pkl')
      
    #########################################
    ##                                     ##
    ##              LOAD vLLM              ##
    ##                                     ##
    #########################################
    
    if not os.path.isfile(generation_filename):
        logger.info(f'Result does not exist : {generation_filename}')
        
        # model, tokenizer = load_vllm_model_and_tokenizer(model_name=model_name, tensor_parallel_size=args.tensor_parallel_size, use_lora=True)
        model, tokenizer = load_vllm_model_and_tokenizer(model_name=model_name, trained_path=output_dir, tensor_parallel_size=args.tensor_parallel_size)
        
        generation_param_dict = get_vllm_param(args, do_sampling=False)
        greedy_param = generation_param_dict.get('greedy')
        
        
        demonstrations = get_demonstrations_str(dataset_name=train_dataset_name, 
                                                num_demonstration=args.generation.num_demonstrations,
                                                template_id=args.generation.template_id)
        system_message = DATASET_TO_SYS_MESSAGE.get(train_dataset_name)
        
        
        inference_str_list = list()
        for sample in tqdm(dataset, total=len(dataset), desc='Formatting inputs'):
        # for sample in dataset:
            question = sample.get('question')
                                    
            inference_str = make_inference_template(dataset_name=train_dataset_name, 
                                                    demonstration_str=demonstrations,
                                                    query=question, 
                                                    template_id=args.generation.template_id)
        
        
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": inference_str},
            ]
            
            formatted_inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,)
            inference_str_list.append(formatted_inputs)
            
        # generations = model.generate(inference_str_list, greedy_param, lora_request=lora_request,)
        generations = model.generate(inference_str_list, greedy_param)
        
        results = list()
        for generation_class, sample in tqdm(zip(generations, dataset), total=len(dataset), desc='Selecting Generations...'):

            prediction = generation_class.outputs[0].text
            sample['full_prediction'] = prediction
            results.append(sample)

        assert 'full_prediction' in results[0].keys(), f'No full_prediction found in {dataset[0].keys()}'
        
        logger.info(f'Save results to : {generation_filename}')
        save_pkl(data=results, path=generation_filename)
    
    results = load_pkl(path=generation_filename)


    #################################
    #                               #
    #       Evaluation Code         #
    #                               #
    #################################
    
    
    if not os.path.isfile(results_filename):
        logger.info(f'Evaluation results does not exist: {results_filename}')
        
        evaluator = MathEvaluator(logger=logger)
        steps_evaluator = StepsEvaluator(logger=logger)    
        detector = FeedbackRequestDetector(logger=logger)
        
        # evaluate correctness
        # new keys : exact_match, soft_exact_match, answer_format
        evaluator.update(results=results)
        evaluated_results = evaluator.evaluate_all()
        
        print('Accuracy Results...')
        evaluator.print_all_results()
        
        # evaluate step generations
        # new keys : has_step, steps
        for sample in evaluated_results:
            full_prediction = sample.get('full_prediction')
            steps_res_dict = steps_evaluator.get_steps(full_prediction, verbose=False)
            sample.update(steps_res_dict)
        steps_evaluator.print_stat()
    
        # evaluate correctness
        # new keys : has_feedback_request
        detector.update(results=evaluated_results)
        final_results = detector.evaluate_all()
        print('Final Results...')
        detector.print_all_results()
        
        save_pkl(data=final_results, path=results_filename)
        
    final_results = load_pkl(results_filename)
    
    
    correct_count = 0
    correct_with_step = 0
    feedback_count = 0
    feedback_with_step = 0
    incorrect_count = 0
    incorrect_with_step = 0
    total_count = len(final_results)
    
    for result in final_results:
        is_correct = result.get('exact_match')
        has_steps = result.get('has_step')
        
        if is_correct:
            correct_count += 1
            if has_steps:
                correct_with_step += 1
        else:
            is_feedback_request = result.get('has_feedback_request')
            if is_feedback_request:
                feedback_count += 1
                if has_steps:
                    feedback_with_step += 1
            else:
                incorrect_count += 1
                if has_steps:
                    incorrect_with_step += 1
    
    # 최종 결과 출력    
    logger.info('*' * 20)
    logger.info('** FINAL RESULTS **')
    logger.info(f'Correct             : {correct_count} / {total_count} ({round(correct_count / total_count * 100, 2)})')
    logger.info(f'Correct with Steps  : {correct_with_step} / {total_count} ({round(correct_with_step / total_count * 100, 2)})')
    logger.info(f'Feedback Request    : {feedback_count} / {total_count} ({round(feedback_count / total_count * 100, 2)})')
    logger.info(f'Feedback with Steps : {feedback_with_step} / {total_count} ({round(feedback_with_step / total_count * 100, 2)})')
    logger.info(f'Incorrect           : {incorrect_count} / {total_count} ({round(incorrect_count / total_count * 100, 2)})')
    logger.info(f'Incorrect with Steps: {incorrect_with_step} / {total_count} ({round(incorrect_with_step / total_count * 100, 2)})')
    logger.info('*' * 20)

        
    logger.info('Done...')
    
    
    
if __name__ == '__main__':
    try:
        start = time()
        main()
        end = time()
        print(f'Total time : {end - start} seconds')
        
    except Exception as error:
        os.system(f'curl -d "Error running preliminary_training/main.py. Error : {error}" ntfy.sh/hjkim-experiments')
        