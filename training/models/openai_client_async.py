from typing import List, Dict

import numpy as np
from openai import AsyncOpenAI  
from openai.types.chat.chat_completion import ChatCompletion, Choice

from inference import (
    make_inference_template,
    get_demonstrations_str,
    DATASET_TO_SYS_MESSAGE,
    PTRUE_TEMPLATE,
    PCORRECT_TEMPLATE,
)

class Generation:
    def __init__(self, choice_object:Choice):
        # keys: finish_reason, index, logprobs, message, stop_reason
        self.generated_text = choice_object.message.content
        logprobs_dict = choice_object.logprobs
        
        # breakpoint()
        # self.tokens = logprobs_dict.tokens
        
        # list of selected tokens == list(str)
        self.tokens = list()
        
        # logprobs_dict.keys() == content
        # type(top_logprobs) == list
        top_logprobs = logprobs_dict.content
        
        # list of dict() : {token, prob}
        # dict()는 각 step에서 top log prob에 대한 {token1:prob1, token2:prob2, ...}
        self.top_probs_dict = list()
        # type(top_logprob)   == dict
        # top_logprobs.keys() == token, bytes, logprob, top_logprobs
        for top_logprob in top_logprobs:
            selected_token = top_logprob.token
            # selected_logprob = top_logprob.get('logprob')
            # selected_prob = np.exp(selected_logprob)
            self.tokens.append(selected_token)
            
            # type(logprobs_list) == list(dict())
            logprobs_list = top_logprob.top_logprobs
            
            # 총 logprob 개수만큼의 {k,v} 쌍으로 저장됨.
            cur_token_dict = dict()
            # logprob_dict.keys() : token, bytes, logprob
            for logprob_dict in logprobs_list:
                cur_token = logprob_dict.token
                cur_logprob = logprob_dict.logprob
                cur_prob = np.exp(cur_logprob)
                
                cur_token_dict[cur_token] = cur_prob
            
            self.top_probs_dict.append(cur_token_dict)
        
        self.num_top_probs = len(self.top_probs_dict[0])
        
    
    def _get_token_prob(self, cur_token:str, index:int=0, neg_token:str=None) -> float:
        token_dict = self.top_probs_dict[index]
        
        cur_token = cur_token.lower().strip()
        if neg_token is not None:
            neg_token = neg_token.lower().strip()
            
        res_prob = 0    
            
        for token, prob in token_dict.items():
            
            flag = False
            token = token.lower().strip()
            
            if neg_token in token:
                flag = True
                
            if cur_token in token:
                if flag:
                    tmp_prob = res_prob
                
                if neg_token is None or neg_token not in token:
                    res_prob += prob
                
                if flag:
                    assert res_prob == tmp_prob, f"res_prob should be same. {res_prob} vs {tmp_prob}"
                
        return res_prob
       
       
    def get_max_prob(self) -> Dict:
        res = list()
        for token_dict in self.top_probs_dict:
            max_prob = 0.0
            max_token = None
            for token, prob in token_dict.items():
                if prob > max_prob:
                    max_prob = prob
                    max_token = token
            res.append({max_token:max_prob})
            
        assert len(self.tokens) == len(res), f"Length of tokens and max prob list are not same. {len(self.tokens)} vs {len(res)}"
        
        return dict(tokens=self.tokens, max_probs=res)
    
    def get_entropy(self, num_top_probs:int=None) -> Dict:
        
        num_top_probs = num_top_probs if num_top_probs is not None else self.num_top_probs
        
        # print(num_top_probs)
        
        # 각 step에서의 entropy의 list
        # res = list(float)
        res = list()
        for token_dict in self.top_probs_dict:
            sorted_dict = sorted(token_dict.items(), key=lambda x: x[1], reverse=True)
            selected_dict = sorted_dict[:num_top_probs]
            selected_probs = [prob for token,prob in selected_dict]
            entropy = sum([-1 * prob * np.log(prob + 1e-40) for prob in selected_probs])
            res.append(entropy)
            # from scipy.stats import entropy
            # _entropy_2 = entropy(selected_probs)
        
        assert len(self.tokens) == len(res), f"Length of tokens and entropy list are not same. {len(self.tokens)} vs {len(res)}"
        
        sum_entropy = sum(res)
        avg_entropy = sum_entropy / len(res)
        return dict(tokens=self.tokens, full=res, sum=sum_entropy, average=avg_entropy)
        
        

class GenerationOutputClass:
    def __init__(self,
                 completion:ChatCompletion) -> None:
        
        # keys : id, choices, created, model, object, usage, prompt_logprobs
        # self.completion = completion
        choices = completion.choices
        
        self.num_generation = len(choices)
        self.generations = list()
        
        for choice in choices:
            # attributes : generated_text, tokens, top_logprobs_dict
            generation = Generation(choice)
            self.generations.append(generation)
            
        usage = completion.usage
        self.completion_tokens = usage.completion_tokens
        self.prompt_tokens = usage.prompt_tokens
        self.total_tokens = usage.total_tokens




class AsyncOpenAIClient:
    def __init__(self, 
                 model_name:str,
                 base_url:str,
                 api_key:str,
                 num_generations:int=1,
                 num_max_tokens:int=500,
                 temperature:float=0.0,
                 top_p:float=1.0,
                 top_logprobs:int=100,
                 dataset_name:str=None,
                 num_demonstrations:int=4,
                 inference_template_id:int=None,
                 ) -> None:
        
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.num_generations = num_generations
        self.num_max_tokens = num_max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_logprobs = top_logprobs
        self.dataset_name = dataset_name
        self.num_demonstrations = num_demonstrations
        self.inference_template_id = inference_template_id
        self.demonstrations = get_demonstrations_str(dataset_name=self.dataset_name, 
                                                     num_demonstration=self.num_demonstrations,
                                                     template_id=self.inference_template_id)
        self.system_message = DATASET_TO_SYS_MESSAGE.get(self.dataset_name)
        assert self.system_message is not None, f"System message for {self.dataset_name} is not defined."
        
        self.messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": None},
        ]
        
        
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
    def _format(self, query:str) -> str:    
        return make_inference_template(dataset_name=self.dataset_name, 
                                       demonstration_str=self.demonstrations,
                                       query=query, 
                                       template_id=self.inference_template_id)
    
    def _reset_messages(self):
        self.messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": None}
        ]
        
        # self.messages = [
        #     {"role": "system", "content": self.system_message},
        #     {"role": "user", "content": None},
        #     {"role": "assistant", "content": "Step 1:"}
        # ]
        
    def _print_message(self, print_last:bool=False):
        print('\n\n[[ Print messages for debugging ]]')
        if print_last:
            print('[[ Print last step only ]]')
            print(f"\n[[ {self.messages[-1]['role']} ]]: {self.messages[-1]['content']}")
        else:
            for message in self.messages:
                print(f"\n[[ {message['role']} ]]: {message['content']}")       
        
    async def generate(self, query:str):
        
        self.messages[1]["content"] = self._format(query) #+ '\n' + self.system_message
                
        # self._print_message()
        
        completion = await self.client.chat.completions.create(
                        model=self.model_name,
                        logprobs=True,
                        top_logprobs=self.top_logprobs,
                        n=self.num_generations,
                        max_tokens=self.num_max_tokens,
                        temperature=self.temperature,
                        messages=self.messages
                    )
        
        # attributes : num_generations(int), generations(class), completion_tokens(int), prompt_tokens(int), total_tokens(int)
        generation = GenerationOutputClass(completion)
        return generation
    
    
    ## DEPRECATED (2025.05.02) ## -> 취소
    ## used in preliminary.sampling_uncertainty.py
    async def generate_with_steps(self, query:str, steps:List=None):
        
        self._reset_messages()
        
        self.messages[1]["content"] = self._format(query) #+ '\n' + self.system_message
                
        if steps is not None:
            
            STEP_SEP = "\n"
            STEP = "Step {index}: {step}"
            
            formatted_steps_list = list()
            for step_index, step in enumerate(steps):
                step_format = STEP.format(index=step_index+1, step=step)
                formatted_steps_list.append(step_format)
                
            steps_str = STEP_SEP.join(formatted_steps_list)
            
            self.messages.append({"role": "assistant", "content": steps_str})
            # 이 부분이 없는게 (no_user_prompt)
            # 이 부분이 포함시켜서 추가 실험 (2025.04.25)
            self.messages.append({"role": "user", "content": "Continue generating the next step while keeping the same format."})
                           
                
        # self._print_message(print_last=True)
        # self._print_message(print_last=False)
        
        completion = await self.client.chat.completions.create(
                                model=self.model_name,
                                logprobs=True,
                                top_logprobs=self.top_logprobs,
                                n=self.num_generations,
                                max_tokens=self.num_max_tokens,
                                temperature=self.temperature,
                                messages=self.messages
                            )
        
        
        # attributes : num_generations(int), generations(class), completion_tokens(int), prompt_tokens(int), total_tokens(int)
        generation = GenerationOutputClass(completion)
        
        return generation
    
    ## TODO : 뭐가 문젠지 모르겠는데 안됨.
    ## used in preliminary.sampling_uncertainty.py
    async def _generate_with_steps(self, query:str, steps:List[str]=None):
        
        self._reset_messages()
        
        self.messages[1]["content"] = self._format(query) #+ '\n' + self.system_message
                
        if steps is not None:
            
            STEP_SEP = "\n"
            STEP = "Step {index}: {step}"
            
            formatted_steps_list = list()
            for step_index, step in enumerate(steps):
                step_format = STEP.format(index=step_index+1, step=step)
                formatted_steps_list.append(step_format)
                
            steps_str = STEP_SEP.join(formatted_steps_list)
            
            self.messages.append({"role": "assistant", "content": steps_str})
            # 이 부분이 없는게 (no_user_prompt)
            # 이 부분이 포함시켜서 추가 실험 (2025.04.25)
            # self.messages.append({"role": "user", "content": "Continue generating the next step while keeping the same format."})
                           
                
        # self._print_message(print_last=True)
        # self._print_message(print_last=False)
        
        
        completion = await self.client.chat.completions.create(
                                model=self.model_name,
                                logprobs=True,
                                top_logprobs=self.top_logprobs,
                                n=self.num_generations,
                                max_tokens=self.num_max_tokens,
                                temperature=self.temperature,
                                messages=self.messages,
                                # for continuous generation
                                extra_body={
                                    'add_generation_prompt':False,
                                    'continue_final_message':True,
                                },
                            )
        
        
        # attributes : num_generations(int), generations(class), completion_tokens(int), prompt_tokens(int), total_tokens(int)
        generation = GenerationOutputClass(completion)
        return generation
    
        
    async def prtue_generate(self, query:str, steps:List):
        sys_msg, user_msg = PTRUE_TEMPLATE[0], PTRUE_TEMPLATE[1]
        
        inference_template = f"Question: {query}\n\n"
        steps_str = ""
        
        res_list = list()
        
        for step_index, step in enumerate(steps):
            step_sys_msg = sys_msg.format(k=step_index+1)
            steps_str += f'Step {step_index+1}: {step}\n'

            final_user_msg = inference_template+steps_str+user_msg
            messages = [
                {"role": "system", "content": step_sys_msg},
                {"role": "user", "content": final_user_msg},
            ]
            
            completion = await self.client.chat.completions.create(
                        model=self.model_name,
                        logprobs=True,
                        top_logprobs=self.top_logprobs,
                        n=self.num_generations,
                        max_tokens=self.num_max_tokens,
                        temperature=self.temperature,
                        messages=messages
                    )
            
            # attributes : num_generations(int), generations(class), completion_tokens(int), prompt_tokens(int), total_tokens(int)
            generation_class = GenerationOutputClass(completion)
            
            generations_list = generation_class.generations
            assert len(generations_list) == 1, f"Number of generation should be 1. {len(generations_list)} (GREEDY GENERATION ONLY)"
            generation = generations_list[0]
            ptrue_prediction = generation.generated_text
            true_prob = generation._get_token_prob('True', index=0)
            
            res_list.append(dict(step=step, ptrue_prediction=ptrue_prediction, true_prob=true_prob))

        return res_list
    
    # steps(list)의 마지막 step만 평가하려고 하는 경우
    async def single_ptrue_generate(self, query:str, steps:List):
        sys_msg, user_msg = PTRUE_TEMPLATE[0], PTRUE_TEMPLATE[1]
        
        inference_template = f"Question: {query}\n\n"
        steps_str = ""
                
        for step_index, step in enumerate(steps):
            step_sys_msg = sys_msg.format(k=step_index+1)
            steps_str += f'Step {step_index+1}: {step}\n'

        final_user_msg = inference_template+steps_str+user_msg
        messages = [
            {"role": "system", "content": step_sys_msg},
            {"role": "user", "content": final_user_msg},
        ]
        
        completion = await self.client.chat.completions.create(
                    model=self.model_name,
                    logprobs=True,
                    top_logprobs=self.top_logprobs,
                    n=self.num_generations,
                    max_tokens=self.num_max_tokens,
                    temperature=self.temperature,
                    messages=messages
                )
        
        # attributes : num_generations(int), generations(class), completion_tokens(int), prompt_tokens(int), total_tokens(int)
        generation_class = GenerationOutputClass(completion)
        
        generations_list = generation_class.generations
        assert len(generations_list) == 1, f"Number of generation should be 1. {len(generations_list)} (GREEDY GENERATION ONLY)"
        generation = generations_list[0]
        ptrue_prediction = generation.generated_text
        true_prob = generation._get_token_prob('True', index=0)
        
        return dict(step=step, 
                    ptrue_prediction=ptrue_prediction, 
                    true_prob=true_prob)

    
    
    
    async def pcorrect_generate(self, query:str, steps:List):
        sys_msg, user_msg = PCORRECT_TEMPLATE[0], PCORRECT_TEMPLATE[1]
        
        inference_template = f"Question: {query}\n\n"
        steps_str = ""
        
        res_list = list()
        
        for step_index, step in enumerate(steps):
            steps_str += f'Step {step_index+1}: {step}\n'

            final_user_msg = inference_template+steps_str+user_msg
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": final_user_msg},
            ]
            
            completion = await self.client.chat.completions.create(
                        model=self.model_name,
                        logprobs=True,
                        top_logprobs=self.top_logprobs,
                        n=self.num_generations,
                        max_tokens=self.num_max_tokens,
                        temperature=self.temperature,
                        messages=messages
                    )
            
            # attributes : num_generations(int), generations(class), completion_tokens(int), prompt_tokens(int), total_tokens(int)
            generation_class = GenerationOutputClass(completion)
            
            generations_list = generation_class.generations
            assert len(generations_list) == 1, f"Number of generation should be 1. {len(generations_list)} (GREEDY GENERATION ONLY)"
            generation = generations_list[0]
            pcorrect_prediction = generation.generated_text
            correct_prob = generation._get_token_prob('Correct', index=0, neg_token='Incorrect')
            
            res_list.append(dict(step=step, pcorrect_prediction=pcorrect_prediction, correct_prob=correct_prob))

        return res_list
    
    
    async def single_pcorrect_generate(self, query:str, steps:List):
        sys_msg, user_msg = PCORRECT_TEMPLATE[0], PCORRECT_TEMPLATE[1]
        
        inference_template = f"Question: {query}\n\n"
        steps_str = ""
                
        for step_index, step in enumerate(steps):
            steps_str += f'Step {step_index+1}: {step}\n'

        final_user_msg = inference_template+steps_str+user_msg
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": final_user_msg},
        ]
        
        # breakpoint()
        
        completion = await self.client.chat.completions.create(
                    model=self.model_name,
                    logprobs=True,
                    top_logprobs=self.top_logprobs,
                    n=self.num_generations,
                    max_tokens=self.num_max_tokens,
                    temperature=self.temperature,
                    messages=messages
                )
            
        # attributes : num_generations(int), generations(class), completion_tokens(int), prompt_tokens(int), total_tokens(int)
        generation_class = GenerationOutputClass(completion)
        
        generations_list = generation_class.generations
        assert len(generations_list) == 1, f"Number of generation should be 1. {len(generations_list)} (GREEDY GENERATION ONLY)"
        generation = generations_list[0]
        pcorrect_prediction = generation.generated_text
        correct_prob = generation._get_token_prob('Correct', index=0, neg_token='Incorrect')
        
        return dict(step=step, 
                    pcorrect_prediction=pcorrect_prediction,
                    correct_prob=correct_prob)