from typing import List, Dict

import numpy as np
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice

from inference import DATASET_TO_TEMPLATE

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
        
        
    def get_max_prob(self) -> List[Dict]:
        res = list()
        for token_dict in self.top_probs_dict:
            max_prob = 0.0
            max_token = None
            for token, prob in token_dict.items():
                if prob > max_prob:
                    max_prob = prob
                    max_token = token
            res.append({max_token:max_prob})
        return res
    
    def get_entropy(self, num_top_probs:int=None) -> Dict:
        
        num_top_probs = num_top_probs if num_top_probs is not None else self.num_top_probs
        
        print(num_top_probs)
        
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
        
        
        sum_entropy = sum(res)
        avg_entropy = sum_entropy / len(res)
        return dict(full=res, sum=sum_entropy, average=avg_entropy)
        
        

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




class OpenAIClient:
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
                 inference_template_id:int=None,
                 system_message:str="You are a helpful assistant that responds only within the scope of the given request."
                #  system_message:str="You are a helpful assistant that responds only within the scope of the given request. Please reason step by step, and put your final answer within \\boxed{}."
                #  system_message:str="Please reason step by step, and put your final answer within \\boxed{}."
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
        self.template = self._get_template(dataset_name, inference_template_id)
        self.system_message = system_message
        # for formatting
        self.target_str = "\\boxed{}"
        
        self.messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": None},
            # {"role": "assistant", "content": "Step 1:"}
        ]
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
    def _get_template(self, dataset_name:str, inference_template_id:int)->str:
        templates_list = DATASET_TO_TEMPLATE.get(dataset_name, None)
        assert templates_list is not None, f"Dataset {dataset_name} is not supported."
        template = templates_list[inference_template_id]
        return template
        
    def _format(self, query:str) -> str:    
        return self.template.format(query=query, target=self.target_str)
    
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
        
    def _print_message(self):
        print('[ Print messages for debugging ]')
        for message in self.messages:
            print(f"[{message['role']}]: {message['content']}")
        
        
    def generate(self, query:str):
        
        self.messages[1]["content"] = self._format(query) #+ '\n' + self.system_message
        
        self._print_message()
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            # logprobs=True,
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
        
        
    