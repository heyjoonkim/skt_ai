from typing import List, Dict

# TODO : 우리가 모델한테 넣어줄 시스템 메세지 작성하기
SYSTEM_MESSAGE = """You are a helpful assistant."""


# TODO : 우리가 학습할 때 사용할 prompt 넣어주기
# 개인적으론 함수 형태로 정의한거 다 예시로 넣어주면 좋긴할듯?
PROMPT = """

Query : {question}
Function : """





# list of [system message, prompt]
PTRUE_TEMPLATE = ['Given the following question and reasoning steps, evaluate whether the last step (Step {k}) is correct. Respond with "True" or "False" only.', 'Respond with "True" or "False" only.']
PCORRECT_TEMPLATE = ['The following is a partial solution process for a given question. Considering both the question and the previous steps, evaluate whether the current step in the solution is valid. As the final output, generate only "Correct" or "Incorrect".', 'Is the final step correct? Answer only in "Correct" or "Incorrect".']

MATH_TEMPLATES = [
    # "{query}\nPlease reason step by step, and put your final answer within {target}.",
    ["Question: ", " Let's think step by step."],
]


DATASET_TO_TEMPLATE = {
    "heyjoonkim/hendrycks_math" : MATH_TEMPLATES,
    "heyjoonkim/math_500" : MATH_TEMPLATES,
}


DATASET_TO_DEMONSTRATIONS = {
    "heyjoonkim/hendrycks_math" : [
        {
            "query" : "A sequence of ten $0$s and/or $1$s is randomly generated. If the probability that the sequence does not contain two consecutive $1$s can be written in the form $\dfrac{m}{n}$, where $m,n$ are relatively prime positive integers, find $m+n$.",
            "steps" : [
                "Let $a_n$ denote the number of sequences of length $n$ that do not contain consecutive $1$s. A sequence of length $n$ must either end in a $0$ or a $1$.",
                "If the string of length $n$ ends in a $0$, this string could have been formed by appending a $0$ to any sequence of length $n-1$, of which there are $a_{n-1}$ such strings.",
                "If the string of length $n$ ends in a $1$, this string could have been formed by appending a $01$ (to avoid consecutive $1$s) to any sequence of length $n-2$, of which there are $a_{n-2}$ such strings.",
                "Thus, we have the recursion\[a_n = a_{n-1} + a_{n-2}\]Solving for initial conditions, we find $a_1 = 2, a_2 = 3$.",
                "Thus we have the Fibonacci sequence with shifted indices; indeed $a_n = F_{n+2}$, so $a_{10} = F_{12} = 144$.",
                "The probability is $\frac{144}{2^{10}} = \frac{9}{64}$, and $m+n=\\boxed{73}$.",
            ]
        },
        {
            "query" : "Our football team has 10 members, of which only 3 are strong enough to play offensive lineman, while all other positions can be played by anyone. In how many ways can we choose a starting lineup consisting of a quarterback, a running back, an offensive lineman, and a wide receiver?",
            "steps" : [
                "There are 3 choices for the offensive lineman position.",
                "Then there are 9 choices for the next position, 8 choices for the  position after, and 7 choices for the last position.",
                "So that's a total of $3\times9\times8\times7 = \\boxed{1512}$."                
            ]
        },
        {
            "query" : "Find the value of $b$ that satisfies the equation $161_{b}+134_{b}=315_{b}$.",
            "steps" : [
                "In the rightmost column there is no carrying, so our base must be greater than 5.",
                "However, in the next column we see that $6_{b}+3_{b}=11_{b}$.",
                "This tells us that $b$ divides into 9 once, with a remainder 1.",
                "Therefore, $b=\\boxed{8}$."
            ]
        },
        {
            "query" : "How many four-digit numbers $N$ have the property that the three-digit number obtained by removing the leftmost digit is one ninth of $N$?",
            "steps" : [
                "Let $a$ denote the leftmost digit of $N$ and let $x$ denote the three-digit number obtained by removing $a$.",
                "Then $N=1000a+x=9x$ and it follows that $1000a=8x$.",
                "Dividing both sides by 8 yields $125a=x$.",
                "All the values of $a$ in the range 1 to 7 result in three-digit numbers, hence there are $\\boxed{7}$ values for $N$."
            ]
        }
    ],
    "heyjoonkim/math_500" : [
        {
            "query" : "A sequence of ten $0$s and/or $1$s is randomly generated. If the probability that the sequence does not contain two consecutive $1$s can be written in the form $\dfrac{m}{n}$, where $m,n$ are relatively prime positive integers, find $m+n$.",
            "steps" : [
                "Let $a_n$ denote the number of sequences of length $n$ that do not contain consecutive $1$s. A sequence of length $n$ must either end in a $0$ or a $1$.",
                "If the string of length $n$ ends in a $0$, this string could have been formed by appending a $0$ to any sequence of length $n-1$, of which there are $a_{n-1}$ such strings.",
                "If the string of length $n$ ends in a $1$, this string could have been formed by appending a $01$ (to avoid consecutive $1$s) to any sequence of length $n-2$, of which there are $a_{n-2}$ such strings.",
                "Thus, we have the recursion\[a_n = a_{n-1} + a_{n-2}\]Solving for initial conditions, we find $a_1 = 2, a_2 = 3$.",
                "Thus we have the Fibonacci sequence with shifted indices; indeed $a_n = F_{n+2}$, so $a_{10} = F_{12} = 144$.",
                "The probability is $\frac{144}{2^{10}} = \frac{9}{64}$, and $m+n=\\boxed{73}$.",
            ]
        },
        {
            "query" : "Our football team has 10 members, of which only 3 are strong enough to play offensive lineman, while all other positions can be played by anyone. In how many ways can we choose a starting lineup consisting of a quarterback, a running back, an offensive lineman, and a wide receiver?",
            "steps" : [
                "There are 3 choices for the offensive lineman position.",
                "Then there are 9 choices for the next position, 8 choices for the  position after, and 7 choices for the last position.",
                "So that's a total of $3\times9\times8\times7 = \\boxed{1512}$."                
            ]
        },
        {
            "query" : "Find the value of $b$ that satisfies the equation $161_{b}+134_{b}=315_{b}$.",
            "steps" : [
                "In the rightmost column there is no carrying, so our base must be greater than 5.",
                "However, in the next column we see that $6_{b}+3_{b}=11_{b}$.",
                "This tells us that $b$ divides into 9 once, with a remainder 1.",
                "Therefore, $b=\\boxed{8}$."
            ]
        },
        {
            "query" : "How many four-digit numbers $N$ have the property that the three-digit number obtained by removing the leftmost digit is one ninth of $N$?",
            "steps" : [
                "Let $a$ denote the leftmost digit of $N$ and let $x$ denote the three-digit number obtained by removing $a$.",
                "Then $N=1000a+x=9x$ and it follows that $1000a=8x$.",
                "Dividing both sides by 8 yields $125a=x$.",
                "All the values of $a$ in the range 1 to 7 result in three-digit numbers, hence there are $\\boxed{7}$ values for $N$."
            ]
        }        
    ],
}


DATASET_TO_SYS_MESSAGE = {
    # "heyjoonkim/hendrycks_math" : "Please reason step by step, and put your final answer within \\boxed{}.",
    # "heyjoonkim/math_500" : "Please reason step by step, and put your final answer within \\boxed{}.",
    "heyjoonkim/hendrycks_math" : "Please reason step by step. Make sure your output matches the structure of the provided demonstration, and place your final answer in \\boxed{}.",
    "heyjoonkim/math_500" : "Please reason step by step. Make sure your output matches the structure of the provided demonstration, and place your final answer in \\boxed{}.",
    
}



def get_demonstrations_str(dataset_name:str, num_demonstration:int=0, template_id:int=0) -> List[Dict]:
    
    demonstrations = DATASET_TO_DEMONSTRATIONS.get(dataset_name, None)
    assert demonstrations is not None, f"Dataset {dataset_name} is not supported."
    
    selected_demonstrations = demonstrations[:num_demonstration]
    
    # query prompt
    inference_template_list = DATASET_TO_TEMPLATE.get(dataset_name, None)
    assert inference_template_list is not None, f"Dataset {dataset_name} is not supported."
    inference_template = inference_template_list[template_id]
    prefix, suffix = inference_template
    
    # placeholders
    STEP_SEP = "\n"
    DEMO_SEP = "\n\n"
    STEP = "Step {index}: {step}"
    
    formatted_demonstrations_list = list()
    for selected_demonstration in selected_demonstrations:
        query = selected_demonstration.get('query')
        steps = selected_demonstration.get('steps')
        
        formatted_demonstration_list = list()
        
        query_prompt = prefix + query + suffix
        formatted_demonstration_list.append(query_prompt)
        
        for index, step in enumerate(steps):
            step_format = STEP.format(index=index+1, step=step)
            formatted_demonstration_list.append(step_format)
            
        formatted_demonstration = STEP_SEP.join(formatted_demonstration_list)
        formatted_demonstrations_list.append(formatted_demonstration)
    
    formatted_demonstrations = DEMO_SEP.join(formatted_demonstrations_list) + DEMO_SEP
    
    return formatted_demonstrations

def make_inference_template(dataset_name:str, demonstration_str:str, query:str, template_id:int) -> str:
    
    inference_template_list = DATASET_TO_TEMPLATE.get(dataset_name, None)
    assert inference_template_list is not None, f"Dataset {dataset_name} is not supported."
    
    inference_template = inference_template_list[template_id]
    prefix, suffix = inference_template
    query_template = prefix + query + suffix
    
    final_template = demonstration_str + query_template
    
    return final_template
    


## system message
# You are a helpful assistant. Always follow the user’s instructions precisely, without making assumptions or adding extra information.
# You are a helpful assistant. Your top priority is to fulfill the user’s request exactly as written.
# You are a helpful assistant. Respond only within the scope of the user’s request. 