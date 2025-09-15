
from typing import List


def _clean_str(string:str) -> str:
    string = string.strip().lower()
    string = string.replace(' ', '')
    
    # TODO: add more filtering rules
    
    return string

def _extract_boxed_content(string:str) -> List:
    results = list()
    start_tag = '\\boxed{'
    start_tag_2 = '\boxed{'
    idx = 0
    while idx < len(string):
        start = string.find(start_tag, idx)
        if start == -1:
            start = string.find(start_tag_2, idx)
        if start == -1:
            break
        i = start + len(start_tag)
        brace_count = 1
        while i < len(string) and brace_count > 0:
            if string[i] == '{':
                brace_count += 1
            elif string[i] == '}':
                brace_count -= 1
            i += 1
        if brace_count == 0:
            content = string[start + len(start_tag): i - 1]
            results.append(content)
            idx = i
        else:
            break
    return results

def extract_math_answer(string:str) -> str:
    matches = _extract_boxed_content(string)

    if len(matches) > 0:
        res = matches[0]
        res = _clean_str(res)
    else:
        res = None
        
    return res