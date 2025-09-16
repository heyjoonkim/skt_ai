
import os

import json
import pickle
from typing import List

# 파일 경로를 리스트로 받아옴.
# 해당 파일들이 실제로 존재하는지 확인
# 존재하면 True, 하나라도 존재하지 않으면 False
# 결과 파일 리스트를 받아서 모두 존재하는지 확인, 존재하지 않으면 모델 로드
def has_all_results(files:List) -> bool:
    for file in files:
        if not os.path.isfile(file):
            return False
    return True

# check if path exists, if not, create it 
def _check_path(path:str):
    directory = os.path.dirname(path)

    if os.path.isdir(directory) is False:
        os.makedirs(directory, exist_ok=True)

##################
#                #
#     Pickle     #
#                #
##################

def load_pkl(path:str):

    if not os.path.exists(path):
        return None

    with open(path, 'rb') as infile:
        pkl_file = pickle.load(infile)
    
    return pkl_file

def save_pkl(data, path):
    _check_path(path)
    
    with open(path, 'wb') as outfile:
        pickle.dump(data, outfile)

################
#              #
#     JSON     #
#              #
################
        
def load_json(path:str):
    
    if not os.path.exists(path):
        return None
    
    with open(path, 'r') as infile:
        data = json.load(infile)
    return data

def save_json(data, path):
    _check_path(path)
    
    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile)
        
        
###################
#                 #
#   DELETE FILE   #
#                 #
###################

def remove_file(path:str):
    if not os.path.isfile(path):
        print(f'File does not exists: {path}')
        return
    os.remove(path)