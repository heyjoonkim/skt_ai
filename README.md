


# 데이터 전처리
- <code>dataset/preprocess.ipynb</code> : 그냥 형이 보내준 데이터 하나의 파일로 합치는 코드
-  뭔가 데이터가 너무 많아서 일단 함수당 500개로 필터링함 (총 22,000개)
- 22,000개 데이터가 들어가있는 jsonl 파일로 저장

# 학습할 수 있게 데이터 전처리
- <code>inference/templates.py</code> : 여기에 inference할 때 사용할 템플릿을 매뉴얼하게 적어서 그냥 박아둠. 어떤 템플릿이 좋을지는 잘 모르겠음 ㅠ
- <code>train/build_dataset.py</code> : 전처리에서 만든 데이터에 inference template 붙여주기 + 토크나이징해서 저장함.

# 학습하기
- <code>train/build_dataset.py</code>에서 토크나이징한 파일 불러와서 Trainer로 학습하기
- <code>scripts/train.sh</code> 돌리면 됨.
    - CUDA_VISIBLE_DEVICES 랑 num_processes 에 따라서 사용하는 gpu 수 지정
    - num_epochs : 학습 몇 에폭 돌릴지? (1~5 사이가 국룰이긴 함.)
    - lrs : 학습할 때 사용할 learning rate 지정 (1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6 중 고르는게 국룰임.)
- 학습할 떄 wandb 에 결과 로깅됨.