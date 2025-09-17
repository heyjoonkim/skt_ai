from huggingface_hub import HfApi, HfFolder, Repository
import os

# 사용자 입력값
model_checkpoint_path = "/skt_ai/output/meta-llama-Llama-3.2-1B-Instruct-1234/PROMPT-3_epoch-1_batch-8_accumulation-1_lr-1e-05"   # 저장된 로컬 모델 경로
repo_id = "Hooooooooooon/Llama-3.2-1B-Instruct-0917-1e5-1"

# HuggingFace 로그인 (토큰은 환경변수에 저장)
hf_token = os.environ.get("HF_TOKEN")
HfFolder.save_token(hf_token)

# API 객체 생성
api = HfApi()

# repo가 없다면 생성
api.create_repo(repo_id=repo_id, exist_ok=True)

# 업로드
api.upload_folder(
    folder_path=model_checkpoint_path,
    repo_id=repo_id,
    commit_message="Upload final trained model"
)

print(f"✅ Model uploaded successfully to https://huggingface.co/{repo_id}")