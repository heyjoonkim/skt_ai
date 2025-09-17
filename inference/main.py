import json
import logging
import os
import csv
from typing import Any, Dict, List, Union

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from templates import PROMPTS, SYSTEM_MESSAGE
from utils import save_pkl

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaFunctionCallInference:
    def __init__(
        self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct", use_8bit: bool = False, use_4bit: bool = True
    ):
        """
        Llama Function Call 추론 클래스 (베이스라인 - 파인튜닝 없음)

        Args:
            model_name: 사용할 Llama 모델명 (기본값: meta-llama/Llama-3.2-1B-Instruct)
            use_8bit: 8bit 양자화 모델 사용 여부
            use_4bit: 4bit 양자화 모델 사용 여부
        """
        self.model_name = model_name
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """모델과 토크나이저를 로드합니다."""
        logger.info(f"Loading model: {self.model_name}")

        # 토크나이저 로드
        
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, padding_side="right")
        except:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", trust_remote_code=True, padding_side="right")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델 로드
        if self.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        elif self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            )

        logger.info("Model loaded successfully")

    def predict_function_call(self, query: Union[str, list[str]]) -> Union[str, list[str]]:
        """
        사용자 쿼리에 대한 Function Call을 예측합니다.
        query가 str이면 단일 예측, list of str이면 배치 예측을 수행합니다.

        Args:
            query: 사용자 명령(str) 또는 사용자 명령 리스트(list of str)

        Returns:
            예측된 Function Call 문자열 또는 리스트
        """
        # query가 str이면 리스트로 변환
        is_single = False
        if isinstance(query, str):
            queries = [query]
            is_single = True
        else:
            queries = query

        # 모델별로 프롬프트 템플릿을 다르게 적용할 수 있도록 일반화
        system_prompt = "당신은 한국어 음성 명령을 Function Call로 변환하는 AI 어시스턴트입니다. 주어진 사용자 명령을 분석하여 적절한 함수 호출 형태로 변환해주세요."

        # 메시지 구조화 (OpenAI/ChatML 스타일 지원)
        messages_list = [
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": PROMPTS[3].format(query=q)},
                {"role": "assistant", "content": ""},
            ]
            for q in queries
        ]

        # tokenizer가 chat template 지원 여부에 따라 분기
        if hasattr(self.tokenizer, "apply_chat_template"):
            # chat template 지원 (예: Llama, ChatGLM 등)
            prompts = [self.tokenizer.apply_chat_template(messages, tokenize=False) for messages in messages_list]
        else:
            # 일반 텍스트 프롬프트 (예: GPT2, T5 등)
            prompts = [f"{system_prompt}\n\n사용자 명령: {q}\n\n" for q in queries]

        # 토크나이징 (batch)
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)

        # 추론 수행 (batch)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        # 응답 디코딩 (batch)
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results = []
        for response in responses:
            resp = response.split("assistant")[-1].strip()
            resp = resp.split("system")[-1].strip()
            resp = resp.split("user")[-1].strip()
            if "<|eot_id|>" in resp:
                resp = resp.split("<|eot_id|>")[0].strip()
            if "<end>" in resp:
                resp = resp.split("<end>")[0].strip() + "<end>"
            results.append(resp)

        # 입력이 단일 str이면 str로 반환, 아니면 list로 반환
        if is_single:
            return results[0]
        else:
            return results

    def batch_predict(self, queries: List[str], batch_size: int = 64) -> List[str]:
        """
        여러 쿼리에 대한 배치 예측을 수행합니다.

        Args:
            queries: 사용자 명령 리스트
            batch_size: 배치 크기

        Returns:
            예측된 Function Call 리스트
        """
        results = []

        for start in tqdm(range(0, len(queries), batch_size)):
            end = min(start + batch_size, len(queries))
            if end >= len(queries):
                batch_queries = queries[start:]
            else:
                batch_queries = queries[start:end]
            batch_results = self.predict_function_call(batch_queries)
            results.extend(batch_results)

        return results

    def evaluate_model(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        모델 성능을 평가합니다.

        Args:
            test_data: 테스트 데이터

        Returns:
            평가 결과 딕셔너리
        """
        correct = 0
        total = len(test_data)

        for item in test_data:
            query = item.get("Query(한글)", "")
            expected_output = item.get("LLM Output", "")

            predicted_output = self.predict_function_call(query)

            if predicted_output.strip() == expected_output.strip():
                correct += 1

        accuracy = correct / total if total > 0 else 0

        return {"accuracy": accuracy, "correct": correct, "total": total}
    

# HJ add
def single_evaluation(query, label, result) -> Dict[str, float]:
    
    if result.strip() == label.strip():
        return 1
    return 0
    

def main():
    """메인 실행 함수"""
    # 베이스라인 모델로 추론 테스트
    # inference = LlamaFunctionCallInference()
    # model_path = "/home/heyjoonkim/data/skt_ai/meta-llama-Llama-3.2-1B-Instruct-1234/SUBSET_PROMPT-4_epoch-1_batch-16_accumulation-1_lr-5e-05"
    model_path='Hooooooooooon/Llama-3.2-1B-Instruct-0917-1e3-2'
    inference = LlamaFunctionCallInference(model_name=model_path)

    test_filename = '/skt_ai/inference/test.csv'
    # output_file = os.path.join(model_path, 'test_results.pkl')
    
    with open(test_filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # 첫 번째 줄(헤더) 건너뛰기

        count = 0
        correct_count = 0
        res = list()
        label_stats = {}  # label별 통계를 저장할 딕셔너리
        
        for row in tqdm(reader):
            query = row[1]   # 두 번째 컬럼 (Query)
            label = row[2]  # 세 번째 컬럼 (LLM Output)
            
            if ';' in label:
                # 이 경우 다중함수콜이라서 일단 제외
                continue
            
            result = inference.predict_function_call(query)
            
            single_res = single_evaluation(query=query, label=label, result=result)
            
            is_correct = 'CORRECT' if single_res == 1 else 'INCORRECT'

            # label별 통계 업데이트
            if label not in label_stats:
                label_stats[label] = {'total': 0, 'correct': 0, 'incorrect': 0}
            
            label_stats[label]['total'] += 1
            if single_res == 1:
                label_stats[label]['correct'] += 1
            else:
                label_stats[label]['incorrect'] += 1

            count += 1
            correct_count += single_res
            
            res.append(dict(query=query, label=label, result=result, correct=single_res))
        
        # label별 정확도 계산
        label_accuracy = {}
        for label, stats in label_stats.items():
            accuracy = round(100 * stats['correct'] / stats['total'], 2) if stats['total'] > 0 else 0
            label_accuracy[label] = accuracy
        
        final_results = {
            'total_samples': count,
            'correct_predictions': correct_count,
            'incorrect_predictions': count - correct_count,
            'accuracy_percentage': round(100 * correct_count / count, 2),
            'accuracy_fraction': f"{correct_count}/{count}",
            'label_statistics': label_stats,
            'label_accuracy': label_accuracy
        }
        
        print("\n" + "=" * 80)
        print("FINAL EVALUATION RESULTS")
        print("=" * 80)
        
        # 전체 통계만 print로 출력
        print("OVERALL STATISTICS:")
        print("-" * 40)
        print(f"{'total_samples':<25}: {final_results['total_samples']}")
        print(f"{'correct_predictions':<25}: {final_results['correct_predictions']}")
        print(f"{'incorrect_predictions':<25}: {final_results['incorrect_predictions']}")
        print(f"{'accuracy_percentage':<25}: {final_results['accuracy_percentage']}%")
        print(f"{'accuracy_fraction':<25}: {final_results['accuracy_fraction']}")
        print("=" * 80)
        
        # label별 상세 통계를 JSON 파일로 저장
        detailed_results = {
            'label_statistics': label_stats,
            'label_accuracy': label_accuracy,
            'sorted_by_accuracy': [
                {
                    'label': label,
                    'total': stats['total'],
                    'correct': stats['correct'],
                    'incorrect': stats['incorrect'],
                    'accuracy': label_accuracy[label]
                }
                for label, stats in sorted(label_stats.items(), key=lambda x: label_accuracy[x[0]], reverse=True)
            ],
            'low_performance_functions': [
                {
                    'label': label,
                    'accuracy': accuracy,
                    'total': label_stats[label]['total'],
                    'correct': label_stats[label]['correct']
                }
                for label, accuracy in sorted(
                    [(label, acc) for label, acc in label_accuracy.items() if acc < 70 and label_stats[label]['total'] >= 3],
                    key=lambda x: x[1]
                )
            ]
        }
        
        json_output_file = os.path.join('/skt/inference', 'evaluation_results_detailed.json')
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        print(f"Detailed label statistics saved to: {json_output_file}")
        
    return

if __name__ == "__main__":
    main()
