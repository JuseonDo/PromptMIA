import sys
import os

# Python 경로에 필요한 디렉토리 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# /data2/PromptMIA를 Python 경로에 추가
sys.path.insert(0, parent_dir)

print(f"Added to Python path: {parent_dir}")

# Import 시도
try:
    from Falcon.utils.model_utils import FalconGenerator, GenerateConfig
except ImportError:
    from utils.model_utils import FalconGenerator, GenerateConfig

import re
import json
from typing import Dict, List, Tuple
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv
import fire

DEFAULT_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 0,
    "repetition_penalty": 1.05,
    "max_new_tokens": 5, 
    "batch_size": 8,
    "seed": 0,
}

PARAM_KEY_MAP = {
    "temperature": "temp",
    "top_p": "top_P",
    "top_k": "top_k",
    "repetition_penalty": "rep",
    "max_new_tokens": "mx",
    "batch_size": "bs",
    "seed": "seed",
}


class QuizLoader:
    """JSONL 파일에서 퀴즈를 로드합니다."""
    
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        self.quizzes = []
        self.quiz_configs = {}
        self.load_quizzes()
    
    def load_quizzes(self):
        """JSONL 파일에서 퀴즈 데이터를 로드합니다."""
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                quiz = json.loads(line.strip())
                self.quizzes.append(quiz)
        
        print(f"Loaded {len(self.quizzes)} quizzes from {self.jsonl_path}")
        
        # 각 퀴즈에 대해 정답 위치를 변경하여 여러 구성 생성
        for quiz in self.quizzes:
            quiz_id = quiz.get('id', len(self.quiz_configs))
            answer = quiz['answer']  # 정답
            wrong_options = quiz['options']  # 오답 선지들 리스트
            processed_passage = quiz['processed_passage']
            
            # 빈칸 표시를 ( ___ )로 변경
            processed_passage = processed_passage.replace('_______', '( ___ )')
            
            # 정답이 각 위치(A, B, C, D)에 오도록 4가지 구성 생성
            positions = ["A", "B", "C", "D"]
            
            for pos in positions:
                # 정답을 해당 위치에 배치
                options_dict = {}
                
                # 나머지 위치에 오답들을 배치
                wrong_idx = 0
                for position in positions:
                    if position == pos:
                        # 현재 위치에는 정답 배치
                        options_dict[position] = answer
                    else:
                        # 현재 위치에는 오답 배치
                        if wrong_idx < len(wrong_options):
                            options_dict[position] = wrong_options[wrong_idx]
                            wrong_idx += 1
                
                # 키 생성: quiz_{id}_{정답위치}
                key = f"quiz_{quiz_id}_{pos}"
                self.quiz_configs[key] = {
                    "text": processed_passage,
                    "options": options_dict,
                    "correct": pos,  # 정답 위치
                    "original_answer": answer,  # 원래 정답 텍스트
                    "quiz_id": quiz_id
                }
        
        print(f"Generated {len(self.quiz_configs)} quiz configurations")


class QuizBuilder:    
    """JSONL에서 로드한 퀴즈를 관리합니다."""
    
    def __init__(self, quiz_loader: QuizLoader = None):
        self.quiz_loader = quiz_loader
        self.quiz_configs = {}
    
    def load_from_jsonl(self, jsonl_path: str):
        """JSONL 파일에서 퀴즈를 로드합니다."""
        self.quiz_loader = QuizLoader(jsonl_path)
        self.quiz_configs = self.quiz_loader.quiz_configs
        print(f"Loaded {len(self.quiz_configs)} quiz configurations")
    
    def build(self, target: str) -> Tuple[str, Dict[str, str], str]:
        """특정 타겟의 퀴즈를 반환"""
        if target not in self.quiz_configs:
            valid_targets = ", ".join(list(self.quiz_configs.keys())[:10])
            raise ValueError(f"Invalid target. Available: {valid_targets}...")
        
        config = self.quiz_configs[target]
        return config["text"], config["options"], config["correct"]


class PromptBuilder:
    """Build chat messages with system prompt."""

    @staticmethod
    def build_chat_messages(paragraph: str, options: Dict[str, str]) -> List[Dict]:
        """Build chat messages with few-shot examples."""
        options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
        
        return [
            {
                "role": "system",
                "content": "You answer multiple choice questions with only a single letter."
            },
            {
                "role": "user",
                "content": (
                    "The capital of France is ( ___ ).\n"
                    "A. Paris\n"
                    "B. London\n"
                    "C. Berlin\n"
                    "D. Madrid\n\n"
                    "Your answer:"
                )
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": (
                    "2 + 2 = ( ___ )\n"
                    "A. 3\n"
                    "B. 4\n"
                    "C. 5\n"
                    "D. 6\n\n"
                    "Your answer:"
                )
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": (
                    "The largest ocean on Earth is ( ___ ).\n"
                    "A. Atlantic Ocean\n"
                    "B. Indian Ocean\n"
                    "C. Pacific Ocean\n"
                    "D. Arctic Ocean\n\n"
                    "Your answer:"
                )
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": (
                    "The sun rises in the ( ___ ).\n"
                    "A. North\n"
                    "B. South\n"
                    "C. West\n"
                    "D. East\n\n"
                    "Your answer:"
                )
            },
            {
                "role": "assistant",
                "content": "D"
            },
            {
                "role": "user",
                "content": (
                    f"{paragraph}\n\n"
                    f"{options_text}\n\n"
                    "Your answer:"
                )
            }
        ]


class ResponseAnalyzer:
    """Analyzes model responses and extracts answers."""
    
    @staticmethod
    def extract_letter(response: str) -> str:
        """Extract the first valid letter (A-D) from response."""
        response = response.strip()
        
        match = re.search(r"\b([A-D])\b", response, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        for char in response:
            if char.upper() in "ABCD":
                return char.upper()
        
        return "?"
    
    @staticmethod
    def analyze_predictions(predictions: List[str], correct: str) -> Dict:
        distribution = Counter(predictions)
        majority = max(distribution.items(), key=lambda x: (x[1], x[0]))[0] if distribution else "?"
        
        return {
            "distribution": dict(distribution),
            "majority": majority,
            "correct": correct,
            "is_correct": majority == correct
        }


class FileManager:
    """Manages file operations for saving results."""

    @staticmethod
    def build_filename(
        target: str,
        samples: int,
        defaults: Dict,
        params: Dict
    ) -> str:
        tags = [f"quiz_target={target}", f"samples={samples}"]
        
        for key, value in params.items():
            if key not in defaults or value == defaults[key]:
                continue
            
            param_name = PARAM_KEY_MAP.get(key, key)
            tags.append(f"{param_name}:{value}")
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return "__".join(tags) + f"__{timestamp}.jsonl"
    
    @staticmethod
    def save_results(
        save_path: str,
        metadata: Dict,
        prompt: str,
        raw_outputs: List[str],
        predictions: List[str]
    ) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "meta", **metadata}, ensure_ascii=False) + "\n")
            
            for i, (raw, pred) in enumerate(zip(raw_outputs, predictions)):
                record = {
                    "type": "sample",
                    "index": i,
                    "prompt": prompt,
                    "output_raw": raw,
                    "output_choice": pred,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    @staticmethod
    def append_final_result(target: str, majority: str, correct: str, results_dir: str) -> None:
        """Append final result to summary file with target label."""
        final_path = os.path.join(results_dir, "final_results.jsonl")
        os.makedirs(results_dir, exist_ok=True)
        
        with open(final_path, "a", encoding="utf-8") as f:
            data = {"target": target, "predict": majority, "correct": correct}
            f.write(json.dumps(data) + "\n")
    
    @staticmethod
    def save_experiment_metadata(
        results_dir: str,
        jsonl_path: str,
        model_name: str,
        samples: int,
        num_repeats: int,
        params: Dict
    ) -> None:
        """실험 메타데이터를 저장합니다."""
        metadata_path = os.path.join(results_dir, "experiment_metadata.json")
        os.makedirs(results_dir, exist_ok=True)
        
        metadata = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "jsonl_path": jsonl_path,
                "model_name": model_name,
                "samples_per_experiment": samples,
                "repeats_per_target": num_repeats,
            },
            "generation_params": params
        }
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


def chunked(lst: List, n: int) -> List[List]:
    """Split a list into chunks of size n."""
    if n <= 0:
        return [lst]
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def run_single_experiment(
    target: str,
    model_name: str,
    samples: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    show_all: bool,
    seed: int,
    debug_mode: bool,
    gen: FalconGenerator,
    quiz_builder: QuizBuilder,
    results_dir: str,
) -> Dict:
    """Run a single quiz experiment and return results."""
    
    paragraph, options, correct = quiz_builder.build(target)
    
    cfg = GenerateConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        return_only_new_text=True,
        do_sample=True,
    )
    
    messages = PromptBuilder.build_chat_messages(paragraph, options)
    prompt_text = gen.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    if seed is not None and seed != 0:
        import torch
        torch.manual_seed(seed)
    
    prompts = [prompt_text] * samples
    raw_outputs = []
    
    print(f"  Generating {samples} samples...")
    for i, chunk in enumerate(chunked(prompts, batch_size)):
        batch_outputs = gen.generate_batch(chunk, cfg)
        raw_outputs.extend(batch_outputs)
    
    predictions = [ResponseAnalyzer.extract_letter(output) for output in raw_outputs]
    analysis = ResponseAnalyzer.analyze_predictions(predictions, correct)
    
    print(f"  Distribution: {' '.join([f'{k}:{v}' for k, v in sorted(analysis['distribution'].items())])}")
    print(f"  Majority: {analysis['majority']} | Correct: {correct} | {'✓' if analysis['is_correct'] else '✗'}")
    
    params = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "seed": seed,
    }
    
    filename = FileManager.build_filename(target, samples, DEFAULT_PARAMS, params)
    save_path = os.path.join(results_dir, filename)
    
    metadata = {
        "model_name": model_name,
        "target": target,
        "options": options,
        "correct": correct,
        "samples": samples,
        "batch_size": batch_size,
        "params": params,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "analysis": analysis,
    }
    
    FileManager.save_results(save_path, metadata, prompt_text, raw_outputs, predictions)
    FileManager.append_final_result(target, analysis['majority'], correct, results_dir)
    
    return analysis


def run(
    jsonl_path: str,
    model_name: str = "tiiuae/falcon-7b-instruct",
    samples: int = 32,
    batch_size: int = 16,
    max_new_tokens: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    show_all: bool = False,
    seed: int = 0,
    debug_mode: bool = False,
    num_repeats: int = 15,
    test_number: int = 1,
) -> None:
    """
    JSONL 파일에서 퀴즈를 로드하여 실험합니다.
    
    Args:
        jsonl_path: 퀴즈 데이터가 담긴 JSONL 파일 경로 (필수)
        model_name: 사용할 모델 이름
        samples: 실험당 샘플 수
        num_repeats: 각 타겟당 반복 횟수
        test_number: 테스트 번호 (결과 디렉토리 이름에 사용)
    """
    
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    load_dotenv()
    base_results_dir = os.getenv("EJ_RESULTS_DIR", "/data2/PromptMIA/Falcon_ej/results").rstrip("/")
    
    sub_folder = f"test_{test_number}"
    results_dir = os.path.join(base_results_dir, sub_folder)
    
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"EXPERIMENT {test_number}: LOADING QUIZZES FROM JSONL")
    print("=" * 70)
    print(f"JSONL Path: {jsonl_path}")
    print(f"Results Directory: {results_dir}")
    
    quiz_builder = QuizBuilder()
    quiz_builder.load_from_jsonl(jsonl_path)
    
    params = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "seed": seed,
    }
    
    FileManager.save_experiment_metadata(
        results_dir=results_dir,
        jsonl_path=jsonl_path,
        model_name=model_name,
        samples=samples,
        num_repeats=num_repeats,
        params=params
    )
    print(f"Experiment metadata saved to: {os.path.join(results_dir, 'experiment_metadata.json')}")
    
    print("\n" + "=" * 70)
    print("INITIALIZING FALCON MODEL")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Samples per experiment: {samples}")
    print(f"Repeats per target: {num_repeats}")
    print("=" * 70 + "\n")
    
    gen = FalconGenerator(model_name=model_name)
    
    targets = list(quiz_builder.quiz_configs.keys())
    print(f"\nTotal quiz targets: {len(targets)}")
    
    total_experiments = len(targets) * num_repeats
    current_exp = 0
    
    for target in targets:
        print("\n" + "=" * 70)
        print(f"TARGET: {target.upper()}")
        print("=" * 70)
        
        for repeat in range(num_repeats):
            current_exp += 1
            print(f"\n[{current_exp}/{total_experiments}] Run {repeat + 1}/{num_repeats} for '{target}'")
            
            run_single_experiment(
                target=target,
                model_name=model_name,
                samples=samples,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                show_all=show_all,
                seed=seed,
                debug_mode=debug_mode,
                gen=gen,
                quiz_builder=quiz_builder,
                results_dir=results_dir,
            )
    
    print("\n" + "=" * 70)
    print(f"EXPERIMENT {test_number} COMPLETED")
    print("=" * 70)
    print(f"Total experiments: {total_experiments}")
    print(f"Results saved to: {results_dir}")
    print(f"Metadata file: {os.path.join(results_dir, 'experiment_metadata.json')}")
    print(f"Summary file: {os.path.join(results_dir, 'final_results.jsonl')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    fire.Fire(run)