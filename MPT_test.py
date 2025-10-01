from MPT.utils.model_utils import MPTGenerator, GenerateConfig

import os
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

# Constants
ORIGINAL_TEXT = (
    "Beginners BBQ Class Taking Place in Missoula!\n"
    "Do you want to get better at making delicious BBQ? You will have the opportunity, "
    "put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, "
    "Tony Balay from Lonestar Smoke Rangers. He will be teaching a beginner level class "
    "for everyone who wants to get better with their culinary skills.\n"
    "He will teach you everything you need to know to compete in a KCBS BBQ competition, "
    "including techniques, recipes, timelines, meat selection and trimming, plus smoker "
    "and fire information.\n"
    "The cost to be in the class is $35 per person, and for spectators it is free. "
    "Included in the cost will be either a t-shirt or apron and you will be tasting "
    "samples of each meat that is prepared."
)

class QuizBuilder:    
    QUIZ_CONFIGS = {
        "event_location": {
            "replace": ("Missoula", "( ___ )"),
            "options": {
                "A": "Helena",
                "B": "Billings", 
                "C": "Missoula",
                "D": "Bozeman"
            },
            "correct": "C"
        },
        "event_day": {
            "replace": ("Thursday", "( ___ )"),
            "options": {
                "A": "Wednesday",
                "B": "Thursday",
                "C": "Friday",
                "D": "Saturday"
            },
            "correct": "B"
        },
        "event_date": {
            "replace": ("September 22nd", "( ___ )"),
            "options": {
                "A": "September 21st",
                "B": "September 22nd",
                "C": "September 23rd",
                "D": "September 29th"
            },
            "correct": "B"
        },
        "instructor_name": {
            "replace": ("Tony Balay", "( ___ )"),
            "options": {
                "A": "Tony Blair",
                "B": "Tony Balay",
                "C": "Tony Bennett",
                "D": "Tony Parker"
            },
            "correct": "B"
        },
        "team_name": {
            "replace": ("Lonestar Smoke Rangers", "( ___ )"),
            "options": {
                "A": "Texas BBQ Masters",
                "B": "Smoke Legends",
                "C": "Lonestar Smoke Rangers",
                "D": "BBQ Champions United"
            },
            "correct": "C"
        },
        "competition_org": {
            "replace": ("KCBS", "( ___ )"),
            "options": {
                "A": "KCBS",
                "B": "IBCA",
                "C": "MBN",
                "D": "SCA"
            },
            "correct": "A"
        },
        "class_cost": {
            "replace": ("$35", "( ___ )"),
            "options": {
                "A": "$25",
                "B": "$30",
                "C": "$35",
                "D": "$40"
            },
            "correct": "C"
        },
        "spectator_cost": {
            "replace": ("free", "( ___ )"),
            "options": {
                "A": "$5",
                "B": "$10",
                "C": "free",
                "D": "$15"
            },
            "correct": "C"
        },
        "included_items": {
            "replace": ("t-shirt or apron", "( ___ )"),
            "options": {
                "A": "hat or gloves",
                "B": "t-shirt or apron",
                "C": "cookbook or hat",
                "D": "gloves or tongs"
            },
            "correct": "B"
        }
    }
    
    @classmethod
    def build(cls, target: str) -> Tuple[str, Dict[str, str], str]:
        if target not in cls.QUIZ_CONFIGS:
            valid_targets = ", ".join(cls.QUIZ_CONFIGS.keys())
            raise ValueError(f"Invalid target. Must be one of: {valid_targets}")
        
        config = cls.QUIZ_CONFIGS[target]
        old_text, new_text = config["replace"]
        paragraph = ORIGINAL_TEXT.replace(old_text, new_text, 1)
        
        return paragraph, config["options"], config["correct"]


class PromptBuilder:
    """Build chat messages with system prompt for MPT."""

    @staticmethod
    def build_chat_messages(paragraph: str, options: Dict[str, str]) -> List[Dict]:
        """Build chat messages with few-shot examples for MPT-7B-Chat."""
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
                    "A. London\n"
                    "B. Madrid\n"
                    "C. Berlin\n"
                    "D. Paris\n\n"
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
        
        # Try to find a letter with word boundaries
        match = re.search(r"\b([A-D])\b", response, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Fall back to finding any A-D character
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
            # Write metadata
            f.write(json.dumps({"type": "meta", **metadata}, ensure_ascii=False) + "\n")
            
            # Write each sample
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
        final_path = os.path.join(results_dir, "final_results_mpt.jsonl")
        os.makedirs(results_dir, exist_ok=True)
        
        with open(final_path, "a", encoding="utf-8") as f:
            data = {"target": target, "predict": majority, "correct": correct}
            f.write(json.dumps(data) + "\n")


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
    gen: MPTGenerator,
    results_dir: str,
) -> Dict:
    """Run a single quiz experiment and return results."""
    
    # Build quiz
    paragraph, options, correct = QuizBuilder.build(target)
    
    # Initialize config
    cfg = GenerateConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        return_only_new_text=True,
        do_sample=True,
    )
    
    # Build chat messages and format for MPT
    messages = PromptBuilder.build_chat_messages(paragraph, options)
    prompt_text = gen.format_chat_prompt(messages)
    
    # Set seed if specified
    if seed is not None and seed != 0:
        import torch
        torch.manual_seed(seed)
    
    # Generate responses in batches
    prompts = [prompt_text] * samples
    raw_outputs = []
    
    print(f"  Generating {samples} samples...")
    for i, chunk in enumerate(chunked(prompts, batch_size)):
        batch_outputs = gen.generate_batch(chunk, cfg, use_chat_format=False)
        raw_outputs.extend(batch_outputs)
    
    # Analyze responses
    predictions = [ResponseAnalyzer.extract_letter(output) for output in raw_outputs]
    analysis = ResponseAnalyzer.analyze_predictions(predictions, correct)
    
    # Display results
    print(f"  Distribution: {' '.join([f'{k}:{v}' for k, v in sorted(analysis['distribution'].items())])}")
    print(f"  Majority: {analysis['majority']} | Correct: {correct} | {'✓' if analysis['is_correct'] else '✗'}")
    
    # Save results
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
    model_name: str = "mosaicml/mpt-7b-chat",
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
    num_repeats: int = 10,
    sub_folder: str = "test_mpt1"
) -> None:

    # Load environment
    load_dotenv()
    base_results_dir = os.getenv("RESULTS_DIR_MPT").rstrip("/")
    results_dir = os.path.join(base_results_dir, sub_folder)
    
    # Initialize model once
    print("=" * 70)
    print("INITIALIZING MPT MODEL")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Samples per experiment: {samples}")
    print(f"Repeats per target: {num_repeats}")
    print("=" * 70 + "\n")
    
    gen = MPTGenerator(
        model_name=model_name,
        cache_env_key="MODEL_CACHE_DIR"
    )
    
    # Define all targets
    targets = ["event_location", "event_day", "event_date", "instructor_name",
               "team_name", "competition_org", "class_cost", "spectator_cost",
               "included_items"]
    
    # Run experiments
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
                results_dir=results_dir,
            )
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 70)
    print(f"Total experiments: {total_experiments}")
    print(f"Results saved to: {results_dir}")
    print(f"Summary file: {os.path.join(results_dir, 'final_results_mpt.jsonl')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    fire.Fire(run)