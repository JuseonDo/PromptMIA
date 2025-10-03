from Falcon.utils.model_utils import FalconGenerator, GenerateConfig
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
    "Watch Survivor Redemption Island Season 22 Episode 11: A Mystery Package "
    "Online S22e11 Free Stream Megavideo\n\n"
    "Article by StreamThatSeries\n\n"
    "Horray!time for another dose of very exciting reality series with lots of twists."
    "You must watch survivor redemption island season 22 episode 11 tonight "
    "with a title of 'Mystery Package' coz surely this will stir things up on the "
    "merge tribe murlonio. The show premiered on May 31, 2000 on CBS. "
    "It is hosted by veteran television personality Jeff Probst. "
    "The first U.S. season of Survivor followed the Swedish series, but subsequently "
    "the show has introduced several twists. Season 22 features the return of "
    "Rob Mariano and Russell Hantz. This is Russell's third time on the show and "
    "Rob's fourth."
)

class QuizBuilder:    
    QUIZ_CONFIGS = {
        "season_A": {
            "replace": ("Season 22", "Season ( ___ )"),
            "options": {"A": "22", "B": "21", "C": "20", "D": "23"},
            "correct": "A"
        },
        "season_B": {
            "replace": ("Season 22", "Season ( ___ )"),
            "options": {"A": "20", "B": "22", "C": "21", "D": "23"},
            "correct": "B"
        },
        "season_C": {
            "replace": ("Season 22", "Season ( ___ )"),
            "options": {"A": "20", "B": "21", "C": "22", "D": "23"},
            "correct": "C"
        },
        "season_D": {
            "replace": ("Season 22", "Season ( ___ )"),
            "options": {"A": "20", "B": "21", "C": "23", "D": "22"},
            "correct": "D"
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
        final_path = os.path.join(results_dir, "final_results_DCBA.jsonl")
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
    gen: FalconGenerator,
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
    
    # Build chat messages and apply template
    messages = PromptBuilder.build_chat_messages(paragraph, options)
    prompt_text = gen.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Set seed if specified
    if seed is not None and seed != 0:
        import torch
        torch.manual_seed(seed)
    
    # Generate responses in batches
    prompts = [prompt_text] * samples
    raw_outputs = []
    
    print(f"  Generating {samples} samples...")
    for i, chunk in enumerate(chunked(prompts, batch_size)):
        batch_outputs = gen.generate_batch(chunk, cfg)
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
    sub_folder: str = "testDCBA",
) -> None:

    # Load environment
    load_dotenv()
    base_results_dir = os.getenv("RESULTS_DIR").rstrip("/")
    results_dir = os.path.join(base_results_dir, sub_folder)
    
    # Initialize model once
    print("=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Samples per experiment: {samples}")
    print(f"Repeats per target: {num_repeats}")
    print("=" * 70 + "\n")
    
    gen = FalconGenerator(model_name=model_name)
    
    # Define all targets
    targets = ["season_A", "season_B", "season_C", "season_D"]
    
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
    print(f"Summary file: {os.path.join(results_dir, 'final_results_ABCD.jsonl')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    fire.Fire(run)