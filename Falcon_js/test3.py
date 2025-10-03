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
    """Builds quiz questions from the original text."""
    
    QUIZ_CONFIGS = {  # season, episode, episode_title, premiere_date, host, rob_times, russell_times
        "season": {
            "replace": ("Season 22", "Season ( ___ )"),
            "options": {"A": "20", "B": "21", "C": "22", "D": "23"},
            "correct": "C"
        },
        "episode": {
            "replace": ("Episode 11", "Episode ( ___ )"),
            "options": {"A": "9", "B": "10", "C": "11", "D": "12"},
            "correct": "C"
        },
        "episode_title": {
            "replace": ("'Mystery Package'", "'( ___ )'"),
            "options": {
                "A": "Mystery Package",
                "B": "Rice Wars",
                "C": "Redemption Island",
                "D": "Final Showdown"
            },
            "correct": "A"
        },
        "premiere_date": {
            "replace": ("May 31, 2000", "( ___ )"),
            "options": {
                "A": "May 28, 2000",
                "B": "May 31, 2000",
                "C": "June 1, 2000",
                "D": "June 15, 2000"
            },
            "correct": "B"
        },
        "host": {
            "replace": ("Jeff Probst", "( ___ )"),
            "options": {
                "A": "Phil Keoghan",
                "B": "Ryan Seacrest",
                "C": "Jeff Probst",
                "D": "Mark Burnett"
            },
            "correct": "C"
        },
        "rob_times": {
            "replace": ("Rob's fourth", "Rob's ( ___ )"),
            "options": {"A": "second", "B": "third", "C": "fourth", "D": "fifth"},
            "correct": "C"
        },
        "russell_times": {
            "replace": ("Russell's third", "Russell's ( ___ )"),
            "options": {"A": "second", "B": "third", "C": "fourth", "D": "fifth"},
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
    def append_final_result(majority: str, correct: str, results_dir: str) -> None:
        """Append final result to summary file."""
        final_path = os.path.join(results_dir, "final_results4.jsonl")
        os.makedirs(results_dir, exist_ok=True)
        
        with open(final_path, "a", encoding="utf-8") as f:
            data = {"predict": majority, "correct": correct}
            f.write(json.dumps(data) + "\n")


def chunked(lst: List, n: int) -> List[List]:
    """Split a list into chunks of size n."""
    if n <= 0:
        return [lst]
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def run(
    target: str = "year",
    model_name: str = "tiiuae/falcon-7b-instruct",
    samples: int = 16,
    batch_size: int = 8,
    max_new_tokens: int = 5,  # 2에서 5로 기본값 변경
    temperature: float = 0.3,  # 0.6에서 0.3으로 낮춤 (더 결정적)
    top_p: float = 0.9,
    top_k: int = 50,  # 0에서 50으로 (다양성 제한)
    repetition_penalty: float = 1.2,  # 1.05에서 1.2로 증가
    show_all: bool = True,
    seed: int = 0,
    debug_mode: bool = True,
) -> None:
    """
    Run quiz experiment with chat template and system prompt.
    
    Args:
        target: Quiz target (season, episode, etc.)
        model_name: HuggingFace model name
        samples: Number of samples to generate
        batch_size: Batch size for generation
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty
        show_all: Show all individual outputs
        seed: Random seed (0 = no seed)
        debug_mode: Show debug information
    """
    # Load environment
    load_dotenv()
    
    # Build quiz
    paragraph, options, correct = QuizBuilder.build(target)
    
    # Initialize model and config
    gen = FalconGenerator(model_name=model_name)
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
    
    # Debug output
    if debug_mode:
        print("\n" + "=" * 50)
        print("PROMPT SENT TO MODEL (with chat template)")
        print("=" * 50)
        print(prompt_text)
        print("=" * 50 + "\n")
        
        print("=" * 50)
        print("GENERATION CONFIG")
        print("=" * 50)
        print(f"Temperature: {temperature}")
        print(f"Top-p: {top_p}")
        print(f"Top-k: {top_k}")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Repetition penalty: {repetition_penalty}")
        print(f"Seed: {seed if seed != 0 else 'Not set'}")
        print("=" * 50 + "\n")
    
    # Set seed if specified
    if seed != 0:
        import torch
        torch.manual_seed(seed)
    
    # Generate responses in batches
    prompts = [prompt_text] * samples
    raw_outputs = []
    
    print("Generating responses...")
    for i, chunk in enumerate(chunked(prompts, batch_size)):
        if debug_mode:
            print(f"  Batch {i+1}/{(samples + batch_size - 1) // batch_size}...", end=" ")
        batch_outputs = gen.generate_batch(chunk, cfg)
        raw_outputs.extend(batch_outputs)
        if debug_mode:
            print(f"Done (first: '{batch_outputs[0][:30]}...')")
    print()
    
    # Analyze responses
    predictions = [ResponseAnalyzer.extract_letter(output) for output in raw_outputs]
    analysis = ResponseAnalyzer.analyze_predictions(predictions, correct)
    
    # Display results
    print("\n" + "=" * 50)
    print("QUIZ")
    print("=" * 50)
    print(paragraph)
    print("\nOptions:")
    for key, value in options.items():
        print(f"  {key}. {value}")
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Samples: {samples} | Batch Size: {batch_size}")
    print(f"Distribution: {' '.join([f'{k}:{v}' for k, v in sorted(analysis['distribution'].items())])}")
    print(f"Majority Vote: {analysis['majority']}")
    print(f"Correct Answer: {correct}")
    print(f"Status: {'✓ CORRECT' if analysis['is_correct'] else '✗ INCORRECT'}")
    
    if show_all:
        print("\n" + "-" * 50)
        print("Individual Outputs")
        print("-" * 50)
        for i, (raw, pred) in enumerate(zip(raw_outputs, predictions)):
            status = "✓" if pred == correct else "✗"
            print(f"[{i:02d}] {status} {pred} | '{raw.strip()}'")
    
    # Debug: Check for bias
    if debug_mode:
        print("\n" + "=" * 50)
        print("BIAS ANALYSIS")
        print("=" * 50)
        total = len(predictions)
        for choice in ["A", "B", "C", "D", "?"]:
            count = predictions.count(choice)
            pct = (count / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"{choice}: {count:3d}/{total} ({pct:5.1f}%) {bar}")
        print("=" * 50 + "\n")
    
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
    
    # Get RESULTS_DIR from .env
    results_dir = os.getenv("RESULTS_DIR").rstrip("/")
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
    print(f"\n[✓] Results saved to: {save_path}")
    
    FileManager.append_final_result(analysis['majority'], correct, results_dir)
    print(f"[✓] Final result appended to: {os.path.join(results_dir)}\n")


if __name__ == "__main__":
    
    for i in range(10):
        fire.Fire(run)