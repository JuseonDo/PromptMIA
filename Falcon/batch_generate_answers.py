#!/usr/bin/env python3
"""
Batch answer generation script for Falcon model with few-shot prompting.
Simplified and refactored version with minimal dependencies.
"""

import sys
import os
import json
import re
import fire
from typing import List, Dict
from collections import Counter
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Falcon.utils.model_utils import FalconGenerator
from Falcon.utils.dataset_models import RefinedWebQADataset


# Few-shot examples for prompting
FEW_SHOT_EXAMPLES = [
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
    {"role": "assistant", "content": "A"},
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
    {"role": "assistant", "content": "B"},
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
    {"role": "assistant", "content": "C"},
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
    {"role": "assistant", "content": "D"},
]


def build_prompt(passage: str, options: Dict[str, str], tokenizer) -> str:
    """Build few-shot prompt for the model."""
    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])

    messages = FEW_SHOT_EXAMPLES + [
        {
            "role": "user",
            "content": f"{passage}\n\n{options_text}\n\nYour answer:"
        }
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_answer(response: str) -> str:
    """Extract answer letter (A-D) from model response."""
    response = response.strip()

    match = re.search(r"\b([A-D])\b", response, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    for char in response:
        if char.upper() in "ABCD":
            return char.upper()

    return "?"


def get_distribution(predictions: List[str]) -> Dict:
    """Get answer distribution."""
    return dict(Counter(predictions))


def run(
    dataset_path: str = "/data2/PromptMIA/Dataset/falcon_dataset_refinedweb_origianl.json",
    model_name: str = "tiiuae/falcon-7b-instruct",
    samples: int = 480,
    batch_size: int = 16,
    output_dir: str = None,
    max_new_tokens: int = 5,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
    repetition_penalty: float = None,
):
    """
    Batch generate answers for quiz dataset.

    Args:
        dataset_path: Path to dataset JSON file
        model_name: Hugging Face model name (default: tiiuae/falcon-7b-instruct)
        samples: Number of samples to generate per question (default: 32)
        batch_size: Batch size for generation (default: 8)
        output_dir: Output directory for results (default: Falcon/results)
        max_new_tokens: Maximum tokens to generate (default: 5)
        temperature: Sampling temperature (optional, uses model default if not set)
        top_p: Top-p sampling (optional, uses model default if not set)
        top_k: Top-k sampling (optional, uses model default if not set)
        repetition_penalty: Repetition penalty (optional, uses model default if not set)
    """
    # Setup
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "results/retests")
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    dataset = RefinedWebQADataset(dataset_path)
    print(f"Loaded {len(dataset)} questions")

    # Initialize model
    print(f"Loading model: {model_name}")
    generator = FalconGenerator(model_name=model_name)

    # Build generation parameters (only include non-None values)
    gen_params = {"max_new_tokens": max_new_tokens}
    if temperature is not None:
        gen_params["temperature"] = temperature
    if top_p is not None:
        gen_params["top_p"] = top_p
    if top_k is not None:
        gen_params["top_k"] = top_k
    if repetition_penalty is not None:
        gen_params["repetition_penalty"] = repetition_penalty

    print(f"Generation parameters: {gen_params}")

    # Results storage
    results_file = os.path.join(
        output_dir,
        f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    # Process each question
    for idx, quiz in enumerate(tqdm(dataset, desc="Processing questions")):
        print(f"\n[{idx+1}/{len(dataset)}] Processing question {quiz.id}")

        # Prepare quiz configurations (A, B, C, D positions)
        for correct_pos in tqdm(["A", "B", "C", "D"], desc=f"  Quiz {quiz.id} positions", leave=False):
            # Build options with correct answer at specified position
            options = {}
            wrong_idx = 0
            for pos in ["A", "B", "C", "D"]:
                if pos == correct_pos:
                    options[pos] = quiz.answer
                else:
                    if wrong_idx < len(quiz.options):
                        options[pos] = quiz.options[wrong_idx]
                        wrong_idx += 1

            # Format passage
            passage = quiz.processed_passage.replace('_______', '( ___ )')

            # Build prompt
            prompt = build_prompt(passage, options, generator.tokenizer)

            # Print first prompt for inspection
            if correct_pos == "A":
                print(f"\n{'='*70}")
                print("PROMPT (first instance):")
                print(f"{'='*70}")
                print(prompt)
                print(f"{'='*70}\n")

            # Generate samples in batches
            prompts = [prompt] * samples
            all_responses = []

            for i in tqdm(range(0, samples, batch_size), desc=f"    Generating position {correct_pos}", leave=False):
                batch = prompts[i:i + batch_size]
                responses = generator.generate(batch, **gen_params)
                all_responses.extend(responses)

            # Extract predictions
            predictions = [extract_answer(resp) for resp in all_responses]
            distribution = get_distribution(predictions)

            # Log results
            print(f"  Position: {correct_pos} | Distribution: {distribution}")

            # Save to file
            with open(results_file, "a", encoding="utf-8") as f:
                result = {
                    "quiz_id": quiz.id,
                    "correct_position": correct_pos,
                    "passage": passage,
                    "options": options,
                    "samples": samples,
                    "raw_outputs": all_responses,
                    "predictions": predictions,
                    "distribution": distribution,
                    "timestamp": datetime.now().isoformat()
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\n✓ Results saved to {results_file}")


if __name__ == "__main__":
    fire.Fire(run)
