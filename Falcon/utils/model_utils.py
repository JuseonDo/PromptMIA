from typing import List, Optional
import os
import dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class FalconGenerator:
    """Simplified Falcon model generator with default values."""

    def __init__(self, model_name: str = "tiiuae/falcon-7b-instruct"):
        """Initialize Falcon generator with minimal configuration."""
        dotenv.load_dotenv()
        cache_dir = os.getenv("MODEL_CACHE_DIR")

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to('cuda' if torch.cuda.is_available() else 'cpu')

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = self.model.device

    @torch.inference_mode()
    def generate(self, prompts: List[str], max_new_tokens: int = 5, **kwargs) -> List[str]:
        """
        Generate text with configurable parameters.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate (default: 5)
            **kwargs: Optional generation parameters (temperature, top_p, top_k, repetition_penalty, etc.)
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # Default generation parameters
        gen_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.6,
            "top_p": 0.9,
            "top_k": 0,
            "repetition_penalty": 1.05,
            "do_sample": True,
            "use_cache": False,  # Required for legacy Falcon model
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        # Override with user-provided kwargs
        gen_params.update(kwargs)

        # Generate
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_params
        )

        # Decode only new tokens
        results = []
        for i, output in enumerate(outputs):
            input_len = (inputs["input_ids"][i] != self.tokenizer.pad_token_id).sum().item()
            new_tokens = output[input_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(text)

        return results
