from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

def load_model(model_name = "tiiuae/falcon-7b-instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/data2/models")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir="/data2/models",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
      )
    return tokenizer, model
