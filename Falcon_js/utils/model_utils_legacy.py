# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# import dotenv
# import os

# dotenv.load_dotenv()
# model_cache_dir = os.getenv("model_cache_dir")

# def load_model(model_name = "tiiuae/falcon-7b-instruct"):
#     tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         cache_dir=model_cache_dir,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         trust_remote_code=True
#       )
#     return tokenizer, model
