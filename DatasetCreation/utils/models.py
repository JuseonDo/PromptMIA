import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import pipeline
import dotenv
from typing import List

dotenv.load_dotenv()
cache_dir = os.getenv("MODEL_CACHE_DIR")

class MLM:
    def __init__(self, model_name: str):
        if "distil" in model_name:
            from transformers import DistilBertTokenizer, DistilBertModel as BertTokenizer, BertModel
        else:
            from transformers import BertTokenizer, BertModel

        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            cache_dir=cache_dir
        )
        self.model = BertModel.from_pretrained(
            "bert-base-uncased",
            cache_dir=cache_dir
        )
        self.model.eval()
        self.pipeline = pipeline(
            'fill-mask', 
            model=self.model_name, 
            tokenizer=self.tokenizer
        )

    def get_masked_predictions(self, texts: List[str], top_k: int = 10):
        results = self.pipeline(texts, top_k=top_k)
        return results


if __name__ == "__main__":
    model = MLM("google-bert/bert-base-uncased")
    results = model.get_masked_predictions(["The capital of France is [MASK]."])
    for res in results:
        print(res)