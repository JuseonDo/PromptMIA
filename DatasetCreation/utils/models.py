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

    def get_masked_predictions(self, texts: List[str], top_k: int = 10, batch_size: int = 16):
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            results += self.pipeline(batch_texts, top_k=top_k)
        return results


if __name__ == "__main__":
    model = MLM("google-bert/bert-base-uncased")
    a = """
[MASK]?
this was a high school project for a president campaign in our government class, yes thats him, for a school project, you guys are crazy
i know his dad from work. very cool and funny guy!!
"""
    results = model.get_masked_predictions([a])
    for res in results:
        print(res)

"""
Pesky?
this was a high school project for a president campaign in our government class, yes thats him, for a school project, you guys are crazy
i know his dad from work. very cool and funny guy!!
"""