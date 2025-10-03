from typing import List
import random
import json

class RefinedWebQAInstance:
    """
        A class to represent a single instance in the RefinedWeb QA dataset.
    """
    def __init__(
            self, 
            id: int,
            original_passage: str, 
            processed_passage: str, 
            answer: str, 
            options: List[str]
        ):
        self.original_passage = original_passage
        self.processed_passage = processed_passage
        self.options = options
        self.answer = answer
        self.id = id

    def to_text(self, answer_position: int = 0):
        options = self.options[:]
        random.shuffle(options)
        options.insert(answer_position, self.answer)
        options = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        return (
            f"Question:\n{self.processed_passage}\n"
            f"Options:\n{options}\n"
            "Your answer:"
        )

class RefinedWebQADataset:
    """
        A class to load and handle the RefinedWeb QA dataset.
    """
    dataset = []
    def __init__(self, path: str = None):
        if path is None:
            path = "Dataset/falcon_dataset_refinedweb.json"
        with open(path) as f:
            if path.endswith(".json"):
                self.dataset = [RefinedWebQAInstance(**line) for line in json.load(f)]

            elif path.endswith(".jsonl"):
                self.datatset = [RefinedWebQAInstance(json.loads(line)) for line in f]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        return self.dataset[idx]
        
    def to_text(self):
        return [line.to_text() for line in self.dataset]


if __name__ == "__main__":
    dataset = RefinedWebQADataset()
    print(dataset[0].to_text())