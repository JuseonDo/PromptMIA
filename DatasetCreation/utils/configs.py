from dataclasses import dataclass

@dataclass
class ModelNames:
    bert: str = "google-bert/bert-base-uncased"
    distilbert: str = "distilbert/distilbert-base-uncased"