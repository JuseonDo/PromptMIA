from datasets import load_dataset

# openllm dataset sample
dataset = load_dataset(
    "togethercomputer/RedPajama-Data-1T",
    streaming=True
)

print(dataset)
print(next(iter(dataset['train'])))