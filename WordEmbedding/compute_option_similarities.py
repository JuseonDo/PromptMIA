import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
with open('/data2/PromptMIA/Dataset/falcon_dataset_refinedweb.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Load sentence transformer model
print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

results = []

for item in data:
    item_id = item['id']
    passage = item['processed_passage']
    answer = item['answer']
    options = item['options']

    # Create all options including the answer
    all_options = [answer] + options

    # Fill blanks with each option
    filled_sentences = []
    for option in all_options:
        filled_sentence = passage.replace('_______', option)
        filled_sentences.append(filled_sentence)

    # Compute embeddings
    embeddings = model.encode(filled_sentences)

    # Calculate pairwise cosine similarities
    similarities = cosine_similarity(embeddings)

    # Store results
    result = {
        'id': item_id,
        'answer': answer,
        'options': options,
        'all_options': all_options,
        'similarity_matrix': similarities.tolist()
    }

    # Calculate similarities between answer and each wrong option
    answer_vs_options = {}
    for i, option in enumerate(options):
        sim = similarities[0][i + 1]  # 0 is answer, 1+ are options
        answer_vs_options[f'answer_vs_option_{i+1}'] = float(sim)

    result['answer_vs_options'] = answer_vs_options

    # Calculate similarities between wrong options
    option_vs_option = {}
    for i in range(len(options)):
        for j in range(i + 1, len(options)):
            sim = similarities[i + 1][j + 1]
            option_vs_option[f'option_{i+1}_vs_option_{j+1}'] = float(sim)

    result['option_vs_option'] = option_vs_option

    results.append(result)

    print(f"Processed item {item_id}")

# Save results
output_file = '/data2/PromptMIA/WordEmbedding/option_similarities.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to {output_file}")

# Print summary statistics
print("\n=== Summary Statistics ===")
all_answer_vs_options = []
all_option_vs_option = []

for result in results:
    all_answer_vs_options.extend(result['answer_vs_options'].values())
    all_option_vs_option.extend(result['option_vs_option'].values())

print(f"Average similarity (answer vs wrong options): {np.mean(all_answer_vs_options):.4f}")
print(f"Average similarity (wrong options vs wrong options): {np.mean(all_option_vs_option):.4f}")
print(f"\nTotal items processed: {len(results)}")
