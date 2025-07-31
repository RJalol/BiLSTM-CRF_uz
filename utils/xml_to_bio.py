from datasets import load_dataset
import json
from sklearn.model_selection import train_test_split
from pathlib import Path

def convert_to_bio_format(dataset, max_len=100):
    bio_data = []

    for item in dataset:
        text = item["text"]
        aspect_terms = item["aspect_terms"]
        tokens = text.split()
        labels = ["O"] * len(tokens)

        for asp in aspect_terms:
            term_tokens = asp["term"].split()
            for i in range(len(tokens) - len(term_tokens) + 1):
                if tokens[i:i + len(term_tokens)] == term_tokens:
                    labels[i] = "B-ASP"
                    for j in range(1, len(term_tokens)):
                        labels[i + j] = "I-ASP"
                    break

        bio_data.append({
            "tokens": tokens[:max_len],
            "labels": labels[:max_len]
        })

    return bio_data

# 1. Load dataset from Hugging Face
dataset = load_dataset("Sanatbek/aspect-based-sentiment-analysis-uzbek")["train"]
dataset = list(dataset)  # ← YANGI: Dataset obyektini listga aylantiramiz

# 2. Split into train/test
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# 3. Convert to BIO format
train_bio = convert_to_bio_format(train_data)
test_bio = convert_to_bio_format(test_data)

# 4. Save to JSON
Path("data").mkdir(exist_ok=True)

with open("data/train_bio.json", "w", encoding="utf-8") as f:
    json.dump(train_bio, f, ensure_ascii=False, indent=2)

with open("data/test_bio.json", "w", encoding="utf-8") as f:
    json.dump(test_bio, f, ensure_ascii=False, indent=2)

print("✅ train_bio.json va test_bio.json yaratildi!")
