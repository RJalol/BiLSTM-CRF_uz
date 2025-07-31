import json

def get_max_seq_len(json_path):
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    
    max_len = max(len(item["tokens"]) for item in data)
    avg_len = sum(len(item["tokens"]) for item in data) / len(data)
    
    print(f"🔍 Max sequence length: {max_len}")
    print(f"📊 Average sequence length: {avg_len:.2f}")
    return max_len

if __name__ == "__main__":
    max_len = get_max_seq_len("data/train_bio.json")
    print(f"\nSuggested MAX_LEN: {max_len + 10}  # Adding buffer")
