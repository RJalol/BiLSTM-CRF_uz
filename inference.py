import torch
import torch.nn as nn
import numpy as np
import json
from model.bilstm_crf_model import BiLSTM_CRF
from gensim.models import KeyedVectors

# 🔧 Sozlamalar
MODEL_PATH = "model/bilstm_crf.pth"
WORD2IDX_PATH = "model/word2idx.json"
TAG2IDX_PATH = "model/tag2idx.json"
EMBEDDING_BIN_PATH = "uzbek-word2vec.bin"
MAX_LEN = 81

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📥 Word2Vec bin yuklash
print("📥 Word2Vec yuklanmoqda...")
w2v_model = KeyedVectors.load_word2vec_format(EMBEDDING_BIN_PATH, binary=True)
embedding_dim = w2v_model.vector_size

# 📘 Mappinglar
with open(WORD2IDX_PATH, encoding="utf-8") as f:
    word2idx = json.load(f)
with open(TAG2IDX_PATH, encoding="utf-8") as f:
    tag2idx = json.load(f)
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

# 📐 Embedding matritsa
embedding_matrix = [np.zeros(embedding_dim), np.random.uniform(-0.25, 0.25, embedding_dim)]
for word in w2v_model.index_to_key:
    embedding_matrix.append(w2v_model[word])
embedding_matrix = torch.tensor(np.array(embedding_matrix), dtype=torch.float)

# 🧠 Model yuklash
model = BiLSTM_CRF(vocab_size=len(word2idx), tagset_size=len(tag2idx),
                   embedding_dim=embedding_dim, hidden_dim=256)
model.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 🔮 Bashorat qilish
def predict_tags(text):
    tokens = text.strip().split()
    ids = [word2idx.get(tok, word2idx["<UNK>"]) for tok in tokens]
    mask = [1] * len(ids)

    # Padding
    pad_len = MAX_LEN - len(ids)
    if pad_len > 0:
        ids += [word2idx["<PAD>"]] * pad_len
        mask += [0] * pad_len
    else:
        ids = ids[:MAX_LEN]
        mask = mask[:MAX_LEN]
        tokens = tokens[:MAX_LEN]

    x = torch.tensor([ids], dtype=torch.long).to(device)
    mask = torch.tensor([mask], dtype=torch.bool).to(device)

    with torch.no_grad():
        preds = model.predict(x, mask)[0]
    tags = [idx2tag[i] for i in preds]
    return list(zip(tokens, tags[:len(tokens)]))

# ✏️ Foydalanuvchidan matn olish
text = input("\n✏️ Matn kiriting (so‘zlar bo‘shliq bilan ajratilgan):\n")
results = predict_tags(text)

# 📌 Natija
print("\n📌 Bashorat:")
for token, tag in results:
    print(f"{token:<15} → {tag}")
