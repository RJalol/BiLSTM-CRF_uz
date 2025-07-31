import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seq_classification_report
from model.bilstm_crf_model import BiLSTM_CRF
from utils.dataset import TokenLabelDataset
from gensim.models import KeyedVectors
import tempfile

# 1. Fayllar va sozlamalar
EMBEDDING_BIN_PATH = "uzbek-word2vec.bin"
MODEL_DIR = "model"
DATA_PATH = "data/train_bio.json"
BATCH_SIZE = 16
EPOCHS = 40
LR = 0.0003
HIDDEN_DIM = 256
MAX_LEN = 81

# 2. Qurilmani aniqlash
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. .bin embedding faylini yuklash
print("📥 Embedding yuklanmoqda...")
w2v_model = KeyedVectors.load_word2vec_format(EMBEDDING_BIN_PATH, binary=True)
embedding_dim = w2v_model.vector_size

word2idx = {"<PAD>": 0, "<UNK>": 1}
embedding_matrix = [np.zeros(embedding_dim), np.random.uniform(-0.25, 0.25, embedding_dim)]

for word in w2v_model.index_to_key:
    word2idx[word] = len(word2idx)
    embedding_matrix.append(w2v_model[word])

embedding_matrix = torch.tensor(np.array(embedding_matrix), dtype=torch.float)

# 4. JSON ma'lumotlarni yuklash
with open(DATA_PATH, encoding='utf-8') as f:
    dataset = json.load(f)

# 5. Tag mapping
tag_set = set(label for item in dataset for label in item['labels'])
tag2idx = {tag: idx for idx, tag in enumerate(sorted(tag_set))}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

# Saqlash
os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "word2idx.json"), "w", encoding="utf-8") as f:
    json.dump(word2idx, f, ensure_ascii=False, indent=2)
with open(os.path.join(MODEL_DIR, "tag2idx.json"), "w", encoding="utf-8") as f:
    json.dump(tag2idx, f, ensure_ascii=False, indent=2)

# 6. Train/Test ajratish
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Vaqtinchalik JSON fayllarga yozib saqlaymiz
tmp_train_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
tmp_val_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")

json.dump(train_data, tmp_train_path, ensure_ascii=False)
json.dump(val_data, tmp_val_path, ensure_ascii=False)
tmp_train_path.close()
tmp_val_path.close()

train_dataset = TokenLabelDataset(tmp_train_path.name, word2idx, tag2idx, max_len=MAX_LEN)
val_dataset = TokenLabelDataset(tmp_val_path.name, word2idx, tag2idx, max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 7. Modelni yaratish
model = BiLSTM_CRF(vocab_size=len(word2idx), tagset_size=len(tag2idx),
                   embedding_dim=embedding_dim, hidden_dim=HIDDEN_DIM)
model.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 8. Trening
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for x, y, mask in train_loader:
        x, y, mask = x.to(device), y.to(device), mask.bool().to(device)
        loss = model.loss(x, y, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {avg_loss:.4f}")

# 9. Baholash
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x, y, mask in val_loader:
        x, y, mask = x.to(device), y.to(device), mask.bool().to(device)
        predictions = model.predict(x, mask)
        for i in range(len(predictions)):
            true_tags = [idx2tag[idx.item()] for idx, m in zip(y[i], mask[i]) if m.item()]
            pred_tags = [idx2tag[idx] for idx in predictions[i]]
            y_true.append(true_tags)
            y_pred.append(pred_tags)

print("\n📊 Validation Result:")
print(seq_classification_report(y_true, y_pred, zero_division=0))

# 10. Modelni saqlash
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "bilstm_crf.pth"))
