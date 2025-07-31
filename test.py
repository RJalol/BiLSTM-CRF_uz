import torch
import json
import numpy as np
import torch.nn as nn
from model.bilstm_crf_model import BiLSTM_CRF
from utils.dataset import TokenLabelDataset
from gensim.models import KeyedVectors

# Sozlamalar
MODEL_PATH = "model/bilstm_crf.pth"
WORD2IDX_PATH = "model/word2idx.json"
TAG2IDX_PATH = "model/tag2idx.json"
EMBEDDING_BIN_PATH = "uzbek-word2vec.bin"
TEST_PATH = "data/test_bio.json"
MAX_LEN = 81

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Word2Vec modelini yuklash
print("📥 Embedding yuklanmoqda...")
w2v_model = KeyedVectors.load_word2vec_format(EMBEDDING_BIN_PATH, binary=True)
embedding_dim = w2v_model.vector_size

# word2idx va tag2idx
with open(WORD2IDX_PATH, encoding="utf-8") as f:
    word2idx = json.load(f)
with open(TAG2IDX_PATH, encoding="utf-8") as f:
    tag2idx = json.load(f)
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

# Embedding matrix
embedding_matrix = [np.zeros(embedding_dim), np.random.uniform(-0.25, 0.25, embedding_dim)]
for word in w2v_model.index_to_key:
    embedding_matrix.append(w2v_model[word])
embedding_matrix = torch.tensor(np.array(embedding_matrix), dtype=torch.float)

# Dataset
test_dataset = TokenLabelDataset(TEST_PATH, word2idx, tag2idx, max_len=MAX_LEN)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

# Model
model = BiLSTM_CRF(vocab_size=len(word2idx), tagset_size=len(tag2idx),
                   embedding_dim=embedding_dim, hidden_dim=256)
model.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Baholash
from seqeval.metrics import classification_report

y_true, y_pred = [], []
with torch.no_grad():
    for x, y, mask in test_loader:
        x, y, mask = x.to(device), y.to(device), mask.bool().to(device)
        preds = model.predict(x, mask)
        for i in range(len(preds)):
            true_tags = [idx2tag[idx.item()] for idx, m in zip(y[i], mask[i]) if m.item()]
            pred_tags = [idx2tag[idx] for idx in preds[i]]
            y_true.append(true_tags)
            y_pred.append(pred_tags)

print("\n📊 Test natijalari:")
print(classification_report(y_true, y_pred, zero_division=0))
