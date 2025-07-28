import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from bilstm_crf_model import BiLSTM_CRF
from data_utils import bio_format
from seqeval.metrics import classification_report as seqeval_report

# ==== Sample Data ====
data = [
    {"text": "Osh juda mazali edi", "aspect_terms": ["Osh"]},
    {"text": "Lag'mon biroz sho'r edi", "aspect_terms": ["Lag'mon"]},
]

# ==== Vocab + Label encoder ====
from collections import Counter
word_counter = Counter()
tag_counter = Counter()

sentences, tags = [], []
for d in data:
    toks, lbls = bio_format(d["text"], d["aspect_terms"])
    word_counter.update(toks)
    tag_counter.update(lbls)
    sentences.append(toks)
    tags.append(lbls)

word2idx = {w: i+2 for i, w in enumerate(word_counter)}
word2idx["<PAD>"] = 0
word2idx["<UNK>"] = 1

tag2idx = {t: i for i, t in enumerate(tag_counter)}
idx2tag = {i: t for t, i in tag2idx.items()}

def encode(seq, mapper, pad_len):
    return [mapper.get(tok, mapper["<UNK>"]) for tok in seq] + [0]*(pad_len - len(seq))

max_len = max(len(s) for s in sentences)
X = [encode(s, word2idx, max_len) for s in sentences]
y = [encode(t, tag2idx, max_len) for t in tags]

X = torch.tensor(X)
y = torch.tensor(y)
mask = (X != 0)

# ==== Train/Test Split ====
X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(X, y, mask, test_size=0.3, random_state=42)

# ==== Model ====
model = BiLSTM_CRF(len(word2idx), len(tag2idx))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ==== Training ====
for epoch in range(20):
    model.train()
    loss = model.loss(X_train, y_train, mask_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ==== Evaluation ====
model.eval()
with torch.no_grad():
    predictions = model.predict(X_test, mask_test)

true_tags = [[idx2tag[i.item()] for i in row[:mask.sum().item()]] for row, mask in zip(y_test, mask_test)]
pred_tags = [[idx2tag[i] for i in row[:mask.sum().item()]] for row, mask in zip(predictions, mask_test)]

print(seqeval_report(true_tags, pred_tags))
