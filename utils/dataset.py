import torch
from torch.utils.data import Dataset
import json

class TokenLabelDataset(Dataset):
    def __init__(self, json_path, word2idx, tag2idx, max_len=100):
        self.data = json.load(open(json_path, encoding='utf-8'))
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def encode(self, seq, mapper):
        ids = [mapper.get(tok, mapper.get('<UNK>')) for tok in seq]
        ids = ids[:self.max_len]
        ids += [mapper.get('<PAD>')] * (self.max_len - len(ids))
        return ids

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        labels = item['labels']
        
        x = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
        y = [self.tag2idx.get(label, self.tag2idx["O"]) for label in labels]

        pad_len = self.max_len - len(x)
        if pad_len < 0:
            x = x[:self.max_len]
            y = y[:self.max_len]
        else:
            x += [self.word2idx["<PAD>"]] * pad_len
            y += [self.tag2idx["O"]] * pad_len

        input_len = min(len(tokens), self.max_len)
        mask = [1] * input_len + [0] * (self.max_len - input_len)

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long), torch.tensor(mask, dtype=torch.uint8)
