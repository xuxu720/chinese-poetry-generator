# src/model_lstm.py
import torch
import torch.nn as nn

class PoetryLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden=512, num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)  # (B, T, E)
        outputs, hidden = self.lstm(emb, hidden)  # (B, T, H)
        logits = self.fc(outputs)  # (B, T, V)
        return logits, hidden
