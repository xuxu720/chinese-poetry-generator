# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import json
import argparse


class PoetryDataset(Dataset):
    def __init__(self, path, vocab_path, max_len=128):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.pad_idx = self.word2idx["<PAD>"]
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        if len(ids) < self.max_len:
            ids = ids + [self.pad_idx] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        input_ids = ids[:-1]   # 前 n-1 个作为输入
        target_ids = ids[1:]   # 后 n-1 个作为预测目标
        return torch.tensor(input_ids), torch.tensor(target_ids)


def get_dataloader(path, vocab_path, batch_size=32, shuffle=True, max_len=128):
    dataset = PoetryDataset(path, vocab_path, max_len=max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="训练/验证数据路径 (jsonl)")
    parser.add_argument("--vocab", type=str, required=True, help="词表路径 (vocab.json)")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--max_len", type=int, default=128, help="最大序列长度")
    args = parser.parse_args()

    loader = get_dataloader(
        args.data,
        args.vocab,
        batch_size=args.batch_size,
        max_len=args.max_len
    )

    for x, y in loader:
        print("输入 shape:", x.shape)
        print("目标 shape:", y.shape)
        break
