# src/evaluate.py
import json
import torch
import argparse
from model_lstm import PoetryLSTM

def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    return vocab, word2idx, idx2word

def evaluate(checkpoint, data_path, vocab_path, batch_size=32, max_len=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, word2idx, idx2word = load_vocab(vocab_path)
    pad_idx = word2idx.get("<PAD>", 0)

    # 读取 token 序列数据
    sequences = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = json.loads(line.strip())
            sequences.append(tokens)

    # 转成 tensor 并 pad
    def pad_sequence(seq, max_len):
        seq = seq[:max_len]
        seq += [pad_idx] * (max_len - len(seq))
        return seq

    x_data = [pad_sequence(seq[:-1], max_len) for seq in sequences]  # 输入
    y_data = [pad_sequence(seq[1:], max_len) for seq in sequences]   # 目标
    x_tensor = torch.tensor(x_data, dtype=torch.long, device=device)
    y_tensor = torch.tensor(y_data, dtype=torch.long, device=device)

    # 模型
    model = PoetryLSTM(len(vocab)).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    total_loss = 0.0
    num_batches = (len(x_tensor) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(x_tensor))
            x_batch = x_tensor[start:end]
            y_batch = y_tensor[start:end]

            logits, _ = model(x_batch)
            loss = criterion(logits.view(-1, len(vocab)), y_batch.view(-1))
            total_loss += loss.item() * (end - start)

    avg_loss = total_loss / len(x_tensor)
    print(f"Average Loss : {avg_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    parser.add_argument("--data", required=True, help="验证集 token 文件 (.jsonl)")
    parser.add_argument("--vocab", required=True, help="词表文件路径 (.json)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.data, args.vocab, args.batch_size, args.max_len)
