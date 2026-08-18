# src/sample.py
import argparse
import torch
import json
from model_lstm import PoetryLSTM

def load_vocab(vocab_path):
    """加载词表并返回 vocab, word2idx, idx2word"""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    return vocab, word2idx, idx2word

def generate(model, device, word2idx, idx2word, start="春", max_len=64, temperature=1.0):
    """根据起始文字生成诗句"""
    model.eval()
    input_ids = [word2idx.get("<SOS>")]
    if start:
        for ch in start:
            input_ids.append(word2idx.get(ch, word2idx["<UNK>"]))
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    hidden = None
    generated = input_ids[:]
    for _ in range(max_len):
        logits, hidden = model(input_tensor, hidden)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        if idx2word[next_id] == "<EOS>":
            break
        generated.append(next_id)
        input_tensor = torch.tensor([[next_id]], dtype=torch.long).to(device)

    # 拼接成字符串，去掉特殊符号
    return "".join(idx2word[i] for i in generated if i not in [word2idx["<SOS>"], word2idx["<EOS>"], word2idx["<PAD>"]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--vocab", type=str, required=True, help="vocab.json 路径")
    parser.add_argument("--start", type=str, default="春風", help="起始文字")
    parser.add_argument("--max_len", type=int, default=64, help="生成最大长度")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载词表
    vocab, word2idx, idx2word = load_vocab(args.vocab)

    # 初始化模型并加载 checkpoint
    model = PoetryLSTM(len(vocab)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # 生成诗句
    poem = generate(
        model, device, word2idx, idx2word,
        start=args.start,
        max_len=args.max_len,
        temperature=args.temperature
    )
    print("=== 生成结果 ===")
    print(poem)
