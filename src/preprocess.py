# preprocess.py
import json
import os
import random
import argparse
from collections import Counter
from tqdm import tqdm

random.seed(42)

def load_raw_json(data_dir):
    poems = []
    for fname in os.listdir(data_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
            poems.extend(data)
    return poems

def clean_poem(poem):
    """简单清洗：只保留中文和常见标点，拼成一首诗的字符串"""
    text = "".join(poem["paragraphs"])
    # 过滤太短或太长的诗
    if len(text) < 10 or len(text) > 200:
        return None
    # 保留常见中文标点
    allowed_punct = set("，。！？、；：")
    clean = []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or ch in allowed_punct:
            clean.append(ch)
    return "".join(clean)

def build_vocab(poems, vocab_size=8000):
    counter = Counter()
    for poem in poems:
        counter.update(list(poem))
    vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
    vocab.extend([w for w, _ in counter.most_common(vocab_size - len(vocab))])
    word2idx = {w: i for i, w in enumerate(vocab)}
    return vocab, word2idx

def encode_poems(poems, word2idx, max_len=128):
    encoded = []
    for poem in poems:
        ids = [word2idx.get(ch, word2idx["<UNK>"]) for ch in poem]
        ids = [word2idx["<SOS>"]] + ids + [word2idx["<EOS>"]]
        if len(ids) > max_len:
            ids = ids[:max_len]
            ids[-1] = word2idx["<EOS>"]
        encoded.append(ids)
    return encoded

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True, help="原始数据目录")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录")
    args = parser.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("加载数据...")
    poems = load_raw_json(raw_dir)
    print("原始数据数量:", len(poems))

    print("清洗数据...")
    poems_clean = [clean_poem(p) for p in tqdm(poems)]
    poems_clean = [p for p in poems_clean if p is not None]
    print("清洗后数据数量:", len(poems_clean))

    print("构建词表...")
    vocab, word2idx = build_vocab(poems_clean, vocab_size=8000)
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print("编码数据...")
    poems_encoded = encode_poems(poems_clean, word2idx, max_len=128)

    # 划分训练/验证/测试
    random.shuffle(poems_encoded)
    n = len(poems_encoded)
    train = poems_encoded[: int(n * 0.8)]
    val = poems_encoded[int(n * 0.8) : int(n * 0.9)]
    test = poems_encoded[int(n * 0.9) :]

    save_jsonl(train, os.path.join(out_dir, "train.jsonl"))
    save_jsonl(val, os.path.join(out_dir, "val.jsonl"))
    save_jsonl(test, os.path.join(out_dir, "test.jsonl"))

    print("处理完成 ✅")
