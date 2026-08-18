# train.py
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset import get_dataloader
from model_lstm import PoetryLSTM
import json

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载词表
    vocab_path = os.path.join(args.data_dir, "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    pad_idx = vocab.index("<PAD>")

    # 数据加载
    train_loader = get_dataloader(
        os.path.join(args.data_dir, "train.jsonl"),
        vocab_path,
        batch_size=args.batch_size,
        shuffle=True,
        max_len=args.max_len
    )
    val_loader = get_dataloader(
        os.path.join(args.data_dir, "val.jsonl"),
        vocab_path,
        batch_size=args.batch_size,
        shuffle=False,
        max_len=args.max_len
    )

    # 模型
    model = PoetryLSTM(
        vocab_size, args.embedding_dim, args.hidden_size,
        args.num_layers, args.dropout, pad_idx
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # 日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    best_val_loss = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)

    step = 0
    for epoch in range(args.epochs): #训练轮次
        model.train() #训练模式设置
        total_loss = 0 
        for batch, (x, y) in enumerate(train_loader): #分批次处理 一次处理batch_size个poem
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad() #梯度清零
            logits, _ = model(x) #前向传播
            loss = criterion(logits.view(-1, vocab_size), y.view(-1)) #损失计算
            loss.backward() #反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) #梯度裁剪
            optimizer.step() #参数更新

            total_loss += loss.item() #损失记录
            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")
                writer.add_scalar("train/loss", loss.item(), step)
            step += 1

        # 验证
        model.eval() #评估模式设置
        val_loss = 0
        with torch.no_grad(): #验证损失计算
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch} Val Loss {val_loss:.4f}")
        writer.add_scalar("val/loss", val_loss, epoch)

        # 保存最好模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"✅ 保存最佳模型 {save_path}")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="训练/验证数据目录（processed 子目录）")
    parser.add_argument("--save_dir", type=str, required=True, help="模型保存目录")
    parser.add_argument("--log_dir", type=str, required=True, help="tensorboard 日志目录")
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()

    train(args)
