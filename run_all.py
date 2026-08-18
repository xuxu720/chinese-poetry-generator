# run_all.py
import os
import subprocess
import time

# ----------------- 配置 -----------------
RAW_ROOT = "data/raw"                 # 原始数据根目录
PROCESSED_ROOT = "data/processed"     # 处理后数据根目录
CHECKPOINT_DIR = "checkpoints"        # 模型保存目录
LOG_DIR = "logs"                       # tensorboard日志目录
GRADIO_PORT = 7860                     # Gradio 前端端口

# 风格选择提示
print("请选择训练风格：")
print("1: 唐诗")
print("2: 宋词")
style_choice = input("输入 1 或 2: ").strip()
if style_choice == "1":
    style = "tang"
elif style_choice == "2":
    style = "song"
else:
    print("输入无效，默认使用唐诗")
    style = "tang"

RAW_DIR = os.path.join(RAW_ROOT, style)
PROCESSED_DIR = os.path.join(PROCESSED_ROOT, style)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ----------------- Step 1: 数据预处理 -----------------
print(f"==== Step 1: 数据预处理 ({style}) ====")
subprocess.run([
    "python", "src/preprocess.py",
    "--raw_dir", RAW_DIR,
    "--out_dir", PROCESSED_DIR
])

# ----------------- Step 2: 模型训练 -----------------
print(f"==== Step 2: 模型训练 ({style}) ====")
subprocess.run([
    "python", "src/train.py",
    "--data_dir", PROCESSED_DIR,
    "--save_dir", CHECKPOINT_DIR,
    "--log_dir", LOG_DIR,
    "--epochs", "20",
    "--batch_size", "64"
])

# ----------------- Step 3: 测试 / 生成样例 -----------------
print(f"==== Step 3: 测试生成样例 ({style}) ====")
subprocess.run([
    "python", "src/sample.py",
    "--checkpoint", os.path.join(CHECKPOINT_DIR, "best_model.pt"),
    "--vocab", os.path.join(PROCESSED_DIR, "vocab.json"),
    "--start", "秋雨",
    "--max_len", "64",
    "--temperature", "1.0"
])

# ----------------- Step 4: 启动 Gradio 前端 -----------------
print("==== Step 4: 启动 Gradio 前端 ====")
time.sleep(2)
subprocess.run([
    "python", "src/ui/app_gradio.py"
])
