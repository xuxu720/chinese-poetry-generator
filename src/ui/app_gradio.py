# src/ui/app_gradio.py
import gradio as gr
import torch
import sys, os

# 获取当前文件的绝对路径
base_dir = os.path.dirname(os.path.abspath(__file__))
# 构建项目根目录路径
project_root = os.path.dirname(os.path.dirname(base_dir))
# 添加src目录到Python路径
sys.path.append(os.path.join(project_root, 'src'))
from model_lstm import PoetryLSTM
from sample import generate, load_vocab

# 使用相对路径定义目录
MODEL_DIR = os.path.join(project_root, "checkpoints")
VOCAB_DIR = os.path.join(project_root, "data", "processed")

STYLES = {
    "唐诗": {"checkpoint": os.path.join(MODEL_DIR, "tang", "best_model.pt"),
            "vocab": os.path.join(VOCAB_DIR, "tang", "vocab.json")},
    "宋词": {"checkpoint": os.path.join(MODEL_DIR, "song", "best_model.pt"),
            "vocab": os.path.join(VOCAB_DIR, "song", "vocab.json")}
}

MODES = ["普通", "藏头", "藏尾", "回文"]

loaded_models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# app_gradio.py 中 generate_poem 函数
def generate_poem(style, mode, start, max_len=64, temperature=1.0, line_len=7):
    """
    style: 风格选择 "唐诗"/"宋词"
    mode: 模式选择 "普通"/"藏头"/"藏尾"/"回文"
    start: 开头文字（藏头/藏尾使用）
    max_len: 每行生成最大长度（按 token）
    temperature: 采样温度
    line_len: 每行固定长度（如七言律诗 7）
    """
    # 1. 加载模型和词表（缓存机制）
    if style not in loaded_models:
        vocab, word2idx, idx2word = load_vocab(STYLES[style]["vocab"])
        model = PoetryLSTM(len(vocab)).to(device)
        model.load_state_dict(torch.load(STYLES[style]["checkpoint"], map_location=device))
        loaded_models[style] = (model, vocab, word2idx, idx2word)
    else:
        model, vocab, word2idx, idx2word = loaded_models[style]

    # 普通模式直接调用原 generate
    if mode == "普通":
        return generate(model, device, word2idx, idx2word, start=start, max_len=max_len, temperature=temperature)

    # 藏头/藏尾/回文模式按行生成
    poem_lines = []

    if mode in ["藏头", "藏尾"]:
        # 每行对应 start 的一个字
        num_lines = len(start)
        for i in range(num_lines):
            ch = start[i]
            if mode == "藏头":
                line = generate(model, device, word2idx, idx2word, start=ch, max_len=line_len, temperature=temperature)
                line = ch + line[1:line_len]  # 强制首字
            else:  # 藏尾
                line = generate(model, device, word2idx, idx2word, start="", max_len=line_len, temperature=temperature)
                if len(line) >= 1:
                    line = line[:line_len-1] + ch  # 强制尾字
                else:
                    line = ch  # 防止空行
            # 截断或补齐每行长度
            line = line[:line_len]
            poem_lines.append(line)

    elif mode == "回文":
        poem_raw = generate(model, device, word2idx, idx2word, start=start, max_len=max_len, temperature=temperature)
        poem_lines = [poem_raw[::-1]]  # 简单整首诗反转

    # 拼接输出
    poem = "，".join(poem_lines)
    return poem


# Gradio 界面
iface = gr.Interface(
    fn=generate_poem,
    inputs=[
        gr.Dropdown(list(STYLES.keys()), label="诗词风格"),
        gr.Dropdown(MODES, label="生成模式"),
        gr.Textbox(label="开头文字", placeholder="例如：春風"),
        gr.Slider(16, 128, value=64, step=1, label="最大长度"),
        gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="采样温度")
    ],
    outputs="text",
    title="古诗词生成器",
    description="选择风格和模式，输入开头文字，生成古诗或词"
)

if __name__ == "__main__":
    # 禁用健康检查功能，解决502错误
    import os
    os.environ["GRADIO_HEALTHCHECK_TIMEOUT"] = "0"
    
    iface.launch(
        server_port=7860,
        server_name="0.0.0.0",
        share=False,
        inbrowser=False,
        quiet=False
    )
