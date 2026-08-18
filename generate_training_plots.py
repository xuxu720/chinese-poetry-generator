import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建保存图片的目录
output_dir = 'training_plots'
os.makedirs(output_dir, exist_ok=True)

# 1. 生成损失函数曲线
def generate_loss_plot():
    # 根据描述生成模拟数据
    epochs = np.arange(1, 41)
    
    # 训练损失：快速下降然后趋于稳定
    train_loss = 3.5 * np.exp(-epochs/5) + 0.2 + 0.05 * np.sin(epochs)
    
    # 验证损失：比训练损失稍高，在30轮后趋于稳定
    val_loss = 3.7 * np.exp(-epochs/6) + 0.3 + 0.08 * np.sin(epochs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', linewidth=2, label='训练损失')
    plt.plot(epochs, val_loss, 'r--', linewidth=2, label='验证损失')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('交叉熵损失', fontsize=12)
    plt.title('训练损失与验证损失曲线', fontsize=14)
    plt.legend(fontsize=10)
    plt.xlim(0, 40)
    plt.ylim(0, 4)
    
    # 添加文本注释
    plt.axvline(x=10, color='g', linestyle=':', alpha=0.5)
    plt.axvline(x=30, color='g', linestyle=':', alpha=0.5)
    plt.text(5, 3.5, '快速学习阶段', fontsize=10, color='g')
    plt.text(15, 1.5, '精细化学习阶段', fontsize=10, color='g')
    plt.text(35, 0.5, '稳定阶段', fontsize=10, color='g')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 2. 生成学习率变化曲线
def generate_lr_plot():
    epochs = np.arange(1, 41)
    
    # 学习率变化：初始为0.001，在15轮和25轮时衰减
    lr = np.ones_like(epochs) * 0.001
    lr[14:] = 0.0001  # 第15轮开始衰减为原来的0.1
    lr[24:] = 0.00001  # 第25轮开始再次衰减为原来的0.1
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, lr, 'g-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('学习率', fontsize=12)
    plt.title('学习率变化曲线', fontsize=14)
    plt.xlim(0, 40)
    plt.ylim(5e-6, 2e-3)
    
    # 添加学习率变化点的标记
    plt.scatter([1, 15, 25], [0.001, 0.0001, 0.00001], color='r', s=50, zorder=5)
    plt.text(1, 0.0015, '初始学习率: 0.001', fontsize=10, color='r')
    plt.text(15, 0.00015, '第一次衰减: 0.0001', fontsize=10, color='r')
    plt.text(25, 0.000015, '第二次衰减: 0.00001', fontsize=10, color='r')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 3. 生成梯度范数变化曲线
def generate_gradient_norm_plot():
    steps = np.arange(1, 1001)  # 假设有1000个训练步骤
    
    # 梯度范数：初始较大，然后逐渐减小并趋于稳定
    base_norm = 2.0 * np.exp(-steps/200)  # 基础衰减趋势
    noise = 0.3 * np.random.randn(len(steps))  # 添加一些随机噪声
    gradient_norm = base_norm + noise + 0.5  # 确保梯度范数始终为正
    
    # 添加梯度裁剪的阈值线（假设设置为2.5）
    clip_threshold = np.ones_like(steps) * 2.5
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, gradient_norm, 'b-', linewidth=1, alpha=0.7, label='梯度L2范数')
    plt.plot(steps, clip_threshold, 'r--', linewidth=1.5, label='梯度裁剪阈值')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('训练步数', fontsize=12)
    plt.ylabel('梯度L2范数', fontsize=12)
    plt.title('梯度范数变化曲线', fontsize=14)
    plt.legend(fontsize=10)
    plt.xlim(0, 1000)
    plt.ylim(0, 4)
    
    # 添加文本注释
    plt.axvline(x=200, color='g', linestyle=':', alpha=0.5)
    plt.text(100, 3.5, '初期：参数更新幅度大', fontsize=10, color='g')
    plt.text(600, 1.0, '后期：参数更新幅度稳定', fontsize=10, color='g')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_norm_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 生成所有图表
if __name__ == "__main__":
    print("生成损失函数曲线...")
    generate_loss_plot()
    print("生成学习率变化曲线...")
    generate_lr_plot()
    print("生成梯度范数变化曲线...")
    generate_gradient_norm_plot()
    print("所有图表已保存到 training_plots 目录！")
    print("\n生成的文件：")
    print(f"1. {os.path.abspath(os.path.join(output_dir, 'loss_curve.png'))}")
    print(f"2. {os.path.abspath(os.path.join(output_dir, 'learning_rate_curve.png'))}")
    print(f"3. {os.path.abspath(os.path.join(output_dir, 'gradient_norm_curve.png'))}")