import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sympy.stats.sampling.sample_numpy import numpy
from torch.utils.data import DataLoader

from src.models.UNet import UNet1D


# 扩散模型的正向和反向过程
def diffusion_process(x, num_steps, beta_start=1e-4, beta_end=0.02):
    """
    正向扩散过程，添加噪声到数据中。
    """
    beta = torch.linspace(beta_start, beta_end, num_steps)
    alpha = 1 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)

    noise = torch.randn_like(x)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod).unsqueeze(1)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod).unsqueeze(1)

    noise_steps = []
    x_steps = []
    for s in range(num_steps):
        noise_step = sqrt_one_minus_alpha_cumprod[s] * noise
        noise_steps.append(noise_step)
        x_t = sqrt_alpha_cumprod[s] * x + noise_step
        x_steps.append(x_t)
    return x_steps, noise_steps


def reverse_diffusion_process(model, x_t, num_steps, beta_start=1e-4, beta_end=0.02):
    """
    逆向扩散过程，从噪声数据生成新数据。
    """
    beta = torch.linspace(beta_start, beta_end, num_steps)
    alpha = 1 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)

    for t in reversed(range(num_steps)):
        x_pred = model(x_t)
        alpha_t = alpha[t]
        alpha_cumprod_t = alpha_cumprod[t]

        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta[t] / torch.sqrt(1 - alpha_cumprod_t)) * x_pred)
        x_t = mean

    return x_t


def reverse_diffusion_process_new(model, signal, steps, beta_start=1e-4, beta_end=0.02):
    for _ in range(steps):
        signal = model(signal) + 1
    return signal



def custom_loss(output, target):
    # 假设输出和目标都是1维张量
    loss = torch.mean((output - target) ** 2)  # 简单的均方误差损失
    return loss

# 训练循环
def train_diffusion_model(model, data_loader, num_steps, num_epochs, learning_rate, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        for x in data_loader:
            x = x.to(device)
            optimizer.zero_grad()

            # 正向扩散
            x_noisy, noise = diffusion_process(x, num_steps)
            x_final_noisy = x_noisy[-1].unsqueeze(1)
            x_pred = model(x_final_noisy).squeeze()
            # 计算损失
            loss = mse_loss(x_pred, x)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    plot_diffusion(x, x_noisy, x_pred)

def plot_diffusion(origin_signal, noises, forcasting_signal):
    # 将所有张量转换为 NumPy 数组
    origin_signal = origin_signal.detach().numpy()
    noises = [noise.detach().numpy() for noise in noises]
    forcasting_signal = forcasting_signal.detach().numpy()

    # 选择步长
    selected_steps = [noises[i - 1] for i in [20, 40, 60, 80, 100]]
    selected_steps.insert(0, origin_signal)  # 插入原始信号
    selected_steps.insert(-1, forcasting_signal)  # 插入预测信号

    # 创建子图
    fig, axes = plt.subplots(len(selected_steps), 1, figsize=(10, 10))

    # 画图
    for idx, (step, ax) in enumerate(zip(selected_steps, axes)):
        ax.plot(step[0].flatten())  # 只画第一个样本的曲线
        ax.set_title(f'Step {idx * 20}')
        ax.grid(True)

    plt.tight_layout()
    plt.show()
