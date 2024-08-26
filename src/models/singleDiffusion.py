import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

    x_t = sqrt_alpha_cumprod[-1] * x + sqrt_one_minus_alpha_cumprod[-1] * noise
    return x_t, noise


def reverse_diffusion_process(model, x_t, num_steps, beta_start=1e-4, beta_end=0.02):
    """
    逆向扩散过程，从噪声数据生成新数据。
    """
    beta = torch.linspace(beta_start, beta_end, num_steps)
    alpha = 1 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)

    for t in reversed(range(num_steps)):
        noise_pred = model(x_t)
        alpha_t = alpha[t]
        alpha_cumprod_t = alpha_cumprod[t]

        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta[t] / torch.sqrt(1 - alpha_cumprod_t)) * noise_pred)
        x_t = mean

    return x_t


# 训练循环
def train_diffusion_model(model, data_loader, num_steps, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        for x in data_loader:
            x = x.to(device)
            optimizer.zero_grad()

            # 正向扩散
            x_noisy, noise = diffusion_process(x, num_steps)
            noise_pred = model(x_noisy)

            # 计算损失
            loss = mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


if __name__ == '__main__':
    # 参数和数据准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet1D(in_channels=1, out_channels=1).to(device)

    # 生成示例数据
    batch_size = 16
    data = torch.sin(torch.linspace(0, 4 * np.pi, 100)).unsqueeze(0).repeat(batch_size, 1).unsqueeze(1)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    # 训练模型
    num_steps = 100
    num_epochs = 20
    learning_rate = 1e-3
    train_diffusion_model(model, data_loader, num_steps, num_epochs, learning_rate)

    # 使用逆向扩散生成数据
    x_t = torch.randn_like(data).to(device)
    generated_data = reverse_diffusion_process(model, x_t, num_steps)
