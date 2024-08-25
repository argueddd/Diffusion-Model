import matplotlib.pyplot as plt
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义 device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.Resize(32),  # 调整图像尺寸为32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=True)
data_loader = DataLoader(mnist, batch_size=64, shuffle=True)

# 定义U-Net模型并移动到 device
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=1
).to(device)

# 定义扩散过程并移动到 device
diffusion = GaussianDiffusion(
    model,
    image_size=32,
    timesteps=1000,
).to(device)


def train():
    global step
    # 优化器
    optimizer = optim.Adam(diffusion.model.parameters(), lr=1e-4)
    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        for step, (images, _) in enumerate(data_loader):
            images = images.to(device)  # 将图像移动到 device
            loss = diffusion(images)  # 计算损失

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Step {step}/{len(data_loader)}, Loss: {loss.item()}")


train()
torch.save(model.state_dict(), '../models/unet_model.pth')
# model.load_state_dict(torch.load('../models/unet_model.pth'))

# 生成图像
# 生成图像并返回所有时间步的结果
sampled_images = diffusion.sample(batch_size=64, return_all_timesteps=True)
sampled_images = sampled_images.cpu().detach().numpy()
# 检查sampled_images的形状
print("Sampled images shape:", sampled_images.shape)  # 预期形状: (batch_size, time_steps, channels, height, width)

# 假设我们要展示每个时间步的前8张图片
batch_size, time_steps, channels, height, width = sampled_images.shape

# 去掉通道维度（假设是单通道图像）
if channels == 1:
    sampled_images = sampled_images.squeeze(2)  # 新形状: (timesteps, batch_size, height, width)

selected_time_steps = [0, 200, 400, 600, 800, time_steps - 1]  # 你可以选择其他时间步
sampled_images = sampled_images[:, selected_time_steps, :, :]  # 选取每个时间步前8张图片

# 可视化
fig, axes = plt.subplots(len(selected_time_steps), 8, figsize=(8, 4))
for i, timestep in enumerate(selected_time_steps):
    for j in range(8):
        axes[i, j].imshow(sampled_images[j, i], cmap='gray')
        axes[i, j].axis('off')

plt.show()
