import torch
import torch.nn as nn
import torch.optim as optim

# 检查是否可以使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义一个简单的全连接神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # 输入层: 10, 隐藏层: 50
        self.fc2 = nn.Linear(50, 1)  # 隐藏层: 50, 输出层: 1

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 初始化模型、损失函数和优化器
model = SimpleNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 生成随机数据
x = torch.randn(100, 10).to(device)  # 100 个样本, 每个样本 10 个特征
y = torch.randn(100, 1).to(device)  # 100 个目标值

# 训练模型
for epoch in range(100):
    model.train()

    # 前向传播
    outputs = model(x)
    loss = criterion(outputs, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

print("训练完成！")
