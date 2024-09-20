import torch
import torch.nn as nn

# 设置随机种子
torch.manual_seed(0)

# 模拟输入数据
batch_size = 20     # 批次大小
seq_length = 10000  # 序列长度
input_dim = 100     # 输入维度
hidden_dim = 128    # 隐藏层维度
num_layers = 3      # LSTM层数
output_dim = 1      # 输出维度（二分类）

# 随机生成输入数据 (batch_size, seq_length, input_dim)
inputs = torch.randn(batch_size, seq_length, input_dim)

# 随机生成二分类标签 (batch_size, 1)
labels = torch.randint(0, 2, (batch_size, 1)).float()


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 定义多层双向LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            bidirectional=True, batch_first=True)

        # 定义一个全连接层，用于输出分类结果
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 双向LSTM，输出是hidden_dim * 2

    def forward(self, x):
        # 初始化 LSTM 隐状态和细胞状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)

        # 通过LSTM层
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # 通过全连接层得到最终输出
        out = self.fc(out)

        return out