import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    def __init__(self, num_samples, seq_length, vocab_size):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_seq = torch.randint(0, self.vocab_size, (self.seq_length,))
        target_seq = torch.randint(0, self.vocab_size, (self.seq_length,))
        return input_seq, target_seq


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src) * (src.size(1) ** 0.5)
        tgt = self.embedding(tgt) * (tgt.size(1) ** 0.5)
        src = src.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        tgt = tgt.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        return output


# 设置模型参数
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048

# 设置数据集参数
num_samples = 1024
seq_length = 10
vocab_size = 20

# 创建数据集和数据加载器
dataset = RandomDataset(num_samples, seq_length, vocab_size)
t = dataset[0]
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 实例化模型
model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模型训练
model.train()
for epoch in range(5):  # 简单训练 5 个 epoch
    optimizer.zero_grad()
    for src, tgt in data_loader:
        output = model(src, tgt[:-1, :])
        loss = criterion(output.view(-1, vocab_size), tgt[1:, :].reshape(-1))
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')


# 模型预测
model.eval()
src, tgt = next(iter(data_loader))
with torch.no_grad():
    output = model(src, tgt[:-1, :])
    print(output.argmax(dim=-1))