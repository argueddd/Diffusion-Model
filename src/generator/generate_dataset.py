import torch
from torch.utils.data import Dataset, DataLoader

from src.generator.create_new_sample import calculate_probabilities_distribution_by_freq, generate_signal, \
    calculate_transfer_matrix, generate_new_ref


class CustomDataset(Dataset):
    def __init__(self, nums, generate_function):
        self.nums = nums
        self.generate_function = generate_function

    def __len__(self):
        return self.nums

    def __getitem__(self, idx):
        data = self.generate_function()
        return torch.tensor(data, dtype=torch.float32)


def generate_dataset():
    riker, reflectance, seismic = generate_signal()
    prob_dic = calculate_probabilities_distribution_by_freq(reflectance)
    transform_matrix = calculate_transfer_matrix(prob_dic)
    new_ref = generate_new_ref(prob_dic, transform_matrix, reflectance)
    return new_ref


# 创建自定义数据集实例
num_samples = 1000
dataset = CustomDataset(num_samples, generate_dataset)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 迭代数据集
for batch in dataloader:
    print(batch)
