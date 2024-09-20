import torch
from torch.utils.data import DataLoader

from src.generator.generate_dataset import CustomDataset, generate_dataset
from src.models.singleDiffusion import train_diffusion_model, reverse_diffusion_process
from src.models.UNet import UNet1D

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = 1024
    dataset = CustomDataset(num_samples, generate_dataset)
    dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = UNet1D(in_channels=1, out_channels=1)

    num_steps = 100
    num_epochs = 20
    learning_rate = 1e-3
    train_diffusion_model(model, dataLoader, num_steps, num_epochs, learning_rate, device)

    x_t = []
    generated_data = reverse_diffusion_process(model, x_t, num_steps)
