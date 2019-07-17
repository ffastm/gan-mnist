import numpy as np
import itertools
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


batch_size = 32
noise_size = 32

class G(nn.Module):
    def __init__(self, noise_size=noise_size):
        super(G, self).__init__()
        self.fc1 = nn.Linear(noise_size, 100)
        self.fc2 = nn.Linear(100, 28*28)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    "mnist", train=True,
    download=True, transform=transform)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size//2,
                          shuffle=True)
if __name__ == "__main__":
    d = D()
    g = G()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(d.parameters(), lr=1e-3)

    for batch_idx, (real_data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        seed = torch.randn(batch_size//2, noise_size)
        generated_data = g(seed)  
        mixed_batch = torch.cat((real_data.view(real_data.shape[0], -1), generated_data), dim=0)
        labels = [1]*(batch_size//2)+[0]*(batch_size//2)
        target = torch.eye(2)[labels] 
        result = d(mixed_batch)
        loss = criterion(result, target) 
        loss.backward()
        optimizer.step()
        print(loss)
