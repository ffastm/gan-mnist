import numpy as np
import itertools
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


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
        return x


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
    loss_list_d , loss_list_g = [], [] 
    d = D()
    g = G()
    criterion = nn.BCEWithLogitsLoss()
    optimizer_d = optim.SGD(d.parameters(), lr=1e-2)
    optimizer_g = optim.SGD(g.parameters(), lr=1e-2)

    for epoch in range(2):
        for batch_idx, (real_data, _) in enumerate(train_loader):
            optimizer_d.zero_grad()
            noise = torch.randn(batch_size//2, noise_size, requires_grad=True)
            generated_data = g(noise)
            mixed_batch = torch.cat((real_data.view(real_data.shape[0], -1), generated_data), dim=0)
            labels = [1]*(batch_size//2)+[0]*(batch_size//2)
            target = torch.eye(2)[labels] 
            result = d(mixed_batch)
            loss_d = criterion(result, target) 
            loss_list_d.append(torch.min(loss_d, torch.FloatTensor([1])))
            #print(loss_d)
            loss_d.backward()
            optimizer_d.step()
            
            optimizer_g.zero_grad() 
            noise = torch.randn(batch_size//2, noise_size, requires_grad=True)
            generated_data = g(noise)
            labels = [1]*(batch_size//2)
            target = torch.eye(2)[labels]
            result = d(generated_data)
            loss_g = criterion(result, target) 
            loss_list_g.append(torch.min(loss_g, torch.FloatTensor([1])))
            #print("\t"*4,loss_g)
            loss_g.backward()
            optimizer_g.step()

    noise = torch.randn(16, noise_size, requires_grad=True)
    generated_data = g(noise) 
    print("here")
    for i, digit in enumerate(generated_data): 
        plt.matshow(digit.view(28, 28).detach().numpy())
        plt.savefig("digit%02d.jpg" % i)

#    plt.plot(loss_list_d, color="b")
#    plt.plot(loss_list_g, color="r")
#    plt.savefig("plot.jpg")
