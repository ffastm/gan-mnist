import numpy as np
import itertools
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda")
cpu = torch.device("cpu")

batch_size = 128
noise_size = 128

class G(nn.Module):
    def __init__(self, noise_size=noise_size):
        super(G, self).__init__()
        self.fc1 = nn.Linear(noise_size, 1024*7*7)
        self.conv1 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(x.shape[0], 1024, 7, 7)
        x = F.relu(self.conv1(nn.Upsample(scale_factor=2, mode='nearest')(x)))
        x = F.relu(self.conv2(nn.Upsample(scale_factor=2, mode='nearest')(x)))
        x = x.view(x.shape[0], -1)
        return x

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
                          shuffle=True,
                          drop_last=True)
if __name__ == "__main__":
    loss_list_d , loss_list_g = [], [] 
    d = D()
    g = G()
    d.to(device)
    g.to(device)
    criterion = nn.BCEWithLogitsLoss()
    criterion.to(device)
    optimizer_d = optim.SGD(d.parameters(), lr=1e-2)
    optimizer_g = optim.SGD(g.parameters(), lr=1e-2)

    for epoch in range(10):
        print(epoch)
        for batch_idx, (real_data, _) in enumerate(itertools.islice(train_loader, 60000//128*128)):
            real_data = real_data.to(device)
            print(batch_idx)
            optimizer_d.zero_grad()
            noise = torch.randn(batch_size//2, noise_size, requires_grad=True).to(device)
            generated_data = g(noise)
            mixed_batch = torch.cat((real_data.view(real_data.shape[0], -1), generated_data), dim=0)
            labels = [1]*(batch_size//2)+[0]*(batch_size//2)
            target = torch.eye(2)[labels].to(device)
            result = d(mixed_batch)
            try:
                loss_d = criterion(result, target) 
            except:
                import ipdb; ipdb.set_trace()
            loss_list_d.append(torch.min(loss_d, torch.FloatTensor([1]).to(device)))
            #print(loss_d)
            loss_d.backward()
            optimizer_d.step()
            
            optimizer_g.zero_grad() 
            noise = torch.randn(batch_size//2, noise_size, requires_grad=True).to(device)
            generated_data = g(noise)
            labels = [1]*(batch_size//2)
            target = torch.eye(2)[labels].to(device)
            result = d(generated_data)
            loss_g = criterion(result, target) 
            loss_list_g.append(torch.min(loss_g, torch.FloatTensor([1]).to(device)))
            #print("\t"*4,loss_g)
            loss_g.backward()
            optimizer_g.step()

        noise = torch.randn(16, noise_size, requires_grad=True).to(device)
        generated_data = g(noise) 
        print("here")
        for i, digit in enumerate(generated_data): 
            digit = digit.view(28, 28).to(cpu).detach().numpy()
            digit = (digit-np.min(digit))/(np.max(digit)-np.min(digit))*255
            print(np.min(digit), np.max(digit))
            np.save("digit%02d_%02d" % (epoch, i), digit.astype(np.uint8))

#    plt.plot(loss_list_d, color="b")
#    plt.plot(loss_list_g, color="r")
#    plt.savefig("plot.jpg")
