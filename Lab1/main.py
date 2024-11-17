from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from utils_train import train_loop


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 11 * 11, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # print(x.shape)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # logits
        return x


def main(opts):
    model = Net()
    model = model.to(opts.device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=opts.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=opts.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    train_loop(model, train_loader, test_loader, optimizer, opts)


if __name__ == "__main__":
    opts_dict = dict(log_every=300, max_epochs=20, batch_size=64, model="Net")
    opts = SimpleNamespace(**opts_dict)
    opts.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device", opts.device)

    main(opts)
