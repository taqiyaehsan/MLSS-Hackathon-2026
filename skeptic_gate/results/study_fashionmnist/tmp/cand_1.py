"""Improved FashionMNIST classifier using a small CNN with normalization and Adam optimizer."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class MyMethod(BaseMethod):
    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        n, c, h, w = X.shape

        # Compute per-channel mean and std for normalization
        mean = X.mean()
        std = X.std()

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(16)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(32)
                self.fc1 = nn.Linear(32 * 7 * 7, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = (x - mean) / (std + 1e-6)  # Normalize input
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.max_pool2d(x, 2)  # 28->14
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.max_pool2d(x, 2)  # 14->7
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        self.model = Net().to(DEVICE)
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        batch_size = 128
        g = torch.Generator().manual_seed(seed)
        self.model.train()
        for epoch in range(10):  # Increase epochs for better training
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model(X[idx]), y[idx])
                loss.backward()
                opt.step()

    def predict(self, X):
        torch.manual_seed(0)  # ensure deterministic batchnorm running stats if any
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(1)
