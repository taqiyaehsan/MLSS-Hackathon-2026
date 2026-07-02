"""PRIMARY CODE for the Colored-MNIST task -- enhanced CNN with dropout, LeakyReLU, and cosine annealing LR schedule."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class MyMethod(BaseMethod):
    def __init__(self):
        super().__init__()
        # Define a conv net with dropout and LeakyReLU activations
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2),           # 28->14
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2),           # 14->7
            nn.Dropout(0.15),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        ).to(DEVICE)

    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        n = X.shape[0]

        # Normalize input to zero mean 0.5 and std 0.5 (simple normalization)
        X = (X - 0.5) / 0.5

        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        batch_size = 128
        epochs = 15

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        for _ in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                opt.zero_grad()
                out = self.model(X[idx])
                loss = loss_fn(out, y[idx])
                loss.backward()
                opt.step()
            scheduler.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = (X - 0.5) / 0.5
            out = self.model(X)
            return out.argmax(1)
