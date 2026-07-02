"""Deeper CNN for FashionMNIST with MixUp augmentation, LeakyReLU, increased dropout, weight decay, label smoothing, and cosine LR scheduler."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class MyMethod(BaseMethod):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 28x28
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 7x7

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=1),  # 6x6

            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256 * 6 * 6, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 10)
        ).to(DEVICE)

    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        n = X.shape[0]
        mean = 0.2860406
        std = 0.35302424

        # Normalize
        X = (X - mean) / std

        # MixUp augmentation function
        def mixup_data(x, y, alpha=0.2, generator=None):
            if alpha > 0 and generator is not None:
                lam = torch.distributions.Beta(alpha, alpha).sample((x.size(0),), generator=generator).to(x.device)
            else:
                lam = torch.ones(x.size(0), device=x.device)
            lam = lam.view(-1, 1, 1, 1)
            perm = torch.randperm(x.size(0), generator=generator).to(x.device)
            mixed_x = lam * x + (1 - lam) * x[perm]
            y_a, y_b = y, y[perm]
            return mixed_x, y_a, y_b, lam.squeeze()

        opt = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing

        g = torch.Generator().manual_seed(seed)
        self.model.train()

        batch_size = 128
        epochs = 20

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        for epoch in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                batch_X = X[idx]
                batch_y = y[idx]

                # MixUp augmentation
                mixed_X, y_a, y_b, lam = mixup_data(batch_X, batch_y, alpha=0.2, generator=g)

                opt.zero_grad()
                logits = self.model(mixed_X)
                loss = (lam * loss_fn(logits, y_a) + (1 - lam) * loss_fn(logits, y_b)).mean()
                loss.backward()
                opt.step()
            scheduler.step()

    def predict(self, X):
        mean = 0.2860406
        std = 0.35302424
        X = (X - mean) / std
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            return logits.argmax(dim=1)
