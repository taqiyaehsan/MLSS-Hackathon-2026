"""PRIMARY CODE for the Colored-MNIST task -- improved CNN classifier with normalization, data augmentation, label smoothing, and more epochs."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class MyMethod(BaseMethod):
    def __init__(self):
        super().__init__()
        # Define a small conv net suitable for 28x28 color images
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 28->14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 14->7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
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
        # Use label smoothing by custom CrossEntropy
        def smooth_cross_entropy(pred, target, smoothing=0.1):
            logprobs = F.log_softmax(pred, dim=1)
            n_class = pred.size(1)
            with torch.no_grad():
                true_dist = torch.zeros_like(pred)
                true_dist.fill_(smoothing / (n_class - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1. - smoothing)
            return torch.mean(torch.sum(-true_dist * logprobs, dim=1))

        batch_size = 128
        epochs = 15

        # Data augmentation: random horizontal flip and random crop with padding=2
        def augment(batch_X):
            # batch_X shape: (B,3,28,28)
            # Random horizontal flip
            flip_mask = torch.rand(batch_X.size(0), generator=g, device=batch_X.device) < 0.5
            batch_X[flip_mask] = batch_X[flip_mask].flip(dims=[3])
            # Pad 2 pixels on each side and then random crop 28x28
            batch_X_padded = F.pad(batch_X, (2, 2, 2, 2), mode='reflect')  # (B,3,32,32)
            n = batch_X.size(0)
            crops = torch.zeros_like(batch_X)
            for i in range(n):
                top = torch.randint(0, 5, (1,), generator=g).item()
                left = torch.randint(0, 5, (1,), generator=g).item()
                crops[i] = batch_X_padded[i,:, top:top+28, left:left+28]
            return crops

        for _ in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                batch_X = X[idx].clone()  # clone to avoid modifying original
                batch_X = augment(batch_X)
                opt.zero_grad()
                out = self.model(batch_X)
                loss = smooth_cross_entropy(out, y[idx], smoothing=0.1)
                loss.backward()
                opt.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = (X - 0.5) / 0.5
            out = self.model(X)
            return out.argmax(1)
