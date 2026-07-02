"""PRIMARY CODE for the Colored-MNIST task -- enhanced CNN with additional conv block, ELU activations, MixUp augmentation, label smoothing, and cosine annealing LR schedule."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class MyMethod(BaseMethod):
    def __init__(self):
        super().__init__()
        # Define a conv net with 3 conv blocks, ELU activations, and dropout
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2),           # 28->14
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2),           # 14->7
            nn.Dropout(0.15),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2),           # 7->3
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ELU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 10),
        ).to(DEVICE)

    def _mixup(self, X, y, alpha=0.4, generator=None):
        if alpha <= 0:
            return X, y
        lam = torch.distributions.Beta(alpha, alpha).sample(generator=generator).item()
        batch_size = X.size(0)
        index = torch.randperm(batch_size, generator=generator)
        mixed_X = lam * X + (1 - lam) * X[index]
        y_a, y_b = y, y[index]
        return mixed_X, y_a, y_b, lam

    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        n = X.shape[0]

        # Normalize input to zero mean 0.5 and std 0.5
        X = (X - 0.5) / 0.5

        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)

        # Label smoothing cross-entropy loss
        class LabelSmoothingCrossEntropy(nn.Module):
            def __init__(self, smoothing=0.1):
                super().__init__()
                self.smoothing = smoothing
                self.confidence = 1.0 - smoothing

            def forward(self, pred, target):
                log_probs = F.log_softmax(pred, dim=-1)
                nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
                smooth_loss = -log_probs.mean(dim=-1)
                loss = self.confidence * nll_loss + self.smoothing * smooth_loss
                return loss.mean()

        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)

        batch_size = 128
        epochs = 20

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        for _ in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                batch_X = X[idx]
                batch_y = y[idx]
                # Apply MixUp
                mixed_X, y_a, y_b, lam = self._mixup(batch_X, batch_y, alpha=0.4, generator=g)

                opt.zero_grad()
                out = self.model(mixed_X)
                loss = lam * loss_fn(out, y_a) + (1 - lam) * loss_fn(out, y_b)
                loss.backward()
                opt.step()
            scheduler.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = (X - 0.5) / 0.5
            out = self.model(X)
            return out.argmax(1)
