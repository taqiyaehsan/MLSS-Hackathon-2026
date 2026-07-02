"""Improved FashionMNIST classifier with a deeper CNN, ELU activations, MixUp augmentation, label smoothing, and OneCycleLR scheduler."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class MyMethod(BaseMethod):
    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        n, c, h, w = X.shape

        mean = X.mean()
        std = X.std()

        # Label smoothing cross entropy
        class LabelSmoothingCrossEntropy(nn.Module):
            def __init__(self, smoothing=0.1):
                super().__init__()
                self.smoothing = smoothing
                self.confidence = 1.0 - smoothing

            def forward(self, pred, target):
                logprobs = F.log_softmax(pred, dim=-1)
                nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
                smooth_loss = -logprobs.mean(dim=-1)
                loss = self.confidence * nll_loss + self.smoothing * smooth_loss
                return loss.mean()

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.gn1 = nn.GroupNorm(4, 32)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.gn2 = nn.GroupNorm(8, 64)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.gn3 = nn.GroupNorm(8, 128)
                self.fc1 = nn.Linear(128 * 3 * 3, 256)
                self.fc2 = nn.Linear(256, 10)
                self.elu = nn.ELU()

            def forward(self, x):
                x = (x - mean) / (std + 1e-6)
                x = self.elu(self.gn1(self.conv1(x)))
                x = F.max_pool2d(x, 2)  # 28->14
                x = self.elu(self.gn2(self.conv2(x)))
                x = F.max_pool2d(x, 2)  # 14->7
                x = self.elu(self.gn3(self.conv3(x)))
                x = F.max_pool2d(x, 2)  # 7->3
                x = x.view(x.size(0), -1)
                x = self.elu(self.fc1(x))
                x = self.fc2(x)
                return x

        self.model = Net().to(DEVICE)
        opt = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)

        batch_size = 128
        g = torch.Generator().manual_seed(seed)
        self.model.train()

        # OneCycleLR scheduler for 15 epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01, total_steps=(n // batch_size) * 15)

        for epoch in range(15):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                batch_x = X[idx].clone()
                batch_y = y[idx]

                # MixUp augmentation
                if batch_x.size(0) > 1:
                    lam = torch.distributions.Beta(0.4, 0.4).sample((batch_x.size(0),), generator=g).to(batch_x.device)
                    lam = torch.max(lam, 1 - lam)  # enforce lam >= 0.5 for stability
                    perm_idx = torch.randperm(batch_x.size(0), generator=g)
                    mixed_x = lam.view(-1,1,1,1) * batch_x + (1 - lam).view(-1,1,1,1) * batch_x[perm_idx]

                    # Mixed labels as one-hot
                    y_a = batch_y
                    y_b = batch_y[perm_idx]
                else:
                    mixed_x = batch_x
                    y_a = batch_y
                    y_b = batch_y
                    lam = torch.ones(batch_x.size(0))

                opt.zero_grad()
                outputs = self.model(mixed_x)

                # Compute mixup loss
                loss = (lam * loss_fn(outputs, y_a) + (1 - lam) * loss_fn(outputs, y_b)).mean()

                loss.backward()
                opt.step()
                scheduler.step()

    def predict(self, X):
        torch.manual_seed(0)
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(1)
