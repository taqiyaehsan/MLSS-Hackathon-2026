"""Improved FashionMNIST classifier with wider CNN, LayerNorm, label smoothing, enhanced augmentation, and cosine annealing with warm restarts."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod

DEVICE = torch.device("cpu")

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

class MyMethod(BaseMethod):
    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        n, c, h, w = X.shape

        mean = X.mean()
        std = X.std()

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                # Increased channels moderately
                self.conv1 = nn.Conv2d(1, 48, 3, padding=1)
                self.ln1 = nn.LayerNorm([48, h, w])  # LayerNorm over channel+spatial
                self.conv2 = nn.Conv2d(48, 96, 3, padding=1)
                self.ln2 = nn.LayerNorm([96, h//2, w//2])
                self.fc1 = nn.Linear(96 * 7 * 7, 256)
                self.ln3 = nn.LayerNorm(256)
                self.fc2 = nn.Linear(256, 10)

            def forward(self, x):
                x = (x - mean) / (std + 1e-6)
                x = F.relu(self.ln1(self.conv1(x)))
                x = F.max_pool2d(x, 2)  # 28->14
                x = F.relu(self.ln2(self.conv2(x)))
                x = F.max_pool2d(x, 2)  # 14->7
                x = x.view(x.size(0), -1)
                x = F.relu(self.ln3(self.fc1(x)))
                x = self.fc2(x)
                return x

        self.model = Net().to(DEVICE)
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)

        batch_size = 128
        g = torch.Generator().manual_seed(seed)
        self.model.train()

        epochs = 15

        for epoch in range(epochs):
            perm = torch.randperm(n, generator=g)

            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                batch_x = X[idx].clone()  # clone to avoid inplace on original

                # Data augmentation: random horizontal flip
                flip_mask = torch.rand(batch_x.size(0), generator=g) < 0.5
                batch_x[flip_mask] = torch.flip(batch_x[flip_mask], dims=[3])

                # Random rotation between -15 and 15 degrees
                angles = (torch.rand(batch_x.size(0), generator=g) * 30 - 15) * 3.141592653589793 / 180
                cos = torch.cos(angles)
                sin = torch.sin(angles)
                zeros = torch.zeros_like(cos)
                ones = torch.ones_like(cos)
                rot_matrices = torch.stack([cos, -sin, zeros, sin, cos, zeros], dim=1).view(-1, 2, 3)

                grid = F.affine_grid(rot_matrices, batch_x.size(), align_corners=False)
                batch_x = F.grid_sample(batch_x, grid, padding_mode='border', align_corners=False)

                # Random brightness jitter +-0.2
                brightness_factors = 1 + (torch.rand(batch_x.size(0), generator=g) - 0.5) * 0.4
                batch_x = batch_x * brightness_factors.view(-1, 1, 1, 1)
                batch_x = batch_x.clamp(0, 1)

                opt.zero_grad()
                loss = loss_fn(self.model(batch_x), y[idx])
                loss.backward()
                opt.step()

            # Cosine annealing with warm restarts every 5 epochs
            restart_period = 5
            cycle_progress = (epoch % restart_period) / restart_period
            lr = 0.001 * 0.5 * (1 + torch.cos(torch.tensor(cycle_progress * 3.141592653589793)))
            for param_group in opt.param_groups:
                param_group['lr'] = lr.item()

    def predict(self, X):
        torch.manual_seed(0)
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(1)
