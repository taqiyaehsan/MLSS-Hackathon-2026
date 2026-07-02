"""Enhanced FashionMNIST classifier with LeakyReLU, dropout, label smoothing, and stronger augmentations."""

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

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.gn1 = nn.GroupNorm(4, 32)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.gn2 = nn.GroupNorm(8, 64)
                self.dropout = nn.Dropout(0.25)
                self.fc1 = nn.Linear(64 * 7 * 7, 256)
                self.fc2 = nn.Linear(256, 10)
                self.act = nn.LeakyReLU(negative_slope=0.1)

            def forward(self, x):
                x = (x - mean) / (std + 1e-6)
                x = self.act(self.gn1(self.conv1(x)))
                x = F.max_pool2d(x, 2)
                x = self.act(self.gn2(self.conv2(x)))
                x = F.max_pool2d(x, 2)
                x = x.view(x.size(0), -1)
                x = self.dropout(self.act(self.fc1(x)))
                x = self.fc2(x)
                return x

        self.model = Net().to(DEVICE)
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        batch_size = 128
        g = torch.Generator().manual_seed(seed)
        self.model.train()

        for epoch in range(20):  # a few more epochs
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                batch_x = X[idx].clone()

                if batch_x.size(0) > 0:
                    # Horizontal flip
                    flip_mask = torch.rand(batch_x.size(0), generator=g) < 0.5
                    batch_x[flip_mask] = torch.flip(batch_x[flip_mask], dims=[3])

                    # Random affine transform: rotation (-15 to 15 degrees), translation (±2 pixels), scale (0.9 to 1.1)
                    angles = (torch.rand(batch_x.size(0), generator=g) - 0.5) * 30  # degrees
                    translations = torch.randint(-2, 3, (batch_x.size(0), 2), generator=g)
                    scales = 0.9 + 0.2 * torch.rand(batch_x.size(0), generator=g)

                    theta = torch.zeros(batch_x.size(0), 2, 3)
                    radians = angles * 3.141592653589793 / 180
                    cos = torch.cos(radians) * scales
                    sin = torch.sin(radians) * scales
                    theta[:,0,0] = cos
                    theta[:,0,1] = -sin
                    theta[:,1,0] = sin
                    theta[:,1,1] = cos
                    theta[:,0,2] = translations[:,0].to(torch.float32) / (w/2)
                    theta[:,1,2] = translations[:,1].to(torch.float32) / (h/2)

                    grid = F.affine_grid(theta, batch_x.size(), align_corners=False)
                    batch_x = F.grid_sample(batch_x, grid, padding_mode='border', align_corners=False)

                opt.zero_grad()
                logits = self.model(batch_x)
                loss = loss_fn(logits, y[idx])
                loss.backward()
                opt.step()

            # Cosine annealing LR schedule with warm restart every 10 epochs
            if (epoch + 1) % 10 == 0:
                for param_group in opt.param_groups:
                    param_group['lr'] = 0.001
            else:
                lr = 0.001 * 0.5 * (1 + torch.cos(torch.tensor(epoch / 10 * 3.141592653589793)))
                for param_group in opt.param_groups:
                    param_group['lr'] = lr.item()

    def predict(self, X):
        torch.manual_seed(0)
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(1)
