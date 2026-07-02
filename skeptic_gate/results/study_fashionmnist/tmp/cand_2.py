"""Enhanced FashionMNIST classifier using a CNN with GroupNorm, data augmentation, Adam optimizer with cosine LR schedule."""

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
                self.gn1 = nn.GroupNorm(4, 32)  # GroupNorm with 4 groups
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.gn2 = nn.GroupNorm(8, 64)  # GroupNorm with 8 groups
                self.fc1 = nn.Linear(64 * 7 * 7, 256)
                self.fc2 = nn.Linear(256, 10)

            def forward(self, x):
                x = (x - mean) / (std + 1e-6)
                x = F.relu(self.gn1(self.conv1(x)))
                x = F.max_pool2d(x, 2)  # 28->14
                x = F.relu(self.gn2(self.conv2(x)))
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

        # Precompute permuted indices for data augmentation consistency

        for epoch in range(15):  # increased epochs for better convergence
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                batch_x = X[idx]

                # Data augmentation: random horizontal flip (prob=0.5) and random affine small translation
                if batch_x.size(0) > 0:
                    # Horizontal flip
                    flip_mask = torch.rand(batch_x.size(0), generator=g) < 0.5
                    batch_x[flip_mask] = torch.flip(batch_x[flip_mask], dims=[3])

                    # Random translation up to 2 pixels
                    translations = torch.randint(-2, 3, (batch_x.size(0), 2), generator=g)
                    grid = F.affine_grid(torch.eye(2,3).unsqueeze(0).repeat(batch_x.size(0),1,1), batch_x.size(), align_corners=False)
                    grid = grid + translations.unsqueeze(1).unsqueeze(1).to(torch.float32) / torch.tensor([w/2, h/2])
                    batch_x = F.grid_sample(batch_x, grid.clamp(-1,1), padding_mode='border', align_corners=False)

                opt.zero_grad()
                loss = loss_fn(self.model(batch_x), y[idx])
                loss.backward()
                opt.step()

            # Cosine annealing LR schedule
            lr = 0.001 * 0.5 * (1 + torch.cos(torch.tensor(epoch / 15 * 3.141592653589793)))
            for param_group in opt.param_groups:
                param_group['lr'] = lr.item()

    def predict(self, X):
        torch.manual_seed(0)  # deterministic if applicable
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(1)
