"""FashionMNIST classifier with residual blocks, Cutout augmentation, GroupNorm, Adam optimizer, and warmup+cosine LR schedule."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class ResidualBlock(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += identity
        return F.relu(out)

class MyMethod(BaseMethod):
    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        n, c, h, w = X.shape

        mean = X.mean()
        std = X.std()

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 48, 3, padding=1, bias=False)
                self.gn1 = nn.GroupNorm(8, 48)

                self.res1 = ResidualBlock(48, 8)

                self.conv2 = nn.Conv2d(48, 96, 3, padding=1, bias=False)
                self.gn2 = nn.GroupNorm(12, 96)

                self.res2 = ResidualBlock(96, 12)

                self.fc1 = nn.Linear(96 * 7 * 7, 320)
                self.fc2 = nn.Linear(320, 10)

            def forward(self, x):
                x = (x - mean) / (std + 1e-6)
                x = F.relu(self.gn1(self.conv1(x)))
                x = self.res1(x)
                x = F.max_pool2d(x, 2)  # 28->14

                x = F.relu(self.gn2(self.conv2(x)))
                x = self.res2(x)
                x = F.max_pool2d(x, 2)  # 14->7

                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        self.model = Net().to(DEVICE)
        opt = torch.optim.Adam(self.model.parameters(), lr=0.0015, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        batch_size = 128
        g = torch.Generator().manual_seed(seed)
        self.model.train()

        def cutout(x, size=7):
            # x: (B, C, H, W), apply cutout per image
            for i in range(x.size(0)):
                cx = torch.randint(0, w, (1,), generator=g).item()
                cy = torch.randint(0, h, (1,), generator=g).item()
                x1 = max(cx - size // 2, 0)
                y1 = max(cy - size // 2, 0)
                x2 = min(cx + size // 2, w)
                y2 = min(cy + size // 2, h)
                x[i, :, y1:y2, x1:x2] = mean
            return x

        epochs = 18

        for epoch in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                batch_x = X[idx].clone()

                # Data augment: random horizontal flip
                flip_mask = torch.rand(batch_x.size(0), generator=g) < 0.5
                batch_x[flip_mask] = torch.flip(batch_x[flip_mask], dims=[3])

                # Random translation up to 2 pixels
                translations = torch.randint(-2, 3, (batch_x.size(0), 2), generator=g)
                base_grid = F.affine_grid(torch.eye(2, 3).unsqueeze(0).repeat(batch_x.size(0), 1, 1), batch_x.size(), align_corners=False)
                translations_f = translations.unsqueeze(1).unsqueeze(1).to(torch.float32) / torch.tensor([w / 2, h / 2])
                grid = base_grid + translations_f
                batch_x = F.grid_sample(batch_x, grid.clamp(-1, 1), padding_mode='border', align_corners=False)

                # Cutout (on CPU, so in-place)
                batch_x = cutout(batch_x, size=7)

                opt.zero_grad()
                loss = loss_fn(self.model(batch_x), y[idx])
                loss.backward()
                opt.step()

            # Learning rate warmup (first 3 epochs), then cosine annealing
            if epoch < 3:
                lr = 0.0015 * (epoch + 1) / 3
            else:
                progress = (epoch - 3) / (epochs - 3)
                lr = 0.0015 * 0.5 * (1 + torch.cos(progress * 3.141592653589793))
            for param_group in opt.param_groups:
                param_group['lr'] = lr.item()

    def predict(self, X):
        torch.manual_seed(0)  # deterministic if applicable
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(1)
