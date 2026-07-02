"""Enhanced FashionMNIST classifier using a CNN with LayerNorm, SiLU activations, MixUp augmentation, Adam optimizer with OneCycleLR schedule."""

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
                self.conv1 = nn.Conv2d(1, 48, 3, padding=1)
                self.ln1 = nn.LayerNorm([48, 28, 28])
                self.conv2 = nn.Conv2d(48, 96, 3, padding=1)
                self.ln2 = nn.LayerNorm([96, 14, 14])
                self.fc1 = nn.Linear(96 * 7 * 7, 256)
                self.ln3 = nn.LayerNorm(256)
                self.fc2 = nn.Linear(256, 10)

            def forward(self, x):
                x = (x - mean) / (std + 1e-6)
                x = self.conv1(x)
                x = self.ln1(x)
                x = F.silu(x)
                x = F.max_pool2d(x, 2)  # 28->14
                x = self.conv2(x)
                x = self.ln2(x)
                x = F.silu(x)
                x = F.max_pool2d(x, 2)  # 14->7
                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                x = self.ln3(x)
                x = F.silu(x)
                x = self.fc2(x)
                return x

        self.model = Net().to(DEVICE)
        opt = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        batch_size = 128
        g = torch.Generator().manual_seed(seed)
        self.model.train()

        # OneCycleLR scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01, total_steps=(n // batch_size) * 15)

        for epoch in range(15):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                batch_x = X[idx].clone()
                batch_y = y[idx]

                # MixUp augmentation with alpha=0.4
                if batch_x.size(0) > 1:
                    lam = torch.distributions.Beta(0.4, 0.4).sample((batch_x.size(0),), generator=g).to(batch_x.device)
                    lam = torch.maximum(lam, 1 - lam)  # ensure lam >= 0.5
                    perm_idx = torch.randperm(batch_x.size(0), generator=g)
                    x2 = batch_x[perm_idx]
                    y2 = batch_y[perm_idx]
                    lam = lam.view(-1, 1, 1, 1)
                    batch_x = batch_x * lam + x2 * (1 - lam)
                    # For labels, we do soft labels for mixup
                    # CrossEntropyLoss doesn't take soft labels, so use log_softmax + negative log likelihood manually
                    logits = self.model(batch_x)
                    log_probs = F.log_softmax(logits, dim=1)
                    y_onehot = F.one_hot(batch_y, num_classes=10).float()
                    y2_onehot = F.one_hot(y2, num_classes=10).float()
                    mixed_labels = y_onehot * lam.view(-1, 1) + y2_onehot * (1 - lam).view(-1, 1)
                    loss = (-mixed_labels * log_probs).sum(dim=1).mean()

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    scheduler.step()
                else:
                    # fallback if batch size 1
                    opt.zero_grad()
                    loss = loss_fn(self.model(batch_x), batch_y)
                    loss.backward()
                    opt.step()
                    scheduler.step()

    def predict(self, X):
        torch.manual_seed(0)  # deterministic if applicable
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(1)
