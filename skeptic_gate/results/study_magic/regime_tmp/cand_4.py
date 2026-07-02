"""PRIMARY CODE for the MAGIC Gamma Telescope task -- enhanced MLP with GELU, layer norm, dropout, weighted loss, LR scheduler, and longer training."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class MyMethod(BaseMethod):
    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        n, d = X.shape

        # Define a 3-layer MLP with GELU activations, LayerNorm and Dropout
        self.model = nn.Sequential(
            nn.Linear(d, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        ).to(DEVICE)

        # Handle class imbalance by weighting the loss
        class_counts = torch.bincount(y)
        class_weights = 1.0 / (class_counts.float() + 1e-6)
        class_weights = class_weights / class_weights.sum() * 2  # normalize weights
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

        opt = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

        g = torch.Generator().manual_seed(seed)
        batch_size = 64

        self.model.train()
        for epoch in range(50):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                output = self.model(X[idx])
                loss = loss_fn(output, y[idx])
                loss.backward()
                opt.step()
            scheduler.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(1)
