"""PRIMARY CODE for the MAGIC Gamma Telescope task -- improved MLP with two hidden layers, GELU, dropout, class weighted loss, Adam optimizer, and cosine annealing LR scheduler."""

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class MyMethod(BaseMethod):
    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        n, d = X.shape

        # Two hidden layers MLP with GELU activations and dropout
        self.model = nn.Sequential(
            nn.Linear(d, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        ).to(DEVICE)

        # Compute class weights to handle class imbalance
        classes, counts = torch.unique(y, return_counts=True)
        weights = counts.float().sum() / (counts.float() * len(classes))
        class_weights = weights.to(DEVICE)

        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        opt = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=30)

        g = torch.Generator().manual_seed(seed)
        batch_size = 128

        self.model.train()
        for epoch in range(30):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model(X[idx]), y[idx])
                loss.backward()
                opt.step()
            scheduler.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(1)
