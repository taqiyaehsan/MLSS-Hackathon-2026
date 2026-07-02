"""PRIMARY CODE for the MAGIC Gamma Telescope task -- enhanced MLP with 2 hidden layers, GELU, dropout, class-weighted loss, Adam optimizer, cosine annealing LR scheduler, and longer training."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class MyMethod(BaseMethod):
    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        n, d = X.shape

        # Define a 2-hidden layer MLP with GELU activations and dropout
        self.model = nn.Sequential(
            nn.Linear(d, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        ).to(DEVICE)

        # Compute class weights to handle imbalance
        labels, counts = torch.unique(y, return_counts=True)
        weights = torch.zeros(2, device=DEVICE)
        weights[labels] = n / (2 * counts.float())

        loss_fn = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

        g = torch.Generator().manual_seed(seed)
        batch_size = 128

        self.model.train()
        for epoch in range(50):  # longer training
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
