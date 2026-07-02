"""PRIMARY CODE for the MAGIC Gamma Telescope task -- improved MLP with GELU, dropout, class-weighted loss, Adam optimizer, and longer training."""

import torch
import torch.nn as nn

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class MyMethod(BaseMethod):
    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        n, d = X.shape

        # Calculate class weights to handle imbalance
        classes, counts = torch.unique(y, return_counts=True)
        weights = counts.float().max() / counts.float()
        class_weights = torch.zeros(2, device=DEVICE)
        class_weights[classes] = weights

        # Define a 3-layer MLP with GELU activations and dropout
        self.model = nn.Sequential(
            nn.Linear(d, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        ).to(DEVICE)

        opt = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        g = torch.Generator().manual_seed(seed)
        batch_size = 128

        self.model.train()
        for epoch in range(30):  # longer training
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model(X[idx]), y[idx])
                loss.backward()
                opt.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(1)
