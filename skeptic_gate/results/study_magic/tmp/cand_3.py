"""PRIMARY CODE for the MAGIC Gamma Telescope task -- improved MLP with GELU, dropout, Adam optimizer, cosine LR scheduler, and balanced batches."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class MyMethod(BaseMethod):
    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        n, d = X.shape

        # Define a 3-layer MLP with GELU activations and dropout
        self.model = nn.Sequential(
            nn.Linear(d, 64),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 2),
        ).to(DEVICE)

        opt = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        # Cosine annealing LR scheduler with T_max=20 epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
        loss_fn = nn.CrossEntropyLoss()

        g = torch.Generator().manual_seed(seed)
        batch_size = 128

        # For balanced batches: precompute indices per class
        class0_idx = (y == 0).nonzero(as_tuple=True)[0]
        class1_idx = (y == 1).nonzero(as_tuple=True)[0]
        n0 = len(class0_idx)
        n1 = len(class1_idx)

        self.model.train()
        for epoch in range(20):
            # Shuffle class indices separately
            perm0 = class0_idx[torch.randperm(n0, generator=g)]
            perm1 = class1_idx[torch.randperm(n1, generator=g)]

            # Number of batches based on smaller class
            batches = max(n0, n1) // (batch_size // 2) + 1

            for i in range(batches):
                start0 = i * (batch_size // 2)
                end0 = start0 + (batch_size // 2)
                start1 = i * (batch_size // 2)
                end1 = start1 + (batch_size // 2)

                batch_idx0 = perm0[start0:end0]
                batch_idx1 = perm1[start1:end1]

                # If running out of examples, wrap around
                if len(batch_idx0) < (batch_size // 2):
                    batch_idx0 = torch.cat([batch_idx0, perm0[:(batch_size // 2) - len(batch_idx0)]])
                if len(batch_idx1) < (batch_size // 2):
                    batch_idx1 = torch.cat([batch_idx1, perm1[:(batch_size // 2) - len(batch_idx1)]])

                idx = torch.cat([batch_idx0, batch_idx1])

                opt.zero_grad()
                loss = loss_fn(self.model(X[idx]), y[idx])
                loss.backward()
                opt.step()
            scheduler.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(1)
