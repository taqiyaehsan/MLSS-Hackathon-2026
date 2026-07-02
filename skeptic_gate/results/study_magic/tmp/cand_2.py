"""PRIMARY CODE for the MAGIC Gamma Telescope task -- enhanced MLP with batch norm, larger capacity, cosine LR schedule, and balanced batching."""

import torch
import torch.nn as nn

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class MyMethod(BaseMethod):
    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        n, d = X.shape

        # Define model: 3-layer MLP with batch norm and ReLU
        self.model = nn.Sequential(
            nn.Linear(d, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        ).to(DEVICE)

        opt = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        # Cosine annealing LR scheduler for smooth LR decay
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

        g = torch.Generator().manual_seed(seed)
        batch_size = 128

        # Prepare balanced batches to handle class imbalance (~65% vs 35%)
        # Separate indices by class
        idx0 = (y == 0).nonzero(as_tuple=True)[0]
        idx1 = (y == 1).nonzero(as_tuple=True)[0]

        self.model.train()
        for epoch in range(20):
            # Shuffle indices for both classes
            perm0 = idx0[torch.randperm(len(idx0), generator=g)]
            perm1 = idx1[torch.randperm(len(idx1), generator=g)]

            # Calculate number of batches
            n_batches = (n + batch_size - 1) // batch_size

            # Generate balanced batches by sampling roughly equal numbers from each class
            for batch_i in range(n_batches):
                # Compute how many samples from each class for this batch
                n0 = batch_size // 2
                n1 = batch_size - n0

                # Handle possible shortage at the end
                start0 = batch_i * n0
                end0 = min(start0 + n0, len(perm0))
                start1 = batch_i * n1
                end1 = min(start1 + n1, len(perm1))

                batch_idx = torch.cat([perm0[start0:end0], perm1[start1:end1]])

                if len(batch_idx) == 0:
                    continue

                opt.zero_grad()
                out = self.model(X[batch_idx])
                loss = loss_fn(out, y[batch_idx])
                loss.backward()
                opt.step()

            scheduler.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(1)
