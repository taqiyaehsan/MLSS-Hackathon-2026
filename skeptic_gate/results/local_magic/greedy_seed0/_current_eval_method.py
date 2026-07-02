"""PRIMARY CODE for the MAGIC Gamma Telescope task -- deeper MLP with batch normalization, label smoothing, and OneCycleLR scheduler."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class MyMethod(BaseMethod):
    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        n, d = X.shape

        class MLP(nn.Module):
            def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout=0.3):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim1)
                self.bn1 = nn.BatchNorm1d(hidden_dim1)
                self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
                self.bn2 = nn.BatchNorm1d(hidden_dim2)
                self.fc3 = nn.Linear(hidden_dim2, output_dim)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                x = self.fc1(x)
                x = self.bn1(x)
                x = F.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                x = self.bn2(x)
                x = F.relu(x)
                x = self.dropout(x)
                return self.fc3(x)

        self.model = MLP(d, 128, 64, 2, dropout=0.3).to(DEVICE)

        opt = torch.optim.Adam(self.model.parameters(), lr=0.05, weight_decay=1e-4)

        # Use OneCycleLR scheduler for sharper convergence
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.05, total_steps=40)

        self.model.train()

        idx0 = (y == 0).nonzero(as_tuple=True)[0]
        idx1 = (y == 1).nonzero(as_tuple=True)[0]
        batch_size = 64

        g = torch.Generator().manual_seed(seed)

        # Label smoothing for CrossEntropyLoss
        # Implement label smoothing manually
        smoothing = 0.1

        for epoch in range(40):
            perm0 = idx0[torch.randperm(len(idx0), generator=g)]
            perm1 = idx1[torch.randperm(len(idx1), generator=g)]

            nb0 = (len(perm0) + batch_size - 1) // batch_size
            nb1 = (len(perm1) + batch_size - 1) // batch_size

            for i in range(max(nb0, nb1)):
                batch_idx = []
                if i < nb0:
                    batch_idx.append(perm0[i*batch_size:(i+1)*batch_size])
                if i < nb1:
                    batch_idx.append(perm1[i*batch_size:(i+1)*batch_size])
                if len(batch_idx) == 0:
                    continue
                idx_batch = torch.cat(batch_idx)

                opt.zero_grad()
                logits = self.model(X[idx_batch])

                # Create smoothed targets
                targets = y[idx_batch]
                n_classes = 2
                with torch.no_grad():
                    true_dist = torch.full_like(logits, smoothing / (n_classes - 1))
                    true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)

                log_probs = F.log_softmax(logits, dim=1)
                loss = -(true_dist * log_probs).sum(dim=1).mean()

                loss.backward()
                opt.step()
                scheduler.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(1)
