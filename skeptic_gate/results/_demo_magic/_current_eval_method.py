"""PRIMARY CODE for the MAGIC Gamma Telescope task -- deeper MLP with dropout, cosine annealing LR scheduler, and balanced batches."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class MyMethod(BaseMethod):
    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        n, d = X.shape

        # Define a deeper MLP with two hidden layers, ReLU, and dropout
        class MLP(nn.Module):
            def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout=0.3):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim1)
                self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
                self.fc3 = nn.Linear(hidden_dim2, output_dim)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                return self.fc3(x)

        self.model = MLP(d, 64, 32, 2, dropout=0.3).to(DEVICE)

        # Adam optimizer with weight decay
        opt = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        # Cosine annealing learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

        self.model.train()

        idx0 = (y == 0).nonzero(as_tuple=True)[0]
        idx1 = (y == 1).nonzero(as_tuple=True)[0]
        batch_size = 96  # slightly smaller batch size for more updates

        g = torch.Generator().manual_seed(seed)

        for epoch in range(20):
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
                loss = loss_fn(self.model(X[idx_batch]), y[idx_batch])
                loss.backward()
                opt.step()

            scheduler.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(1)
