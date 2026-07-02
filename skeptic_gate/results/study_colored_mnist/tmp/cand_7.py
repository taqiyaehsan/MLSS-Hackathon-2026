"""PRIMARY CODE for the Colored-MNIST task -- enhanced CNN with 3 conv blocks, ELU, weight normalization, MixUp augmentation, affine transforms, and cosine annealing warm restarts."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class MyMethod(BaseMethod):
    def __init__(self):
        super().__init__()
        # Define a conv net with 3 conv blocks, batchnorm, weightnorm, dropout and ELU activations
        self.model = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(3, 32, kernel_size=3, padding=1)),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2),           # 28->14
            nn.Dropout(0.1),

            nn.utils.weight_norm(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2),           # 14->7
            nn.Dropout(0.15),

            nn.utils.weight_norm(nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2),           # 7->3
            nn.Dropout(0.2),

            nn.Flatten(),
            nn.utils.weight_norm(nn.Linear(128 * 3 * 3, 256)),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
            nn.utils.weight_norm(nn.Linear(256, 10)),
        ).to(DEVICE)

    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        n = X.shape[0]

        # Normalize input to zero mean 0.5 and std 0.5
        X = (X - 0.5) / 0.5

        # Affine augmentation parameters
        max_rotate = 10  # degrees
        max_translate = 2  # pixels

        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        batch_size = 128
        epochs = 20

        # Use cosine annealing with warm restarts to improve optimization
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2)

        for epoch in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]

                batch = X[idx]
                labels = y[idx]

                # Apply random affine augmentation
                angles = (torch.rand(batch.shape[0], generator=g) * 2 - 1) * max_rotate
                translations = (torch.rand(batch.shape[0], 2, generator=g) * 2 - 1) * max_translate

                angles_rad = angles * 3.14159265 / 180.0

                cos = torch.cos(angles_rad)
                sin = torch.sin(angles_rad)

                tx = translations[:,0] * 2 / 28
                ty = translations[:,1] * 2 / 28

                affine_matrices = torch.zeros(batch.shape[0], 2, 3, device=batch.device)
                affine_matrices[:,0,0] = cos
                affine_matrices[:,0,1] = -sin
                affine_matrices[:,0,2] = tx
                affine_matrices[:,1,0] = sin
                affine_matrices[:,1,1] = cos
                affine_matrices[:,1,2] = ty

                grid = F.affine_grid(affine_matrices, batch.size(), align_corners=False)
                batch = F.grid_sample(batch, grid, padding_mode='border', align_corners=False)

                # MixUp augmentation
                if batch.shape[0] > 1:
                    lam = torch.distributions.Beta(0.4, 0.4).sample((batch.shape[0],), generator=g).to(batch.device)
                    lam = torch.max(lam, 1 - lam)  # ensure lam >= 0.5
                    perm_idx = torch.randperm(batch.shape[0], generator=g)
                    batch_perm = batch[perm_idx]
                    labels_perm = labels[perm_idx]

                    lam = lam.view(-1,1,1,1)  # for broadcasting
                    batch = lam * batch + (1 - lam) * batch_perm

                    # For labels, do soft labels as one-hot mix
                    labels_onehot = F.one_hot(labels, num_classes=10).float()
                    labels_perm_onehot = F.one_hot(labels_perm, num_classes=10).float()
                    lam_flat = lam.view(-1,1)
                    mixed_labels = lam_flat * labels_onehot + (1 - lam_flat) * labels_perm_onehot

                    out = self.model(batch)
                    loss = -(mixed_labels * F.log_softmax(out, dim=1)).sum(dim=1).mean()
                else:
                    out = self.model(batch)
                    loss = loss_fn(out, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()
            scheduler.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = (X - 0.5) / 0.5
            out = self.model(X)
            return out.argmax(1)
