"""PRIMARY CODE for the Colored-MNIST task -- enhanced CNN with 3 conv blocks, dropout, LeakyReLU, cosine annealing LR, and mild random affine augmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class MyMethod(BaseMethod):
    def __init__(self):
        super().__init__()
        # Define a conv net with 3 conv blocks, batchnorm, dropout and LeakyReLU activations
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2),           # 28->14
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2),           # 14->7
            nn.Dropout(0.15),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2),           # 7->3
            nn.Dropout(0.2),

            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        ).to(DEVICE)

        # Define a mild random affine augmentation for training
        self.augment = nn.Sequential(
            nn.Identity()  # placeholder, will apply augmentation manually in fit
        )

    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        n = X.shape[0]

        # Normalize input to zero mean 0.5 and std 0.5 (simple normalization)
        X = (X - 0.5) / 0.5

        # Define random affine augmentation parameters
        # small rotations and translations
        max_rotate = 10  # degrees
        max_translate = 2  # pixels

        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        batch_size = 128
        epochs = 15

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        for _ in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]

                batch = X[idx]

                # On CPU, perform augmentation manually with affine grid/sample
                # Random rotation and translation per image
                angles = (torch.rand(batch.shape[0], generator=g) * 2 - 1) * max_rotate
                translations = (torch.rand(batch.shape[0], 2, generator=g) * 2 - 1) * max_translate

                # Convert angles to radians
                angles_rad = angles * 3.14159265 / 180.0

                # Construct affine matrices
                # Each matrix is 2x3
                cos = torch.cos(angles_rad)
                sin = torch.sin(angles_rad)

                # Affine matrix for each sample: [ [cos, -sin, tx], [sin, cos, ty] ]
                # Normalize translations from pixels to [-1,1] relative to 28x28
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

                opt.zero_grad()
                out = self.model(batch)
                loss = loss_fn(out, y[idx])
                loss.backward()
                opt.step()
            scheduler.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = (X - 0.5) / 0.5
            out = self.model(X)
            return out.argmax(1)
