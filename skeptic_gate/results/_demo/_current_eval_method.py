"""Enhanced FashionMNIST classifier with data augmentation, dropout, label smoothing, and cosine LR scheduler."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class MyMethod(BaseMethod):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.25)
        # Define a CNN with dropout
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # increase channels
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 10)
        ).to(DEVICE)

    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        n = X.shape[0]
        mean = 0.2860406
        std = 0.35302424

        # Normalize
        X = (X - mean) / std

        # Data augmentation: random horizontal flip and random affine with small translation
        def augment(batch):
            # batch shape (B,1,28,28)
            if batch.shape[0] == 0:
                return batch
            # Random horizontal flip
            flip_mask = torch.rand(batch.shape[0], generator=torch.Generator().manual_seed(seed)) < 0.5
            batch[flip_mask] = torch.flip(batch[flip_mask], dims=[3])
            # Random translation up to 2 pixels
            max_trans = 2
            translations = torch.randint(-max_trans, max_trans+1, (batch.shape[0], 2), generator=torch.Generator().manual_seed(seed))
            grid = F.affine_grid(torch.eye(2,3).unsqueeze(0).repeat(batch.shape[0],1,1), batch.size(), align_corners=False)
            grid = grid + translations.unsqueeze(1).unsqueeze(1).float() / 14.0  # normalize by half image size
            batch = F.grid_sample(batch, grid, padding_mode='border', align_corners=False)
            return batch

        opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing to regularize

        g = torch.Generator().manual_seed(seed)
        self.model.train()

        batch_size = 128
        epochs = 15

        # Cosine annealing LR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        for epoch in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                batch_X = X[idx]
                batch_X = augment(batch_X)
                opt.zero_grad()
                loss = loss_fn(self.model(batch_X), y[idx])
                loss.backward()
                opt.step()
            scheduler.step()

    def predict(self, X):
        mean = 0.2860406
        std = 0.35302424
        X = (X - mean) / std
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            return logits.argmax(dim=1)
