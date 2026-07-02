"""PRIMARY CODE for the Colored-MNIST task -- improved CNN with Mixup augmentation and cosine LR scheduling."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod

DEVICE = torch.device("cpu")

def mixup_data(x, y, alpha=0.4, generator=None):
    '''Returns mixed inputs, pairs of targets, and lambda'''    
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample(generator) if generator is not None else torch.distributions.Beta(alpha, alpha).sample()
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size, generator=generator if generator is not None else None)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

class MyMethod(BaseMethod):
    def __init__(self):
        super().__init__()
        # Slightly larger conv net
        self.model = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 28->14
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 14->7
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        ).to(DEVICE)

    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        n = X.shape[0]

        # Normalize input to zero mean 0.5 and std 0.5
        X = (X - 0.5) / 0.5

        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=0.002, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(reduction='none')  # for mixup

        batch_size = 128
        epochs = 12

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        for _ in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                batch_x = X[idx]
                batch_y = y[idx]

                # Apply mixup
                mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, alpha=0.4, generator=g)

                opt.zero_grad()
                out = self.model(mixed_x)
                loss_a = loss_fn(out, y_a)
                loss_b = loss_fn(out, y_b)
                loss = (lam * loss_a + (1 - lam) * loss_b).mean()

                loss.backward()
                opt.step()
            scheduler.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = (X - 0.5) / 0.5
            out = self.model(X)
            return out.argmax(1)
