from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def _compute_fisher(self, net, data_loader, criterion, max_batches=100):
        # Compute diagonal Fisher information matrix for EWC
        net.eval()
        fisher = {}
        for n, p in net.named_parameters():
            fisher[n] = torch.zeros_like(p, device=DEVICE)

        count = 0
        for batch_idx, sample in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            if isinstance(sample, dict):
                inputs = sample["image"]
                targets = sample["age_group"]
            else:
                inputs, targets = sample
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            net.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for n, p in net.named_parameters():
                if p.grad is not None:
                    fisher[n] += (p.grad.data ** 2) / max_batches
            count += 1

        return fisher

    def run(self, net, retain_loader, forget_loader, val_loader):
        epochs = 2
        lr = 0.005
        ascent_steps = 4  # increased gradient ascent steps on forget
        descent_steps = 1  # gradient descent on retain
        ewc_lambda_max = 1500.0  # stronger EWC regularization max

        net.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Compute Fisher information on retain set for EWC
        fisher = self._compute_fisher(net, retain_loader, criterion)

        # Backup current parameters
        params_backup = {n: p.clone().detach() for n, p in net.named_parameters()}

        net.train()

        forget_data = list(forget_loader)
        retain_data = list(retain_loader)

        total_batches = max(len(forget_data), len(retain_data))

        for ep in range(epochs):
            # Cosine ramp-up for EWC strength
            ewc_lambda = ewc_lambda_max * (1 - 0.5 * (1 + torch.cos(torch.tensor(ep / epochs * 3.1415926535))))
            ewc_lambda = ewc_lambda.item()

            for i in range(total_batches):
                # Gradient ascent on forget set
                if i < len(forget_data):
                    for step in range(ascent_steps):
                        sample = forget_data[i]
                        if isinstance(sample, dict):
                            inputs = sample["image"]
                            targets = sample["age_group"]
                        else:
                            inputs, targets = sample
                        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                        optimizer.zero_grad()
                        outputs = net(inputs)
                        loss = criterion(outputs, targets)
                        # Gradient ascent (maximize loss)
                        (-loss).backward()
                        optimizer.step()

                    # After ascent steps, add small gaussian noise to parameters to escape local minima
                    noise_std = 1e-3
                    with torch.no_grad():
                        for p in net.parameters():
                            p.add_(torch.randn_like(p) * noise_std)

                # Gradient descent on retain set with EWC penalty
                if i < len(retain_data):
                    for _ in range(descent_steps):
                        sample = retain_data[i]
                        if isinstance(sample, dict):
                            inputs = sample["image"]
                            targets = sample["age_group"]
                        else:
                            inputs, targets = sample
                        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                        optimizer.zero_grad()
                        outputs = net(inputs)
                        loss = criterion(outputs, targets)

                        ewc_loss = 0
                        for n, p in net.named_parameters():
                            ewc_loss += (fisher[n] * (p - params_backup[n])**2).sum()
                        loss += ewc_lambda * ewc_loss

                        loss.backward()
                        optimizer.step()

            scheduler.step()

        net.eval()
