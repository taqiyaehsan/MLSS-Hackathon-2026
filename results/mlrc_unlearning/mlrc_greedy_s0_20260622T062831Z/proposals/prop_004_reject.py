from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def _compute_fisher(self, net, data_loader, criterion, samples=200):
        # Compute empirical Fisher Information diagonal for parameters on retain set
        net.eval()
        fisher = {}
        for n, p in net.named_parameters():
            fisher[n] = torch.zeros_like(p, device=DEVICE)

        count = 0
        for batch_idx, sample in enumerate(data_loader):
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
                    fisher[n] += (p.grad.detach() ** 2) * inputs.size(0)  # weighted by batch size

            count += inputs.size(0)
            if count >= samples:
                break

        for n in fisher:
            fisher[n] /= count
        return fisher

    def run(self, net, retain_loader, forget_loader, val_loader):
        net.to(DEVICE)
        net.train()

        criterion = nn.CrossEntropyLoss()

        # Save original parameters for EWC penalty
        orig_params = {n: p.detach().clone() for n, p in net.named_parameters()}

        # Compute Fisher Information on retain set (small subset for speed)
        fisher = self._compute_fisher(net, retain_loader, criterion, samples=500)

        # Step 1: Gradient ascent on forget set to increase loss
        forget_optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        forget_epochs = 2

        for epoch in range(forget_epochs):
            net.train()
            for batch_idx, sample in enumerate(forget_loader):
                if isinstance(sample, dict):
                    inputs = sample["image"]
                    targets = sample["age_group"]
                else:
                    inputs, targets = sample
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                forget_optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                # Gradient ascent: maximize the loss on forget set
                (-loss).backward()
                forget_optimizer.step()

        # Step 2: Fine-tune on retain set with EWC penalty to preserve retain knowledge
        retain_optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        retain_epochs = 2

        lambda_ewc = 1000.0  # penalty strength

        for epoch in range(retain_epochs):
            net.train()
            for batch_idx, sample in enumerate(retain_loader):
                if isinstance(sample, dict):
                    inputs = sample["image"]
                    targets = sample["age_group"]
                else:
                    inputs, targets = sample
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                retain_optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                # EWC penalty
                ewc_loss = 0
                for n, p in net.named_parameters():
                    ewc_loss += (fisher[n] * (p - orig_params[n]) ** 2).sum()

                total_loss = loss + (lambda_ewc * ewc_loss)
                total_loss.backward()
                retain_optimizer.step()

        net.eval()
