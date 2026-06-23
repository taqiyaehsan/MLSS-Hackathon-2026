from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def _compute_fisher(self, net, loader, criterion, n_batches=20):
        net.eval()
        fisher = {n: torch.zeros_like(p, device=DEVICE) for n, p in net.named_parameters() if p.requires_grad}
        count = 0
        for batch_idx, sample in enumerate(loader):
            if batch_idx >= n_batches:
                break
            if isinstance(sample, dict):
                inputs = sample["image"] if "image" in sample else sample[next(iter(sample))]
                targets = sample["age_group"] if "age_group" in sample else sample[next(iter(sample))]
            else:
                inputs, targets = sample
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            net.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for n, p in net.named_parameters():
                if p.grad is not None and p.requires_grad:
                    fisher[n] += (p.grad.detach() ** 2)
            count += 1
        for n in fisher:
            fisher[n] /= count
        return fisher

    def run(self, net, retain_loader, forget_loader, val_loader):
        net.to(DEVICE)
        net.train()

        criterion = nn.CrossEntropyLoss()

        # Save original parameters for EWC penalty
        orig_params = {n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad}

        # Compute approximate Fisher information on retain set (small subset)
        fisher = self._compute_fisher(net, retain_loader, criterion, n_batches=20)

        # Hyperparameters
        lambda_ewc = 1000.0  # strength of EWC penalty
        forget_epochs = 2
        retain_epochs = 2
        lr_forget = 0.01
        lr_retain = 0.001

        # Optimizer for forget and retain phases
        optimizer = optim.SGD(net.parameters(), lr=lr_forget, momentum=0.9, weight_decay=5e-4)

        # 1) Forget phase: gradient ascent on forget set with uniform KL loss to push predictions away
        uniform_dist = torch.full((10,), 1/10, device=DEVICE)  # CIFAR-10 uniform

        kl_loss = nn.KLDivLoss(reduction='batchmean')

        for ep in range(forget_epochs):
            net.train()
            for sample in forget_loader:
                if isinstance(sample, dict):
                    inputs = sample["image"] if "image" in sample else sample[next(iter(sample))]
                else:
                    inputs, _ = sample
                inputs = inputs.to(DEVICE)

                optimizer.zero_grad()
                outputs = net(inputs)
                log_probs = nn.functional.log_softmax(outputs, dim=1)
                # Gradient ascent to maximize KL divergence to uniform = minimize negative KL
                loss = -kl_loss(log_probs, uniform_dist.expand_as(log_probs))
                loss.backward()
                optimizer.step()

        # 2) Retain phase: finetune on retain set with cross-entropy + EWC penalty
        optimizer = optim.SGD(net.parameters(), lr=lr_retain, momentum=0.9, weight_decay=5e-4)

        for ep in range(retain_epochs):
            net.train()
            for sample in retain_loader:
                if isinstance(sample, dict):
                    inputs = sample["image"] if "image" in sample else sample[next(iter(sample))]
                    targets = sample["age_group"] if "age_group" in sample else sample[next(iter(sample))]
                else:
                    inputs, targets = sample
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = net(inputs)
                ce_loss = criterion(outputs, targets)

                # EWC penalty
                ewc_loss = 0.0
                for n, p in net.named_parameters():
                    if p.requires_grad:
                        ewc_loss += (fisher[n] * (p - orig_params[n]) ** 2).sum()
                ewc_loss = (lambda_ewc / 2) * ewc_loss

                loss = ce_loss + ewc_loss
                loss.backward()
                optimizer.step()

        net.eval()
