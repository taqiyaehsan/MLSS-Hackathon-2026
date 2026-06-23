from copy import deepcopy
import torch
from torch import nn, optim
import torch.nn.functional as F
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        epochs = 1
        forget_ascent_steps = 3  # Number of gradient ascent epochs on forget set
        criterion = nn.CrossEntropyLoss()

        # Save original model for distillation
        net_orig = deepcopy(net).to(DEVICE)
        net_orig.eval()

        # Move net to DEVICE
        net.to(DEVICE)

        optimizer_f = optim.SGD(net.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-4)
        optimizer_r = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

        scheduler_r = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_r, T_max=epochs)

        def symmetric_kl(p_logits, q_logits, eps=1e-8):
            p = F.softmax(p_logits, dim=1)
            q = F.softmax(q_logits, dim=1)
            p_log = F.log_softmax(p_logits, dim=1)
            q_log = F.log_softmax(q_logits, dim=1)
            kl_pq = torch.sum(p * (p_log - q_log), dim=1)
            kl_qp = torch.sum(q * (q_log - p_log), dim=1)
            return (kl_pq + kl_qp) * 0.5

        def uniform_kl_loss(logits):
            # KL divergence between model prediction and uniform distribution
            log_probs = F.log_softmax(logits, dim=1)
            num_classes = logits.shape[1]
            uniform_prob = 1.0 / num_classes
            # KL(p||u) = sum p log (p/u) = sum p log p - sum p log u
            # Here we do KL(u||p) = sum u log(u/p) = sum u (log u - log p) = const - sum u log p
            # We want to push predictions toward uniform, so maximize KL divergence from current to uniform
            # Using KL(uniform||p) = constant - sum uniform * log p, minimizing sum uniform * log p pushes towards uniform
            # For ascent, maximize KL(uniform||p) => maximize -sum uniform*log p => minimize sum uniform*log p
            loss = torch.mean(torch.sum((1.0 / num_classes) * (-log_probs), dim=1))
            return loss

        # Phase 1: Gradient ascent on forget set to push predictions toward uniform
        net.train()
        for _ in range(forget_ascent_steps):
            for batch_idx, sample in enumerate(forget_loader):
                if isinstance(sample, dict):
                    inputs = sample["image"] if "image" in sample else sample["inputs"] if "inputs" in sample else None
                    targets = sample["age_group"] if "age_group" in sample else sample["labels"] if "labels" in sample else None
                    if inputs is None or targets is None:
                        inputs, targets = sample["inputs"], sample["labels"]
                else:
                    inputs, targets = sample
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                optimizer_f.zero_grad()
                outputs = net(inputs)

                loss = uniform_kl_loss(outputs)  # loss to minimize to push logits toward uniform
                # Gradient ascent: maximize loss => minimize negative loss
                (-loss).backward()
                optimizer_f.step()

        # Phase 2: Fine-tune on retain set with cross-entropy + symmetric KL distillation
        for ep in range(epochs):
            net.train()
            for batch_idx, sample in enumerate(retain_loader):
                if isinstance(sample, dict):
                    inputs = sample["image"] if "image" in sample else sample["inputs"] if "inputs" in sample else None
                    targets = sample["age_group"] if "age_group" in sample else sample["labels"] if "labels" in sample else None
                    if inputs is None or targets is None:
                        inputs, targets = sample["inputs"], sample["labels"]
                else:
                    inputs, targets = sample
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer_r.zero_grad()
                outputs = net(inputs)
                with torch.no_grad():
                    outputs_orig = net_orig(inputs)

                ce_loss = criterion(outputs, targets)
                kl_loss = symmetric_kl(outputs, outputs_orig).mean()

                loss = ce_loss + kl_loss
                loss.backward()
                optimizer_r.step()
            scheduler_r.step()

        net.eval()
