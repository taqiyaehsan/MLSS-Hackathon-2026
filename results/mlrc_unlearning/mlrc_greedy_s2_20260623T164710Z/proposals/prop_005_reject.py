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
        net.to(DEVICE)

        # Save original model for distillation
        net.eval()
        original_net = deepcopy(net).to(DEVICE)
        original_net.eval()

        # --- Phase 1: Gradient ascent on forget set to degrade forget data memorization ---
        forget_epochs = 2  # small number to limit runtime
        kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        optimizer_forget = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

        uniform_target = None  # will create per-batch

        for ep in range(forget_epochs):
            net.train()
            for batch_idx, sample in enumerate(forget_loader):
                if isinstance(sample, dict):
                    inputs = sample["image"] if "image" in sample else sample[0]
                else:
                    inputs = sample[0]
                inputs = inputs.to(DEVICE)

                optimizer_forget.zero_grad()

                outputs = net(inputs)
                log_probs = F.log_softmax(outputs, dim=1)

                if uniform_target is None or uniform_target.size(0) != outputs.size(0) or uniform_target.size(1) != outputs.size(1):
                    uniform_target = torch.full_like(outputs, fill_value=1.0 / outputs.size(1))

                # KL divergence from outputs to uniform: KL(outputs || uniform)
                # For gradient ascent we maximize KL(outputs||uniform) = E[log outputs - log uniform]
                # PyTorch's KLDivLoss expects input=log_probs and target=probs
                kl = kl_loss_fn(log_probs, uniform_target)

                # Gradient ascent: maximize kl => minimize -kl
                loss = -kl

                loss.backward()
                optimizer_forget.step()

        # --- Phase 2: Fine-tune on retain set with cross-entropy + symmetric KL distillation ---
        retain_epochs = 1
        criterion = nn.CrossEntropyLoss()
        optimizer_retain = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_retain, T_max=retain_epochs)

        for ep in range(retain_epochs):
            net.train()
            for batch_idx, sample in enumerate(retain_loader):
                if isinstance(sample, dict):
                    inputs = sample["image"] if "image" in sample else sample[0]
                    targets = sample["age_group"] if "age_group" in sample else sample[1]
                else:
                    inputs, targets = sample
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer_retain.zero_grad()
                outputs = net(inputs)

                with torch.no_grad():
                    original_outputs = original_net(inputs)

                ce_loss = criterion(outputs, targets)

                p = F.log_softmax(outputs, dim=1)
                q = F.softmax(original_outputs, dim=1)
                kl_pq = F.kl_div(p, q, reduction='batchmean')

                p2 = F.softmax(outputs, dim=1)
                q2 = F.log_softmax(original_outputs, dim=1)
                kl_qp = F.kl_div(q2, p2, reduction='batchmean')

                sym_kl = (kl_pq + kl_qp) / 2

                loss = ce_loss + 0.5 * sym_kl

                loss.backward()
                optimizer_retain.step()
            scheduler.step()

        net.eval()
