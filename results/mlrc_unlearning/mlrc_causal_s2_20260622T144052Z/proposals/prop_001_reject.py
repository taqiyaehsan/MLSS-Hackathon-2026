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
        net.train()

        # Save original model outputs for distillation on retain set
        original_net = deepcopy(net).to(DEVICE)
        original_net.eval()

        # Phase 1: Gradient ascent on forget set pushing outputs toward uniform
        forget_epochs = 1
        optimizer_forget = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        uniform_target = None  # create on the fly per batch

        for _ in range(forget_epochs):
            for batch in forget_loader:
                if isinstance(batch, dict):
                    inputs = batch["image"]
                else:
                    inputs, _ = batch
                inputs = inputs.to(DEVICE)

                optimizer_forget.zero_grad()
                outputs = net(inputs)
                log_probs = F.log_softmax(outputs, dim=1)

                if uniform_target is None or uniform_target.shape[0] != outputs.shape[0]:
                    uniform_target = torch.full_like(outputs, 1.0 / outputs.shape[1])

                # KL divergence KL(uniform || model) = sum u * (log u - log p)
                # We want to maximize KL(model || uniform), so minimize negative KL(model || uniform)
                # But torch.nn.functional.kl_div expects input as log_probs and target as probs
                # So kl_div(log_probs, uniform_target) computes KL(model || uniform)
                # For gradient ascent, minimize negative kl_div
                kl_loss = F.kl_div(log_probs, uniform_target, reduction='batchmean')
                loss = -kl_loss  # ascent
                loss.backward()
                optimizer_forget.step()

        # Phase 2: Fine-tune on retain set with CE + symmetric KL distillation to original net
        retain_epochs = 2
        optimizer_retain = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

        for _ in range(retain_epochs):
            for batch in retain_loader:
                if isinstance(batch, dict):
                    inputs = batch["image"]
                    targets = batch["age_group"]
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer_retain.zero_grad()

                outputs = net(inputs)
                ce_loss = nn.CrossEntropyLoss()(outputs, targets)

                with torch.no_grad():
                    orig_outputs = original_net(inputs)

                p = F.log_softmax(outputs, dim=1)
                q = F.softmax(orig_outputs, dim=1)

                kl1 = F.kl_div(p, q, reduction='batchmean')
                kl2 = F.kl_div(F.log_softmax(orig_outputs, dim=1), F.softmax(outputs, dim=1), reduction='batchmean')
                sym_kl = (kl1 + kl2) * 0.5

                loss = ce_loss + sym_kl
                loss.backward()
                optimizer_retain.step()

        net.eval()
