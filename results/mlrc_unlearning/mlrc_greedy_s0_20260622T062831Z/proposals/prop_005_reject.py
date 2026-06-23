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

        # Backup original model outputs on retain_loader for distillation
        net.eval()
        original_outputs = []
        with torch.no_grad():
            for sample in retain_loader:
                if isinstance(sample, dict):
                    inputs = sample["image"]
                else:
                    inputs = sample[0]
                inputs = inputs.to(DEVICE)
                out = net(inputs)
                original_outputs.append(out.detach())
        # Flatten original outputs
        original_outputs = torch.cat(original_outputs, dim=0)

        # Step 1: Add Gaussian noise to convolutional weights
        with torch.no_grad():
            for name, param in net.named_parameters():
                if 'conv' in name and param.requires_grad:
                    noise = torch.randn_like(param) * 0.6  # sigma=0.6 as in Seif
                    param.add_(noise)

        # Step 2: Finetune on retain_loader with CE + symmetric KL distillation
        criterion_ce = nn.CrossEntropyLoss()

        def symmetric_kl(p_logits, q_logits):
            p = F.log_softmax(p_logits, dim=1)
            q = F.log_softmax(q_logits, dim=1)
            p_soft = p.exp()
            q_soft = q.exp()
            kl1 = F.kl_div(p, q_soft, reduction='batchmean')
            kl2 = F.kl_div(q, p_soft, reduction='batchmean')
            return (kl1 + kl2) / 2

        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        epochs = 2

        # To iterate original outputs in parallel with retain_loader batches
        # We assume retain_loader batch size is consistent
        batch_size = None
        for sample in retain_loader:
            if isinstance(sample, dict):
                batch_size = sample["image"].size(0)
            else:
                batch_size = sample[0].size(0)
            break
        # Split original_outputs into batches
        original_outputs_batches = torch.split(original_outputs, batch_size)

        net.train()
        for ep in range(epochs):
            for batch_idx, sample in enumerate(retain_loader):
                if isinstance(sample, dict):
                    inputs = sample["image"]
                    targets = sample["age_group"]
                else:
                    inputs, targets = sample
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = net(inputs)

                loss_ce = criterion_ce(outputs, targets)

                # Distillation loss to original outputs
                orig_out = original_outputs_batches[batch_idx].to(DEVICE)
                loss_kl = symmetric_kl(outputs, orig_out)

                loss = loss_ce + 1.0 * loss_kl

                loss.backward()
                optimizer.step()

        net.eval()
