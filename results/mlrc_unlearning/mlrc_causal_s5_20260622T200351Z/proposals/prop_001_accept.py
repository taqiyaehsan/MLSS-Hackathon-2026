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
        criterion = nn.CrossEntropyLoss()
        # Save original model for distillation
        net_orig = deepcopy(net).to(DEVICE)
        net_orig.eval()

        optimizer = optim.SGD(net.parameters(), lr=0.001,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)

        def symmetric_kl(p_logits, q_logits, eps=1e-8):
            p = F.softmax(p_logits, dim=1)
            q = F.softmax(q_logits, dim=1)
            p_log = F.log_softmax(p_logits, dim=1)
            q_log = F.log_softmax(q_logits, dim=1)
            kl_pq = torch.sum(p * (p_log - q_log), dim=1)
            kl_qp = torch.sum(q * (q_log - p_log), dim=1)
            return (kl_pq + kl_qp) * 0.5

        for ep in range(epochs):
            net.train()
            for batch_idx, sample in enumerate(retain_loader):
                if isinstance(sample, dict):
                    inputs = sample["image"]
                    targets = sample["age_group"]
                else:
                    inputs, targets = sample
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = net(inputs)
                with torch.no_grad():
                    outputs_orig = net_orig(inputs)

                ce_loss = criterion(outputs, targets)
                kl_loss = symmetric_kl(outputs, outputs_orig).mean()

                loss = ce_loss + kl_loss
                loss.backward()
                optimizer.step()
            scheduler.step()

        net.eval()
