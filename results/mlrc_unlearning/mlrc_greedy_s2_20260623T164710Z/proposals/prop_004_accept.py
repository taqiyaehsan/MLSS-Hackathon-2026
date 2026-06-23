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
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Save original model predictions for retain set to distill
        net.eval()
        original_net = deepcopy(net).to(DEVICE)
        original_net.eval()

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
                    original_outputs = original_net(inputs)

                ce_loss = criterion(outputs, targets)

                # Compute symmetric KL divergence between current and original outputs
                p = F.log_softmax(outputs, dim=1)
                q = F.softmax(original_outputs, dim=1)
                kl_pq = F.kl_div(p, q, reduction='batchmean')
                p2 = F.softmax(outputs, dim=1)
                q2 = F.log_softmax(original_outputs, dim=1)
                kl_qp = F.kl_div(q2, p2, reduction='batchmean')
                sym_kl = (kl_pq + kl_qp) / 2

                loss = ce_loss + 0.5 * sym_kl
                loss.backward()
                optimizer.step()
            scheduler.step()

        net.eval()
