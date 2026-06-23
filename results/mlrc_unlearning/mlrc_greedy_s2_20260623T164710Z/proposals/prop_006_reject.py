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
        epochs = 1
        forget_ascent_steps = 3  # small number of ascent epochs on forget set
        criterion = nn.CrossEntropyLoss()

        # Save original model predictions for retain set distillation
        net.eval()
        original_net = deepcopy(net).to(DEVICE)
        original_net.eval()

        # --- Step 1: Gradient ascent on forget set with uniform-target KL divergence ---
        optimizer_forget = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        for _ in range(forget_ascent_steps):
            net.train()
            for sample in forget_loader:
                if isinstance(sample, dict):
                    inputs = sample["image"] if "image" in sample else sample[0]
                else:
                    inputs = sample[0]
                inputs = inputs.to(DEVICE)

                optimizer_forget.zero_grad()
                outputs = net(inputs)
                log_probs = F.log_softmax(outputs, dim=1)
                # Uniform distribution target
                uniform_dist = torch.full_like(outputs, 1.0 / outputs.size(1))
                kl_loss = F.kl_div(log_probs, uniform_dist, reduction='batchmean')
                # Gradient ascent -> maximize kl_loss, so minimize negative kl_loss
                loss = -kl_loss
                loss.backward()
                optimizer_forget.step()

        # --- Step 2: Fine-tune on retain set with combined CE + symmetric KL distillation ---
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for ep in range(epochs):
            net.train()
            for sample in retain_loader:
                if isinstance(sample, dict):
                    inputs = sample["image"] if "image" in sample else sample[0]
                    targets = sample["age_group"] if "age_group" in sample else sample[1]
                else:
                    inputs, targets = sample
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = net(inputs)
                with torch.no_grad():
                    original_outputs = original_net(inputs)

                ce_loss = criterion(outputs, targets)

                # Symmetric KL divergence between current and original outputs
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
