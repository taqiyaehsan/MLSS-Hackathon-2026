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
        forget_ascent_steps = 3  # small number of ascent epochs on forget set
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        net.to(DEVICE)
        net.train()

        # Save original model for distillation
        original_net = deepcopy(net).to(DEVICE)
        original_net.eval()

        # --- Step 1: Gradient ascent on forget set to push predictions towards uniform ---
        # Use KL divergence loss: KL(f(x) || uniform)
        uniform_prob = torch.full((1, net.fc.out_features), 1.0 / net.fc.out_features, device=DEVICE)

        for _ in range(forget_ascent_steps):
            for batch in forget_loader:
                if isinstance(batch, dict):
                    inputs = batch["image"]
                else:
                    inputs = batch[0]
                inputs = inputs.to(DEVICE)

                optimizer.zero_grad()
                outputs = net(inputs)
                log_probs = F.log_softmax(outputs, dim=1)

                # Broadcast uniform_prob to batch size
                uniform_targets = uniform_prob.expand_as(log_probs)

                # KL divergence KL(f(x)||uniform) = sum f(x) * (log f(x) - log uniform)
                # Here f(x) = softmax outputs
                probs = F.softmax(outputs, dim=1)
                kl_div = torch.sum(probs * (log_probs - torch.log(uniform_targets)), dim=1).mean()

                # Gradient ascent: maximize kl_div, so minimize -kl_div
                loss = -kl_div
                loss.backward()
                optimizer.step()

        # --- Step 2: Fine-tune on retain set with CE + symmetric KL distillation ---
        net.train()
        for ep in range(epochs):
            for batch in retain_loader:
                if isinstance(batch, dict):
                    inputs = batch["image"]
                    targets = batch["age_group"]
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = net(inputs)
                with torch.no_grad():
                    original_outputs = original_net(inputs)

                ce_loss = criterion(outputs, targets)

                # Symmetric KL divergence
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
