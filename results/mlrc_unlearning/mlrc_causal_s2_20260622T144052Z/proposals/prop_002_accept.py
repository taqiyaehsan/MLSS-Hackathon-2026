from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        net.to(DEVICE)
        net.train()

        # Save original model for distillation
        net_orig = deepcopy(net).to(DEVICE)
        net_orig.eval()

        # Hyperparameters
        forget_ascent_epochs = 1  # single pass over forget set
        retain_finetune_epochs = 2  # few passes over retain set
        lr_forget = 0.01
        lr_retain = 0.001

        # Losses
        ce_loss = nn.CrossEntropyLoss()
        kl_loss = nn.KLDivLoss(reduction='batchmean')

        # Uniform target distribution for forget set
        # Assuming CIFAR-10 classes = 10
        n_classes = 10
        uniform_dist = torch.full((1, n_classes), 1.0 / n_classes, device=DEVICE)

        # --- Phase 1: Gradient ascent on forget set to push predictions toward uniform ---
        optimizer_forget = optim.SGD(net.parameters(), lr=lr_forget, momentum=0.9, weight_decay=5e-4)

        net.train()
        for _ in range(forget_ascent_epochs):
            for batch in forget_loader:
                if isinstance(batch, dict):
                    inputs = batch['image']
                else:
                    inputs, _ = batch
                inputs = inputs.to(DEVICE)

                optimizer_forget.zero_grad()

                outputs = net(inputs)
                log_probs = nn.functional.log_softmax(outputs, dim=1)

                # KL divergence from uniform to model predictions
                # We want to maximize KL(P_model || uniform) to push predictions to uniform,
                # so minimize negative KL(P_model || uniform) = - sum p log(p/u) = sum p (log u - log p) = const - entropy(p)
                # But we actually want to maximize entropy, so minimize negative entropy.
                # Using KLDivLoss with log_probs and uniform_dist as target (which is fixed).
                # KLDivLoss expects input=log_probs, target=probs. So compute KL(log_probs || uniform) = sum p log(p/u)
                # To do gradient ascent on KL, we minimize negative KL.

                # Compute KLDivLoss(log_probs, uniform_dist) = sum uniform * (log uniform - log p) but uniform is constant,
                # so we want KL(P||uniform) but KLDivLoss computes KL(target||input), so we swap inputs and targets.

                # So to compute KL(P||uniform) we implement manually:
                probs = torch.exp(log_probs)
                kl = torch.sum(probs * (torch.log(probs + 1e-10) - torch.log(uniform_dist + 1e-10)), dim=1).mean()

                loss = -kl  # gradient ascent on KL(P||uniform)

                loss.backward()
                optimizer_forget.step()

        # --- Phase 2: Fine-tune on retain set with CE + symmetric KL distillation ---

        optimizer_retain = optim.SGD(net.parameters(), lr=lr_retain, momentum=0.9, weight_decay=5e-4)

        net.train()
        for epoch in range(retain_finetune_epochs):
            for batch in retain_loader:
                if isinstance(batch, dict):
                    inputs = batch['image']
                    targets = batch['age_group']
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer_retain.zero_grad()

                outputs = net(inputs)
                outputs_orig = net_orig(inputs)

                loss_ce = ce_loss(outputs, targets)

                log_probs = nn.functional.log_softmax(outputs, dim=1)
                log_probs_orig = nn.functional.log_softmax(outputs_orig, dim=1)

                # Symmetric KL divergence: KL(P||Q) + KL(Q||P)
                kl_pq = kl_loss(log_probs, log_probs_orig.exp().detach())
                kl_qp = kl_loss(log_probs_orig, log_probs.exp())
                loss_kl = kl_pq + kl_qp

                loss = loss_ce + loss_kl

                loss.backward()
                optimizer_retain.step()

        net.eval()
