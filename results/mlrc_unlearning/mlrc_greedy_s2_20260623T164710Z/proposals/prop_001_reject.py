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
        # Move model to device
        net.to(DEVICE)
        net.train()

        # Save original model for distillation
        net_orig = deepcopy(net).to(DEVICE)
        net_orig.eval()

        # Parameters
        forget_epochs = 1  # very lightweight gradient ascent
        retain_epochs = 1  # baseline fine-tuning epochs
        lr_forget = 0.001
        lr_retain = 0.001

        # Losses
        ce_loss = nn.CrossEntropyLoss()

        # Uniform distribution for forget set KL
        n_classes = None

        # Prepare optimizer for ascent on forget set
        optimizer_forget = optim.SGD(net.parameters(), lr=lr_forget, momentum=0.9, weight_decay=5e-4)

        # First: gradient ascent on forget set with KL divergence to uniform
        # This encourages forgetting by pushing predictions to max entropy

        # Lazy get n_classes from first batch
        for batch in forget_loader:
            if isinstance(batch, dict):
                inputs = batch["image"] if "image" in batch else next(iter(batch.values()))
                targets = batch["age_group"] if "age_group" in batch else None
            else:
                inputs, targets = batch
            inputs = inputs.to(DEVICE)
            with torch.no_grad():
                outputs = net_orig(inputs)
            n_classes = outputs.shape[1]
            break

        uniform_dist = torch.full((inputs.size(0), n_classes), 1.0 / n_classes, device=DEVICE)

        net.train()
        for ep in range(forget_epochs):
            for batch in forget_loader:
                if isinstance(batch, dict):
                    inputs = batch["image"] if "image" in batch else next(iter(batch.values()))
                    targets = batch["age_group"] if "age_group" in batch else None
                else:
                    inputs, targets = batch
                inputs = inputs.to(DEVICE)

                optimizer_forget.zero_grad()
                outputs = net(inputs)
                log_probs = F.log_softmax(outputs, dim=1)

                # KL divergence between model prediction and uniform distribution
                # We want to maximize KL(pred || uniform) so do gradient ascent on -KL
                # Here KL(pred||uniform) = sum pred * (log pred - log uniform)
                # We maximize KL, so loss = -KL
                # Compute pred as softmax
                probs = F.softmax(outputs, dim=1)
                kl_div = torch.sum(probs * (log_probs - torch.log(uniform_dist)), dim=1).mean()
                loss = -kl_div  # ascent

                loss.backward()
                optimizer_forget.step()

        # Now fine-tune on retain set with combined CE + symmetric KL to original model
        optimizer_retain = optim.SGD(net.parameters(), lr=lr_retain, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_retain, T_max=retain_epochs)

        for ep in range(retain_epochs):
            net.train()
            for batch in retain_loader:
                if isinstance(batch, dict):
                    inputs = batch["image"] if "image" in batch else next(iter(batch.values()))
                    targets = batch["age_group"] if "age_group" in batch else None
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer_retain.zero_grad()

                outputs = net(inputs)
                outputs_orig = net_orig(inputs).detach()

                # CE loss
                loss_ce = ce_loss(outputs, targets)

                # Symmetric KL divergence between outputs and outputs_orig
                log_prob = F.log_softmax(outputs, dim=1)
                prob = F.softmax(outputs, dim=1)
                log_prob_orig = F.log_softmax(outputs_orig, dim=1)
                prob_orig = F.softmax(outputs_orig, dim=1)

                kl1 = F.kl_div(log_prob, prob_orig, reduction='batchmean')
                kl2 = F.kl_div(log_prob_orig, prob, reduction='batchmean')
                loss_kl = (kl1 + kl2) / 2

                loss = loss_ce + loss_kl
                loss.backward()
                optimizer_retain.step()
            scheduler.step()

        net.eval()
