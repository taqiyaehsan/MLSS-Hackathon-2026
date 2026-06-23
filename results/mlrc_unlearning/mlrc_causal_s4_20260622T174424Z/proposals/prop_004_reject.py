from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        epochs = 1
        forget_ascent_steps = 2  # small number of ascent epochs on forget set
        criterion = nn.CrossEntropyLoss()
        kl_loss = nn.KLDivLoss(reduction='batchmean')

        net.to(DEVICE)
        net.train()

        # Keep a frozen copy of the original model for distillation
        net_orig = deepcopy(net)
        net_orig.eval()
        net_orig.to(DEVICE)
        for param in net_orig.parameters():
            param.requires_grad = False

        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Phase 1: Gradient ascent on forget set to push predictions towards uniform
        uniform_target = None  # will initialize per batch
        optimizer_forget = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

        for _ in range(forget_ascent_steps):
            net.train()
            for sample in forget_loader:
                if isinstance(sample, dict):
                    inputs = sample["image"]
                else:
                    inputs, _ = sample
                inputs = inputs.to(DEVICE)

                optimizer_forget.zero_grad()
                outputs = net(inputs)
                log_probs = nn.functional.log_softmax(outputs, dim=1)

                # Create uniform target distribution
                if uniform_target is None or uniform_target.size(0) != inputs.size(0) or uniform_target.size(1) != outputs.size(1):
                    uniform_target = torch.full_like(outputs, 1.0 / outputs.size(1))

                # KL divergence loss: KL(net || uniform) = sum p log(p/q), here q=uniform
                # We want to maximize this loss (gradient ascent) to push p away from uniform,
                # but to unlearn we want to minimize confidence, so maximize KL divergence from original predictions to uniform by gradient ascent on negative KL(net || uniform).
                # So we do gradient ascent on negative KL divergence = gradient descent on KL divergence.
                # To do gradient ascent on KL(net||uniform), we minimize negative kl_loss.

                loss_kl_uniform = kl_loss(log_probs, uniform_target)
                loss = -loss_kl_uniform  # gradient ascent

                loss.backward()
                optimizer_forget.step()

        # Phase 2: Fine-tune on retain set with CE + symmetric KL distillation
        for ep in range(epochs):
            net.train()
            for sample in retain_loader:
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

                # Cross-entropy loss on retain data
                loss_ce = criterion(outputs, targets)

                # Compute log softmax and softmax for KL divergence
                log_probs = nn.functional.log_softmax(outputs, dim=1)
                probs_orig = nn.functional.softmax(outputs_orig, dim=1)

                # Symmetric KL divergence: KL(net || net_orig) + KL(net_orig || net)
                kl1 = kl_loss(log_probs, probs_orig)
                kl2 = kl_loss(nn.functional.log_softmax(outputs_orig, dim=1), nn.functional.softmax(outputs, dim=1))
                loss_kl = (kl1 + kl2) / 2

                loss = loss_ce + 0.5 * loss_kl

                loss.backward()
                optimizer.step()
            scheduler.step()

        net.eval()
