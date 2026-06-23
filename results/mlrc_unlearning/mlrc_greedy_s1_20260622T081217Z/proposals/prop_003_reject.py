from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        # Keep a frozen copy of the original model for distillation
        net_orig = deepcopy(net).to(DEVICE).eval()
        for param in net_orig.parameters():
            param.requires_grad = False

        epochs = 2  # Increase epochs slightly for distillation benefit
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kl = nn.KLDivLoss(reduction='batchmean')

        optimizer = optim.SGD(net.parameters(), lr=0.001,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)

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
                with torch.no_grad():
                    outputs_orig = net_orig(inputs)

                loss_ce = criterion_ce(outputs, targets)

                # Compute symmetric KL divergence between outputs and outputs_orig
                log_probs = nn.functional.log_softmax(outputs, dim=1)
                probs_orig = nn.functional.softmax(outputs_orig, dim=1)
                kl_1 = criterion_kl(log_probs, probs_orig)

                log_probs_orig = nn.functional.log_softmax(outputs_orig, dim=1)
                probs = nn.functional.softmax(outputs, dim=1)
                kl_2 = criterion_kl(log_probs_orig, probs)

                loss_kl = (kl_1 + kl_2) * 0.5

                loss = loss_ce + 0.5 * loss_kl

                loss.backward()
                optimizer.step()
            scheduler.step()

        net.eval()
