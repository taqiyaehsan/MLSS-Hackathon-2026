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
