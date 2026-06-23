from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        epochs = 2  # modest increase to allow distillation effect
        ce_criterion = nn.CrossEntropyLoss()
        kl_criterion = nn.KLDivLoss(reduction='batchmean')

        net.to(DEVICE)
        # Save original model outputs on retain set for distillation
        net_orig = deepcopy(net).eval()
        net_orig.to(DEVICE)
        
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for ep in range(epochs):
            net.train()
            for batch_idx, sample in enumerate(retain_loader):
                if isinstance(sample, dict):
                    inputs = sample["image"] if "image" in sample else sample["inputs"]
                    targets = sample["age_group"] if "age_group" in sample else sample["targets"]
                else:
                    inputs, targets = sample
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()

                outputs = net(inputs)
                with torch.no_grad():
                    outputs_orig = net_orig(inputs)

                # Cross entropy loss on retain data
                ce_loss = ce_criterion(outputs, targets)

                # Symmetric KL divergence distillation loss
                log_probs = nn.functional.log_softmax(outputs, dim=1)
                probs_orig = nn.functional.softmax(outputs_orig, dim=1)
                kl_loss1 = kl_criterion(log_probs, probs_orig)

                log_probs_orig = nn.functional.log_softmax(outputs_orig, dim=1)
                probs = nn.functional.softmax(outputs, dim=1)
                kl_loss2 = kl_criterion(log_probs_orig, probs)

                kl_loss = (kl_loss1 + kl_loss2) * 0.5

                loss = ce_loss + kl_loss
                loss.backward()
                optimizer.step()
            scheduler.step()

        net.eval()
