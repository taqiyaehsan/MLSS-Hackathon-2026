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

        # Save original model outputs for distillation
        original_net = deepcopy(net).to(DEVICE)
        original_net.eval()

        epochs = 1
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kl = nn.KLDivLoss(reduction='batchmean')

        optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
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
                    orig_outputs = original_net(inputs)

                # Cross entropy loss on retained data
                loss_ce = criterion_ce(outputs, targets)

                # Symmetric KL divergence between current and original model predictions
                log_probs = nn.functional.log_softmax(outputs, dim=1)
                orig_probs = nn.functional.softmax(orig_outputs, dim=1)
                loss_kl = criterion_kl(log_probs, orig_probs) + criterion_kl(
                    nn.functional.log_softmax(orig_outputs, dim=1),
                    nn.functional.softmax(outputs, dim=1))
                loss_kl *= 0.5

                loss = loss_ce + loss_kl
                loss.backward()
                optimizer.step()

            scheduler.step()

        net.eval()
