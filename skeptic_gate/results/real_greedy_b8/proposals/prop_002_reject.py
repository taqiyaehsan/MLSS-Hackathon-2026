from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        # Save original net for distillation
        net.to(DEVICE)
        original_net = deepcopy(net).to(DEVICE)
        original_net.eval()

        # Hyperparameters
        epochs = 2  # small number for cheap unlearning
        lr = 0.01
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kl = nn.KLDivLoss(reduction='batchmean')

        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for ep in range(epochs):
            net.train()
            # Step 1: Gradient ascent on forget set
            for batch in forget_loader:
                if isinstance(batch, dict):
                    inputs = batch["image"]
                    targets = batch["age_group"]
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss_forget = criterion_ce(outputs, targets)
                # Gradient ascent: maximize loss on forget data
                (-loss_forget).backward()
                optimizer.step()

            # Step 2: Gradient descent on retain set with distillation
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
                    orig_outputs = original_net(inputs)
                # Cross entropy loss on retain data
                loss_retain = criterion_ce(outputs, targets)
                # Distillation loss: encourage net output to be close to original net logits
                log_probs = nn.functional.log_softmax(outputs / 2.0, dim=1)
                soft_targets = nn.functional.softmax(orig_outputs / 2.0, dim=1)
                loss_distill = criterion_kl(log_probs, soft_targets) * (2.0 * 2.0)  # temp squared

                loss = loss_retain + 0.7 * loss_distill
                loss.backward()
                optimizer.step()

            scheduler.step()

        net.eval()
