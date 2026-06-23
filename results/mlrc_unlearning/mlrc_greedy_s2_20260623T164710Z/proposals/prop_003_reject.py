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

        # Save original model predictions function for distillation if needed
        # (Not used here but can be extended)

        # Step 1: Gradient ascent on forget set with uniform-target KL divergence
        # Define optimizer for ascent with smaller lr
        ascent_optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        # Use one epoch over forget_loader for ascent
        criterion_kl = nn.KLDivLoss(reduction='batchmean')

        uniform_prob = None  # to be initialized after first batch

        net.train()
        for batch_idx, sample in enumerate(forget_loader):
            if isinstance(sample, dict):
                inputs = sample["image"]
            else:
                inputs = sample[0]  # only inputs needed
            inputs = inputs.to(DEVICE)

            # Forward
            ascent_optimizer.zero_grad()
            logits = net(inputs)
            log_probs = nn.functional.log_softmax(logits, dim=1)

            # Initialize uniform distribution tensor if needed
            if uniform_prob is None or uniform_prob.size(0) != inputs.size(0) or uniform_prob.size(1) != logits.size(1):
                uniform_prob = torch.full_like(logits, 1.0 / logits.size(1))

            # KL divergence between model prediction and uniform distribution
            # We want to maximize KLDivLoss, so minimize negative KLDivLoss (gradient ascent)
            loss = criterion_kl(log_probs, uniform_prob)
            (-loss).backward()  # gradient ascent
            ascent_optimizer.step()

        # Step 2: Fine-tune on retain set with cross-entropy only
        epochs = 1
        criterion_ce = nn.CrossEntropyLoss()
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
                loss = criterion_ce(outputs, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()

        net.eval()
