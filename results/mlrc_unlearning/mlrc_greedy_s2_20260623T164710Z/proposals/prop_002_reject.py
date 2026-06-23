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

        # Save a copy of the original model for distillation
        net_orig = deepcopy(net).to(DEVICE)
        net_orig.eval()

        # Loss functions
        ce_loss = nn.CrossEntropyLoss()
        kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

        # Optimizer and scheduler for retain fine-tuning
        epochs = 1
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Step 1: Gradient ascent on forget set to push predictions toward uniform
        # Use a small number of iterations to keep runtime low
        net.train()
        forget_steps = 5
        forget_lr = 0.01
        optimizer_forget = optim.SGD(net.parameters(), lr=forget_lr, momentum=0.9, weight_decay=5e-4)

        uniform_dist = None  # To be initialized per batch

        for _ in range(forget_steps):
            for batch_idx, sample in enumerate(forget_loader):
                if isinstance(sample, dict):
                    inputs = sample["image"]
                else:
                    inputs = sample[0]
                inputs = inputs.to(DEVICE)

                optimizer_forget.zero_grad()
                outputs = net(inputs)
                log_probs = nn.functional.log_softmax(outputs, dim=1)
                if uniform_dist is None or uniform_dist.size(0) != outputs.size(0):
                    uniform_dist = torch.full_like(outputs, 1.0 / outputs.size(1))

                # KL divergence from model to uniform: KL(f(x) || uniform)
                # We maximize it by gradient ascent, so negate for loss
                loss_kl = kl_loss_fn(log_probs, uniform_dist)
                loss = -loss_kl  # Gradient ascent
                loss.backward()
                optimizer_forget.step()

        # Step 2: Fine-tune on retain set with cross-entropy + symmetric KL distillation
        # Combined loss: CE + 0.5 * (KL(net||orig) + KL(orig||net))
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
                outputs_orig = net_orig(inputs).detach()

                ce = ce_loss(outputs, targets)

                log_probs = nn.functional.log_softmax(outputs, dim=1)
                probs_orig = nn.functional.softmax(outputs_orig, dim=1)
                kl_fw = kl_loss_fn(log_probs, probs_orig)  # KL(net||orig)

                log_probs_orig = nn.functional.log_softmax(outputs_orig, dim=1)
                probs = nn.functional.softmax(outputs, dim=1)
                kl_bw = kl_loss_fn(log_probs_orig, probs)  # KL(orig||net)

                sym_kl = 0.5 * (kl_fw + kl_bw)

                loss = ce + sym_kl
                loss.backward()
                optimizer.step()
            scheduler.step()

        net.eval()
