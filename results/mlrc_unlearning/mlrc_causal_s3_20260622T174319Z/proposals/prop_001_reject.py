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
        net.to(DEVICE)
        net.train()
        epochs_forget_ascent = 1
        epochs_retain_finetune = 2

        # Save original model for distillation
        net_orig = deepcopy(net).to(DEVICE)
        net_orig.eval()

        # Prepare optimizer for forget ascent: small lr, ascent
        optimizer_forget = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        # Uniform distribution target for forget set logits
        num_classes = None
        for batch in forget_loader:
            if isinstance(batch, dict):
                inputs = batch["image"] if "image" in batch else next(iter(batch.values()))
            else:
                inputs = batch[0]
            inputs = inputs.to(DEVICE)
            outputs = net(inputs)
            num_classes = outputs.size(1)
            break
        uniform_target = torch.full((inputs.size(0), num_classes), 1.0 / num_classes, device=DEVICE)

        # Gradient ascent on forget set to maximize KL divergence to original predictions (towards uniform)
        # We use KL(net_output || uniform) so maximize KL is ascent on this loss
        for _ in range(epochs_forget_ascent):
            net.train()
            for batch in forget_loader:
                if isinstance(batch, dict):
                    inputs = batch["image"] if "image" in batch else next(iter(batch.values()))
                else:
                    inputs = batch[0]
                inputs = inputs.to(DEVICE)
                batch_size = inputs.size(0)
                outputs = net(inputs)
                log_probs = F.log_softmax(outputs, dim=1)
                # Create uniform target for this batch size
                uniform = torch.full((batch_size, outputs.size(1)), 1.0 / outputs.size(1), device=DEVICE)
                # KL divergence KL(output || uniform) = sum output * (log output - log uniform)
                # We want to maximize this, so do gradient ascent on this loss
                loss = F.kl_div(log_probs, uniform, reduction='batchmean')
                optimizer_forget.zero_grad()
                # Gradient ascent: negate loss
                (-loss).backward()
                optimizer_forget.step()

        # Fine-tune on retain set with combined CE and symmetric KL distillation to original model
        criterion_ce = nn.CrossEntropyLoss()
        optimizer_retain = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_retain, T_max=epochs_retain_finetune)

        for epoch in range(epochs_retain_finetune):
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
                with torch.no_grad():
                    orig_outputs = net_orig(inputs)

                loss_ce = criterion_ce(outputs, targets)

                # Symmetric KL divergence
                log_probs = F.log_softmax(outputs, dim=1)
                orig_probs = F.softmax(orig_outputs, dim=1)

                kl1 = F.kl_div(log_probs, orig_probs, reduction='batchmean')
                kl2 = F.kl_div(F.log_softmax(orig_outputs, dim=1), F.softmax(outputs, dim=1), reduction='batchmean')
                loss_kl = (kl1 + kl2) / 2

                loss = loss_ce + 0.5 * loss_kl

                loss.backward()
                optimizer_retain.step()
            scheduler.step()

        net.eval()
