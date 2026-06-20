from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def _compute_fisher(self, net, data_loader, criterion, max_batches=100):
        # Compute diagonal Fisher information matrix for EWC
        net.eval()
        fisher = {}
        for n, p in net.named_parameters():
            fisher[n] = torch.zeros_like(p, device=DEVICE)

        count = 0
        for batch_idx, sample in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            if isinstance(sample, dict):
                inputs = sample["image"]
                targets = sample["age_group"]
            else:
                inputs, targets = sample
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            net.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for n, p in net.named_parameters():
                if p.grad is not None:
                    fisher[n] += (p.grad.data ** 2) / max_batches
            count += 1

        return fisher

    def run(self, net, retain_loader, forget_loader, val_loader):
        # Parameters
        epochs = 2
        base_lr = 0.01
        ascent_steps = 3  # increase gradient ascent steps on forget set
        descent_steps = 1
        ewc_lambda = 800.0  # maintain strong EWC
        distill_lambda = 5.0  # weight for knowledge distillation loss

        net.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Step 1: Compute Fisher information on retain set for EWC penalty
        fisher = self._compute_fisher(net, retain_loader, criterion)
        params_backup = {n: p.clone().detach() for n, p in net.named_parameters()}

        # Save original net outputs on retain set for distillation
        net.eval()
        retain_outputs = []
        retain_inputs = []
        with torch.no_grad():
            for sample in retain_loader:
                if isinstance(sample, dict):
                    inputs = sample["image"]
                    targets = sample["age_group"]
                else:
                    inputs, targets = sample
                inputs = inputs.to(DEVICE)
                outputs = net(inputs)
                retain_outputs.append(outputs.detach())
                retain_inputs.append(inputs.detach())

        net.train()

        # Convert loaders to list for repeatable interleaving
        forget_data = list(forget_loader)
        retain_data = list(retain_loader)

        for ep in range(epochs):
            max_batches = max(len(forget_data), len(retain_data))
            for i in range(max_batches):
                # Adaptive learning rate for ascent steps: cosine ramp-up
                lr_scale = 0.5 * (1 + torch.cos(torch.tensor(ep / epochs * 3.1415926535)))
                ascent_lr = base_lr * (0.5 + 0.5 * lr_scale.item())
                for param_group in optimizer.param_groups:
                    param_group['lr'] = ascent_lr

                # Gradient ascent steps on forget set
                if i < len(forget_data):
                    sample = forget_data[i]
                    if isinstance(sample, dict):
                        inputs = sample["image"]
                        targets = sample["age_group"]
                    else:
                        inputs, targets = sample
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                    for _ in range(ascent_steps):
                        optimizer.zero_grad()
                        outputs = net(inputs)
                        loss = criterion(outputs, targets)
                        # Gradient ascent: maximize loss on forget set
                        loss = -loss
                        loss.backward()
                        optimizer.step()

                # Restore base_lr for descent steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = base_lr

                # Gradient descent step on retain set with EWC and distillation loss
                if i < len(retain_data):
                    sample = retain_data[i]
                    if isinstance(sample, dict):
                        inputs = sample["image"]
                        targets = sample["age_group"]
                    else:
                        inputs, targets = sample
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                    optimizer.zero_grad()
                    outputs = net(inputs)
                    ce_loss = criterion(outputs, targets)

                    # EWC penalty
                    ewc_loss = 0
                    for n, p in net.named_parameters():
                        ewc_loss += (fisher[n] * (p - params_backup[n])**2).sum()

                    # Knowledge distillation: match original outputs on retain inputs
                    # Find corresponding original outputs batch (approximate by index)
                    if i < len(retain_outputs):
                        with torch.no_grad():
                            orig_out = retain_outputs[i].to(DEVICE)
                        distill_loss = mse_loss(outputs, orig_out)
                    else:
                        distill_loss = 0

                    total_loss = ce_loss + (ewc_lambda * ewc_loss) + (distill_lambda * distill_loss)
                    total_loss.backward()
                    optimizer.step()

            scheduler.step()

        net.eval()
