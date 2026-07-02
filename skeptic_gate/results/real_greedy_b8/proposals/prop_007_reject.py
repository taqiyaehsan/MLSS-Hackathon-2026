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
        base_lr = 0.005
        ascent_base_lr = 0.01
        descent_steps = 1  # gradient descent on retain
        ewc_lambda = 1000.0  # strength of EWC regularization
        distill_lambda = 0.7  # weight for distillation loss

        net.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Save original net output for distillation
        net.eval()
        with torch.no_grad():
            # Collect outputs for retain set
            retain_outputs = []
            for sample in retain_loader:
                if isinstance(sample, dict):
                    inputs = sample["image"]
                else:
                    inputs = sample[0]
                inputs = inputs.to(DEVICE)
                out = net(inputs)
                retain_outputs.append(out.detach())

        # Compute Fisher information on retain set for EWC penalty
        fisher = self._compute_fisher(net, retain_loader, criterion)
        # Save current parameters as reference
        params_backup = {n: p.clone().detach() for n, p in net.named_parameters()}

        net.train()

        # Convert forget_loader and retain_loader to list for repeatable iteration
        forget_data = list(forget_loader)
        retain_data = list(retain_loader)

        # Prepare retain outputs iterator for distillation
        retain_outputs_iter = iter(retain_outputs)

        for ep in range(epochs):
            max_batches = max(len(forget_data), len(retain_data))
            for i in range(max_batches):
                # Adaptive ascent lr with cosine annealing
                ascent_lr = ascent_base_lr * (0.5 * (1 + torch.cos(torch.tensor(ep * max_batches + i * 1.0 / max_batches * 3.14159265359))))
                ascent_lr = ascent_lr.item() if isinstance(ascent_lr, torch.Tensor) else ascent_lr

                # Gradient ascent step on forget set
                if i < len(forget_data):
                    sample = forget_data[i]
                    if isinstance(sample, dict):
                        inputs = sample["image"]
                        targets = sample["age_group"]
                    else:
                        inputs, targets = sample
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                    # Manual gradient ascent step with adjusted lr
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    # Gradient ascent: maximize loss on forget set
                    loss = -loss
                    loss.backward()

                    # Scale gradients for ascent step
                    for p in net.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(-ascent_lr / base_lr)  # negate and scale compared to descent lr
                    optimizer.step()

                # Gradient descent step on retain set with EWC penalty and distillation
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
                    loss_cls = criterion(outputs, targets)

                    # EWC penalty
                    ewc_loss = 0
                    for n, p in net.named_parameters():
                        ewc_loss += (fisher[n] * (p - params_backup[n])**2).sum()

                    # Distillation: use stored original output on retain set (batchwise)
                    # We approximate by cycling through retain_outputs
                    try:
                        orig_out = next(retain_outputs_iter).to(DEVICE)
                    except StopIteration:
                        retain_outputs_iter = iter(retain_outputs)
                        orig_out = next(retain_outputs_iter).to(DEVICE)

                    # Distill on logits with MSE
                    loss_distill = mse_loss(outputs, orig_out)

                    loss = loss_cls + ewc_lambda * ewc_loss + distill_lambda * loss_distill
                    loss.backward()
                    optimizer.step()

            scheduler.step()

        net.eval()
