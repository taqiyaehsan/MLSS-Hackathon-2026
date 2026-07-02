from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        # Keep a copy of the original net for EWC estimation
        original_net = deepcopy(net).to(DEVICE)
        original_net.eval()

        # Estimate Fisher diagonal on retain_loader for EWC penalty
        net.train()
        fisher_diag = {n: torch.zeros_like(p, device=DEVICE) for n, p in net.named_parameters() if p.requires_grad}
        criterion = nn.CrossEntropyLoss()
        fisher_samples = 0
        max_fisher_batches = 20
        for i, batch in enumerate(retain_loader):
            if i >= max_fisher_batches:
                break
            if isinstance(batch, dict):
                inputs = batch["image"]
                targets = batch["age_group"]
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            net.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for n, p in net.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_diag[n] += (p.grad.detach() ** 2)
            fisher_samples += 1

        for n in fisher_diag:
            fisher_diag[n] /= fisher_samples

        # Hyperparameters
        epochs = 2  # small number of epochs for efficiency
        lr = 0.01
        ascent_lr = 0.01
        ewc_lambda = 100.0  # regularization strength
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        forget_iter = iter(forget_loader)

        for epoch in range(epochs):
            net.train()
            for retain_batch in retain_loader:
                # Process one batch from retain_loader
                if isinstance(retain_batch, dict):
                    inputs_r = retain_batch["image"]
                    targets_r = retain_batch["age_group"]
                else:
                    inputs_r, targets_r = retain_batch
                inputs_r, targets_r = inputs_r.to(DEVICE), targets_r.to(DEVICE)

                # Try to get a batch from forget_loader, if exhausted restart
                try:
                    forget_batch = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    try:
                        forget_batch = next(forget_iter)
                    except StopIteration:
                        # no forget data, skip ascent
                        forget_batch = None

                optimizer.zero_grad()

                # Forward retain batch
                outputs_r = net(inputs_r)
                loss_r = criterion(outputs_r, targets_r)

                # EWC penalty
                ewc_loss = 0
                for n, p in net.named_parameters():
                    if p.requires_grad and n in fisher_diag:
                        ewc_loss += (fisher_diag[n] * (p - original_net.state_dict()[n].to(DEVICE)).pow(2)).sum()
                ewc_loss = (ewc_lambda / 2) * ewc_loss

                total_loss = loss_r + ewc_loss

                # If forget batch available, do gradient ascent step on forget data
                if forget_batch is not None:
                    if isinstance(forget_batch, dict):
                        inputs_f = forget_batch["image"]
                        targets_f = forget_batch["age_group"]
                    else:
                        inputs_f, targets_f = forget_batch
                    inputs_f, targets_f = inputs_f.to(DEVICE), targets_f.to(DEVICE)

                    outputs_f = net(inputs_f)
                    loss_f = criterion(outputs_f, targets_f)

                    # Gradient ascent on forget loss: maximize loss_f
                    # So subtract loss_f gradient from total gradient
                    # We do this by backward on total_loss - loss_f

                    # Compute backward on total_loss - loss_f
                    (total_loss - loss_f).backward()
                else:
                    # Just backward total_loss
                    total_loss.backward()

                optimizer.step()

        net.eval()
